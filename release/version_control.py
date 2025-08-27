"""
Version Control and Rollback Management for Prompt Engineering
Supports blue-green deployments and feature flag rollouts
"""

import os
import json
import shutil
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptVersion:
    """Represents a versioned prompt artifact"""
    
    def __init__(self, version: str, prompt_template: str, spec: Dict, 
                 guardrails: Dict, datasets: Dict[str, Any]):
        self.version = version
        self.prompt_template = prompt_template
        self.spec = spec
        self.guardrails = guardrails
        self.datasets = datasets
        self.created_at = datetime.utcnow()
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for version integrity"""
        content = f"{self.prompt_template}{json.dumps(self.spec, sort_keys=True)}{json.dumps(self.guardrails, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary"""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "spec": self.spec,
            "guardrails": self.guardrails
        }

class VersionManager:
    """Manages prompt versions, deployments, and rollbacks"""
    
    def __init__(self, base_path: str = "release"):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / "versions"
        self.deployments_path = self.base_path / "deployments"
        
        # Create directories
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.deployments_path.mkdir(parents=True, exist_ok=True)
        
        self.current_version = self._get_current_version()
    
    def create_version(self, version: str, description: str = "") -> PromptVersion:
        """Create a new version from current prompt artifacts"""
        logger.info(f"Creating version {version}")
        
        # Load current artifacts
        prompt_template = self._load_prompt_template()
        spec = self._load_spec()
        guardrails = self._load_guardrails_config()
        datasets = self._load_datasets()
        
        # Create version object
        prompt_version = PromptVersion(
            version=version,
            prompt_template=prompt_template,
            spec=spec,
            guardrails=guardrails,
            datasets=datasets
        )
        
        # Save version artifacts
        version_dir = self.versions_path / version
        version_dir.mkdir(exist_ok=True)
        
        # Save prompt template
        with open(version_dir / "template.txt", "w") as f:
            f.write(prompt_template)
        
        # Save spec
        with open(version_dir / "spec.json", "w") as f:
            json.dump(spec, f, indent=2)
        
        # Save guardrails
        with open(version_dir / "guardrails.json", "w") as f:
            json.dump(guardrails, f, indent=2)
        
        # Save datasets
        datasets_dir = version_dir / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        for name, data in datasets.items():
            with open(datasets_dir / f"{name}.json", "w") as f:
                json.dump(data, f, indent=2)
        
        # Save version metadata
        metadata = {
            "version": version,
            "description": description,
            "created_at": prompt_version.created_at.isoformat(),
            "checksum": prompt_version.checksum,
            "files": {
                "template": "template.txt",
                "spec": "spec.json",
                "guardrails": "guardrails.json",
                "datasets": list(datasets.keys())
            }
        }
        
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Version {version} created successfully")
        return prompt_version
    
    def deploy_version(self, version: str, environment: str, 
                      deployment_type: str = "blue_green") -> Dict[str, Any]:
        """Deploy a specific version to an environment"""
        logger.info(f"Deploying version {version} to {environment}")
        
        # Validate version exists
        if not self._version_exists(version):
            raise ValueError(f"Version {version} not found")
        
        # Create deployment record
        deployment = {
            "deployment_id": f"dep_{int(datetime.utcnow().timestamp())}",
            "version": version,
            "environment": environment,
            "deployment_type": deployment_type,
            "deployed_at": datetime.utcnow().isoformat(),
            "status": "deploying",
            "previous_version": self.get_active_version(environment)
        }
        
        try:
            # Perform deployment based on type
            if deployment_type == "blue_green":
                self._blue_green_deployment(version, environment)
            elif deployment_type == "rolling":
                self._rolling_deployment(version, environment)
            elif deployment_type == "feature_flag":
                self._feature_flag_deployment(version, environment)
            else:
                raise ValueError(f"Unknown deployment type: {deployment_type}")
            
            deployment["status"] = "deployed"
            deployment["completed_at"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            deployment["status"] = "failed"
            deployment["error"] = str(e)
            logger.error(f"Deployment failed: {e}")
            raise
        
        # Save deployment record
        deployment_file = self.deployments_path / f"{deployment['deployment_id']}.json"
        with open(deployment_file, "w") as f:
            json.dump(deployment, f, indent=2)
        
        # Update active version
        self._set_active_version(environment, version)
        
        logger.info(f"Deployment {deployment['deployment_id']} completed")
        return deployment
    
    def rollback(self, environment: str, target_version: str = None) -> Dict[str, Any]:
        """Rollback to a previous version"""
        current_version = self.get_active_version(environment)
        
        if target_version is None:
            # Find previous version from deployment history
            target_version = self._get_previous_version(environment)
        
        if not target_version:
            raise ValueError("No previous version found for rollback")
        
        logger.info(f"Rolling back {environment} from {current_version} to {target_version}")
        
        # Create rollback deployment
        deployment = self.deploy_version(target_version, environment, "blue_green")
        deployment["rollback"] = True
        deployment["rolled_back_from"] = current_version
        
        return deployment
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available versions"""
        versions = []
        
        for version_dir in self.versions_path.iterdir():
            if version_dir.is_dir():
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        versions.append(metadata)
        
        # Sort by creation date
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions
    
    def get_version_details(self, version: str) -> Dict[str, Any]:
        """Get detailed information about a specific version"""
        version_dir = self.versions_path / version
        
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")
        
        # Load metadata
        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        # Load artifacts
        with open(version_dir / "template.txt") as f:
            template = f.read()
        
        with open(version_dir / "spec.json") as f:
            spec = json.load(f)
        
        # Add deployment history
        deployments = self._get_version_deployments(version)
        
        return {
            **metadata,
            "template_preview": template[:200] + "..." if len(template) > 200 else template,
            "spec_summary": {
                "goal": spec.get("goal", {}).get("description", ""),
                "inputs": len(spec.get("inputs", [])),
                "outputs": spec.get("outputs", {}).get("schema", {}).get("type", "")
            },
            "deployments": deployments
        }
    
    def get_active_version(self, environment: str) -> Optional[str]:
        """Get currently active version for an environment"""
        active_file = self.deployments_path / f"{environment}_active.txt"
        
        if active_file.exists():
            return active_file.read_text().strip()
        
        return None
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions and highlight differences"""
        v1_details = self.get_version_details(version1)
        v2_details = self.get_version_details(version2)
        
        differences = {
            "metadata_changes": {},
            "template_changed": False,
            "spec_changes": {},
            "summary": ""
        }
        
        # Compare templates
        v1_template = self._load_version_template(version1)
        v2_template = self._load_version_template(version2)
        differences["template_changed"] = v1_template != v2_template
        
        # Compare specs (simplified)
        v1_spec = self._load_version_spec(version1)
        v2_spec = self._load_version_spec(version2)
        
        # Check key spec changes
        if v1_spec.get("goal") != v2_spec.get("goal"):
            differences["spec_changes"]["goal"] = "modified"
        
        if len(v1_spec.get("inputs", [])) != len(v2_spec.get("inputs", [])):
            differences["spec_changes"]["inputs"] = "count_changed"
        
        # Generate summary
        changes = []
        if differences["template_changed"]:
            changes.append("prompt template")
        if differences["spec_changes"]:
            changes.append("specification")
        
        differences["summary"] = f"Changes in: {', '.join(changes)}" if changes else "No significant changes"
        
        return differences
    
    def _version_exists(self, version: str) -> bool:
        """Check if version exists"""
        return (self.versions_path / version).exists()
    
    def _blue_green_deployment(self, version: str, environment: str):
        """Perform blue-green deployment"""
        logger.info(f"Performing blue-green deployment of {version} to {environment}")
        
        # In a real implementation, this would:
        # 1. Start new instances with the new version (green)
        # 2. Run health checks
        # 3. Switch traffic from old instances (blue) to new ones
        # 4. Terminate old instances
        
        # For now, simulate the deployment
        import time
        time.sleep(1)  # Simulate deployment time
        
        logger.info("Blue-green deployment completed")
    
    def _rolling_deployment(self, version: str, environment: str):
        """Perform rolling deployment"""
        logger.info(f"Performing rolling deployment of {version} to {environment}")
        
        # Simulate rolling deployment
        import time
        time.sleep(0.5)
        
        logger.info("Rolling deployment completed")
    
    def _feature_flag_deployment(self, version: str, environment: str):
        """Perform feature flag based deployment"""
        logger.info(f"Performing feature flag deployment of {version} to {environment}")
        
        # In real implementation, this would use feature flags to gradually
        # roll out the new version to a percentage of users
        
        import time
        time.sleep(0.3)
        
        logger.info("Feature flag deployment completed")
    
    def _set_active_version(self, environment: str, version: str):
        """Set active version for environment"""
        active_file = self.deployments_path / f"{environment}_active.txt"
        active_file.write_text(version)
    
    def _get_previous_version(self, environment: str) -> Optional[str]:
        """Get previous version from deployment history"""
        deployments = []
        
        for deployment_file in self.deployments_path.glob("dep_*.json"):
            with open(deployment_file) as f:
                deployment = json.load(f)
                if deployment["environment"] == environment and deployment["status"] == "deployed":
                    deployments.append(deployment)
        
        # Sort by deployment time
        deployments.sort(key=lambda x: x["deployed_at"], reverse=True)
        
        # Return second most recent (skip current)
        if len(deployments) >= 2:
            return deployments[1]["version"]
        
        return None
    
    def _get_version_deployments(self, version: str) -> List[Dict[str, Any]]:
        """Get deployment history for a version"""
        deployments = []
        
        for deployment_file in self.deployments_path.glob("dep_*.json"):
            with open(deployment_file) as f:
                deployment = json.load(f)
                if deployment["version"] == version:
                    deployments.append(deployment)
        
        return sorted(deployments, key=lambda x: x["deployed_at"], reverse=True)
    
    def _load_prompt_template(self) -> str:
        """Load current prompt template"""
        template_path = Path("prompts/find_capital/template.txt")
        if template_path.exists():
            return template_path.read_text()
        return ""
    
    def _load_spec(self) -> Dict:
        """Load current specification"""
        import yaml
        spec_path = Path("prompts/find_capital/spec.yml")
        if spec_path.exists():
            with open(spec_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_guardrails_config(self) -> Dict:
        """Load current guardrails configuration"""
        config_path = Path("config/governance.yml")
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_datasets(self) -> Dict[str, Any]:
        """Load current datasets"""
        import pandas as pd
        datasets = {}
        
        dataset_files = ["golden.csv", "edge_cases.csv", "adversarial.csv"]
        for filename in dataset_files:
            dataset_path = Path(f"datasets/{filename}")
            if dataset_path.exists():
                df = pd.read_csv(dataset_path)
                datasets[filename.replace(".csv", "")] = df.to_dict("records")
        
        return datasets
    
    def _load_version_template(self, version: str) -> str:
        """Load template for specific version"""
        template_path = self.versions_path / version / "template.txt"
        if template_path.exists():
            return template_path.read_text()
        return ""
    
    def _load_version_spec(self, version: str) -> Dict:
        """Load spec for specific version"""
        spec_path = self.versions_path / version / "spec.json"
        if spec_path.exists():
            with open(spec_path) as f:
                return json.load(f)
        return {}
    
    def _get_current_version(self) -> Optional[str]:
        """Get current version from development environment"""
        return self.get_active_version("development")

class DeploymentOrchestrator:
    """Orchestrates deployments across environments with safety checks"""
    
    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager
        self.environments = ["development", "staging", "production"]
    
    def promote_version(self, version: str, target_env: str) -> Dict[str, Any]:
        """Promote version through environments with safety checks"""
        logger.info(f"Promoting version {version} to {target_env}")
        
        # Check promotion path
        if not self._can_promote_to(version, target_env):
            raise ValueError(f"Version {version} cannot be promoted to {target_env}")
        
        # Run pre-deployment checks
        checks = self._run_pre_deployment_checks(version, target_env)
        
        if not checks["passed"]:
            raise ValueError(f"Pre-deployment checks failed: {checks['failures']}")
        
        # Deploy version
        deployment = self.version_manager.deploy_version(version, target_env)
        
        # Run post-deployment validation
        validation = self._run_post_deployment_validation(version, target_env)
        
        if not validation["passed"]:
            logger.warning(f"Post-deployment validation failed: {validation['failures']}")
            # Auto-rollback if critical validations fail
            if validation["critical_failures"]:
                logger.info("Initiating auto-rollback due to critical failures")
                self.version_manager.rollback(target_env)
        
        return {
            **deployment,
            "pre_checks": checks,
            "post_validation": validation
        }
    
    def _can_promote_to(self, version: str, target_env: str) -> bool:
        """Check if version can be promoted to target environment"""
        # Define promotion path
        promotion_path = {
            "staging": ["development"],
            "production": ["staging"]
        }
        
        if target_env == "development":
            return True  # Can always deploy to dev
        
        required_envs = promotion_path.get(target_env, [])
        
        for env in required_envs:
            if self.version_manager.get_active_version(env) != version:
                return False
        
        return True
    
    def _run_pre_deployment_checks(self, version: str, environment: str) -> Dict[str, Any]:
        """Run pre-deployment safety checks"""
        checks = {
            "version_exists": self.version_manager._version_exists(version),
            "evaluation_passed": True,  # Would run evaluation suite
            "security_scan": True,      # Would run security scan
            "performance_test": True    # Would run performance tests
        }
        
        passed = all(checks.values())
        failures = [k for k, v in checks.items() if not v]
        
        return {
            "passed": passed,
            "failures": failures,
            "details": checks
        }
    
    def _run_post_deployment_validation(self, version: str, environment: str) -> Dict[str, Any]:
        """Run post-deployment validation"""
        # Simulate validation checks
        checks = {
            "health_check": True,       # API health check
            "smoke_test": True,         # Basic functionality test
            "performance_check": True,  # Response time check
            "error_rate_check": True    # Error rate within limits
        }
        
        critical_checks = ["health_check", "smoke_test"]
        
        passed = all(checks.values())
        failures = [k for k, v in checks.items() if not v]
        critical_failures = [k for k in failures if k in critical_checks]
        
        return {
            "passed": passed,
            "failures": failures,
            "critical_failures": critical_failures,
            "details": checks
        }

# Command-line interface for version management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Version Management")
    parser.add_argument("command", choices=["create", "deploy", "rollback", "list", "compare"])
    parser.add_argument("--version", help="Version identifier")
    parser.add_argument("--environment", help="Target environment")
    parser.add_argument("--description", help="Version description")
    parser.add_argument("--compare-with", help="Version to compare with")
    
    args = parser.parse_args()
    
    vm = VersionManager()
    
    if args.command == "create":
        if not args.version:
            args.version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version = vm.create_version(args.version, args.description or "")
        print(f"Created version {args.version}")
    
    elif args.command == "deploy":
        if not args.version or not args.environment:
            print("Error: --version and --environment required for deploy")
            exit(1)
        
        deployment = vm.deploy_version(args.version, args.environment)
        print(f"Deployed version {args.version} to {args.environment}")
        print(f"Deployment ID: {deployment['deployment_id']}")
    
    elif args.command == "rollback":
        if not args.environment:
            print("Error: --environment required for rollback")
            exit(1)
        
        rollback = vm.rollback(args.environment, args.version)
        print(f"Rolled back {args.environment} to version {rollback['version']}")
    
    elif args.command == "list":
        versions = vm.list_versions()
        print("\nAvailable versions:")
        for v in versions:
            print(f"  {v['version']} - {v['created_at']} - {v.get('description', '')}")
    
    elif args.command == "compare":
        if not args.version or not args.compare_with:
            print("Error: --version and --compare-with required for compare")
            exit(1)
        
        diff = vm.compare_versions(args.version, args.compare_with)
        print(f"\nComparison between {args.version} and {args.compare_with}:")
        print(f"Summary: {diff['summary']}")
        if diff['template_changed']:
            print("  - Prompt template changed")
        for change in diff['spec_changes']:
            print(f"  - Spec {change}: {diff['spec_changes'][change]}")