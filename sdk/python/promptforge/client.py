"""
PromptForge Enterprise Client
Main SDK for managing prompts across teams
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import hashlib
from dataclasses import dataclass
from enum import Enum

from langfuse import Langfuse, observe
from .prompt import Prompt, PromptVersion
from .evaluation import EvaluationRunner
from .team import TeamManager


class PromptStatus(Enum):
    """Prompt lifecycle status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class PromptForgeClient:
    """
    Enterprise client for PromptForge prompt management system
    
    Example:
        client = PromptForgeClient(team="risk")
        prompt = client.get_prompt("portfolio_analysis")
        result = prompt.execute(portfolio_data=data)
    """
    
    def __init__(
        self,
        team: str,
        config_path: Optional[str] = None,
        environment: str = "development"
    ):
        """
        Initialize PromptForge client
        
        Args:
            team: Team identifier (risk, compliance, trading, etc.)
            config_path: Path to configuration directory
            environment: Environment (development, staging, production)
        """
        self.team = team
        self.environment = environment
        self.config_path = Path(config_path or "prompts")
        
        # Initialize components
        self.langfuse = Langfuse()
        self.team_manager = TeamManager(self.config_path / "_registry" / "teams.yaml")
        self.evaluation_runner = EvaluationRunner()
        
        # Load registry
        self._load_registry()
        
        # Validate team permissions
        self._validate_team_access()
    
    def _load_registry(self):
        """Load prompt registry"""
        registry_path = self.config_path / "_registry" / "index.json"
        if registry_path.exists():
            with open(registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"prompts": {}}
    
    def _validate_team_access(self):
        """Validate team has appropriate access"""
        if not self.team_manager.team_exists(self.team):
            raise ValueError(f"Team '{self.team}' not found in configuration")
    
    @observe(name="create_prompt")
    def create_prompt(
        self,
        name: str,
        template: str,
        description: str,
        template_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Prompt:
        """
        Create a new prompt for the team
        
        Args:
            name: Prompt name (alphanumeric with underscores)
            template: Prompt template content or template name
            description: Human-readable description
            template_type: Template to use (e.g., 'chain_of_thought')
            tags: Optional tags for categorization
        
        Returns:
            Created Prompt object
        """
        # Check permissions
        if not self.team_manager.has_permission(self.team, "create_prompt"):
            raise PermissionError(f"Team '{self.team}' cannot create prompts")
        
        # Create prompt directory structure
        prompt_path = self.config_path / self.team / name
        prompt_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (prompt_path / "versions").mkdir(exist_ok=True)
        (prompt_path / "tests").mkdir(exist_ok=True)
        (prompt_path / "evaluations").mkdir(exist_ok=True)
        
        # Load template if specified
        if template_type:
            template = self._load_template(template_type)
        
        # Create prompt configuration
        prompt_config = {
            "metadata": {
                "name": name,
                "team": self.team,
                "description": description,
                "created": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "status": PromptStatus.DEVELOPMENT.value,
                "tags": tags or []
            },
            "prompt": {
                "template": template,
                "variables": {},
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "model": "gpt-4"
                }
            },
            "evaluation": {
                "metrics": self.team_manager.get_team_evaluation_overrides(self.team),
                "test_suite": "tests/",
                "custom_rules": f"evaluations/{self.team}_rules.py"
            }
        }
        
        # Save prompt configuration
        config_file = prompt_path / "prompt.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(prompt_config, f, default_flow_style=False)
        
        # Update registry
        self._update_registry(f"{self.team}/{name}", prompt_config["metadata"])
        
        # Create Prompt object
        return Prompt(
            name=name,
            team=self.team,
            path=prompt_path,
            config=prompt_config
        )
    
    @observe(name="get_prompt")
    def get_prompt(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Prompt:
        """
        Get a prompt by name
        
        Args:
            name: Prompt name
            version: Specific version (default: latest)
        
        Returns:
            Prompt object
        """
        prompt_path = self.config_path / self.team / name
        
        if not prompt_path.exists():
            # Check if prompt exists in other teams
            for team_prompt in self.registry.get("prompts", {}).keys():
                if team_prompt.endswith(f"/{name}"):
                    team = team_prompt.split("/")[0]
                    if self.team_manager.has_permission(self.team, f"access_{team}_prompts"):
                        prompt_path = self.config_path / team / name
                        break
            else:
                raise ValueError(f"Prompt '{name}' not found")
        
        # Load prompt configuration
        config_file = prompt_path / "prompt.yaml"
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Load specific version if requested
        if version:
            version_path = prompt_path / "versions" / version
            if version_path.exists():
                with open(version_path / "prompt.yaml") as f:
                    config = yaml.safe_load(f)
        
        return Prompt(
            name=name,
            team=config["metadata"]["team"],
            path=prompt_path,
            config=config
        )
    
    @observe(name="test_prompt")
    def test_prompt(
        self,
        prompt: Prompt,
        test_data: Optional[Dict[str, Any]] = None,
        test_suite: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run tests for a prompt
        
        Args:
            prompt: Prompt to test
            test_data: Optional test data
            test_suite: Specific test suite to run
        
        Returns:
            Test results
        """
        # Run evaluation
        results = self.evaluation_runner.run(
            prompt=prompt,
            test_data=test_data,
            test_suite=test_suite
        )
        
        # Check against team thresholds
        thresholds = self.team_manager.get_team_evaluation_overrides(self.team)
        
        passed = all(
            results.get(metric, 0) >= threshold
            for metric, threshold in thresholds.items()
            if metric.endswith("_threshold")
        )
        
        # Log to Langfuse
        self.langfuse.score(
            name=f"test_{prompt.name}",
            value=1.0 if passed else 0.0,
            metadata=results
        )
        
        return {
            "passed": passed,
            "results": results,
            "thresholds": thresholds
        }
    
    @observe(name="deploy_prompt")
    def deploy_prompt(
        self,
        prompt: Prompt,
        environment: str,
        require_approval: bool = True
    ) -> Dict[str, Any]:
        """
        Deploy a prompt to an environment
        
        Args:
            prompt: Prompt to deploy
            environment: Target environment (staging, production)
            require_approval: Whether to require approval
        
        Returns:
            Deployment status
        """
        # Check permissions
        permission_needed = f"deploy_{environment}"
        if not self.team_manager.has_permission(self.team, permission_needed):
            if require_approval:
                return self._request_approval(prompt, environment)
            else:
                raise PermissionError(f"Team '{self.team}' cannot deploy to {environment}")
        
        # Run pre-deployment validation
        test_results = self.test_prompt(prompt)
        if not test_results["passed"]:
            raise ValueError(f"Prompt failed validation: {test_results['results']}")
        
        # Update prompt status
        prompt.config["metadata"]["status"] = environment
        prompt.save()
        
        # Update registry
        registry_key = f"{prompt.team}/{prompt.name}"
        if registry_key in self.registry["prompts"]:
            self.registry["prompts"][registry_key]["status"] = environment
            self._save_registry()
        
        # Create deployment record
        deployment = {
            "prompt": prompt.name,
            "team": prompt.team,
            "version": prompt.config["metadata"]["version"],
            "environment": environment,
            "timestamp": datetime.utcnow().isoformat(),
            "deployed_by": self.team,
            "test_results": test_results["results"]
        }
        
        # Log deployment
        self.langfuse.trace(
            name="deployment",
            metadata=deployment
        )
        
        return deployment
    
    @observe(name="compare_versions")
    def compare_versions(
        self,
        prompt_name: str,
        version1: str,
        version2: str,
        test_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare two versions of a prompt
        
        Args:
            prompt_name: Name of the prompt
            version1: First version
            version2: Second version
            test_data: Test data for comparison
        
        Returns:
            Comparison results
        """
        # Load both versions
        prompt1 = self.get_prompt(prompt_name, version1)
        prompt2 = self.get_prompt(prompt_name, version2)
        
        # Run evaluations
        results1 = self.test_prompt(prompt1, test_data)
        results2 = self.test_prompt(prompt2, test_data)
        
        # Compare metrics
        comparison = {
            "version1": {
                "version": version1,
                "results": results1["results"],
                "passed": results1["passed"]
            },
            "version2": {
                "version": version2,
                "results": results2["results"],
                "passed": results2["passed"]
            },
            "improvements": {},
            "regressions": {}
        }
        
        # Calculate improvements and regressions
        for metric in results1["results"]:
            if metric in results2["results"]:
                diff = results2["results"][metric] - results1["results"][metric]
                if diff > 0:
                    comparison["improvements"][metric] = diff
                elif diff < 0:
                    comparison["regressions"][metric] = abs(diff)
        
        return comparison
    
    def list_prompts(
        self,
        team: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available prompts
        
        Args:
            team: Filter by team
            status: Filter by status
            tags: Filter by tags
        
        Returns:
            List of prompt metadata
        """
        prompts = []
        
        for prompt_key, prompt_data in self.registry.get("prompts", {}).items():
            # Apply filters
            if team and not prompt_key.startswith(f"{team}/"):
                continue
            
            if status and prompt_data.get("status") != status:
                continue
            
            if tags:
                prompt_tags = prompt_data.get("tags", [])
                if not any(tag in prompt_tags for tag in tags):
                    continue
            
            # Check access permissions
            prompt_team = prompt_key.split("/")[0]
            if prompt_team != self.team:
                if not self.team_manager.has_permission(self.team, f"access_{prompt_team}_prompts"):
                    continue
            
            prompts.append(prompt_data)
        
        return prompts
    
    def _load_template(self, template_type: str) -> str:
        """Load a prompt template"""
        template_path = self.config_path / "_templates" / f"{template_type}.jinja2"
        
        if not template_path.exists():
            # Try base templates
            template_path = self.config_path / "_templates" / "base" / f"{template_type}.jinja2"
        
        if template_path.exists():
            with open(template_path) as f:
                return f.read()
        else:
            raise ValueError(f"Template '{template_type}' not found")
    
    def _update_registry(self, prompt_key: str, metadata: Dict[str, Any]):
        """Update the prompt registry"""
        self.registry["prompts"][prompt_key] = {
            "name": metadata["name"],
            "team": metadata["team"],
            "current_version": metadata["version"],
            "status": metadata["status"],
            "created": metadata["created"],
            "updated": datetime.utcnow().isoformat(),
            "tags": metadata.get("tags", []),
            "dependencies": [],
            "metrics": {}
        }
        
        self._save_registry()
    
    def _save_registry(self):
        """Save the registry to disk"""
        registry_path = self.config_path / "_registry" / "index.json"
        with open(registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _request_approval(
        self,
        prompt: Prompt,
        environment: str
    ) -> Dict[str, Any]:
        """Request approval for deployment"""
        approval_request = {
            "prompt": prompt.name,
            "team": prompt.team,
            "requested_by": self.team,
            "environment": environment,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        # In a real implementation, this would create a ticket
        # and notify approvers via Slack/email
        
        return {
            "status": "approval_required",
            "request": approval_request,
            "message": f"Approval required for deploying {prompt.name} to {environment}"
        }


class PromptForgeAdmin(PromptForgeClient):
    """Administrative client with elevated permissions"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize admin client"""
        super().__init__(team="platform", config_path=config_path)
    
    def approve_deployment(
        self,
        approval_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Approve a deployment request"""
        # Implementation for approval workflow
        pass
    
    def rollback_prompt(
        self,
        prompt_name: str,
        team: str,
        version: str
    ) -> Dict[str, Any]:
        """Rollback a prompt to a previous version"""
        # Implementation for rollback
        pass
    
    def generate_analytics_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate analytics report across all teams"""
        # Implementation for analytics
        pass