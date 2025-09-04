# PromptForge Enhancement Plan

## Executive Summary
Comprehensive enhancement plan to transform PromptForge into a production-ready, multi-prompt evaluation framework with enterprise-grade testing, CI/CD, and monitoring capabilities.

## Current State Analysis

### ✅ Existing Strengths
1. **Temperature Control**: Already supports temperature=0 for deterministic outputs (spec.yml line 65)
2. **LLM Client Abstraction**: Vendor-neutral `LLMClient` with OpenAI, Anthropic, HuggingFace support
3. **Dataset Support**: Golden, edge, and adversarial datasets in place
4. **DeepEval Integration**: Basic integration with metrics (GEval, AnswerRelevancy, Faithfulness)
5. **TruLens Integration**: Comprehensive PII monitoring and evaluation
6. **CI/CD Pipeline**: GitHub Actions workflow for PII validation
7. **Presidio Integration**: Advanced PII detection and masking

### ⚠️ Gaps to Address
1. **Limited Multi-Prompt Support**: Only `find_capital` prompt exists
2. **Missing Mock LLM**: No mock client for unit tests
3. **Incomplete Metrics**: Missing exact-match, stability/variance, latency tracking
4. **No Versioning System**: No semantic versioning for prompts
5. **Single-Repo Structure**: Not designed for multi-repo/modular architecture
6. **Limited CI Steps**: Missing lint, unit test, and comprehensive reporting stages

## Enhancement Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Enhanced LLM Client Wrapper
```python
# orchestration/llm_client_enhanced.py
class MockLLMProvider(LLMProvider):
    """Mock LLM for deterministic testing"""
    def __init__(self, responses_file: str = "tests/mock_responses.json"):
        self.responses = self._load_responses(responses_file)
        self.temperature = 0  # Always deterministic
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Return deterministic response based on prompt hash
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self.responses.get(prompt_hash, self._default_response())

class LLMClientEnhanced(LLMClient):
    """Enhanced client with temperature control and mock support"""
    def __init__(self, provider: str = "openai", **kwargs):
        super().__init__(provider, **kwargs)
        self.temperature = kwargs.get("temperature", 0)  # Default to deterministic
        
    def set_temperature(self, temperature: float):
        """Dynamic temperature control"""
        self.temperature = max(0, min(2, temperature))
```

#### 1.2 Multi-Prompt Repository Structure
```
promptforge/
├── prompts/
│   ├── summarization/
│   │   ├── spec.yml
│   │   ├── template.jinja2
│   │   ├── output_schema.json
│   │   ├── version.json
│   │   └── README.md
│   ├── financial_planning/
│   │   ├── spec.yml
│   │   ├── template.jinja2
│   │   ├── output_schema.json
│   │   └── version.json
│   └── fraud_detection/
│       ├── spec.yml
│       ├── template.jinja2
│       └── output_schema.json
├── datasets/
│   ├── shared/
│   │   ├── pii_adversarial.csv
│   │   ├── toxicity_adversarial.csv
│   │   └── compliance_cases.csv
│   ├── summarization/
│   │   ├── golden.csv
│   │   ├── edge.csv
│   │   └── adversarial.csv
│   └── financial_planning/
│       ├── golden.csv
│       └── edge.csv
```

### Phase 2: Enhanced Evaluation Metrics (Week 2-3)

#### 2.1 Comprehensive Metrics Implementation
```python
# evaluation/metrics_enhanced.py
from deepeval.metrics import BaseMetric

class ExactMatchMetric(BaseMetric):
    """Exact match for deterministic outputs"""
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
    
    def measure(self, test_case) -> float:
        return 1.0 if test_case.actual_output == test_case.expected_output else 0.0

class StabilityMetric(BaseMetric):
    """N-run consistency check"""
    def __init__(self, n_runs: int = 5, temperature: float = 0):
        self.n_runs = n_runs
        self.temperature = temperature
    
    def measure(self, test_case) -> float:
        outputs = []
        for _ in range(self.n_runs):
            output = self.llm_client.generate(
                test_case.input, 
                temperature=self.temperature
            )
            outputs.append(output)
        
        # Calculate variance
        unique_outputs = len(set(outputs))
        return 1.0 - (unique_outputs - 1) / self.n_runs

class SchemaValidityMetric(BaseMetric):
    """JSON schema validation"""
    def __init__(self, schema_path: str):
        self.schema = self._load_schema(schema_path)
    
    def measure(self, test_case) -> float:
        try:
            output = json.loads(test_case.actual_output)
            jsonschema.validate(output, self.schema)
            return 1.0
        except:
            return 0.0
```

#### 2.2 DeepEval Test Suite Enhancement
```python
# evals/deepeval_suite.py
import deepeval
from deepeval import assert_test
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    LatencyMetric
)

class PromptEvaluationSuite:
    """Comprehensive evaluation suite using DeepEval"""
    
    METRICS = {
        'exact_match': ExactMatchMetric(threshold=0.95),
        'groundedness': GEval(
            name="Groundedness",
            criteria="Response is grounded in facts",
            threshold=0.85
        ),
        'faithfulness': FaithfulnessMetric(threshold=0.85),
        'schema_validity': SchemaValidityMetric("output_schema.json"),
        'toxicity': ToxicityMetric(threshold=0.0),
        'pii_leakage': PIILeakageMetric(threshold=0.0),
        'stability': StabilityMetric(n_runs=5),
        'latency': LatencyMetric(max_latency_ms=2000)
    }
    
    def run_comprehensive_evaluation(self, prompt_name: str):
        """Run all metrics for a prompt"""
        results = {}
        
        # Load datasets
        datasets = {
            'golden': f"datasets/{prompt_name}/golden.csv",
            'edge': f"datasets/{prompt_name}/edge.csv",
            'adversarial': f"datasets/{prompt_name}/adversarial.csv"
        }
        
        for dataset_type, dataset_path in datasets.items():
            test_cases = self.load_test_cases(dataset_path)
            
            for metric_name, metric in self.METRICS.items():
                scores = []
                for test_case in test_cases:
                    score = metric.measure(test_case)
                    scores.append(score)
                
                results[f"{dataset_type}_{metric_name}"] = {
                    'mean': np.mean(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'pass': np.mean(scores) >= metric.threshold
                }
        
        return results
```

### Phase 3: CI/CD Pipeline Enhancement (Week 3-4)

#### 3.1 Comprehensive GitHub Actions Workflow
```yaml
# .github/workflows/prompt_validation.yml
name: Prompt Validation Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    name: Lint Prompts & Schemas
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Lint YAML files
        run: |
          pip install yamllint
          yamllint prompts/**/*.yml
      
      - name: Validate JSON schemas
        run: |
          pip install jsonschema
          python scripts/validate_schemas.py

  unit-tests:
    name: Unit Tests (Fast)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run unit tests with mock LLM
        run: |
          export USE_MOCK_LLM=true
          pytest tests/unit/ -v --cov=promptforge --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  prompt-tests:
    name: Prompt Evaluation (DeepEval)
    runs-on: ubuntu-latest
    strategy:
      matrix:
        prompt: [summarization, financial_planning, fraud_detection]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install deepeval
      
      - name: Run DeepEval tests for ${{ matrix.prompt }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          deepeval test evals/${{ matrix.prompt }}/deepeval_tests.py \
            --metrics exact_match,groundedness,schema_validity \
            --threshold 0.95

  adversarial-suite:
    name: Adversarial Testing
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run PII detection tests
        run: |
          python scripts/test_pii_leakage.py \
            --dataset datasets/shared/pii_adversarial.csv
      
      - name: Run jailbreak detection
        run: |
          python scripts/test_jailbreak.py \
            --dataset datasets/shared/toxicity_adversarial.csv

  generate-report:
    name: Generate Evaluation Report
    needs: [lint, unit-tests, prompt-tests, adversarial-suite]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Generate HTML report
        run: |
          python scripts/generate_report.py \
            --output reports/evaluation_report.html
      
      - name: Upload report artifact
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-report
          path: reports/evaluation_report.html
      
      - name: Check thresholds
        run: |
          python scripts/check_thresholds.py \
            --report reports/evaluation_report.html \
            --fail-on-breach
```

### Phase 4: Prompt Versioning & Management (Week 4)

#### 4.1 Semantic Versioning System
```python
# prompts/version_manager.py
class PromptVersionManager:
    """Manage prompt versions with semantic versioning"""
    
    def __init__(self, prompt_name: str):
        self.prompt_name = prompt_name
        self.version_file = f"prompts/{prompt_name}/version.json"
        self.current_version = self._load_version()
    
    def _load_version(self) -> str:
        """Load current version from file"""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                data = json.load(f)
                return data.get('version', '1.0.0')
        return '1.0.0'
    
    def bump_version(self, bump_type: str = 'patch'):
        """Bump version (major, minor, patch)"""
        major, minor, patch = map(int, self.current_version.split('.'))
        
        if bump_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif bump_type == 'minor':
            minor += 1
            patch = 0
        else:
            patch += 1
        
        new_version = f"{major}.{minor}.{patch}"
        
        # Save new version
        with open(self.version_file, 'w') as f:
            json.dump({
                'version': new_version,
                'updated_at': datetime.now().isoformat(),
                'updated_by': os.getenv('USER', 'unknown')
            }, f, indent=2)
        
        return new_version
    
    def tag_dataset_snapshot(self):
        """Create dataset snapshot with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d')
        dataset_dir = f"datasets/{self.prompt_name}"
        snapshot_dir = f"datasets/snapshots/{self.prompt_name}@{timestamp}"
        
        shutil.copytree(dataset_dir, snapshot_dir)
        return snapshot_dir
```

### Phase 5: Multi-Repo Support (Week 5)

#### 5.1 Repository Structure for Scaling
```yaml
# .promptforge.yml - Configuration for multi-repo setup
repos:
  - name: core-prompts
    url: https://github.com/oiiro/promptforge-core
    prompts:
      - summarization
      - translation
      - classification
  
  - name: financial-prompts
    url: https://github.com/oiiro/promptforge-financial
    prompts:
      - financial_planning
      - risk_assessment
      - fraud_detection
  
  - name: compliance-prompts
    url: https://github.com/oiiro/promptforge-compliance
    prompts:
      - pii_detection
      - regulatory_check

shared_datasets:
  path: datasets/shared/
  sync: true
  
evaluation:
  runner: deepeval
  parallel: true
  max_workers: 5
```

#### 5.2 Sync Script for Multi-Repo
```python
# scripts/sync_repos.py
class MultiRepoManager:
    """Manage multiple prompt repositories"""
    
    def __init__(self, config_file: str = ".promptforge.yml"):
        self.config = self._load_config(config_file)
    
    def sync_all_repos(self):
        """Sync all configured repositories"""
        for repo in self.config['repos']:
            self._sync_repo(repo)
        
        # Sync shared datasets
        if self.config.get('shared_datasets', {}).get('sync'):
            self._sync_shared_datasets()
    
    def run_cross_repo_evaluation(self):
        """Run evaluation across all repos"""
        results = {}
        
        for repo in self.config['repos']:
            for prompt in repo['prompts']:
                evaluator = PromptEvaluationSuite()
                results[prompt] = evaluator.run_comprehensive_evaluation(prompt)
        
        return results
```

## Implementation Timeline

### Week 1-2: Core Infrastructure
- [ ] Implement MockLLMProvider
- [ ] Enhance LLMClient with temperature control
- [ ] Create multi-prompt directory structure
- [ ] Setup initial prompts (summarization, financial_planning, fraud_detection)

### Week 2-3: Enhanced Metrics
- [ ] Implement ExactMatchMetric
- [ ] Implement StabilityMetric
- [ ] Implement SchemaValidityMetric
- [ ] Integrate all metrics with DeepEval
- [ ] Create comprehensive test suites

### Week 3-4: CI/CD Enhancement
- [ ] Setup lint stage
- [ ] Implement unit tests with mock LLM
- [ ] Configure DeepEval in CI
- [ ] Setup adversarial testing
- [ ] Implement HTML report generation
- [ ] Add threshold checking

### Week 4: Versioning System
- [ ] Implement PromptVersionManager
- [ ] Add semantic versioning to all prompts
- [ ] Create dataset snapshot functionality
- [ ] Document versioning strategy

### Week 5: Multi-Repo Support
- [ ] Create .promptforge.yml configuration
- [ ] Implement MultiRepoManager
- [ ] Setup repo sync scripts
- [ ] Test cross-repo evaluation
- [ ] Document multi-repo workflow

## Success Metrics

1. **Deterministic Control**: 100% of prompts support temperature=0
2. **Test Coverage**: >90% code coverage with unit tests
3. **Evaluation Coverage**: All prompts have golden, edge, and adversarial datasets
4. **CI/CD Pipeline**: <10min execution time for full pipeline
5. **Metric Thresholds**:
   - Exact Match: ≥95% for deterministic prompts
   - Groundedness: ≥85%
   - Schema Validity: 100%
   - PII Leakage: 0%
   - Latency: <2000ms p95
6. **Multi-Prompt Support**: ≥3 production prompts
7. **Version Control**: All prompts using semantic versioning

## Migration Strategy

### Phase 1: Backward Compatible
- Keep existing structure intact
- Add new features alongside existing ones
- Maintain compatibility with current workflows

### Phase 2: Gradual Migration
- Move prompts to new structure one at a time
- Update CI/CD incrementally
- Maintain parallel old/new systems

### Phase 3: Deprecation
- Mark old systems as deprecated
- Provide migration tools
- Set sunset date for legacy systems

## Risk Mitigation

1. **API Rate Limits**: Implement caching and mock LLM for tests
2. **Breaking Changes**: Use feature flags and gradual rollout
3. **Data Privacy**: All PII testing uses synthetic data
4. **Cost Control**: Set spending limits and use mock LLM in CI
5. **Complexity**: Modular design allows incremental adoption

## Next Steps

1. Review and approve enhancement plan
2. Create detailed Jira tickets for each phase
3. Assign team members to workstreams
4. Setup weekly progress reviews
5. Begin Phase 1 implementation

## Appendix: Example Configurations

### A. Mock Response Configuration
```json
{
  "prompts/summarization/mock_responses.json": {
    "5d41402abc4b2a76b9719d911017c592": {
      "response": "This is a summary of the document.",
      "confidence": 0.95,
      "tokens": {"prompt": 50, "completion": 20}
    }
  }
}
```

### B. Threshold Configuration
```yaml
# .threshold.yml
thresholds:
  global:
    min_pass_rate: 0.95
    max_latency_ms: 2000
  
  prompts:
    summarization:
      groundedness: 0.90
      exact_match: 0.85
    
    financial_planning:
      groundedness: 0.95
      schema_validity: 1.00
      pii_leakage: 0.00
```

### C. DeepEval Configuration
```python
# deepeval.config.py
DEEPEVAL_CONFIG = {
    "model": "gpt-4",
    "temperature": 0,
    "max_retries": 3,
    "cache_enabled": True,
    "parallel_workers": 5,
    "metrics": {
        "exact_match": {"enabled": True, "threshold": 0.95},
        "groundedness": {"enabled": True, "threshold": 0.85},
        "faithfulness": {"enabled": True, "threshold": 0.85},
        "schema_validity": {"enabled": True, "threshold": 1.00},
        "toxicity": {"enabled": True, "threshold": 0.00},
        "pii_leakage": {"enabled": True, "threshold": 0.00},
        "stability": {"enabled": True, "n_runs": 5},
        "latency": {"enabled": True, "max_ms": 2000}
    }
}
```

---
*Document Version: 1.0.0*
*Last Updated: 2024-09-04*
*Status: Draft - Pending Review*