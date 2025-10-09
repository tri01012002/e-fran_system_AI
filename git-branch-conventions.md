# Git Branch Naming Conventions

## Introduction

A well-defined branch naming convention improves team collaboration, makes the repository easier to navigate, and integrates seamlessly with CI/CD pipelines and project management tools.

## Branch Naming Format
### Components

1. **Type**: Category of work being done
2. **Description**: Brief, kebab-case description
3. **Ticket ID** (optional): Reference to issue tracker

## Branch Types

| Type | Purpose | Example |
|------|---------|---------|
| `feat` / `feature` | New features | `feat/add-yolo-detection` |
| `fix` / `bugfix` | Bug fixes | `fix/memory-leak-inference` |
| `hotfix` | Urgent production fixes | `hotfix/critical-auth-bug` |
| `refactor` | Code refactoring | `refactor/model-architecture` |
| `docs` | Documentation only | `docs/api-reference` |
| `test` | Adding or updating tests | `test/unit-tests-preprocessing` |
| `perf` | Performance improvements | `perf/optimize-batch-inference` |
| `style` | Code style/formatting | `style/linting-fixes` |
| `chore` | Maintenance tasks | `chore/update-dependencies` |
| `build` | Build system changes | `build/docker-configuration` |
| `ci` | CI/CD changes | `ci/github-actions-workflow` |
| `experiment` / `exp` | ML experiments | `exp/focal-loss-training` |

## Naming Rules

### ✅ DO

- Use lowercase letters
- Use hyphens (`-`) to separate words (kebab-case)
- Keep it short but descriptive (max 50 characters)
- Use imperative mood: `add-feature` not `added-feature`
- Be specific about what the branch does
- Include ticket/issue numbers when applicable

### ❌ DON'T

- Use spaces or special characters (except `/` and `-`)
- Use camelCase or snake_case
- Create overly long names
- Use generic names like `fix-bug` or `updates`
- Include your name in the branch (Git tracks authors)

## Examples

### Feature Development

```bash
# Good
feat/user-authentication
feat/yolo-object-detection
feat/batch-prediction-api
feat/JIRA-123-image-preprocessing
feature/rolex-classification-model

# Bad
feature  # Too generic
feat/addUserAuth  # camelCase
feature_new_model  # snake_case
feat/this-is-a-very-long-branch-name-that-describes-everything-in-detail  # Too long
```

### Bug Fixes

```bash
# Good
fix/tensor-shape-mismatch
fix/null-pointer-dataloader
bugfix/JIRA-456-memory-leak
fix/model-loading-timeout

# Bad
fix/bug  # Not descriptive
fix/Fix-Bug-123  # Don't capitalize
fix/fixing_the_error  # Wrong tense & format
```

### Hotfixes

```bash
# Good
hotfix/production-api-crash
hotfix/critical-auth-vulnerability
hotfix/model-inference-failure

# Bad
hotfix/urgent  # Not specific
hotfix/production-fix  # Too generic
```

### ML Experiments

```bash
# Good
exp/focal-loss-experiment
exp/efficientnet-b7-training
exp/data-augmentation-ablation
experiment/hyperparameter-tuning-lr

# Bad
exp/test  # Too vague
exp/new-model  # Not specific
```

## AI/ML Project Examples

### Model Development

```bash
feat/implement-efficientnet-backbone
feat/add-rolex-p01-detector
feat/multi-label-classification
feat/model-ensemble-voting
```

### Data & Preprocessing

```bash
feat/image-augmentation-pipeline
feat/dataset-loader-caching
refactor/preprocessing-transforms
fix/corrupted-image-handling
```

### Training & Experiments

```bash
exp/learning-rate-scheduler
exp/mixed-precision-training
perf/gradient-accumulation
feat/distributed-training-setup
```

### Inference & Deployment

```bash
feat/batch-inference-endpoint
perf/tensorrt-optimization
fix/cuda-memory-overflow
feat/model-versioning-system
```

### API Development

```bash
feat/prediction-api-endpoint
feat/websocket-streaming
fix/request-validation-error
refactor/response-serialization
```