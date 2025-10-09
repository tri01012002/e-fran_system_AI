# Conventional Commits Guide for AI Engineers

## Introduction

Conventional Commits is a specification for writing clear and meaningful commit messages. It provides a structured format that makes it easy to understand what changed, why it changed, and helps automate versioning and changelog generation.

## Format

The commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Core Components

1. **Type**: Describes the kind of change
2. **Scope** (optional): Specifies what part of the codebase is affected
3. **Description**: Brief summary of the change
4. **Body** (optional): Detailed explanation
5. **Footer** (optional): Breaking changes or issue references

## Commit Types

| Type | Description | When to Use |
|------|-------------|-------------|
| `feat` | New feature | Adding new functionality |
| `fix` | Bug fix | Fixing a bug or error |
| `docs` | Documentation | Changes to documentation only |
| `style` | Code style | Formatting, missing semicolons, etc. |
| `refactor` | Code refactoring | Neither fixes a bug nor adds a feature |
| `perf` | Performance | Code changes that improve performance |
| `test` | Testing | Adding or updating tests |
| `build` | Build system | Changes to build process or dependencies |
| `ci` | CI/CD | Changes to CI configuration files |
| `chore` | Maintenance | Other changes that don't modify src or test files |
| `revert` | Revert | Reverting a previous commit |

## Python AI Engineering Examples

### Feature Development

```bash
# Adding a new model
git commit -m "feat(model): add EfficientNet-B4 for rolex authentication

- Implemented EfficientNet-B4 architecture
- Added preprocessing pipeline for 224x224 input
- Achieved 94.5% validation accuracy
- Model supports batch inference"

# Adding a new preprocessing function
git commit -m "feat(preprocessing): add image augmentation pipeline

Implemented augmentation pipeline with:
- Random rotation (±15°)
- Color jittering (brightness, contrast)
- Horizontal flipping
- Normalization using ImageNet stats"

# Adding a new API endpoint
git commit -m "feat(api): add batch prediction endpoint

- New /api/v1/predict/batch endpoint
- Supports up to 32 images per request
- Async processing with Redis queue
- Returns predictions with confidence scores"
```

### Bug Fixes

```bash
# Fixing model inference issue
git commit -m "fix(inference): resolve tensor shape mismatch in batch prediction

Fixed IndexError when batch size doesn't match model input dimensions.
Added dynamic reshaping for variable batch sizes."

# Fixing data loading bug
git commit -m "fix(dataloader): handle corrupted images gracefully

- Added try-catch for PIL.Image.open()
- Skip corrupted images and log warnings
- Prevents entire batch from failing"

# Fixing memory leak
git commit -m "fix(model): resolve GPU memory leak in inference loop

Clear CUDA cache after each batch and move tensors to CPU
after prediction to prevent OOM errors in long-running services."
```

### Performance Improvements

```bash
# Optimizing inference
git commit -m "perf(inference): reduce prediction latency by 40%

- Implement TorchScript compilation
- Enable CUDA graph capture
- Add batch processing for single image requests
- Latency reduced from 150ms to 90ms"

# Optimizing data pipeline
git commit -m "perf(dataloader): implement parallel image preprocessing

Use multiprocessing.Pool for image decoding and augmentation.
Training throughput increased from 120 to 280 images/sec."
```

### Refactoring

```bash
# Code refactoring
git commit -m "refactor(model): extract feature extraction to separate module

- Created FeatureExtractor base class
- Moved feature extraction logic from model files
- Improved code reusability across models"

# Restructuring codebase
git commit -m "refactor(structure): reorganize model definitions

- Moved all model architectures to core/models/
- Separated model configs to config/models.yaml
- Updated import paths across codebase"
```

### Documentation

```bash
# API documentation
git commit -m "docs(api): add comprehensive docstrings to prediction endpoints

Added detailed docstrings with:
- Parameter descriptions and types
- Return value specifications
- Usage examples
- Error handling notes"

# Model documentation
git commit -m "docs(model): document training pipeline and hyperparameters

- Added training procedure documentation
- Documented hyperparameter tuning results
- Included model architecture diagrams
- Added performance benchmarks"
```

### Testing

```bash
# Unit tests
git commit -m "test(model): add unit tests for preprocessing pipeline

- Test image normalization
- Test augmentation transformations
- Test edge cases (grayscale, alpha channel)
- Coverage increased to 85%"

# Integration tests
git commit -m "test(api): add integration tests for prediction endpoints

Added tests for:
- Single image prediction
- Batch prediction
- Error handling
- Response format validation"
```

### Dependencies and Build

```bash
# Updating dependencies
git commit -m "build(deps): upgrade PyTorch to 2.1.0

- Updated torch from 2.0.1 to 2.1.0
- Updated torchvision to compatible version
- Verified model compatibility
- All tests passing"

# Adding new dependency
git commit -m "build(deps): add albumentations for image augmentation

- Added albumentations==1.3.1
- Replaced custom augmentation pipeline
- Provides more robust transformations"
```

### Breaking Changes

```bash
# Breaking API change
git commit -m "feat(api): restructure prediction response format

BREAKING CHANGE: Response format changed from flat structure
to nested JSON with metadata.

Before:
{
  "prediction": "authentic",
  "confidence": 0.95
}

After:
{
  "result": {
    "prediction": "authentic",
    "confidence": 0.95
  },
  "metadata": {
    "model_version": "v2.0",
    "inference_time_ms": 120
  }
}"
```

### AI/ML Specific Scopes

Common scopes for AI engineering projects:

- `(model)`: Model architecture or weights
- `(training)`: Training pipeline or scripts
- `(inference)`: Inference or prediction logic
- `(preprocessing)`: Data preprocessing
- `(dataset)`: Dataset handling
- `(metrics)`: Evaluation metrics
- `(pipeline)`: ML pipeline
- `(api)`: API endpoints
- `(deployment)`: Deployment configuration
- `(experiment)`: Experiment tracking

## Advanced Examples

### Model Update with Metrics

```bash
git commit -m "feat(model): upgrade to YOLOv11 for object detection

Migrated from YOLOv8 to YOLOv11:

Performance improvements:
- mAP@0.5: 0.87 → 0.92 (+5.7%)
- Inference speed: 45ms → 32ms (-28.9%)
- Model size: 89MB → 67MB (-24.7%)

Changes:
- Updated model architecture
- Retrained on expanded dataset (50k → 75k images)
- Adjusted NMS threshold to 0.45
- Updated postprocessing pipeline

Refs: #123"
```

### Experiment Tracking

```bash
git commit -m "feat(experiment): add new loss function for imbalanced dataset

Implemented focal loss to handle class imbalance:
- Alpha: 0.25
- Gamma: 2.0

Results (vs CrossEntropyLoss):
- Rare class F1: 0.62 → 0.78
- Overall accuracy: 0.91 → 0.93
- Training converges 20% faster

Experiment ID: exp-focal-loss-20251009"
```

### Configuration Changes

```bash
git commit -m "chore(config): update model inference thresholds

Adjusted confidence thresholds based on production metrics:
- Rolex P01: 0.75 → 0.82
- Rolex P02: 0.80 → 0.85
- LV P04: 0.70 → 0.78

Reduces false positive rate by 15% while maintaining recall."
```

## Best Practices

1. **Be Specific**: Clearly describe what changed and why
2. **Keep it Concise**: Summary should be under 72 characters
3. **Use Imperative Mood**: "add feature" not "added feature"
4. **Include Metrics**: For model changes, include performance metrics
5. **Reference Issues**: Link to relevant tickets or issues
6. **Document Breaking Changes**: Always highlight breaking changes
7. **Separate Concerns**: One commit per logical change


