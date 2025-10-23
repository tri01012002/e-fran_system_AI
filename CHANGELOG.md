# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Add Classify Base
- Add Red Frame service
- Add unit tests
- Add CI/CD pipeline

## [0.1.2] - 2025-10-23

### ✨ New Features

- **RedFrame Service**: Image quality assessment and frame validation service
  - `get_redframe_service()`: Factory function with intelligent caching
  - Support for two operation modes:
    - **Basic Mode**: Simple IQA using standard LIQE (pyiqa)
      - Full image quality validation
      - No additional parameters required
      - Fast and lightweight
    - **Advanced Mode**: Full RedFrame validation with custom LIQE
      - XGBoost-based RedFrame classification
      - Geometric frame/position validation
      - Cropped region quality analysis
      - Distortion type detection (blur, noise, etc.)
      - Requires `crop_method` parameter
  
- **RedFrame Service Caching**:
  - Singleton pattern per configuration to avoid redundant model loading
  - Cache key based on: category, part, advanced_mode, device
  - Threshold parameter doesn't trigger model reload
  - Optimized for API/service deployment scenarios

- **IQA Service**: Unified LIQE-based image quality assessment
  - `LIQEServices` class with dual model support
  - Standard LIQE via `pyiqa` (basic mode)
  - Custom LIQE with distortion detection (advanced mode)
  - Class-level model caching for memory efficiency

### 🔧 Technical Details

- **New Services Added**:
  - `raiki_sdk.services.RedFrameService`: Main RedFrame validation class
  - `raiki_sdk.services.LIQEServices`: LIQE IQA service with caching
  - `raiki_sdk.services.get_redframe_service()`: Cached factory function

- **New Schema**:
  - `RedFrameResult`: Result schema for RedFrame validation
    - `result: bool` - Overall validation result
    - `iqa_result: Optional[IqaResult]` - IQA details
    - `error_type: Optional[str]` - Error classification

- **Dependencies**:
  - Added `pyiqa>=0.1.14.1` for standard LIQE model
  - Added `xgboost>=3.1.1` for RedFrame classification
  - Added `joblib>=1.5.2` for model loading
  - Added `pandas>=2.3.3` for feature preparation

### 📖 Documentation

- Added comprehensive RedFrame Service guide in README
- Mode comparison table (Basic vs Advanced)
- Caching behavior documentation
- Usage examples for both modes

## [0.1.1] - 2025-10-21

Changes from v0.1.0 to v0.1.1.

### ✨ New Features
- **Model Package Registry System**: Centralized registry for managing model packages
  - `ModelRegistry` class with lazy loading and caching capabilities
  - Automatic detection and loading of model packages from `model_packages/` directory
  - Support for both authentication and classifier model types
  
- **Part Versioning**: Support for different product parts with individual versioning
  - Part identifiers: P00 (classifier models), P01, P02, etc. (authentication models)
  - Each part can have independent model versions
  
- **Model Versioning**: Flexible version management system
  - Version normalization (e.g., "v1.3" → "v1_3")
  - Support for semantic versioning (v1_1, v1_2, v1_3, etc.)
  - Backward compatible with existing version formats
  
- **New API Functions**:
  - `get_model(category, part, model_version)`: Unified model loader (auto-detects auth vs classifier)
  - `get_auth_model(category, part, model_version)`: Load authentication model packages
  - `list_models()`: List all cached models with metadata
  - `list_available_models()`: Scan and list all available model packages without loading

### ⚙️ Others Changes
- **Model Package Structure**: Standardized package organization
  - Factory pattern with `get_detector()` and `get_authenticator()` functions
  - Category-based organization (rolex, lv, etc.)
  - Lazy loading to optimize memory usage
  
- **Utility Functions**:
  - `_normalize_version_str()`: Version string normalization
  - `_normalize_category_str()`: Category name normalization
  
- **Dependencies**:
  - Added `huggingface-hub==0.35.3` for model management
  - Added `timm==1.0.20` for model architectures

See full release details

## [0.1.0] - 2025-10-10

### Added
- Initial release of Raiki SDK
- Abstract base classes:
  - `AuthenticateBase` for authentication systems
  - `YoloBase` for YOLO-based detection systems
- Pydantic schemas for type-safe data validation:
  - `AuthResult` for authentication results
  - `YoloResult` for detection results
  - `YoloValidation` for validation results
- Image preprocessing utilities:
  - **Processors**: `resize_pwd`, `padding_crop_yolo`, `remove_white_padding`, `pil_to_tensor`
  - **Filters**: `CLAHE`, `LoGFilter`, `AdjustBrightnessContrast`, `UnsharpMask`
- Comprehensive documentation and examples
- Package structure with proper `__init__.py` exports
- Dependencies: numpy, pillow, pydantic, ultralytics

[Unreleased]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.2...HEAD
[0.1.2]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.1...v0.1.2
[0.1.1]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.0...v0.1.1
[0.1.0]: https://gitlab.com/dat.tram/raiki-sdk/-/releases/v0.1.0