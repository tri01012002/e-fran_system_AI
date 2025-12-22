# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Fix some bugs and improve performance


## [0.1.5] - 2025-12-22

### ✨ Improvements
- Support OMG_P00 v1.1 - omega classification model

### 🔧 Fixes
- Fix `position = {}` errors

## [0.1.4] - 2025-12-09

### ✨ New Features

- **Expanded Omega Model Package Support**:
  - **Omega (OMG)**: OMG_P01, OMG_P02, OMG_P03, OMG_P05, and OMG_P06.

- **Replace YoloBase to DetectorBase**:

### 📦 Model Package Version

| Part | Package Version |
|------|-------------|
| OMG_P01 | v1.0 |
| OMG_P02 | v1.1 |
| OMG_P03 | v1.2 |
| OMG_P05 | v1.0 |
| OMG_P06 | v1.0 |

### 🎯 Breaking Changes

None - This is a purely additive release.

### 📖 Documentation

- Updated README with comprehensive Omega model package inventory
- Added supported omega brands
- Changed `YoloBase()` to `DetectorBase()`

## [0.1.3] - 2025-10-24

### ✨ New Features

- **Expanded Model Package Support**: Full support for multiple brands and parts
  - **Louis Vuitton (LV)**: LV_P01, LV_P02, LV_P03, LV_P04, LV_P04case4, LV_P05
  - **Rolex**: Rolex_P01, Rolex_P02, Rolex_P03, Rolex_P04, Rolex_P05, Rolex_P08, Rolex_P09, Rolex_P10
  - Standardized model package structure across all brands
  - Centralized management through ModelRegistry

- **Unified API for Model Loading**:
  - `get_auth_feature(category, part, model_version, device)`: Simple API to load any model package
    - Automatically discovers and loads detector and authenticator
    - Supports multiple devices: "cuda", "mps", "cpu"
    - Lazy loading for memory efficiency
    - Built-in caching to avoid redundant loading

### 📦 Model Package Inventory

#### Louis Vuitton (LV) Models
| Part | Description |
|------|-------------|
| LV_P01 | Primary part authentication |
| LV_P02 | Secondary part authentication |
| LV_P03 | Tertiary part authentication |
| LV_P04 | Quaternary part authentication |
| LV_P04case4 | Special variant of P04 |
| LV_P05 | Additional part authentication |

#### Rolex Models
| Part | Description |
|------|-------------|
| Rolex_P01 | Primary part authentication |
| Rolex_P02 | Secondary part authentication |
| Rolex_P03 | Tertiary part authentication |
| Rolex_P04 | Quaternary part authentication |
| Rolex_P05 | Quinary part authentication |
| Rolex_P08 | Octary part authentication |
| Rolex_P09 | Nonary part authentication |
| Rolex_P10 | Denary part authentication |

### 🔧 Technical Details

- **Model Package Organization**:
  - Each brand has dedicated directory: `raiki_sdk/model_packages/{category}/`
  - Each part has versioned subdirectory: `{category}_{part}_{version}/`
  - Standardized structure with `detector.py`, `authenticator.py`, `config.py`
  - Factory functions in `__init__.py` for easy instantiation

- **AI Engineer Workflow**:
  - Simple import: `from raiki_sdk import get_auth_feature`
  - Single function call to load models: `get_auth_feature(category, part, version, device)`
  - Automatic detection of model type
  - Centralized model management and versioning

### 🎯 Breaking Changes

None - This is a purely additive release.

### 📖 Documentation

- Updated README with comprehensive model package inventory
- Added supported brands and parts reference
- Updated usage examples with both APIs: `get_model()` and `get_auth_feature()`

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
  - `DetectorBase` for YOLO-based detection systems
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

[Unreleased]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.6...HEAD
[0.1.5]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.4...v0.1.5
[0.1.4]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.3...v0.1.4
[0.1.3]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.2...v0.1.3
[0.1.2]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.1...v0.1.2
[0.1.1]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.0...v0.1.1
[0.1.0]: https://gitlab.com/dat.tram/raiki-sdk/-/releases/v0.1.0