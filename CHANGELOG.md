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

### Documentation
- Added README with installation and usage guide
- Added examples for base classes and preprocessing
- Added API reference documentation

[Unreleased]: https://gitlab.com/dat.tram/raiki-sdk/-/compare/v0.1.0...HEAD
[0.1.0]: https://gitlab.com/dat.tram/raiki-sdk/-/releases/v0.1.0