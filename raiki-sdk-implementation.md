<!-- 5acb9aa4-9458-4cab-a61a-7bbe547721af e28c01a1-308b-46ff-87aa-341c4892b8bd -->
# Raiki SDK Implementation Plan

## Overview

Create `raiki-sdk` as a standalone Python package that provides:

1. Abstract base classes for authentication and YOLO detection with Pydantic validation
2. Image preprocessing utilities (transforms, filters, masks)
3. Red Frame detection service using XGBoost

## Project Structure

```
raiki-sdk/
├── raiki_sdk/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── authenticate_base.py    # AuthenticateBase abstract class
│   │   ├── detector_base.py            # DetectorBase abstract class  
│   │   └── schemas.py              # Pydantic output models
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── transforms.py           # resize_pwd, padding_yolo
│   │   ├── filters.py              # CLAHE, UnsharpMask, LoG, Gabor
│   │   └── masks.py                # GridMask variants
│   ├── services/
│   │   ├── __init__.py
│   │   └── red_frame.py            # RedFrameService class
│   └── utils/
│       ├── __init__.py
│       └── helpers.py              # Common utilities
├── tests/
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_preprocessing.py
│   └── test_red_frame.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
├── setup.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Implementation Details

### 1. Base Classes (`raiki_sdk/base/`)

**authenticate_base.py:**

- `AuthenticateBase` abstract class with methods:
  - `forward(image) -> AuthResult`: Main inference method
  - `get_prob(output) -> tuple[float, str]`: Extract probability and label
  - `generate_heatmap() -> HeatmapResult`: Generate attention heatmap

**detector_base.py:**

- `DetectorBase` abstract class with methods:
  - `validate(yolo_results, frame) -> YoloValidation`: Validate detection
  - `detect(image, frame, iqa) -> YoloResult`: Run detection pipeline

**schemas.py:**

- `AuthResult(BaseModel)`: probability, label, metadata
- `YoloResult(BaseModel)`: position, confidence, image_detect
- `YoloValidation(BaseModel)`: is_valid, position dict
- `HeatmapResult(BaseModel)`: heatmap_bytes, probability, label
- `RedFrameResult(BaseModel)`: status ("good"|"wide"|"angle"|"out-of-frame"), ratio

### 2. Preprocessing (`raiki_sdk/preprocessing/`)

**transforms.py:**

- `resize_pwd(img, output_size, color_padding)` - from function_common.py
- `padding_yolo(image, position, p, remove_padding)` - from function_common.py
- `remove_white_padding(image, threshold)` - from function_common.py

**filters.py:**

- `LoGFilter` - Laplacian of Gaussian from image_preprocessor.py
- `AdjustBrightnessContrast` - from image_preprocessor.py
- `CLAHE` - from image_preprocessor.py
- `UnsharpMask` - from image_preprocessor.py
- `CLAHE_P04_m2`, `UnsharpMask_P04_m2` - grayscale variants
- `GaborFeatureExtractor` - from image_preprocessor.py
- `CLAHE_P10`, `preprocess_image` - Rolex P10 preprocessing

**masks.py:**

- `GridMaskLVP02` - 3x3 grid masking
- `Grid6x6Mask` - 6x6 grid with corner/center masking

### 3. Red Frame Service (`raiki_sdk/services/`)

**red_frame.py:**

- `RedFrameService` class:
  - `__init__(model_path: str)`: Load XGBoost model
  - `predict(image, position, frame) -> RedFrameResult`: Predict frame quality
  - `calculate_area(image, frame) -> float`: Helper method
  - `calculate_area_ratio(image, frame, position) -> float`: Helper method

### 4. Package Configuration

**setup.py:**

```python
from setuptools import setup, find_packages

setup(
    name="raiki-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "pydantic>=2.0.0",
        "numpy<2",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "pandas>=2.0.0",
        "scikit-image>=0.21.0"
    ],
    python_requires=">=3.9"
)
```

**requirements.txt:** Same as install_requires

### 5. Documentation

**README.md:**

- Installation instructions (`pip install git+https://...`)
- Quick start guide
- API reference
- Examples for each component

**examples/:**

- `basic_usage.py`: Simple authentication flow
- `advanced_usage.py`: Custom implementation with all features

### 6. Testing

**tests/:**

- `test_base.py`: Test abstract classes can be inherited correctly
- `test_preprocessing.py`: Test all preprocessing functions with dummy images
- `test_red_frame.py`: Test RedFrameService with mock XGBoost model

## Key Files to Reference from rolex-kag-api

1. `/rolex-kag-api/features/base/FeatureBase.py` → Abstract methods for base classes
2. `/rolex-kag-api/features/base/FeatureBaseKag.py` → Implementation patterns
3. `/rolex-kag-api/features/utils/function_common.py` → Transform functions
4. `/rolex-kag-api/core/image_preprocessor.py` → Filter classes
5. `/rolex-kag-api/features/RedFrame/RedframeP01.py` → RedFrame logic

### To-dos[text]

- [x] Create raiki-sdk folder structure with all necessary __init__.py files
- [x] Implement Pydantic schemas in raiki_sdk/base/schemas.py
- [x] Create AuthenticateBase abstract class in raiki_sdk/base/authenticate_base.py
- [x] Create DetectorBase abstract class in raiki_sdk/base/yolo_base.py
- [x] Validate import raiki-sdk
- [x] Port transform functions to raiki_sdk/preprocessing/transforms.py
- [x] Port filter classes to raiki_sdk/preprocessing/filters.py
- [ ] Implement RedFrameService in raiki_sdk/services/red_frame.py
- [ ] Create setup.py, requirements.txt, and .gitignore
- [ ] Write comprehensive README.md with installation and usage examples
- [ ] Create example scripts in examples/ directory
- [ ] Write unit tests for all components in tests/ directory