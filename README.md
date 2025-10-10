# Raiki SDK 🔍

**Raiki SDK** - Base interfaces and utilities for authentication and detection systems.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Abstract Base Classes**: Ready-to-extend interfaces for authentication and YOLO-based detection
- **Pydantic Schemas**: Type-safe data validation for detection results
- **Image Preprocessing**: Professional image processing utilities
  - Resize and padding operations
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - LoG (Laplacian of Gaussian) filtering
  - Brightness/Contrast adjustments
  - White padding removal

## 📦 Installation

> **Note:** This is a private repository. You need authentication to install.

### Prerequisites: Create GitLab Personal Access Token

1. Go to GitLab: **Settings → Access Tokens** → [Create new token](https://gitlab.com/-/profile/personal_access_tokens)
2. Set token name: `raiki-sdk-access`
3. Select scopes: `read_repository`
4. Click **Create personal access token**
5. **Copy the token** (shown only once!)

### Install from Private Repository

#### Using Personal Access Token

```bash
# Replace YOUR_TOKEN with your actual token
# Latest version from main branch
pip install git+https://oauth2:YOUR_TOKEN@gitlab.com/dat.tram/raiki-sdk.git

# Or with uv (recommended)
uv pip install git+https://oauth2:YOUR_TOKEN@gitlab.com/dat.tram/raiki-sdk.git
```

#### Install Specific Version

```bash
# Install version 0.1.0 with token
pip install git+https://oauth2:YOUR_TOKEN@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0

# Or with uv
uv pip install git+https://oauth2:YOUR_TOKEN@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

#### Using SSH (if you have SSH key configured)

```bash
# Latest version
pip install git+ssh://git@gitlab.com/dat.tram/raiki-sdk.git

# Specific version
pip install git+ssh://git@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

### Add to requirements.txt

```txt
# Replace YOUR_TOKEN with your actual token
raiki-sdk @ git+https://oauth2:YOUR_TOKEN@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

⚠️ **Security Warning:** Do NOT commit your token to git! Use environment variables instead:

```bash
# Set environment variable
export GITLAB_TOKEN=your_token_here

# Use in requirements.txt or scripts
raiki-sdk @ git+https://oauth2:${GITLAB_TOKEN}@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

### Add to pyproject.toml

```toml
[project]
dependencies = [
    # Option 1: With token (not recommended for public repos)
    "raiki-sdk @ git+https://oauth2:YOUR_TOKEN@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0",
    
    # Option 2: With SSH (recommended)
    # "raiki-sdk @ git+ssh://git@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0",
]
```

### Using Environment Variables (Recommended for CI/CD)

```bash
# .env file (add to .gitignore!)
GITLAB_TOKEN=your_token_here

# Install using environment variable
pip install git+https://oauth2:${GITLAB_TOKEN}@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

In CI/CD pipelines (GitLab CI, GitHub Actions, etc.), add the token as a secret environment variable.

## 🎯 Quick Start

### Using Base Classes

```python
from raiki_sdk import AuthenticateBase, YoloBase
from raiki_sdk import AuthResult, YoloResult

class MyAuthenticator(AuthenticateBase):
    """Custom authentication implementation"""
    
    def authenticate(self, image_path: str) -> AuthResult:
        # Your authentication logic here
        return AuthResult(
            is_authentic=True,
            confidence=0.95,
            message="Product is authentic"
        )

class MyDetector(YoloBase):
    """Custom YOLO detection implementation"""
    
    def __init__(self):
        super().__init__(model_path="path/to/your/model.pt")
    
    def detect(self, image_path: str) -> YoloResult:
        # Your detection logic here
        pass
```

### Using Preprocessing Utilities

```python
from raiki_sdk import (
    resize_pwd,
    padding_crop_yolo,
    CLAHE,
    LoGFilter,
    AdjustBrightnessContrast
)
from PIL import Image

# Load image
image = Image.open("product.jpg")

# Resize with aspect ratio preservation
resized = resize_pwd(image, target_size=(640, 640))

# Apply CLAHE enhancement
clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
enhanced = clahe(resized)

# Apply LoG filter for edge detection
log_filter = LoGFilter(sigma=1.5, kernel_size=5)
edges = log_filter(enhanced)

# Adjust brightness and contrast
adjuster = AdjustBrightnessContrast(brightness_factor=1.2, contrast_factor=1.1)
final_image = adjuster(edges)
```

## 📚 API Reference

### Base Classes

#### `AuthenticateBase`
Abstract base class for authentication systems.

**Methods:**
- `authenticate(image_path: str) -> AuthResult`: Implement your authentication logic

#### `YoloBase`
Abstract base class for YOLO-based detection systems.

**Methods:**
- `detect(image_path: str) -> YoloResult`: Detect objects in image
- `validate(image_path: str) -> YoloValidation`: Validate detection results

### Schemas

#### `AuthResult`
- `is_authentic: bool` - Authentication result
- `confidence: float` - Confidence score (0-1)
- `message: str` - Result message
- `metadata: dict` - Additional metadata

#### `YoloResult`
- `boxes: List` - Detected bounding boxes
- `confidences: List[float]` - Detection confidences
- `class_ids: List[int]` - Class IDs
- `class_names: List[str]` - Class names

### Preprocessing Functions

#### Image Processors
- `resize_pwd(image, target_size)` - Resize with aspect ratio
- `padding_crop_yolo(image, target_size)` - Pad and crop for YOLO
- `remove_white_padding(image)` - Remove white borders
- `pil_to_tensor(image)` - Convert PIL to tensor

#### Image Filters
- `CLAHE()` - Contrast enhancement
- `LoGFilter()` - Edge detection
- `AdjustBrightnessContrast()` - Brightness/contrast adjustment
- `UnsharpMask()` - Image sharpening

## 🔧 Development

### Setup Development Environment

#### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://gitlab.com/dat.tram/raiki-sdk.git
cd raiki-sdk

# Sync dependencies (creates venv automatically)
uv sync

# The package is now installed in editable mode
# Any code changes will be reflected immediately
```

#### Option 2: Using traditional pip

```bash
# Clone the repository
git clone https://gitlab.com/dat.tram/raiki-sdk.git
cd raiki-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Verify Installation

```bash
# Test imports
python -c "from raiki_sdk import AuthenticateBase, YoloBase; print('✅ Success!')"

# Check version
python -c "import raiki_sdk; print(f'Version: {raiki_sdk.__version__}')"

# Run validation script
python test_import.py
```

## 📋 Requirements

- Python >= 3.10
- numpy >= 2.2.6
- pillow >= 11.3.0
- pydantic >= 2.12.0
- ultralytics >= 8.3.208

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Danny Tram** - [GitLab](https://gitlab.com/dat.tram)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please use the [GitLab Issues](https://gitlab.com/dat.tram/raiki-sdk/-/issues) page.

## 🗺️ Roadmap

- [ ] Add Classify Base
- [ ] Add Red Frame service
- [ ] Add unit tests
- [ ] Add CI/CD pipeline

## 📊 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

### Latest Release: v0.1.0 (2025-10-10)
- Initial SDK release with base classes and preprocessing utilities
- See full changelog for details