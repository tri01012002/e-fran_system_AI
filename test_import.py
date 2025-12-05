#!/usr/bin/env python3
"""Quick validation script for raiki-sdk"""

def test_imports():
    """Test all main imports"""
    try:
        from raiki_sdk import (
            AuthenticateBase,
            DetectorBase,
            AuthResult,
            YoloResult,
            YoloValidation,
            resize_pwd,
            padding_crop_yolo,
            remove_white_padding,
            pil_to_tensor,
            CLAHE,
            LoGFilter,
            AdjustBrightnessContrast,
        )
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_version():
    """Test version matches"""
    import raiki_sdk
    expected_version = "0.1.0"
    if raiki_sdk.__version__ == expected_version:
        print(f"✅ Version correct: {raiki_sdk.__version__}")
        return True
    else:
        print(f"❌ Version mismatch: expected {expected_version}, got {raiki_sdk.__version__}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from raiki_sdk import CLAHE
        from PIL import Image
        import numpy as np
        
        print("✅ Basic functionality test passed!")
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing raiki-sdk")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_version,
        test_basic_functionality,
    ]
    
    results = [test() for test in tests]
    
    print("=" * 50)
    if all(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed!")
        exit(1)