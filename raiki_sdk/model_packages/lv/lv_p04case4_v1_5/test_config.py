# test_config_with_model.py
from raiki_sdk import get_auth_feature

def test_model_instantiation():
    try:
        # Thử load model với config này
        model_package = get_auth_feature(
            category="bag-lv",
            part="LV_P04case4", 
            model_version="v1_5",
            device="cpu"
        )
        
        print(f"✓ Model loaded: {model_package}")
        print(f"  - Detector: {model_package.detector}")
        print(f"  - Authenticator: {model_package.authenticator}")
        print("✅ Config is working correctly!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_instantiation()