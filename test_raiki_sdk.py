import numpy as np
import cv2
from typing import Annotated
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from raiki_sdk import get_auth_feature, get_classification_feature
from raiki_sdk.services import get_brand_detection_service  # 🆕 NEW

# from core.logging import APP_LOGGER, MODEL_RESULTS_LOGGER

check_router = APIRouter()
classify_router = APIRouter()
brand_detect_router = APIRouter() 

@check_router.post("/")
async def check_part_test(
    checkfile: Annotated[UploadFile, File()],
    part: str,
    category: str = "rolex",
    model_version: str = "v1_3",
    frame: str = None
):
    try:
        # 1. Get model from registry
        model_package = get_auth_feature(category, part, model_version, device="cpu")
        print(f"Model package: {model_package}")
        detector = model_package.detector
        authenticator = model_package.authenticator

        # 2. Get image from request
        contents = checkfile.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Image is None")
            return JSONResponse(
                content={
                    "result": "failed",
                    "error": "image_none"
                }
            )

        # 3. YOLO detection
        try:
            yolo_result = detector.detect(image)
            print(f"Yolo result: {yolo_result}")
        except Exception as e:
            print(f"[/check/ {category} - {part}] - YOLO error {e}")
            return JSONResponse(
                content={
                    "result": "failed",
                    "error": "yolo_error"
                }
            )

        print(f"[/check/ {category} - {part}] - YOLO ok")

        # 4. Validate YOLO detection
        validation = detector.validate(yolo_result)
        print(f"Validation: {validation}")
        position = validation.position
        if not validation.is_valid:
            print(f"[/check/ {category} - {part}] - Validation failed")
            return JSONResponse(
                content={
                    "result": "failed",
                    "error": "validation_failed"
                }
            )

        print(f"[/check/ {category} - {part}] - Validation ok")

        # 5. Check Red Frame
        # redframe_service = get_redframe_service(category, part, advanced_mode=False, device="cpu", threshold=3.2)
        # redframe_result = redframe_service.forward(image, frame, position)
        # print(f"Red Frame result: {redframe_result}")

        # 5. Crop detector region
        cropped_image = detector.crop(image, position)
        cropped_image.save("cropped_image.jpg")
        print(f"[/check/ {category} - {part}] - Cropped image ok")

        # 6. Authenticate
        authentication = authenticator.forward(cropped_image)
        print(f"Authentication: {authentication}")
        print(f"[/check/ {category} - {part}] - Authentication ok")

        return JSONResponse(
            content={
                "result": "ok",
                "is_valid": validation.is_valid,
                "position": validation.position,
                "real_prob": authentication.probability,
                "label": authentication.label,
                "metadata": authentication.metadata
            }
        )
    except Exception as e:
        print(f"[/check/ {category} - {part}] - Error {e}")
        return JSONResponse(
            content={
                "result": "failed",
                "error": str(e)
            }
        )

@classify_router.post(
    "/",
    summary="[PRODUCTION] Classify the image into a category and predict objects from the image. .",
    description="""
**PRODUCTION**
Classify the image into a category and predict objects from the image.  
**Input:** `checkfile` (image)
**Steps:**
1. Get image from request
2. Detect category (*rolex*, *bag-lv*) using YOLO model
3. Detect objects using the corresponding category model
4. Return top **K=3** predictions
---
**DEVELOPMENT**
- Generate heatmap from the image (for visualization)
""")
async def classify(
    checkfile: Annotated[UploadFile, File()],
    category: str = "rolex",
    part: str = "P00",
    model_version: str = "v1_3",
    topk: int=3,
):
    """
        Steps:
        1. Get image from request
        2. Detect category from image and get feature (if not provided). (Repeat through each features classify until valid)
        3. predict object from image
    """
    #1
    contents = checkfile.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #2. Get model from registry
    model_package = get_classification_feature(category, model_version, device="cpu")
    classifier = model_package.classifier
    detector = model_package.detector

    #3. YOLO detection
    detector_result = detector.detect(image)
    print(f"Detector result: {detector_result}")
    validation = detector.validate(detector_result)
    print(f"Validation: {validation}")
    if not validation.is_valid:
        return JSONResponse(
            content={
                "result": "failed",
                "error": "validation_failed"
            }
        )
    image_region = detector.crop(image, validation.position)
    image_region.save("image_region.jpg")
    print(f"[/classify/ {category} - {part}] - Validation ok")

    #4. Classify
    result = classifier.forward(image_region)
    print(f"Result: {result}")
    
    topks = classifier.get_topk(result, topk) # ClassificationResultProb
    print(f"Topks: {topks}")
    topks_payload = topks.model_dump()
    return JSONResponse(
        content={
            "result": "ok",
            "category": category,
            "predict": topks_payload,
        }
    )


@brand_detect_router.post(
    "/",
    summary="Test Brand Detection + Classification Pipeline",
    description="""
**TEST ENDPOINT**
Full pipeline: Auto-detect brand → Classify

**Input:** `checkfile` (image)

**Pipeline:**
1. Brand Detection (Watch: Rolex/Omega, Bag: LV)
2. Classification using detected category

**Output:** Detection result + Top K predictions
""")
async def test_brand_detection(
    checkfile: Annotated[UploadFile, File()],
    topk: int = 3,
    device: str = "cpu",
):
    """
    Test brand detection + classification pipeline.
    """
    # 1. Read image
    contents = checkfile.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"result": "failed", "error": "invalid_image"}
        )
    
    print(f"[Brand Detection] Image shape: {image.shape}")
    
    # 2. Brand Detection
    detection_service = get_brand_detection_service(device=device)
    detection_result = detection_service.detect(image)
    
    if detection_result is None:
        return JSONResponse(
            status_code=400,
            content={
                "result": "failed",
                "error": "category_not_detected",
                "message": "Could not detect any supported brand"
            }
        )
    
    print(f"[Brand Detection] Detected: {detection_result.category} "
          f"conf={detection_result.confidence:.2f}")
    
    # Handle unknown watch
    if detection_result.category == "unknown_watch":
        return JSONResponse(
            status_code=400,
            content={
                "result": "failed",
                "error": "unsupported_watch_brand",
                "message": "Detected watch but brand not supported (Rolex/Omega only)",
                "detection": {
                    "category": detection_result.category,
                    "confidence": detection_result.confidence,
                    "bbox": detection_result.bbox
                }
            }
        )
    
    # 3. Get classification model
    category = detection_result.category
    
    # Map category to model version (you may want to configure this)
    MODEL_VERSIONS = {
        "rolex": "v1_3",
        "omg": "v1_1",
        "bag-lv": "v1_1",
    }
    
    model_version = MODEL_VERSIONS.get(category, "v1_1")
    
    try:
        feature = get_classification_feature(
            category=category,
            model_version=model_version,
            device=device
        )
        classifier = feature.classifier
        print(f"[Classification] Loaded {category} classifier {model_version}")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "result": "failed",
                "error": "model_load_error",
                "message": str(e),
                "detection": {
                    "category": category,
                    "confidence": detection_result.confidence
                }
            }
        )
    
    # 4. Use cropped image if available, else use full image
    if detection_result.cropped_image is not None:
        pil_img = detection_result.cropped_image
        print(f"[Classification] Using cropped image: {pil_img.size}")
    else:
        # Fallback to full image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb)
        print(f"[Classification] Using full image")
    
    # Save cropped for debugging
    pil_img.save("brand_detect_cropped.jpg")
    
    # 5. Classification
    try:
        result = classifier.forward(pil_img)
        topks = classifier.get_topk(result, topk)
        print(f"[Classification] Result: {topks}")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "result": "failed",
                "error": "classification_error",
                "message": str(e)
            }
        )
    
    # 6. Return results
    return JSONResponse(
        content={
            "result": "ok",
            "detection": {
                "category": detection_result.category,
                "confidence": detection_result.confidence,
                "bbox": detection_result.bbox
            },
            "classification": {
                "category": category,
                "model_version": model_version,
                "predict": topks.model_dump()
            }
        }
    )


@brand_detect_router.post("/detect-only")
async def test_detection_only(
    checkfile: Annotated[UploadFile, File()],
    device: str = "cpu",
):
    """
    Test brand detection only (no classification).
    """
    contents = checkfile.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"result": "failed", "error": "invalid_image"}
        )
    
    detection_service = get_brand_detection_service(device=device)
    detection_result = detection_service.detect(image)
    
    if detection_result is None:
        return JSONResponse(
            content={
                "result": "not_detected",
                "message": "No brand detected"
            }
        )
    
    # Save cropped image
    if detection_result.cropped_image:
        detection_result.cropped_image.save("detected_crop.jpg")
    
    return JSONResponse(
        content={
            "result": "ok",
            "detection": {
                "category": detection_result.category,
                "confidence": detection_result.confidence,
                "bbox": detection_result.bbox,
                "has_cropped_image": detection_result.cropped_image is not None
            }
        }
    )