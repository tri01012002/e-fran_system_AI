import numpy as np
import cv2
from typing import Annotated
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from raiki_sdk import get_auth_feature, get_classification_feature
# from core.logging import APP_LOGGER, MODEL_RESULTS_LOGGER

check_router = APIRouter()
classify_router = APIRouter()

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
        model_package = get_auth_feature(category, part, model_version, device="cuda")
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
            yolo_result = detector.yolo_detect(image)
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
        # redframe_service = get_redframe_service(category, part, advanced_mode=False, device="cuda", threshold=3.2)
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
    model_package = get_classification_feature(category, part, model_version, device="cuda")
    classifier = model_package.classifier
    detector = model_package.detector

    #3. YOLO detection
    detector_result = detector.yolo_detect(image)
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
            "classes": classifier.labels,
        }
    )
