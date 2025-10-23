import numpy as np
import cv2
from typing import Annotated
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from raiki_sdk import get_auth_feature, get_redframe_service
# from core.logging import APP_LOGGER, MODEL_RESULTS_LOGGER

router = APIRouter()

@router.post("/")

async def check_part_test(
    checkfile: Annotated[UploadFile, File()],
    part: str,
    category: str = "rolex",
    model_version: str = "v1_3",
    frame: str = None
):
    try:
        # 1. Get model from registry
        model_package = get_auth_feature(category, part, model_version, device="mps")
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
        redframe_service = get_redframe_service(category, part, advanced_mode=True, device="mps", threshold=0.1)
        redframe_result = redframe_service.forward(image, frame, position, detector.crop)
        print(f"Red Frame result: {redframe_result}")

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
