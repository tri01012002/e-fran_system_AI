from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_CONF, YOLO_IMGSZ, PADD_CROP, COLOR_PADDING, TARGET_SIZE
from PIL import Image
import cv2
from raiki_sdk.services import get_brand_detection_service


class RolexP00Detector(DetectorBase):
    """
    RolexP00Detector: uses the shared brand-detection service (Rolex/Omega/CTR/LV)
    instead of its own YOLO model. detect()/validate()/crop() are kept separate
    (instead of only run_pipeline()) so it stays compatible with routes that call
    them individually, like /api/v1/classify/.
    """

    def __init__(self, device: str):
        """
        Initialize the shared brand detection service.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = get_brand_detection_service(device=device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.padd_crop = PADD_CROP
        self.color_padding = COLOR_PADDING
        self.image_detect = None
        self.iqa_image = None
        self.output_size = TARGET_SIZE

    def detect(self, image):
        """
        Run the shared brand-detection service on the input image.

        Args:
            image: Input image (ndarray, BGR - OpenCV format).

        Returns:
            The raw detection result object from the brand detection service
            (has .category, .bbox, .cropped_image, .confidence, ...).
        """
        return self.model.detect(image=image)

    def validate(self, result) -> YoloValidation:
        """
        Validate the brand-detection result: only accept it if the detected
        category is 'rolex'.

        Args:
            result: Result object returned by detect() (brand detection service).

        Returns:
            YoloValidation: is_valid=True with position/image_crop if a Rolex
            was detected, otherwise is_valid=False.
        """
        if result is None or getattr(result, "category", None) != "rolex":
            return YoloValidation(is_valid=False, position=[], class_id=None)

        return YoloValidation(
            is_valid=True,
            position=result.bbox,
            class_id=None,
            image_crop=result.cropped_image
        )

    def crop(self, image, position) -> Image.Image:
        """
        Crop the detected region from the input image with padding.

        If the brand detection service already returned a cropped image
        (validation.image_crop), prefer reusing that instead of re-cropping,
        since it may already be preprocessed. Falls back to manual crop
        with padding when no pre-cropped image is available.

        Args:
            image: Input image (OpenCV BGR format)
            position: Coordinates [x_min, y_min, x_max, y_max] of detected object

        Returns:
            Image.Image: Cropped region as a PIL Image (RGB format)
        """
        cropped_image = padding_crop_yolo(image, position, p=self.padd_crop)

        self.image_detect = cropped_image
        # Convert from BGR (OpenCV) to RGB (PIL)
        return Image.fromarray(cv2.cvtColor(self.image_detect, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: any, is_crop: bool = True) -> YoloValidation:
        """
        Convenience all-in-one pipeline: detect + validate + (optionally)
        reuse the already-cropped image from the brand detection service,
        without needing a separate crop() call.
        """
        result = self.model.detect(image=image)
        if result is None or result.category != "rolex":
            return YoloValidation(is_valid=False, position=[], class_id=None)

        return YoloValidation(
            is_valid=True,
            position=result.bbox,
            class_id=None,
            image_crop=result.cropped_image
        )