from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, PADD_CROP, COLOR_PADDING, TARGET_SIZE
from PIL import Image
import cv2
class OMP00Detector(DetectorBase):
    """
    OMP00Detector: A custom YOLO-based object detector that extends DetectorBase.
    Handles detection, validation, and cropping of detected regions.
    """

    def __init__(self, device: str):
        """
        Initialize the YOLO model with pre-defined configuration.
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.padd_crop = PADD_CROP
        self.color_padding = COLOR_PADDING
        self.image_detect = None
        self.iqa_image = None
        self.output_size = TARGET_SIZE

    def detect(self, image) -> YoloResult:
        """
        Perform object detection using YOLO model.
        
        Args:
            image: Input image (can be file path or ndarray)
            
        Returns:
            YoloResult: YOLO detection results for the given image.
        """
        results = self.model.predict(
            source=image,
            show=False,
            save=False,
            save_crop=False,
            conf=self.yolo_conf,
            line_width=0,
            show_labels=False,
            show_conf=False
        )
        return results[0]

    def validate(self, result: YoloResult) -> YoloValidation:
        """
        Validate detection results:
        - Select the watch box (cls_id = 0) with the highest confidence
        - Select the Omega logo box (cls_id = 2) with the highest confidence
        - The Omega logo box must be inside the watch box
        """

        # Return invalid if no detections are found

        best_watch_box = None
        best_watch_conf = -1

        best_omega_box = None
        best_omega_conf = -1

        # Iterate over all detected boxes
        for box in result.boxes:
            cls_id = int(box.cls)          # Class ID
            conf = box.conf.item()         # Confidence score

            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            # Watch class (cls_id = 0)
            if cls_id == 0:
                # Keep the watch box with the highest confidence
                if conf > best_watch_conf:
                    best_watch_conf = conf
                    best_watch_box = [x_min, y_min, x_max, y_max]

            # Omega logo class (cls_id = 1)
            elif cls_id == 1:
                # Keep the Omega logo box with the highest confidence
                if conf > best_omega_conf:
                    best_omega_conf = conf
                    best_omega_box = [x_min, y_min, x_max, y_max]

        # No watch or no Omega logo detected
        if best_watch_box is None or best_omega_box is None:
            return YoloValidation(is_valid=False, position=[], class_id=None)

        wx1, wy1, wx2, wy2 = best_watch_box
        ox1, oy1, ox2, oy2 = best_omega_box

        # Check if the Omega logo box is fully inside the watch box
        if not (ox1 >= wx1 and oy1 >= wy1 and ox2 <= wx2 and oy2 <= wy2):
            return YoloValidation(is_valid=False, position=[], class_id=None)

        # Valid result: watch contains Omega logo
        return YoloValidation(
            is_valid=True,
            position=best_watch_box,
            class_id=0
        )

    def crop(self, image, position) -> Image.Image:
        """
        Crop the detected region from the input image with padding.
        
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
        Execute the full detection pipeline.
        
        Args:
            image: Input image (any format supported by base class)
            is_crop: Whether to perform cropping after detection.
            
        Returns:
            YoloValidation: Validation result of the detection.
        """
        return super().run_pipeline(image, is_crop)
