from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, PADD_CROP, TARGET_SIZE, YOLO_INDEX
from PIL import Image
import cv2

class LVP00Detector(DetectorBase):
    """
    LVP00Detector: A custom YOLO-based object detector that extends DetectorBase.
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

    def validate(self, results) -> YoloValidation:
        """
        Validate detection results and select the most confident box.
        
        Args:
            yolo_results: YOLO detection results.
            
        Returns:
            YoloValidation: Object containing validation status, 
                            detected position, and class ID.
        """
        # Return invalid if no detections are found
        if len(results) == 0 or len(results[0].boxes) == 0:
            return YoloValidation(is_valid=False, position=[], class_id=None)

        best_box = None
        best_conf = -1
        valid = False

        # Iterate over all detected boxes
        for box in results[0].boxes:
            cls_id = int(box.cls)           # Class ID
            conf = box.conf.item()          # Confidence score
            # Model
            # Validate box has index of bag
            if cls_id == YOLO_INDEX:
                valid = True
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # Keep the box with the highest confidence score
                if conf > best_conf:
                    best_conf = conf
                    best_box = [x_min, y_min, x_max, y_max]

        return YoloValidation(is_valid=valid, position=best_box, class_id=None)

    def crop(self, image, position) -> Image.Image:
        """
        Crop the detected region from the input image with padding.
        
        Args:
            image: Input image (OpenCV BGR format)
            position: Coordinates [x_min, y_min, x_max, y_max] of detected object
            
        Returns:
            Image.Image: Cropped region as a PIL Image (RGB format)
        """
        cropped_image = padding_crop_yolo(image, position, p=PADD_CROP)

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
