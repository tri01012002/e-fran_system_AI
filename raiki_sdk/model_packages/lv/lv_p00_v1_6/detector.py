from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_CONF, YOLO_IMGSZ, PADD_CROP, COLOR_PADDING, TARGET_SIZE
from PIL import Image
import cv2
from raiki_sdk.services import get_brand_detection_service
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
        self.model = get_brand_detection_service(device=device)
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
        # Use brand detection service to get detection result
        result = self.model.detect(image=image)
        return result

    def validate(self, results) -> YoloValidation:
        """
        Validate detection results from brand detection service.
        
        Args:
            results: Detection results from brand_detection_service.
            
        Returns:
            YoloValidation: Object containing validation status, 
                            detected position, and cropped image.
        """
        # Return invalid if results is None
        if results is None:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        # Check category
        if hasattr(results, 'category') and results.category != "bag-lv":
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        # Return valid with bbox and cropped image
        return YoloValidation(
            is_valid=True,
            position=results.bbox if hasattr(results, 'bbox') else None,
            class_id=None,
            image_crop=results.cropped_image if hasattr(results, 'cropped_image') else None
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
        result = self.model.detect(image=image)
        if result.category != "bag-lv":
            return YoloValidation(is_valid = False)
        
        return YoloValidation(
            is_valid = True,
            position= result.bbox,
            image_crop = result.cropped_image
        )
        # class YoloValidation(BaseModel):
        #     model_config = ConfigDict(arbitrary_types_allowed=True)
        #     is_valid: bool
        #     position: list[int] | dict | None = None 
        #     class_id: int | None = None
        #     image_crop: Optional[Image.Image | dict] = None