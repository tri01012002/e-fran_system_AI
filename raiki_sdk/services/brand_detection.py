from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import cv2
import logging
from PIL import Image
from ultralytics import YOLO
from raiki_sdk.preprocessing import resize_pwd
from raiki_sdk.base import DetectionResult
from .config import (
    WATCH_DETECTOR_WEIGHTS,
    WATCH_DETECTOR_IMGSZ,
    WATCH_DETECTOR_CONFIDENCE,
    WATCH_CLASS_MAP,
    WATCH_BRAND_RULES,
)
from raiki_sdk.services.classify_brand import get_main_model as get_watch_brand_classifier

_logger = logging.getLogger(__name__)

# ============================================================================
# BRAND DETECTOR
# ============================================================================

class BrandDetector:
    """
    Unified brand detector using YOLO model.
    
    Detects luxury product brands from images using a single YOLO model
    trained on 4 classes. Handles both watches (Rolex/Omega) and bags (LV).
    
    Detection Logic:
        - If "lv" detected → return "bag-lv"
        - If "watch" + "rolex_logo" → return "rolex"
        - If "watch" + "omega_logo" → return "omg"
        - If "watch" only (no logo) → return "unknown_watch"
        - If nothing detected → return None
    
    Attributes:
        model_path: Path to YOLO weights file
        class_map: Mapping from class_id to class_name
        brand_rules: Mapping from logo class to category name
        imgsz: Input image size for YOLO (default: 640)
        confidence: Detection confidence threshold
        device: Compute device ("cuda", "mps", "cpu")
    
    Example:
        >>> detector = BrandDetector(device="cuda")
        >>> result = detector.detect(image)
        >>> if result:
        ...     print(f"Detected: {result.category}")
    """
    
    # Primary object classes (used for cropping)
    PRIMARY_OBJECTS = ("watch", "lv")
    
    def __init__(
        self,
        device: str = "cuda",
        confidence: Optional[float] = None,
    ):
        """
        Initialize BrandDetector.
        
        Args:
            device: Compute device for inference
            confidence: Override default confidence threshold
        """
        self.model_path = WATCH_DETECTOR_WEIGHTS
        self.class_map = WATCH_CLASS_MAP
        self.brand_rules = WATCH_BRAND_RULES
        self.imgsz = WATCH_DETECTOR_IMGSZ
        self.confidence = confidence or WATCH_DETECTOR_CONFIDENCE
        self.device = device
        self._model = None
        self._watch_brand_classifier = get_watch_brand_classifier(device=self.device)
    
    @property
    def model(self):
        """
        Lazy-load YOLO model on first access.
        
        Returns:
            YOLO model loaded on specified device
        """
        if self._model is None:
            self._model = YOLO(self.model_path).to(self.device)
            _logger.info(f"Loaded brand detector on {self.device}")
        return self._model
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO inference.
        
        Converts to grayscale (to focus on shape, not color),
        stacks to 3 channels, and resizes with white padding.
        
        Args:
            image: Input BGR image (H, W, C)
        
        Returns:
            Preprocessed image (imgsz, imgsz, 3)
        """
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Stack to 3 channels (YOLO expects 3-channel input)
        image_3ch = cv2.merge([gray, gray, gray])
        
        # Resize with aspect ratio preservation and white padding
        return resize_pwd(
            img=image_3ch,
            output_size=self.imgsz,
            color_padding=(255, 255, 255)
        )
    
    def _parse_detections(self, results) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[str]]:
        """
        Parse YOLO results into detected classes and best bounding .
        
        Args:
            results: YOLO inference results
        
        Returns:
            Tuple of:
                - detected: Dict mapping class_name to highest confidence
                - best_box: Bounding box of primary object (watch/lv)
                - best_box_class: Class name of the primary object
        """
        detected: Dict[str, float] = {}
        best_box: Optional[np.ndarray] = None
        best_box_class: Optional[str] = None
        best_box_conf: float = 0.0
        
        for r in results:
            if r.boxes is None:
                continue
                
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                cls_name = self.class_map.get(cls_id)
                
                if cls_name is None:
                    continue
                
                # Store highest confidence per class
                if cls_name not in detected or conf > detected[cls_name]:
                    detected[cls_name] = conf
                
                # Track best primary object (watch or lv)
                if cls_name in self.PRIMARY_OBJECTS and conf > best_box_conf:
                    best_box = box.xyxy[0].cpu().numpy()
                    best_box_class = cls_name
                    best_box_conf = conf
        
        return detected, best_box, best_box_class
    
    def _scale_bbox_to_original(
        self,
        box: np.ndarray,
        orig_h: int,
        orig_w: int
    ) -> Tuple[int, int, int, int]:
        """
        Scale bounding box from preprocessed image back to original image coordinates.
        
        Args:
            box: Bounding box [x1, y1, x2, y2] in preprocessed image space
            orig_h: Original image height
            orig_w: Original image width
        
        Returns:
            Tuple (x1, y1, x2, y2) in original image coordinates
        """
        # Calculate scale factor used in preprocessing
        scale = self.imgsz / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # Calculate padding offsets (image is centered)
        x_offset = (self.imgsz - new_w) // 2
        y_offset = (self.imgsz - new_h) // 2
        
        # Scale back to original coordinates
        x1 = max(0, int((box[0] - x_offset) / scale))
        y1 = max(0, int((box[1] - y_offset) / scale))
        x2 = min(orig_w, int((box[2] - x_offset) / scale))
        y2 = min(orig_h, int((box[3] - y_offset) / scale))
        
        return x1, y1, x2, y2
    
    def _crop_and_convert(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Image.Image:
        """
        Crop image region and convert to PIL Image (RGB).
        
        Args:
            image: Original BGR image
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Cropped PIL Image in RGB format
        """
        x1, y1, x2, y2 = bbox
        cropped_np = image[y1:y2, x1:x2].copy()
        cropped_rgb = cv2.cvtColor(cropped_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_rgb)
    
    def _determine_category(
        self,
        detected: Dict[str, float],
        best_box_class: Optional[str],
        cropped_watch: Optional[Image.Image] = None
    ) -> Tuple[Optional[str], float]:
        """
        Determine final category based on detected classes.
        
        Decision Logic:
            1. LV detected → "bag-lv"
            2. Watch + brand logo → brand category
            3. Watch only → "unknown_watch"
            4. Nothing useful → None
        
        Args:
            detected: Dict of detected classes with confidences
            best_box_class: Class of primary detected object
        
        Returns:
            Tuple of (category, confidence) or (None, 0.0)
        """
        print(detected)
        # Priority 1: LV bag
        if "lv" in detected:
            return self.brand_rules.get("lv", "bag-lv"), detected["lv"]
        
        # Priority 2: Watch with brand
        if "watch" in detected:
            for logo_class, category in self.brand_rules.items():
                if logo_class == "lv":
                    continue
                if logo_class in detected:
                    return category, detected[logo_class]

            print("cropped_watch is not None:", cropped_watch is not None)
            print("_watch_brand_classifier is not None:", self._watch_brand_classifier is not None)
            if cropped_watch is not None and self._watch_brand_classifier is not None:
                try:
                    label, prob = self._watch_brand_classifier.forward(cropped_watch)
                    if prob >= 0.4:
                        _logger.info(
                            "Watch brand classified by classifier: %s (%.3f)",
                            label, prob
                        )
                        print(f"detected['watch'] : {detected['watch']},  prob: {prob}")
                        combined_conf = min(detected["watch"],  prob)
                        return label, combined_conf
                    else:
                        _logger.warning(
                            "Watch brand classifier low confidence: %.3f, keep unknown_watch",
                            prob,
                        )
                except Exception as e:
                    _logger.exception("Error when running watch brand classifier: %s", e)

            _logger.warning("Watch detected but brand could not be determined")
            return "unknown_watch", detected["watch"]

        return None, 0.0
    
    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        Detect brand from image.
        
        Args:
            image: Input BGR image (numpy array)
        
        Returns:
            DetectionResult if brand detected, None otherwise
        """
        # Preprocess
        processed = self.preprocess(image)
        
        # Run inference
        results = self.model(processed, verbose=False, conf=self.confidence)
        
        # Parse detections
        detected, best_box, best_box_class = self._parse_detections(results)
        if not detected:
            _logger.debug("No objects detected")
            return None

        _logger.debug(f"Detected classes: {detected}")

        cropped_pil: Optional[Image.Image] = None
        bbox_tuple: Optional[Tuple[int, int, int, int]] = None

        if best_box is not None:
            orig_h, orig_w = image.shape[:2]
            bbox_tuple = self._scale_bbox_to_original(best_box, orig_h, orig_w)
            cropped_pil = self._crop_and_convert(image, bbox_tuple)

        # truyền thêm cropped_watch
        category, confidence = self._determine_category(
            detected=detected,
            best_box_class=best_box_class,
            cropped_watch=cropped_pil if best_box_class == "watch" else None,
        )

        if category is None:
            return None

        return DetectionResult(
            category=category,
            confidence=confidence,
            cropped_image=cropped_pil,
            bbox=bbox_tuple
        )

# ============================================================================
# SERVICE
# ============================================================================

class BrandDetectionService:
    """
    Brand detection service with caching.
    
    Provides a high-level interface for brand detection with lazy
    model loading. Uses singleton pattern for efficient resource usage.
    
    Example:
        >>> from raiki_sdk.services import get_brand_detection_service
        >>> service = get_brand_detection_service(device="cuda")
        >>> result = service.detect(image)
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize service.
        
        Args:
            device: Compute device for inference
        """
        self.device = device
        self._detector: Optional[BrandDetector] = None
    
    @property
    def detector(self) -> BrandDetector:
        """
        Get brand detector (lazy initialization).
        
        Returns:
            BrandDetector instance
        """
        if self._detector is None:
            self._detector = BrandDetector(device=self.device)
            _logger.info(f"Initialized BrandDetector on {self.device}")
        return self._detector
    
    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        Detect brand from image.
        
        Args:
            image: Input BGR image (numpy array, H x W x 3)
        
        Returns:
            DetectionResult if brand detected, None otherwise
            
        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")
        
        return self.detector.detect(image)