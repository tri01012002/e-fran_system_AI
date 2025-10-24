from raiki_sdk.base import YoloBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ
from PIL import Image
import cv2
import numpy as np
import torch


class RolexP03Detector(YoloBase):

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.position = {}
        self.image_detect = None
        self.iqa_image = None
        self.output_size = 224
        self.color_padding = (85, 85, 85)

    def yolo_detect(self, image, frame=None, iqa=True):
        img_input_yolo_cv2 = image
        img_input_yolo_pil = Image.fromarray(cv2.cvtColor(img_input_yolo_cv2, cv2.COLOR_BGR2GRAY))

        results = self.model.predict(source=img_input_yolo_pil, verbose=False, show=False, save=False,
                                     save_crop=False, conf=self.yolo_conf, line_width=0,
                                     show_labels=False, show_conf=False, imgsz=self.yolo_imgsz)
        # validation = self.validate(results, frame)
        #
        # if not validation.is_valid:
        #     return False
        #
        # self.position = validation.position
        #
        # img_padding_yolo_cv2 = padding_crop_yolo(img_input_yolo_cv2, self.position, p=0)
        #
        # self.image_detect = self.resize_pwd_training_compatible(
        #     img_padding_yolo_cv2,
        #     output_size=self.output_size,
        #     color_padding=self.color_padding
        # )
        #
        # if iqa:
        #     self.iqa_image = img_padding_yolo_cv2

        return results

    def validate(self, yolo_results, frame=None):
        boxesn = yolo_results[0].boxes.xyxyn.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().to(torch.int).numpy()

        for i in range(len(classes)):
            if (classes[i] == 1 and yolo_results[0].boxes[i].conf < 0.55):
                return False, {}

        bbox_class_inner = [boxesn[i] for i in range(len(classes)) if classes[i] == 0]
        bbox_class_outer = [boxesn[i] for i in range(len(classes)) if classes[i] == 1]

        valid = False
        position = None
        p03_boxesn = []

        if bbox_class_inner and bbox_class_outer:
            for bbox1 in bbox_class_inner:
                x_min1, y_min1, x_max1, y_max1 = bbox1
                for bbox2 in bbox_class_outer:
                    x_min2, y_min2, x_max2, y_max2 = bbox2
                    w = x_max2 - x_min2
                    p = 0 * w
                    if x_min1 >= x_min2 and y_min1 >= y_min2 and x_max1 <= x_max2 and y_max1 <= y_max2:
                        p03_boxesn.append(
                            [max(0, x_min2 - p), max(0, y_min2 - p), min(x_max2 + p, 1), min(1, y_max2 + p)])

        if frame:
            p03_boxesn = [boxn for boxn in p03_boxesn if
                          boxn[0] > frame[0] and boxn[1] > frame[1] and boxn[2] < frame[2] and boxn[3] < frame[3]]

        if len(p03_boxesn) > 0:
            valid = True
            image_h, image_w = yolo_results[0].boxes.orig_shape
            x1n, y1n, x2n, y2n = p03_boxesn[0]
            x1, y1, x2, y2 = int(x1n * image_w), int(y1n * image_h), int(x2n * image_w), int(y2n * image_h)
            position = [x1, y1, x2, y2]

        return YoloValidation(is_valid=valid, position=position, class_id=None)

    def crop(self, image, position) -> Image.Image:
        cropped_image = padding_crop_yolo(image, position)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)

    def resize_pwd_training_compatible(self, img, output_size=224, color_padding=(85, 85, 85)):
        if isinstance(img, Image.Image):
            img = np.array(img)

        h, w = img.shape[:2]

        scale = min(output_size / h, output_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((output_size, output_size, 3), color_padding, dtype=np.uint8)

        y_offset = (output_size - new_h) // 2
        x_offset = (output_size - new_w) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        return canvas