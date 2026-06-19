"""
YOLO Detector for Louis Vuitton P04 Case4
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo, crop_center
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, COLOR_PADDING
from PIL import Image
import cv2


class LVP04Case4Detector(DetectorBase):

    def __init__(self, device: str):

        self.model = YOLO(YOLO_WEIGHTS).to(device)

        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ

        # tránh lỗi chưa khởi tạo
        self.img_rsed_square = None

        # print("YOLO Weights:", YOLO_WEIGHTS)
        # print("YOLO Classes:", self.model.names)

    def detect(self, image) -> YoloResult:

        # resize trước khi detect
        self.img_rsed_square = resize_pwd(
            image,
            output_size=YOLO_IMGSZ,
            color_padding=COLOR_PADDING
        )

        # dùng RGB thay vì Gray
        pil_img = Image.fromarray(
            cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2RGB)
        )

        results = self.model.predict(
            source=pil_img,
            verbose=False,
            show=False,
            save=False,
            save_crop=False,
            conf=self.yolo_conf,
            line_width=0,
            show_labels=False,
            show_conf=False,
            imgsz=self.yolo_imgsz
        )

        return results[0]

    def validate(self, results: YoloResult) -> YoloValidation:

        if len(results.boxes) == 0:
            # print("No detection")
            return YoloValidation(
                is_valid=False,
                position=None,
                class_id=None
            )

        best_box = None
        best_conf = -1
        best_cls = None

        for box in results.boxes:

            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            # print(
            #     f"Detect cls={cls_id}, "
            #     f"name={results.names.get(cls_id)}, "
            #     f"conf={conf}"
            # )

            # chỉ lấy class material
            if cls_id == 2:

                x_min, y_min, x_max, y_max = (
                    box.xyxy[0].cpu().numpy().astype(int)
                )

                if conf > best_conf:
                    best_conf = conf
                    best_box = [x_min, y_min, x_max, y_max]
                    best_cls = cls_id

        # không detect được class cần thiết
        if best_box is None:
            return YoloValidation(
                is_valid=False,
                position=None,
                class_id=None
            )

        return YoloValidation(
            is_valid=True,
            position=best_box,
            class_id=best_cls,
            confidence=best_conf,
            class_name=results.names.get(best_cls)
        )

    def crop(self, image, position) -> Image.Image:

        # detect chưa chạy hoặc lỗi
        if self.img_rsed_square is None:

            self.img_rsed_square = resize_pwd(
                image,
                output_size=YOLO_IMGSZ,
                color_padding=COLOR_PADDING
            )

        if position is None:

            image_detect = Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

            return crop_center(image_detect)

        cropped_image = padding_crop_yolo(
            self.img_rsed_square,
            position
        )

        image_detect = resize_pwd(
            cropped_image,
            output_size=512,
            color_padding=COLOR_PADDING
        )

        return Image.fromarray(
            cv2.cvtColor(image_detect, cv2.COLOR_BGR2RGB)
        )

    def run_pipeline(self, image, is_crop: bool = True):
        return super().run_pipeline(image, is_crop)