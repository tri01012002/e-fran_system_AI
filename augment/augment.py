import os
import cv2
import albumentations as A

# Thư mục ảnh đầu vào / đầu ra
INPUT_DIR = "LV_Marylebone_N41215"
OUTPUT_DIR = "augment_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Các phép augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),

    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(-0.1, 0.1),
        rotate=(-20, 20),
        shear=(-10, 10),
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=15,
        p=0.3
    ),

    A.GaussianBlur(
        blur_limit=(3, 7),
        p=0.2
    ),

    A.GaussNoise(p=0.2),
])


# Số ảnh augment tạo ra cho mỗi ảnh gốc
NUM_AUGMENTATIONS = 5

for filename in os.listdir(INPUT_DIR):
    input_path = os.path.join(INPUT_DIR, filename)

    # Chỉ xử lý các file ảnh
    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ):
        continue

    image = cv2.imread(input_path)

    if image is None:
        print(f"Không đọc được: {filename}")
        continue

    for i in range(NUM_AUGMENTATIONS):
        augmented = transform(image=image)
        aug_image = augmented["image"]

        name, ext = os.path.splitext(filename)
        output_name = f"{name}_aug_{i}{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        cv2.imwrite(output_path, aug_image)

    print(f"Đã augment: {filename}")

print("Hoàn thành!")
