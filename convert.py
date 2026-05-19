import os
import scipy.io
from PIL import Image

dataset_path = "ShanghaiTech/part_A/test_data"

images_path = os.path.join(dataset_path, "images")
gt_path = os.path.join(dataset_path, "ground-truth")

labels_output = os.path.join(dataset_path, "labels")
os.makedirs(labels_output, exist_ok=True)

for img_file in os.listdir(images_path):

    if img_file.endswith(".jpg"):

        img_path = os.path.join(images_path, img_file)

        img = Image.open(img_path)
        w, h = img.size

        mat_file = "GT_" + img_file.replace(".jpg", ".mat")
        mat_path = os.path.join(gt_path, mat_file)

        mat = scipy.io.loadmat(mat_path)

        points = mat["image_info"][0][0][0][0][0]

        label_file = os.path.join(
            labels_output,
            img_file.replace(".jpg", ".txt")
        )

        with open(label_file, "w") as f:

            for p in points:

                x, y = p

                x_center = x / w
                y_center = y / h

                box_w = 20 / w
                box_h = 20 / h

                f.write(
                    f"0 {x_center} {y_center} {box_w} {box_h}\n"
                )

print("Conversion complete!")