import os
import cv2
import numpy as np


def generate_mock_density_map(image_shape, points, sigma=45):
    h, w = image_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1.0

    density = cv2.GaussianBlur(density, (0, 0), sigma)
    return density


def create_heatmap_overlay(image, density_map):
    normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)

    return heatmap, overlay


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "samples", "test.jpg")
    output_dir = os.path.join(base_dir, "outputs", "heatmaps")

    print("Base directory:", base_dir)
    print("Input path:", input_path)
    print("Path exists:", os.path.exists(input_path))
    print("Is file:", os.path.isfile(input_path))

    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_path)

    if image is None:
        print("Image not found or could not be opened by OpenCV.")
        return

    h, w = image.shape[:2]

    mock_points = [
        (w // 4, h // 4),
        (w // 3, h // 3),
        (w // 2, h // 2),
        (2 * w // 3, h // 2),
        (3 * w // 4, 2 * h // 3),
        (w // 2, 3 * h // 4)
    ]

    density_map = generate_mock_density_map(image.shape, mock_points)
    heatmap, overlay = create_heatmap_overlay(image, density_map)

    estimated_count = len(mock_points)

    cv2.putText(
        overlay,
        f"Estimated Count: {estimated_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    heatmap_path = os.path.join(output_dir, "density_heatmap.jpg")
    overlay_path = os.path.join(output_dir, "density_overlay.jpg")

    cv2.imwrite(heatmap_path, heatmap)
    cv2.imwrite(overlay_path, overlay)

    print("Done.")
    print("Saved:", heatmap_path)
    print("Saved:", overlay_path)


if __name__ == "__main__":
    main()