def run_prediction(image, filename: str):
    height, width = image.shape[:2]

    predicted_count = (width * height) // 50000

    if predicted_count < 10:
        density_level = "low"
    elif predicted_count < 30:
        density_level = "medium"
    else:
        density_level = "high"

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "predicted_count": predicted_count,
        "density_level": density_level
    }