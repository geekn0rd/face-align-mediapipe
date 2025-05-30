import cv2 as cv
import numpy as np


def draw_landmarks_on_original(original_img, landmarks, crop_coords, rotation_matrix):
    """
    Transform landmarks from cropped face coordinates back to original image coordinates.

    Parameters:
        original_img: Original input image
        landmarks: List of (x, y) coordinates in normalized space (0-1) relative to cropped face
        crop_coords: (x1, y1, x2, y2) coordinates of the crop in rotated image
        rotation_matrix: 2x3 affine transformation matrix used for rotation

    Returns:
        Original image with landmarks drawn
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    # Create inverse rotation matrix
    inv_rotation_matrix = cv.invertAffineTransform(rotation_matrix)

    # Draw landmarks with a slightly larger radius and different color for better visibility
    for nx, ny in landmarks:
        # Step 1: Convert normalized coordinates to cropped image coordinates
        x_crop = int(nx * crop_width)
        y_crop = int(ny * crop_height)

        # Step 2: Add crop offset to get coordinates in rotated image
        x_rot = x_crop + crop_x1
        y_rot = y_crop + crop_y1

        # Step 3: Convert to original image coordinates using inverse rotation
        point = np.array([[x_rot, y_rot]], dtype=np.float32)
        original_point = cv.transform(
            point.reshape(-1, 1, 2), inv_rotation_matrix
        ).reshape(-1, 2)

        x = int(original_point[0, 0])
        y = int(original_point[0, 1])

        # Ensure coordinates are within image bounds
        x = max(0, min(x, original_img.shape[1] - 1))
        y = max(0, min(y, original_img.shape[0] - 1))

        # Draw a slightly larger circle for better visibility
        cv.circle(original_img, (x, y), 2, (0, 0, 255), -1)  # Red color, filled circle

    return original_img


if __name__ == "__main__":
    import sys

    import cv2

    original_img = cv.imread("/Users/a2m/Desktop/test_pp.png")
    if original_img is None:
        print("Could not load original image.")
        sys.exit()

    # Example: Load landmarks and crop coords (you need to get these from your pipeline)
    import pickle

    # This is just an example - replace these with your actual data loading logic
    with open("landmarks.pkl", "rb") as f:
        landmarks = pickle.load(f)
    with open("crop_coords.pkl", "rb") as f:
        crop_coords = pickle.load(f)
    with open("rotation_matrix.pkl", "rb") as f:
        rotation_matrix = pickle.load(f)

    annotated_img = draw_landmarks_on_original(
        original_img, landmarks, crop_coords, rotation_matrix
    )
    cv.imwrite("annotated_original.jpg", annotated_img)
    print("Annotated original image saved.")
