import cv2 as cv
import numpy as np
from retinaface import RetinaFace


def align_and_crop_face(img):
    detections = RetinaFace.detect_faces(img)
    if not detections or "face_1" not in detections:
        print("No face detected!")
        return None, None, None, None

    face_data = detections["face_1"]
    x1, y1, x2, y2 = face_data["facial_area"]
    m1, m2 = face_data["landmarks"]["right_eye"]
    m3, m4 = face_data["landmarks"]["left_eye"]

    delta_y = m4 - m2
    delta_x = m3 - m1
    angle_deg = np.degrees(np.arctan2(delta_y, delta_x))
    rotation_center = ((m1 + m3) / 2, (m2 + m4) / 2)

    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D(rotation_center, angle_deg, 1.0)
    rotated_img = cv.warpAffine(
        img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT
    )

    # Transform facial area coordinates to rotated image
    original_corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]).T
    transformed_corners = M @ original_corners

    rotated_x1 = int(np.min(transformed_corners[0]))
    rotated_y1 = int(np.min(transformed_corners[1]))
    rotated_x2 = int(np.max(transformed_corners[0]))
    rotated_y2 = int(np.max(transformed_corners[1]))

    padding = 100
    crop_x1 = max(0, rotated_x1 - padding)
    crop_y1 = max(0, rotated_y1 - padding)
    crop_x2 = min(w, rotated_x2 + padding)
    crop_y2 = min(h, rotated_y2 + padding)

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        print("Invalid crop dimensions. Returning full rotated image.")
        return rotated_img, (0, 0, w, h), img.shape, M

    aligned_face_crop = rotated_img[crop_y1:crop_y2, crop_x1:crop_x2]

    return aligned_face_crop, (crop_x1, crop_y1, crop_x2, crop_y2), img.shape, M


if __name__ == "__main__":
    img_path = "/Users/a2m/Desktop/test_pp.png"
    img = cv.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        exit()

    cropped_face, crop_coords, original_shape, rotation_matrix = align_and_crop_face(
        img
    )
    if cropped_face is not None:
        cv.imwrite("face_aligned_cropped.jpg", cropped_face)
        print("Saved aligned cropped face.")
        print("Crop coords:", crop_coords)
        print("Original image shape:", original_shape)
