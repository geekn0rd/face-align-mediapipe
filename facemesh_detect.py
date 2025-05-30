import mediapipe as mp


def detect_facemesh_landmarks(cropped_face_img):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:

        image_rgb = cropped_face_img[:, :, ::-1]  # BGR to RGB
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("No face mesh landmarks detected.")
            return []

        face_landmarks = results.multi_face_landmarks[0]

        # Return landmarks as list of (x, y) normalized coordinates
        return [(lm.x, lm.y) for lm in face_landmarks.landmark]


if __name__ == "__main__":
    import cv2

    img = cv2.imread("face_aligned_cropped.jpg")
    if img is None:
        print("Could not load cropped face image for facemesh detection.")
        exit()

    landmarks = detect_facemesh_landmarks(img)
    print(f"Detected {len(landmarks)} landmarks.")
