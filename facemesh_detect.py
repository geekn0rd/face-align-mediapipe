import os
import sys

# Redirect stderr to suppress MediaPipe warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

# Suppress MediaPipe warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
os.environ["GLOG_minloglevel"] = "2"  # Suppress glog logging

import logging

import cv2
import mediapipe as mp
from absl import logging as absl_logging
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Restore stderr
sys.stderr = stderr

# Configure logging to suppress MediaPipe warnings
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# Initialize absl logging to suppress STDERR warning
absl_logging.use_absl_handler()
absl_logging.set_verbosity(absl_logging.ERROR)

# Initialize the face landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create the face landmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.1,  # Lowered tracking confidence to be more lenient
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
)

# Create the face landmarker
landmarker = FaceLandmarker.create_from_options(options)


def detect_facemesh_landmarks(cropped_face_img, frame_timestamp_ms=0):
    # Print input image shape and type
    print(f"Input image shape: {cropped_face_img.shape}")
    print(f"Input image dtype: {cropped_face_img.dtype}")
    print(f"Frame timestamp (ms): {frame_timestamp_ms}")

    # Validate input image
    if cropped_face_img is None or cropped_face_img.size == 0:
        print("Error: Invalid input image")
        return []

    if cropped_face_img.shape[0] < 10 or cropped_face_img.shape[1] < 10:
        print(f"Error: Image too small: {cropped_face_img.shape}")
        return []

    # Save the input image for debugging with timestamp
    debug_filename = f"debug_frame_{frame_timestamp_ms}_before_rgb.jpg"
    cv2.imwrite(debug_filename, cropped_face_img)
    print(f"Saved debug image: {debug_filename}")

    # Convert to RGB using cv2.cvtColor
    image_rgb = cv2.cvtColor(cropped_face_img, cv2.COLOR_BGR2RGB)

    # Save the RGB image for debugging - convert back to BGR for saving
    debug_rgb_filename = f"debug_frame_{frame_timestamp_ms}_after_rgb.jpg"
    cv2.imwrite(debug_rgb_filename, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved RGB debug image: {debug_rgb_filename}")

    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect face landmarks using detect_for_video for VIDEO mode
    face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if not face_landmarker_result.face_landmarks:
        print("No face mesh landmarks detected.")
        print("Note: This could be due to:")
        print("1. Face not clearly visible in the image")
        print("2. Face orientation is too extreme")
        print("3. Poor lighting conditions")
        print("4. Face is too small in the image")
        print("5. Tracking confidence dropped between frames")
        return []

    # Get the first face's landmarks
    face_landmarks = face_landmarker_result.face_landmarks[0]
    landmarks = [(lm.x, lm.y) for lm in face_landmarks]

    # Print some statistics about the landmarks
    x_coords = [lm[0] for lm in landmarks]
    y_coords = [lm[1] for lm in landmarks]
    print(f"Landmark coordinate ranges:")
    print(f"X: min={min(x_coords):.3f}, max={max(x_coords):.3f}")
    print(f"Y: min={min(y_coords):.3f}, max={max(y_coords):.3f}")
    print(f"Successfully detected {len(landmarks)} landmarks")

    return landmarks


if __name__ == "__main__":
    img = cv2.imread("face_aligned_cropped.jpg")
    if img is None:
        print("Could not load cropped face image for facemesh detection.")
        exit()

    landmarks = detect_facemesh_landmarks(img)
    print(f"Detected {len(landmarks)} landmarks.")
