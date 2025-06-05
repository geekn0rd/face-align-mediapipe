import logging  # Added for logging.getLogger
import os
import sys

# Redirect stderr to suppress MediaPipe warnings (kept from original)
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

# Suppress MediaPipe warnings (kept from original)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
os.environ["GLOG_minloglevel"] = "2"  # Suppress glog logging

import cv2
import mediapipe as mp
import numpy as np
from absl import logging as absl_logging  # Kept from original

# Import modules using the 'python' and 'vision' namespaces
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Restore stderr (kept from original)
sys.stderr = stderr

# Configure logging to suppress MediaPipe warnings (kept from original)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# Initialize absl logging to suppress STDERR warning (kept from original)
absl_logging.use_absl_handler()
absl_logging.set_verbosity(absl_logging.ERROR)

# Initialize the face landmarker
# Using vision.FaceLandmarker, python.BaseOptions, vision.FaceLandmarkerOptions, vision.RunningMode directly

# Create the face landmarker options
# Kept running_mode as VIDEO because detect_facemesh_landmarks uses frame_timestamp_ms for detect_for_video
# Updated output_facial_transformation_matrixes to True as per the "new snippet" style
# Kept other confidence parameters from the original code.
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(  # Using python.BaseOptions
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.3,
    # min_face_presence_confidence=0.3,
    # min_tracking_confidence=0.1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

# Create the face landmarker using vision.FaceLandmarker
# Ensure the model file 'face_landmarker_v2_with_blendshapes.task' is accessible
try:
    landmarker = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error creating FaceLandmarker: {e}")
    print(
        "Please ensure 'face_landmarker_v2_with_blendshapes.task' is in the correct path."
    )
    # Depending on desired behavior, you might want to exit or raise the exception
    sys.exit(1)  # Exit if landmarker cannot be created


def detect_facemesh_landmarks(cropped_face_img, frame_timestamp_ms=0, frame_num=None):
    # Print input image shape and type (kept from original)
    print(f"Input image shape: {cropped_face_img.shape}")
    print(f"Input image dtype: {cropped_face_img.dtype}")
    print(f"Frame timestamp (ms): {frame_timestamp_ms}")
    if frame_num is not None:
        print(f"Frame number: {frame_num}")

    # Validate input image (kept from original)
    if cropped_face_img is None or cropped_face_img.size == 0:
        print("Error: Invalid input image")
        return []

    if cropped_face_img.shape[0] < 10 or cropped_face_img.shape[1] < 10:
        print(f"Error: Image too small: {cropped_face_img.shape}")
        return []

    # Save the input image for debugging with frame number if available, else timestamp (kept from original)
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    if frame_num is not None:
        debug_filename = os.path.join(debug_dir, f"debug_frame_{frame_num:03d}.jpg")
    else:
        debug_filename = os.path.join(
            debug_dir, f"debug_frame_{frame_timestamp_ms}.jpg"
        )
    cv2.imwrite(debug_filename, cropped_face_img)
    print(f"Saved debug image: {debug_filename}")

    # Convert to RGB using cv2.cvtColor (kept from original)
    image_rgb = cv2.cvtColor(cropped_face_img, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image (kept from original)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect face landmarks using detect_for_video for VIDEO mode (kept from original)
    # The 'landmarker' object is now initialized with the updated options
    face_landmarker_result = landmarker.detect(mp_image)

    if not face_landmarker_result.face_landmarks:
        print("No face mesh landmarks detected.")
        # Optional detailed notes (kept from original, commented out for brevity but can be restored)
        # print("Note: This could be due to:")
        # print("1. Face not clearly visible in the image")
        # print("2. Face orientation is too extreme")
        # print("3. Poor lighting conditions")
        # print("4. Face is too small in the image")
        # print("5. Tracking confidence dropped between frames")
        return []

    # Get the first face's landmarks (kept from original)
    face_landmarks = face_landmarker_result.face_landmarks[0]
    landmarks = [(lm.x, lm.y) for lm in face_landmarks]  # Output format preserved

    # Print some statistics about the landmarks (kept from original)
    # x_coords = [lm[0] for lm in landmarks]
    # y_coords = [lm[1] for lm in landmarks]
    # print(f"Landmark coordinate ranges:")
    # print(f"X: min={min(x_coords):.3f}, max={max(x_coords):.3f}")
    # print(f"Y: min={min(y_coords):.3f}, max={max(y_coords):.3f}")
    print(f"Successfully detected {len(landmarks)} landmarks")

    # Example: Accessing face_blendshapes if needed (newly available due to options change)
    if face_landmarker_result.face_blendshapes:
        print(
            f"Detected {len(face_landmarker_result.face_blendshapes[0])} blendshape categories."
        )
        # You can process blendshapes here if desired, e.g.:
        # for category in face_landmarker_result.face_blendshapes[0]:
        #     print(f"Blendshape: {category.category_name}, Score: {category.score:.4f}")

    # Example: Accessing facial_transformation_matrixes (newly available)
    if face_landmarker_result.facial_transformation_matrixes:
        print("Facial transformation matrix available.")
        # Process matrix here: face_landmarker_result.facial_transformation_matrixes[0]

    return landmarks


if __name__ == "__main__":  # Kept original __main__ block
    img = cv2.imread("face_aligned_cropped.jpg")  # Make sure this image exists
    if img is None:
        print("Could not load cropped face image for facemesh detection.")
        # Create a dummy image for testing if not found
        print("Creating a dummy black image for demonstration...")
        img = np.zeros((256, 256, 3), dtype=np.uint8)  # Example dummy image
        cv2.imwrite(
            "face_aligned_cropped.jpg", img
        )  # Save it so it can be loaded next time if needed
        # exit() # Optionally exit if image must be real

    # Using a dummy timestamp for the test call, as in original implications
    landmarks = detect_facemesh_landmarks(img, frame_timestamp_ms=1)
    print(f"Detected {len(landmarks)} landmarks in the test image.")

    # To properly test, you might want to draw these landmarks
    if landmarks and img is not None:
        output_img = img.copy()
        height, width, _ = output_img.shape
        for x_norm, y_norm in landmarks:
            cv2.circle(
                output_img,
                (int(x_norm * width), int(y_norm * height)),
                2,
                (0, 255, 0),
                -1,
            )
        cv2.imwrite("face_landmarks_output.jpg", output_img)
        print("Saved test image with landmarks to face_landmarks_output.jpg")
        # cv2.imshow("Landmarks", output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # It's good practice to close the landmarker when done, especially if it were in a class or long-running app.
    # For a single script run, it might not be strictly necessary but good for completeness.
    # landmarker.close() # Uncomment if you want to explicitly close.
