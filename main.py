import argparse
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

import draw_landmarks as draw_landmarks
import face_align as face_align
import facemesh_detect


def run_pipeline(img, keypoints_dir=None, frame_num=None):
    """
    Run the face detection and landmark pipeline on an image.

    Parameters:
        img (np.ndarray): Input image
        keypoints_dir (str, optional): Directory to save keypoints. If None, keypoints won't be saved.
        frame_num (int, optional): Frame number for timestamp calculation

    Returns:
        np.ndarray: Annotated image with landmarks drawn
    """
    # Step 1: Align and crop face
    cropped_face, crop_coords, original_shape, rotation_matrix = (
        face_align.align_and_crop_face(img)
    )
    if cropped_face is None:
        print("Face alignment and cropping failed.")
        return None

    # Step 2: Detect FaceMesh landmarks on cropped image
    # Calculate timestamp in milliseconds (assuming 30 fps)
    frame_timestamp_ms = (
        int(frame_num * (1000.0 / 30.0)) if frame_num is not None else 0
    )
    landmarks = facemesh_detect.detect_facemesh_landmarks(
        cropped_face, frame_timestamp_ms, frame_num=frame_num
    )
    if not landmarks:
        print("No landmarks detected.")
        return None

    # Step 3: Draw landmarks back on original image
    annotated_img = draw_landmarks.draw_landmarks_on_original(
        img.copy(), landmarks, crop_coords, rotation_matrix
    )

    # Step 4: Save keypoints if keypoints_dir is provided
    if keypoints_dir is not None and frame_num is not None:
        os.makedirs(keypoints_dir, exist_ok=True)

        # Convert landmarks to original image coordinates
        original_keypoints = []
        for lm in landmarks:
            # Get coordinates in cropped image space
            x_crop = lm[0] * cropped_face.shape[1]
            y_crop = lm[1] * cropped_face.shape[0]
            z = lm[2] if len(lm) > 2 else 0.0

            # Convert to original image coordinates
            x_orig = x_crop + crop_coords[0]
            y_orig = y_crop + crop_coords[1]

            # Apply inverse rotation if needed
            if rotation_matrix is not None:
                # Create a 2x1 point matrix
                point = np.array([[x_orig], [y_orig]], dtype=np.float32)
                # Get inverse rotation matrix
                inv_rotation = cv.invertAffineTransform(rotation_matrix)
                # Apply inverse rotation
                rotated_point = np.dot(inv_rotation, np.vstack([point, [[1.0]]]))
                x_orig, y_orig = rotated_point[0, 0], rotated_point[1, 0]

            # Normalize coordinates to [0,1] range
            x_orig = x_orig / original_shape[1]
            y_orig = y_orig / original_shape[0]

            original_keypoints.append([x_orig, y_orig, z])

        original_keypoints = np.array(original_keypoints)

        # Save keypoints in original image space
        np.save(
            os.path.join(keypoints_dir, f"face_landmarks_{frame_num:05d}.npy"),
            original_keypoints,
        )
        print(f"Saved keypoints for frame {frame_num} to {keypoints_dir}")

    return annotated_img


def process_video(video_path, output_frames_dir, output_video_path):
    """
    Process the video for face detection and landmark annotation.

    Parameters:
        video_path (str): Path to the input video
        output_frames_dir (str): Directory to save annotated frames
        output_video_path (str): Path to save the compiled output video
    """
    os.makedirs(output_frames_dir, exist_ok=True)
    keypoints_dir = os.path.join(output_frames_dir, "keypoints")
    os.makedirs(keypoints_dir, exist_ok=True)
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_paths = []
    all_landmarks = []  # List to store landmarks for each frame
    idx = 0
    successful_frames = 0  # Counter for successful frames
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_img = run_pipeline(
                frame, keypoints_dir=keypoints_dir, frame_num=idx
            )
            if processed_img is not None:
                successful_frames += 1
            # Load landmarks if saved, else fill with -1s of correct shape
            landmarks_path = os.path.join(
                keypoints_dir, f"face_landmarks_{idx:05d}.npy"
            )
            if os.path.exists(landmarks_path):
                frame_landmarks = np.load(landmarks_path).tolist()
            else:
                # Try to infer landmark size from previous frame, else default to 478 (MediaPipe FaceMesh)
                if (
                    all_landmarks
                    and all_landmarks[-1] is not None
                    and all_landmarks[-1] != [[-1, -1, -1]]
                ):
                    n_points = len(all_landmarks[-1])
                else:
                    n_points = 478
                frame_landmarks = [[-1, -1, -1]] * n_points
            all_landmarks.append(frame_landmarks)
            if processed_img is None:
                processed_img = frame  # Save untouched frame if no face
            frame_file = os.path.join(output_frames_dir, f"frame_{idx:05d}.jpg")
            cv.imwrite(frame_file, processed_img)
            frame_paths.append(frame_file)
            idx += 1
            pbar.set_postfix(successful=f"{successful_frames}/{frame_count}")
            pbar.update(1)
    cap.release()
    # Save all landmarks as a list to a file
    np.save(
        os.path.join(output_frames_dir, "all_landmarks_list.npy"),
        np.array(all_landmarks, dtype=object),
    )
    # Compile frames into video
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame_file in tqdm(frame_paths, desc="Compiling video"):
        img = cv.imread(frame_file)
        out.write(img)
    out.release()
    print(f"Annotated video saved as '{output_video_path}'.")
    print(
        f"All frame landmarks saved as 'all_landmarks_list.npy' in {output_frames_dir}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face annotation on image or video.")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    args = parser.parse_args()

    if args.video:
        video_path = args.video
        output_frames_dir = "video_frames_output"
        output_video_path = "annotated_video.mp4"
        process_video(video_path, output_frames_dir, output_video_path)
    else:
        # Default to image mode
        image_path = args.image or "/Users/a2m/Desktop/test_pp.png"
        img = cv.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            exit()
        save_dir = "keypoints_output"
        processed_img = run_pipeline(img, save_dir)
        if processed_img is not None:
            cv.imwrite("annotated_original.jpg", processed_img)
            print("Annotated original image saved as 'annotated_original.jpg'.")
