# Face Annotation Pipeline

This project provides a pipeline for face detection, alignment, and landmark annotation on images and videos.

## Usage

### Annotate an Image

To annotate a single image and save the result:

```sh
python main.py --image path/to/your/image.png
```

- The annotated image will be saved as `annotated_original.jpg`.
- Keypoints will be saved in the `keypoints_output/` directory.

### Annotate a Video

To annotate every frame of a video and compile the results into a new video:

```sh
python main.py --video path/to/your/video.mp4
```

- Each frame (annotated or untouched if no face is found) will be saved in the `video_frames_output/` directory.
- The compiled annotated video will be saved as `annotated_video.mp4`.

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Notes

- Progress bars are shown for video processing using `tqdm`.
- If a frame does not contain a face, the original frame is saved without annotation.
