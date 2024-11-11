import os
import os.path as osp
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from lib.landmarks_pytorch import LandmarksEstimation
from align import align_crop_image

def process_video(input_path, output_path, transform_size=256):
    """Process a video file, detecting and aligning faces in each frame.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save processed video
        transform_size (int): Size of output frames
    """
    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (transform_size, transform_size))

    # Build landmark estimator
    le = LandmarksEstimation(type="2D")

    # Process each frame
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Landmark estimation
            img_tensor = torch.tensor(np.transpose(frame_rgb, (2, 0, 1))).float().cuda()
            with torch.no_grad():
                landmarks, detected_faces = le.detect_landmarks(
                    img_tensor.unsqueeze(0), detected_faces=None
                )

            # Align and crop face
            if len(landmarks) > 0:
                processed_frame = align_crop_image(
                    image=frame_rgb,
                    landmarks=np.asarray(landmarks[0].detach().cpu().numpy()),
                    transform_size=transform_size
                )
            else:
                # If no face detected, create blank frame or skip
                processed_frame = np.zeros((transform_size, transform_size, 3), dtype=np.uint8)

            # Convert back to BGR for writing
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            out.write(processed_frame_bgr)
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()

def main():
    """Process a single video file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input video path")
    parser.add_argument(
        "--size", type=int, default=256, help="set output size of cropped frames"
    )
    args = parser.parse_args()

    # Get input path
    input_path = osp.abspath(osp.expanduser(args.input))
    
    # Get the source file name and create new output path
    src_file_name = osp.basename(input_path)
    src_dir = osp.dirname(input_path)
    output_path = osp.join(src_dir, f"processed_{src_file_name}")

    # Create output directory if it doesn't exist
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    print(f"#. Processing video: {input_path}")
    print(f"#. Output will be saved to: {output_path}")

    try:
        process_video(input_path, output_path, args.size)
        print("#. Processing complete!")
    except Exception as e:
        print(f"#. Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    # python align_video.py --input path/to/input.mp4 --size 256
