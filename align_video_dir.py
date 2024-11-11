import os
import os.path as osp
import argparse
import logging
from datetime import datetime
from align_video import process_video  # Import from original script
import gc
import torch  # Add this import at the top


def process_directory(input_dir, output_dir, transform_size=256, tight_crop=False):
    """Process all videos in a directory and its subdirectories.

    Args:
        input_dir (str): Path to input directory
        output_dir (str): Path to output directory
        transform_size (int): Size of output frames
        tight_crop (bool): Enable tighter face cropping
    """
    # Create output directory if it doesn't exist
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Set up logging
    log_file = osp.join(
        output_dir, f'processing_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    failed_videos = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding subdirectory in output
        rel_path = osp.relpath(root, input_dir)
        curr_output_dir = osp.join(output_dir, rel_path)
        if not osp.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        # Process each video file
        for file in files:
            if file.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):  # Add more formats if needed
                input_path = osp.join(root, file)
                output_path = osp.join(curr_output_dir, file)
                print(f"\n#. Processing video: {input_path}")
                print(f"#. Output will be saved to: {output_path}")

                try:
                    process_video(input_path, output_path, transform_size, tight_crop)
                    logging.info(f"Successfully processed: {input_path}")

                    # Clear CUDA cache and run garbage collection
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                except Exception as e:
                    error_msg = f"Error processing {input_path}: {str(e)}"
                    logging.error(error_msg)
                    failed_videos.append((input_path, str(e)))

                    # Clear CUDA cache and run garbage collection even if processing fails
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue

    # Log summary of failures
    if failed_videos:
        logging.error("\nSummary of failed videos:")
        for video, error in failed_videos:
            logging.error(f"- {video}: {error}")
    else:
        logging.info("\nAll videos processed successfully!")


def main():
    """Process videos in directory maintaining directory structure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input directory path")
    parser.add_argument(
        "--size", type=int, default=256, help="set output size of cropped image"
    )
    parser.add_argument('--tight-crop', action='store_true', help='enable tighter face cropping')
    args = parser.parse_args()

    # Get input path
    input_path = osp.abspath(osp.expanduser(args.input))

    if not osp.isdir(input_path):
        raise ValueError("Input path must be a directory")

    # Get the source directory name and create new output path
    src_dir_name = osp.basename(input_path)
    output_path = osp.join(osp.dirname(input_path), f"processed_{src_dir_name}")

    print(f"#. Processing directory: {input_path}")
    print(f"#. Output directory: {output_path}")

    try:
        process_directory(input_path, output_path, args.size, args.tight_crop)
        print("#. Processing complete! Check log file for details.")
    except Exception as e:
        print(f"#. Critical error occurred: {str(e)}")
        logging.error(f"Critical error: {str(e)}")


if __name__ == "__main__":
    main()
    # python align_video_dir.py --input path/to/source_dir --size 256
