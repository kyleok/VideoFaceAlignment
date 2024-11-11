# FFHQFaceAlignment with Video Support

This repository is forked from [@chi0tzp/FFHQFaceAlignment](https://github.com/chi0tzp/FFHQFaceAlignment) with added video processing capabilities and minor code optimizations. The tool allows for FFHQ-style face alignment for both images and videos.

## Features
- Original image-to-image face alignment
- New video-to-video face alignment
- Batch video processing support

## Installation

This project has been tested with Python 3.10. Please ensure you have Conda installed before proceeding.

```bash
# Create and activate conda environment
conda create -n ffhqfacealignment python==3.10
conda activate ffhqfacealignment

# Install PyTorch (adjust URL based on your CUDA version)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```



## Usage


```bash
# First, you need to download the pretrained SFD [1] model using
python download.py
```



```bash
# Same as original repo
python align.py --input-dir=<directory of original images> --output-dir=<directory of cropped images> --size=<cropped image resolution>
```

```bash
# Process single video - outputs to same directory with "processed_" prefix
python align_video.py --input path/to/input.mp4 --size 256
# Output will be saved as: path/to/processed_input.mp4
```

```bash
# Process batch of videos in a directory - maintains directory structure
python align_video_dir.py --input path/to/source_dir --size 256
# Creates a new directory: path/to/processed_source_dir/
```

## Credits

 - [Face Detector [1]](https://github.com/sfzhang15/SFD) 
 - [Face alignment [2]](https://github.com/1adrianb/face-alignment)
 - [@chi0tzp/FFHQFaceAlignment](https://github.com/chi0tzp/FFHQFaceAlignment)



## References 

[1] Zhang, Shifeng, et al. "S3fd: Single shot scale-invariant face detector." *Proceedings of the IEEE international conference on computer vision*. 2017.

[2] Bulat, Adrian, and Georgios  Tzimiropoulos. "How far are we from solving the 2D & 3D face  alignment problem?(and a dataset of 230,000 3d facial landmarks)." *Proceedings of the IEEE International Conference on Computer Vision*. 2017.

