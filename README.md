# SectionAligner
This project provides a comprehensive toolset for processing, analyzing, and aligning tissue images from OME-TIFF files. It includes features for image preprocessing, tissue detection, cropping, and alignment.

## Installation

Clone this repository and navigate into the project directory. Ensure you have Python 3.8 or later installed. It's recommended to use a virtual environment.

```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
pip install -r requirements.txt
```

## Usage 

The main script can be run from the command line, providing various options for processing the images:

```bash
python main.py --input_path "path/to/your/image.ome.tiff" --output_dir "path/to/output" --num_tissue 8 --pixel_size [0.5073519424785282, 0.5073519424785282] ---crop_only False
```

### Key Options

    --num_tissue: Number of tissues to detect (default is 8).
    --pixel_size: Pixel size in microns. (IMPORTANT as it determines how much to downsample and will effect end results)
    --crop_only: Only identify tissues and crop, without alignment (default is False).

### All Options

    --num_tissue: The number of tissues to detect. Default is 8.
    --level: Pyramid level of the image. Default is 0, which is the original image size.
    --thresh: Threshold value for binarization. If not set, Otsu's method is used.
    --kernel_size: Size of the structuring element used for morphological operations. Default is 100.
    --holes_thresh: Area threshold for removing small holes. Default is 5000.
    --scale_factor: Scale factor for downscaling images. Default is 8.
    --padding: Padding for bounding boxes during cropping. Default is 50.
    --connect: Connectivity for connected components. Default is 2.
    --pixel_size: Physical pixel size of the image in microns. Default is [0.5073519424785282, 0.5073519424785282].
    --output_dir: Output directory for saving images. Default is ./outputs.
    --input_path: Input path for reading images. Default is the specified path to an OME-TIFF file.
    --output_file_basename: Basename for output files. Default is aligned_tissue.
    --align_upsample_factor: Upsample factor for aligning images. Default is 2.
    --optimize: Whether to optimize alignment parameters using Optuna. Boolean. Default is True.
    --crop_only: If True, only identify tissues and crop, without aligning. Boolean. Default is False.

## Features

**Tissue Detection**: Identifies tissues within OME-TIFF images. \
**Image Cropping**: Crops identified tissues for focused analysis. \
**Image Alignment**: Aligns tissue slices for better comparison and analysis. \
**Optimization**: Uses Optuna for optimizing image alignment parameters.

## Contact 

Ted Zhang [tedz@andrew.cmu.edu] \
Bob Murphy [murphy@andrew.cmu.edu]




