import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
import tifffile
from tqdm import tqdm
from aicsimageio.writers import OmeTiffWriter
from ome_utils import get_converted_physical_size
from pint import Quantity
from aicsimageio import types
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def read_ome_tiff(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Read an OME-TIFF file and return the image data and metadata.
    
    Args:
        file_path: Path to the OME-TIFF file
        
    Returns:
        Tuple of (image_data, metadata) where image_data is a numpy array with dimensions (z, c, y, x)
    """
    print(f"Reading OME-TIFF file: {file_path}")
    
    with tifffile.TiffFile(file_path) as tif:
        # Check if it's an OME-TIFF
        is_ome = tif.is_ome
        if is_ome:
            print("File is an OME-TIFF")
        else:
            print("File is not an OME-TIFF, but will try to read it as a regular TIFF")
        
        # Read the image data
        image = tif.asarray()
        print(f"Raw image shape: {image.shape}")
        
        # Extract metadata
        if hasattr(tif, 'ome_metadata') and tif.ome_metadata is not None:
            ome_meta = tif.ome_metadata
            print("Found OME metadata")
        else:
            ome_meta = {}
            print("No OME metadata found")
        
        # Try to determine the dimension order from series
        series = tif.series[0]
        axes = series.axes
        print(f"Detected axes: {axes}")
        
        # Reshape to ensure we have (z, c, y, x) format
        # This is a bit tricky and may need adjustments for specific files
        if len(image.shape) == 2:
            # Single 2D image, assume (y, x)
            print("Reshaping 2D image to (1, 1, y, x)")
            image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # Add z and c dimensions
        elif len(image.shape) == 3:
            if axes.upper().startswith('ZYX'):
                # (z, y, x), add c dimension
                print("Reshaping (z, y, x) to (z, 1, y, x)")
                image = np.expand_dims(image, axis=1)
            elif axes.upper().startswith('CYX'):
                # (c, y, x), add z dimension
                print("Reshaping (c, y, x) to (1, c, y, x)")
                image = np.expand_dims(image, axis=0)
            else:
                # Default assumption: (z, y, x)
                print(f"Unknown 3D format with axes {axes}, assuming (z, y, x), reshaping to (z, 1, y, x)")
                image = np.expand_dims(image, axis=1)
        elif len(image.shape) == 4:
            # Check if we need to transpose
            if axes.upper() == 'ZCYX':
                print("Image is already in (z, c, y, x) format")
                pass  # Already in the right format
            elif axes.upper() == 'ZYXC':
                print("Transposing from (z, y, x, c) to (z, c, y, x)")
                image = np.transpose(image, (0, 3, 1, 2))
            elif axes.upper() == 'CZYX':
                print("Transposing from (c, z, y, x) to (z, c, y, x)")
                image = np.transpose(image, (1, 0, 2, 3))
            else:
                print(f"Unknown 4D format with axes {axes}, assuming it's already (z, c, y, x)")
        else:
            # Handle other dimensions like TZCYX (5D)
            if len(image.shape) == 5 and axes.upper().startswith('TZCYX'):
                print(f"Detected 5D image with time dimension, extracting first timepoint")
                image = image[0]  # Take only the first timepoint
            else:
                raise ValueError(f"Unexpected image shape: {image.shape} with axes {axes}")
        
        print(f"Reshaped image to (z, c, y, x): {image.shape}")
        
        metadata = {
            "original_shape": image.shape,
            "axes": axes,
            "ome_metadata": ome_meta
        }
        
        return image, metadata


def tile_image_xy(
    image: np.ndarray,
    num_tiles: int,
    overlap: int,
    output_dir: str,
    metadata: Optional[Dict] = None,
    filename_prefix: str = "tile"
) -> Dict:
    """
    Split a 4D image (z, c, y, x) into tiles by cropping only in the x and y dimensions.
    Each tile contains the full z stack and all channels.
    
    Args:
        image: 4D numpy array with dimensions (z, c, y, x)
        num_tiles: Total number of tiles to create (will be arranged in an approximately square grid)
        overlap: Overlap between adjacent tiles in pixels (same for both x and y)
        output_dir: Directory to save the tiles
        metadata: Optional additional metadata to include in the output JSON
        filename_prefix: Prefix for the tile filenames
    
    Returns:
        Dictionary with metadata about the tiling process
    """
    # Get image dimensions
    z_dim, c_dim, y_dim, x_dim = image.shape
    print(f"Tiling image with shape (z={z_dim}, c={c_dim}, y={y_dim}, x={x_dim})")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate grid shape (try to make it approximately square)
    y_tiles = int(np.ceil(np.sqrt(num_tiles)))
    x_tiles = int(np.ceil(num_tiles / y_tiles))
    
    # Use the same overlap value for both dimensions
    y_overlap = overlap
    x_overlap = overlap
    
    # Calculate base tile sizes and remainders
    y_base = y_dim // y_tiles
    x_base = x_dim // x_tiles
    y_remainder = y_dim % y_tiles
    x_remainder = x_dim % x_tiles
    
    print(f"Base tile size without overlap: (y={y_base}, x={x_base})")
    print(f"Remainders: y={y_remainder}, x={x_remainder}")
    
    # Initialize metadata dictionary
    tiling_metadata = {
        "original_shape": image.shape,
        "tile_size": (z_dim, c_dim, y_base + y_overlap, x_base + x_overlap),
        "overlap": (0, 0, y_overlap, x_overlap),
        "tiles": []
    }

    # print out shape of tile
    print(f"Tile size: {y_base + y_overlap}x{x_base + x_overlap}")
    
    # Include additional metadata if provided
    if metadata:
        tiling_metadata["source_metadata"] = metadata

    original_ome_xml = metadata.get("ome_metadata", None) if metadata else None
    
    # Counter for tile index
    tile_idx = 0
    
    # Create a progress bar for the tiling process
    total_tiles = y_tiles * x_tiles
    progress_bar = tqdm(total=total_tiles, desc="Tiling image")
    
    # Iterate through y and x dimensions to extract tiles
    for y_idx in range(y_tiles):
        for x_idx in range(x_tiles):
            if tile_idx >= num_tiles:
                break
                
            # Calculate base positions without overlap
            y_start = y_idx * y_base
            x_start = x_idx * x_base
            
            # Add overlap only where needed (not at image edges)
            if y_idx > 0:  # Not first row
                y_start -= y_overlap
            
            if x_idx > 0:  # Not first column
                x_start -= x_overlap
            
            # Calculate end positions, including remainder distribution
            if y_idx == y_tiles - 1:  # Last row
                y_end = y_dim  # Go to the end of the image
            else:
                y_end = y_start + y_base
                if y_idx < y_remainder:  # Distribute remainder pixels
                    y_end += 1
                if y_idx < y_tiles - 1:  # Add overlap if not last tile
                    y_end += y_overlap
            
            if x_idx == x_tiles - 1:  # Last column
                x_end = x_dim  # Go to the end of the image
            else:
                x_end = x_start + x_base
                if x_idx < x_remainder:  # Distribute remainder pixels
                    x_end += 1
                if x_idx < x_tiles - 1:  # Add overlap if not last tile
                    x_end += x_overlap
            
            # Ensure we don't exceed image boundaries (shouldn't be necessary now, but keep as safety)
            y_end = min(y_end, y_dim)
            x_end = min(x_end, x_dim)
            
            tile = image[:, :, y_start:y_end, x_start:x_end]
            
            # Print detailed tile information for debugging
            print(f"\nTile {tile_idx}:")
            print(f"  Position: y={y_start}:{y_end}, x={x_start}:{x_end}")
            print(f"  Shape: {tile.shape}")
            print(f"  Grid position: row={y_idx+1}/{y_tiles}, col={x_idx+1}/{x_tiles}")
            
            # Generate filename for this tile
            tile_filename = f"{filename_prefix}_y{y_start:04d}_x{x_start:04d}.tif"
            tile_path = os.path.join(output_dir, tile_filename)
            
            # Save the tile as TIFF
            tifffile.imwrite(
                tile_path,
                tile,
                metadata={'axes': 'ZCYX'}, 
                description=original_ome_xml,  # This preserves the full OME-XML metadata
            )
            
            # Calculate summed intensity for the tile
            summed_intensity = float(np.sum(tile))
            
            # Record metadata for this tile
            tile_metadata = {
                "index": tile_idx,
                "filename": tile_filename,
                "position": {
                    "z": (0, z_dim),  # Full z dimension
                    "c": (0, c_dim),  # Full c dimension
                    "y": (y_start, y_end),
                    "x": (x_start, x_end)
                },
                "shape": tile.shape,
                "summed_intensity": summed_intensity
            }
            
            tiling_metadata["tiles"].append(tile_metadata)
            tile_idx += 1
            progress_bar.update(1)
            
        # Stop outer loop too if we've reached the requested number of tiles
        if tile_idx >= num_tiles:
            break
    
    progress_bar.close()
    
    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, f"{filename_prefix}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(tiling_metadata, f, indent=2)
    
    print(f"Created {len(tiling_metadata['tiles'])} tiles in {output_dir}")
    print(f"Metadata saved to {metadata_path}")
    
    # Verify coverage
    min_y = min(tile["position"]["y"][0] for tile in tiling_metadata["tiles"])
    max_y = max(tile["position"]["y"][1] for tile in tiling_metadata["tiles"])
    min_x = min(tile["position"]["x"][0] for tile in tiling_metadata["tiles"])
    max_x = max(tile["position"]["x"][1] for tile in tiling_metadata["tiles"])
    
    assert min_y == 0, f"Missing coverage at start of Y axis: {min_y}"
    assert max_y == y_dim, f"Missing coverage at end of Y axis: {max_y} vs {y_dim}"
    assert min_x == 0, f"Missing coverage at start of X axis: {min_x}"
    assert max_x == x_dim, f"Missing coverage at end of X axis: {max_x} vs {x_dim}"
    
    return tiling_metadata


def process_ome_tiff(
    input_file: str,
    output_dir: str,
    num_tiles: int,
    overlap: int,
    filename_prefix: str = "tile"
) -> Dict:
    """
    Process an OME-TIFF file by reading it, tiling it in x and y dimensions only, and saving the tiles.
    
    Args:
        input_file: Path to the input OME-TIFF file
        output_dir: Directory to save the tiles
        num_tiles: Total number of tiles to create
        overlap: Overlap between adjacent tiles in pixels
        filename_prefix: Prefix for the tile filenames
    
    Returns:
        Dictionary with metadata about the tiling process
    """
    # Read the OME-TIFF file
    image, metadata = read_ome_tiff(input_file)

    physical_pixel_sizes: dict[str, Quantity] = get_converted_physical_size(tifffile.TiffFile(input_file))
    
    # Tile the image only in x and y dimensions
    tiling_metadata = tile_image_xy(
        image=image,
        num_tiles=num_tiles,
        overlap=overlap,
        output_dir=output_dir,
        metadata=metadata,
        filename_prefix=filename_prefix,
    )
    
    # return tiling_metadata


def stitch_tiles(metadata_file: str) -> np.ndarray:
    """
    Stitch tiles back together based on the metadata file.
    
    Args:
        metadata_file: Path to the metadata JSON file
    
    Returns:
        Stitched image as numpy array with dimensions (z, c, y, x)
    """
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Get original dimensions
    z_dim, c_dim, y_dim, x_dim = metadata['original_shape']
    y_overlap, x_overlap = metadata['overlap'][2:]  # Get y,x overlap values
    
    # Create empty array for stitched image
    stitched = np.zeros((z_dim, c_dim, y_dim, x_dim), dtype=np.float32)
    
    # Create count array to handle overlapping regions
    count = np.zeros((y_dim, x_dim), dtype=np.float32)
    
    # Get directory of metadata file for relative tile paths
    base_dir = os.path.dirname(metadata_file)
    
    # Sort tiles by position for better debugging
    tiles = sorted(metadata['tiles'], 
                  key=lambda x: (x['position']['y'][0], x['position']['x'][0]))
    
    print(f"Stitching {len(tiles)} tiles...")
    print("\nTile positions and overlaps:")
    
    for i, tile_info in enumerate(tqdm(tiles)):
        # Load tile
        tile_path = os.path.join(base_dir, tile_info['filename'])
        tile = tifffile.imread(tile_path)
        
        # Get position
        y_start, y_end = tile_info['position']['y']
        x_start, x_end = tile_info['position']['x']
        
        # Calculate overlaps for this tile
        has_top_overlap = y_start > 0
        has_bottom_overlap = y_end < y_dim
        has_left_overlap = x_start > 0
        has_right_overlap = x_end < x_dim
        
        print(f"\nTile {i}:")
        print(f"  Position: y={y_start}:{y_end}, x={x_start}:{x_end}")
        print(f"  Overlaps: top={has_top_overlap}, bottom={has_bottom_overlap}, "
              f"left={has_left_overlap}, right={has_right_overlap}")
        
        # Add tile to stitched image
        stitched[:, :, y_start:y_end, x_start:x_end] += tile
        
        # Increment count for overlapping regions
        count[y_start:y_end, x_start:x_end] += 1
    
    # Average overlapping regions
    print("\nAveraging overlapping regions...")
    for z in range(z_dim):
        for c in range(c_dim):
            stitched[z, c] = np.divide(stitched[z, c], count, where=count > 0)
    
    # Print overlap statistics
    unique_counts = np.unique(count)
    print("\nOverlap statistics:")
    print(f"Number of different overlap counts: {len(unique_counts)}")
    for c in unique_counts:
        pixels = np.sum(count == c)
        percentage = (pixels / (y_dim * x_dim)) * 100
        print(f"  {pixels} pixels ({percentage:.2f}%) have {c} overlapping tiles")
    
    # Verify dimensions
    print(f"\nOriginal shape: {metadata['original_shape']}")
    print(f"Stitched shape: {stitched.shape}")
    
    if stitched.shape != tuple(metadata['original_shape']):
        raise ValueError("Stitched image dimensions do not match original!")
    
    return stitched

def verify_stitching(input_file: str, metadata_file: str):
    """
    Verify that stitching tiles produces an image matching the original.
    
    Args:
        input_file: Path to original image
        metadata_file: Path to tile metadata JSON
    """
    # Read original image
    original_img, _ = read_ome_tiff(input_file)
    
    # Stitch tiles
    stitched_img = stitch_tiles(metadata_file)
    
    # Compare dimensions
    print("\nDimension comparison:")
    print(f"Original shape: {original_img.shape}")
    print(f"Stitched shape: {stitched_img.shape}")
    
    # Compare content (allowing for small floating point differences)
    max_diff = np.max(np.abs(original_img - stitched_img))
    mean_diff = np.mean(np.abs(original_img - stitched_img))
    print(f"\nImage content comparison:")
    print(f"Maximum pixel difference: {max_diff}")
    print(f"Mean pixel difference: {mean_diff}")
    
    # Save stitched image for visual comparison
    output_dir = os.path.dirname(metadata_file)
    stitched_path = os.path.join(output_dir, "stitched_verification.tif")
    tifffile.imwrite(stitched_path, stitched_img)
    print(f"\nSaved stitched image to: {stitched_path}")

    # Add validation checks
    print("\nValidating tile coverage...")
    validate_tile_coverage(metadata_file)
    
    print("\nGenerating coverage visualization...")
    visualize_tile_coverage(metadata_file)

def visualize_tile_coverage(metadata_file: str, channel: int = 0, z_slice: int = 0):
    """
    Create a visualization of tile coverage and overlap regions.
    
    Args:
        metadata_file: Path to the metadata JSON file
        channel: Which channel to visualize
        z_slice: Which z slice to visualize
    """
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Get dimensions
    z_dim, c_dim, y_dim, x_dim = metadata['original_shape']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Tile boundaries
    ax1.set_title("Tile Boundaries")
    
    # Draw each tile as a rectangle
    colors = plt.cm.rainbow(np.linspace(0, 1, len(metadata['tiles'])))
    for tile_info, color in zip(metadata['tiles'], colors):
        y_start, y_end = tile_info['position']['y']
        x_start, x_end = tile_info['position']['x']
        
        rect = patches.Rectangle(
            (x_start, y_start), 
            x_end - x_start, 
            y_end - y_start,
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        ax1.add_patch(rect)
    
    ax1.set_xlim(-5, x_dim + 5)
    ax1.set_ylim(-5, y_dim + 5)
    
    # Plot 2: Overlap heatmap
    ax2.set_title("Overlap Regions")
    
    # Create overlap count array
    count = np.zeros((y_dim, x_dim))
    for tile_info in metadata['tiles']:
        y_start, y_end = tile_info['position']['y']
        x_start, x_end = tile_info['position']['x']
        count[y_start:y_end, x_start:x_end] += 1
    
    im = ax2.imshow(count, cmap='viridis')
    plt.colorbar(im, ax=ax2, label='Number of overlapping tiles')
    
    # Save the visualization
    output_dir = os.path.dirname(metadata_file)
    plt.savefig(os.path.join(output_dir, 'tile_coverage_visualization.png'))
    plt.close()

def validate_tile_coverage(metadata_file: str) -> bool:
    """
    Perform detailed validation of tile coverage.
    
    Args:
        metadata_file: Path to the metadata JSON file
    
    Returns:
        bool: True if validation passes
    """
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    z_dim, c_dim, y_dim, x_dim = metadata['original_shape']
    y_overlap, x_overlap = metadata['overlap'][2:]
    
    # Create coverage map
    coverage = np.zeros((y_dim, x_dim), dtype=int)
    
    for tile_info in metadata['tiles']:
        y_start, y_end = tile_info['position']['y']
        x_start, x_end = tile_info['position']['x']
        coverage[y_start:y_end, x_start:x_end] += 1
    
    # Check for uncovered regions
    uncovered = coverage == 0
    if np.any(uncovered):
        y_uncovered, x_uncovered = np.where(uncovered)
        print("ERROR: Found uncovered pixels:")
        for y, x in zip(y_uncovered[:10], x_uncovered[:10]):  # Show first 10 uncovered pixels
            print(f"  Pixel at y={y}, x={x}")
        if len(y_uncovered) > 10:
            print(f"  ... and {len(y_uncovered) - 10} more")
        return False
    
    # Analyze overlap regions
    unique_counts = np.unique(coverage)
    print("\nOverlap analysis:")
    print(f"Found {len(unique_counts)} different overlap counts")
    for count in unique_counts:
        pixels = np.sum(coverage == count)
        percentage = (pixels / (y_dim * x_dim)) * 100
        print(f"  {pixels} pixels ({percentage:.2f}%) have {count} overlapping tiles")
    
    # Check overlap consistency
    print("\nChecking overlap consistency:")
    for i, tile1 in enumerate(metadata['tiles']):
        y1_start, y1_end = tile1['position']['y']
        x1_start, x1_end = tile1['position']['x']
        
        # Count expected overlaps with other tiles
        expected_overlaps = 0
        if y1_start > 0: expected_overlaps += 1  # top overlap
        if y1_end < y_dim: expected_overlaps += 1  # bottom overlap
        if x1_start > 0: expected_overlaps += 1  # left overlap
        if x1_end < x_dim: expected_overlaps += 1  # right overlap
        
        actual_overlaps = 0
        for j, tile2 in enumerate(metadata['tiles']):
            if i == j: continue
            y2_start, y2_end = tile2['position']['y']
            x2_start, x2_end = tile2['position']['x']
            
            # Check if tiles overlap
            if (y1_start < y2_end and y1_end > y2_start and 
                x1_start < x2_end and x1_end > x2_start):
                actual_overlaps += 1
        
        if actual_overlaps != expected_overlaps:
            print(f"WARNING: Tile {i} has {actual_overlaps} overlaps but expected {expected_overlaps}")
            print(f"  Position: y={y1_start}:{y1_end}, x={x1_start}:{x1_end}")
    
    print("\nValidation complete!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tile OME-TIFF images in x and y dimensions only")
    parser.add_argument("--input_file", default='/hive/users/tedz/SectionAligner/SectionAligner/transformed_output.ome.tif', help="Input OME-TIFF file")
    parser.add_argument("--output_dir", default='/hive/users/tedz/SectionAligner/SectionAligner/new_alignment_output/tiles', help="Output directory for tiles")
    parser.add_argument("--num-tiles", type=int, default=100, help="Total number of tiles to create")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap between tiles in pixels")
    parser.add_argument("--prefix", default="tile", help="Prefix for tile filenames")
    parser.add_argument("--verify", type=bool, default=False, help="Verify stitching after tiling")
    
    args = parser.parse_args()
    
    # Process the image
    process_ome_tiff(
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_tiles=args.num_tiles,
        overlap=args.overlap,
        filename_prefix=args.prefix
    )
    
    # Verify if requested
    if args.verify:
        metadata_file = os.path.join(args.output_dir, f"{args.prefix}_metadata.json")
        verify_stitching(args.input_file, metadata_file)