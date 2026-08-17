import numpy as np
import os
from pathlib import Path
import json
import tifffile
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple
from collections import defaultdict


def load_tile_metadata(metadata_file: str) -> Dict:
    """
    Load the tile metadata JSON file created during the tiling process.
    
    Args:
        metadata_file: Path to the metadata JSON file
        
    Returns:
        Dictionary containing the tiling metadata
    """
    with open(metadata_file, 'r') as f:
        return json.load(f)


def stitch_segmentation_masks(
    tiles_dir: str,
    metadata_file: str,
    output_path: str,
    relabel_strategy: str = "global"
) -> np.ndarray:
    """
    Stitch segmentation mask tiles back together, handling label conflicts in overlap regions.
    
    Args:
        tiles_dir: Directory containing the segmentation mask tiles
        metadata_file: Path to the metadata JSON file from the tiling process
        output_path: Path to save the stitched mask
        relabel_strategy: Strategy for handling label conflicts:
                         - "global": Relabel all cells to ensure global uniqueness
                         - "local": Keep original labels where possible and only resolve conflicts
        
    Returns:
        Stitched segmentation mask as a numpy array
    """
    print(f"Loading metadata from {metadata_file}")
    metadata = load_tile_metadata(metadata_file)
    
    # Get original image dimensions
    original_shape = tuple(metadata["original_shape"])
    z_dim, c_dim, y_dim, x_dim = original_shape
    
    print(f"Original image shape: (z={z_dim}, c={c_dim}, y={y_dim}, x={x_dim})")
    
    # For segmentation masks, we expect a single channel
    # Create an empty array for the stitched mask, using a suitable dtype
    stitched_mask = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint32)
    
    # Create a map to track which tiles contribute to each region
    # This will help resolve conflicts in overlap regions
    tile_contributions = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)
    
    # Get list of tiles
    tiles = metadata["tiles"]
    
    # If using global relabeling, we need to track the maximum label used
    max_label = 0
    
    # Dictionary to map original labels to new global labels
    # Structure: {tile_index: {original_label: new_global_label}}
    label_mapping = defaultdict(dict)
    
    print(f"Loading and analyzing {len(tiles)} mask tiles...")
    
    # First pass: Load all tiles and identify cell labels to build global mapping
    if relabel_strategy == "global":
        for tile_idx, tile_meta in enumerate(tqdm(tiles, desc="Analyzing tiles")):
            # Load the segmentation mask tile
            tile_filename = tile_meta["filename"]
            tile_path = os.path.join(tiles_dir, tile_filename)
            
            if not os.path.exists(tile_path):
                print(f"Warning: Tile file {tile_path} not found, skipping")
                continue
                
            mask_tile = tifffile.imread(tile_path)
            
            # For segmentation masks, we expect a single channel (so c dimension should be absent)
            # If it has a channel dimension, take the first channel
            if len(mask_tile.shape) == 4 and mask_tile.shape[1] == 1:
                mask_tile = mask_tile[:, 0, :, :]
            
            # Find unique labels in this tile (excluding background, which is typically 0)
            unique_labels = np.unique(mask_tile)
            unique_labels = unique_labels[unique_labels > 0]
            
            # Create global mapping for this tile's labels
            for label in unique_labels:
                # Assign a new global label
                max_label += 1
                label_mapping[tile_idx][label] = max_label
    
    # Second pass: Place tiles in the stitched mask, resolving conflicts in overlap regions
    print(f"Stitching mask tiles...")
    for tile_idx, tile_meta in enumerate(tqdm(tiles, desc="Stitching masks")):
        # Get tile position
        z_start, z_end = tile_meta["position"]["z"]
        y_start, y_end = tile_meta["position"]["y"]
        x_start, x_end = tile_meta["position"]["x"]
        
        # Load the segmentation mask tile
        tile_filename = tile_meta["filename"]
        tile_path = os.path.join(tiles_dir, tile_filename)
        
        if not os.path.exists(tile_path):
            continue  # Skip if file not found (already warned in first pass)
            
        mask_tile = tifffile.imread(tile_path)
        
        # Handle dimension differences (segmentation may have lost the channel dimension)
        if len(mask_tile.shape) == 4 and mask_tile.shape[1] == 1:
            mask_tile = mask_tile[:, 0, :, :]
        
        # Apply global relabeling if needed
        if relabel_strategy == "global":
            # Create a copy of the mask to avoid modifying the original during relabeling
            relabeled_mask = np.zeros_like(mask_tile)
            
            # Apply the label mapping
            for original_label in np.unique(mask_tile):
                if original_label == 0:  # Skip background
                    continue
                    
                global_label = label_mapping[tile_idx].get(original_label)
                if global_label is not None:
                    relabeled_mask[mask_tile == original_label] = global_label
            
            mask_tile = relabeled_mask
        
        # Place the tile in the stitched mask
        # For segmentation masks, we prioritize cells in non-overlap regions
        # and resolve conflicts in overlap regions
        for z in range(z_start, z_end):
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    # Calculate relative position in the tile
                    rel_z = z - z_start
                    rel_y = y - y_start
                    rel_x = x - x_start
                    
                    # Get the label at this position in the tile
                    label = mask_tile[rel_z, rel_y, rel_x]
                    
                    if label == 0:  # Skip background
                        continue
                    
                    # Check if this position already has a non-zero label
                    existing_label = stitched_mask[z, y, x]
                    
                    if existing_label == 0:
                        # If there's no existing label, use this tile's label
                        stitched_mask[z, y, x] = label
                        tile_contributions[z, y, x] += 1
                    else:
                        # If there's already a label, we're in an overlap region
                        # For segmentation masks, conflicts can be resolved in different ways
                        
                        # Option 1: Keep the larger label
                        # This is a simple heuristic and may not be ideal for all cases
                        # stitched_mask[z, y, x] = max(existing_label, label)
                        
                        # Option 2: Keep the label from the tile with higher summed intensity
                        # This might be relevant if higher intensity suggests more confident segmentation
                        # We would need to track which tile contributed which label
                        
                        # Option 3: Keep the label from the tile closer to the center
                        # This assumes segmentation is more reliable away from tile edges
                        # For simplicity, here we just keep the first label encountered
                        # stitched_mask[z, y, x] = existing_label
                        
                        # For this implementation, we'll use a simple "keep label if it's not background" approach
                        # We could implement more sophisticated strategies based on specific needs
                        if label > 0:
                            # Just increment the contribution counter, we'll resolve conflicts later
                            tile_contributions[z, y, x] += 1
    
    # Post-process the mask to resolve inconsistencies
    # For this simple example, we'll just leave it as is
    # More sophisticated methods would analyze and resolve cell fragments
    
    # Save the stitched mask
    print(f"Saving stitched mask to {output_path}")
    tifffile.imwrite(output_path, stitched_mask)
    
    # Optionally, also save the tile contributions map for debugging
    contributions_path = output_path.replace('.tif', '_contributions.tif')
    tifffile.imwrite(contributions_path, tile_contributions)
    
    return stitched_mask


def resolve_label_conflicts(
    stitched_mask: np.ndarray,
    tile_positions: List[Tuple],
    overlap: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Resolve label conflicts in overlap regions of a stitched segmentation mask.
    This function would implement more sophisticated conflict resolution strategies.
    
    Args:
        stitched_mask: The initial stitched mask with potential conflicts
        tile_positions: List of tile positions [(z_start, z_end, y_start, y_end, x_start, x_end), ...]
        overlap: Tuple of (z_overlap, c_overlap, y_overlap, x_overlap)
        
    Returns:
        Conflict-resolved segmentation mask
    """
    # This would be a more sophisticated conflict resolution approach
    # For now, it's just a placeholder
    
    # Possible approaches:
    # 1. Connected component analysis to merge fragmented cells
    # 2. Watershed-based merging
    # 3. Graph-based approaches where nodes are labels and edges are overlap regions
    
    return stitched_mask


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stitch segmentation mask tiles back together")
    parser.add_argument("tiles_dir", help="Directory containing the segmentation mask tiles")
    parser.add_argument("metadata_file", help="Path to the metadata JSON file from the tiling process")
    parser.add_argument("output_path", help="Path to save the stitched mask")
    parser.add_argument("--relabel", choices=["global", "local"], default="global",
                        help="Strategy for handling label conflicts")
    
    args = parser.parse_args()
    
    stitch_segmentation_masks(
        tiles_dir=args.tiles_dir,
        metadata_file=args.metadata_file,
        output_path=args.output_path,
        relabel_strategy=args.relabel
    )