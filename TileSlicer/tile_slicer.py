"""
Gaussian Splatting PLY Tile Slicer

Slices a large Gaussian Splatting PLY file into smaller tiles based on XY position.
Similar to how Unreal Engine handles landscape tiling for better performance.
Not generate empty tiles if no points fall within them.

Usage:
# Preview (info only, no slicing)
python tile_slicer.py input.ply --tile-size 100 --trim-percentile 0.1 --info-only

# Slice into tiles
# Trim outliers using percentile (ignore sparse outer 1% of points for bbox calculation. Avoid almost empty tiles)
python tile_slicer.py input.ply --tile-size 100 --trim-percentile 0.1 --output-dir ./tiles
"""

import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from collections import defaultdict
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice a Gaussian Splatting PLY file into tiles"
    )
    parser.add_argument("input", type=str, help="Input PLY file path")
    parser.add_argument(
        "--tile-size",
        type=float,
        required=True,
        help="Tile size in world units (e.g., 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tiles",
        help="Output directory for tile files (default: ./tiles)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="tile",
        help="Prefix for output tile files (default: tile)",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show info about the PLY file and estimated tiles, don't slice",
    )
    parser.add_argument(
        "--trim-percentile",
        type=float,
        default=0,
        help="Trim outliers by percentile (e.g., 1 means use 1st-99th percentile for bbox). "
             "Points outside trimmed bbox are assigned to edge tiles. Default: 0 (no trimming)",
    )
    return parser.parse_args()


def load_ply(filepath: str) -> PlyData:
    """Load PLY file and return PlyData object."""
    print(f"Loading PLY file: {filepath}")
    start = time.time()
    plydata = PlyData.read(filepath)
    elapsed = time.time() - start
    print(f"Loaded in {elapsed:.2f} seconds")
    return plydata


def get_bounding_box(vertices, trim_percentile: float = 0) -> dict:
    """Calculate bounding box from vertex data.

    Args:
        vertices: Vertex data array
        trim_percentile: If > 0, use percentile bounds instead of min/max.
                        E.g., trim_percentile=1 uses 1st to 99th percentile.
                        This excludes sparse outliers from bbox calculation.
    """
    if trim_percentile > 0:
        low_p = trim_percentile
        high_p = 100 - trim_percentile
        return {
            "min_x": float(np.percentile(vertices["x"], low_p)),
            "max_x": float(np.percentile(vertices["x"], high_p)),
            "min_y": float(np.percentile(vertices["y"], low_p)),
            "max_y": float(np.percentile(vertices["y"], high_p)),
            "min_z": float(np.percentile(vertices["z"], low_p)),
            "max_z": float(np.percentile(vertices["z"], high_p)),
        }
    else:
        return {
            "min_x": float(np.min(vertices["x"])),
            "max_x": float(np.max(vertices["x"])),
            "min_y": float(np.min(vertices["y"])),
            "max_y": float(np.max(vertices["y"])),
            "min_z": float(np.min(vertices["z"])),
            "max_z": float(np.max(vertices["z"])),
        }


def calculate_tile_grid(bbox: dict, tile_size: float) -> dict:
    """Calculate the tile grid parameters."""
    # Calculate grid dimensions
    x_range = bbox["max_x"] - bbox["min_x"]
    y_range = bbox["max_y"] - bbox["min_y"]

    num_tiles_x = int(np.ceil(x_range / tile_size))
    num_tiles_y = int(np.ceil(y_range / tile_size))

    return {
        "num_tiles_x": num_tiles_x,
        "num_tiles_y": num_tiles_y,
        "total_tiles": num_tiles_x * num_tiles_y,
        "origin_x": bbox["min_x"],
        "origin_y": bbox["min_y"],
        "tile_size": tile_size,
    }


def assign_points_to_tiles(vertices, grid: dict) -> dict:
    """Assign each point to a tile based on XY position."""
    print("Assigning points to tiles...")
    start = time.time()

    x = vertices["x"]
    y = vertices["y"]

    # Calculate tile indices for each point
    tile_x = np.floor((x - grid["origin_x"]) / grid["tile_size"]).astype(int)
    tile_y = np.floor((y - grid["origin_y"]) / grid["tile_size"]).astype(int)

    # Clamp to valid range (handles edge cases at max boundary)
    tile_x = np.clip(tile_x, 0, grid["num_tiles_x"] - 1)
    tile_y = np.clip(tile_y, 0, grid["num_tiles_y"] - 1)

    # Group point indices by tile
    tiles = defaultdict(list)
    for idx in range(len(vertices)):
        tile_key = (int(tile_x[idx]), int(tile_y[idx]))
        tiles[tile_key].append(idx)

    elapsed = time.time() - start
    print(f"Assignment completed in {elapsed:.2f} seconds")
    print(f"Non-empty tiles: {len(tiles)} / {grid['total_tiles']}")

    return tiles


def create_tile_ply(vertices, indices: list, original_plydata: PlyData) -> PlyData:
    """Create a new PlyData object for a tile with selected vertices."""
    # Get the original vertex element to preserve property structure
    original_element = original_plydata["vertex"]

    # Extract the subset of vertices
    tile_vertices = vertices[indices]

    # Create new element with same properties
    new_element = PlyElement.describe(tile_vertices, "vertex")

    return PlyData([new_element], text=False, byte_order="<")


def save_tiles(
    vertices,
    tiles: dict,
    grid: dict,
    original_plydata: PlyData,
    output_dir: Path,
    prefix: str,
):
    """Save each tile as a separate PLY file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving tiles to: {output_dir}")
    total_tiles = len(tiles)

    for i, ((tx, ty), indices) in enumerate(tiles.items()):
        # Calculate world bounds for this tile
        tile_min_x = grid["origin_x"] + tx * grid["tile_size"]
        tile_min_y = grid["origin_y"] + ty * grid["tile_size"]
        tile_max_x = tile_min_x + grid["tile_size"]
        tile_max_y = tile_min_y + grid["tile_size"]

        # Create filename with tile coordinates
        filename = f"{prefix}_{tx}_{ty}.ply"
        filepath = output_dir / filename

        # Create and save tile PLY
        tile_ply = create_tile_ply(vertices, indices, original_plydata)
        tile_ply.write(str(filepath))

        # Progress update
        if (i + 1) % 10 == 0 or (i + 1) == total_tiles:
            print(f"  Saved {i + 1}/{total_tiles} tiles...")

    print(f"All tiles saved!")


def print_info(vertices, bbox: dict, grid: dict, tiles: dict = None,
               actual_bbox: dict = None, trim_percentile: float = 0):
    """Print information about the PLY file and tile grid."""
    print("\n" + "=" * 60)
    print("PLY FILE INFO")
    print("=" * 60)
    print(f"Total vertices (Gaussian splats): {len(vertices):,}")

    # Show actual bounding box if trimming was applied
    if actual_bbox and trim_percentile > 0:
        print(f"\nActual Bounding Box (all points):")
        print(f"  X: {actual_bbox['min_x']:.2f} to {actual_bbox['max_x']:.2f} (range: {actual_bbox['max_x'] - actual_bbox['min_x']:.2f})")
        print(f"  Y: {actual_bbox['min_y']:.2f} to {actual_bbox['max_y']:.2f} (range: {actual_bbox['max_y'] - actual_bbox['min_y']:.2f})")
        print(f"  Z: {actual_bbox['min_z']:.2f} to {actual_bbox['max_z']:.2f} (range: {actual_bbox['max_z'] - actual_bbox['min_z']:.2f})")
        print(f"\nTrimmed Bounding Box ({trim_percentile:.1f}th - {100-trim_percentile:.1f}th percentile):")
    else:
        print(f"\nBounding Box:")
    print(f"  X: {bbox['min_x']:.2f} to {bbox['max_x']:.2f} (range: {bbox['max_x'] - bbox['min_x']:.2f})")
    print(f"  Y: {bbox['min_y']:.2f} to {bbox['max_y']:.2f} (range: {bbox['max_y'] - bbox['min_y']:.2f})")
    print(f"  Z: {bbox['min_z']:.2f} to {bbox['max_z']:.2f} (range: {bbox['max_z'] - bbox['min_z']:.2f})")

    print(f"\nTile Grid (tile size: {grid['tile_size']}):")
    print(f"  Grid dimensions: {grid['num_tiles_x']} x {grid['num_tiles_y']} = {grid['total_tiles']} tiles")

    if tiles:
        non_empty = len(tiles)
        print(f"  Non-empty tiles: {non_empty}")
        print(f"  Empty tiles: {grid['total_tiles'] - non_empty}")

        # Statistics about points per tile
        points_per_tile = [len(indices) for indices in tiles.values()]
        print(f"\nPoints per tile:")
        print(f"  Min: {min(points_per_tile):,}")
        print(f"  Max: {max(points_per_tile):,}")
        print(f"  Average: {np.mean(points_per_tile):,.0f}")
        print(f"  Median: {np.median(points_per_tile):,.0f}")

    print("=" * 60)


def main():
    args = parse_args()

    # Load PLY file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    plydata = load_ply(str(input_path))
    vertices = plydata["vertex"].data

    # Calculate bounding boxes
    actual_bbox = get_bounding_box(vertices, trim_percentile=0)

    if args.trim_percentile > 0:
        print(f"Using {args.trim_percentile:.1f}th - {100-args.trim_percentile:.1f}th percentile for bounding box")
        bbox = get_bounding_box(vertices, trim_percentile=args.trim_percentile)
    else:
        bbox = actual_bbox
        actual_bbox = None  # Don't show twice

    grid = calculate_tile_grid(bbox, args.tile_size)

    if args.info_only:
        # Just show info without actually assigning/saving
        print_info(vertices, bbox, grid, actual_bbox=actual_bbox,
                   trim_percentile=args.trim_percentile)
        return 0

    # Assign points to tiles
    tiles = assign_points_to_tiles(vertices, grid)

    # Print info
    print_info(vertices, bbox, grid, tiles, actual_bbox=actual_bbox,
               trim_percentile=args.trim_percentile)

    # Save tiles
    output_dir = Path(args.output_dir)
    save_tiles(vertices, tiles, grid, plydata, output_dir, args.prefix)

    print(f"\nDone! Tiles saved to: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
