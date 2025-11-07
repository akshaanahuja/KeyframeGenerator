#!/usr/bin/env python3
"""
generate_train_csv.py - Create CSV index for training dataset triplets

Usage Examples:
    # Minimal usage
    python generate_train_csv.py

    # Generate line maps with custom Canny thresholds
    python generate_train_csv.py --gen-lines --canny 80,160 --relative-paths

    # Quick test run
    python generate_train_csv.py --dry-run --limit 50

    # Full run with relative paths and line generation
    python generate_train_csv.py --gen-lines --relative-paths --num-workers 8
"""

import argparse
import csv
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)


def natural_sort_key(text: str) -> List:
    """
    Generate a key for natural sorting (e.g., 'frame2.jpg' < 'frame10.jpg').
    
    Args:
        text: String to generate sort key for
        
    Returns:
        List of strings and integers for sorting
    """
    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()
    
    return [convert(c) for c in re.split(r'(\d+)', text)]


def discover_triplets(seq_dir: Path, exts: List[str], min_seq_len: int = 3, stride: int = 1) -> List[Tuple[str, str, str]]:
    """
    Discover triplets (I0, It, I1) from a sequence directory.
    
    Args:
        seq_dir: Path to sequence directory
        exts: List of valid image extensions (e.g., ['.jpg', '.png'])
        min_seq_len: Minimum number of frames required
        stride: Sliding window stride for sequences with >3 frames
        
    Returns:
        List of (frame0_path, frame_middle_path, frame1_path) tuples
    """
    # Gather all images matching extensions
    images = []
    for ext in exts:
        images.extend(seq_dir.glob(f'*{ext}'))
        images.extend(seq_dir.glob(f'*{ext.upper()}'))
    
    if not images:
        return []
    
    # Natural sort by filename
    images.sort(key=lambda p: natural_sort_key(p.name))
    
    if len(images) < min_seq_len:
        return []
    
    triplets = []
    
    if len(images) == 3:
        # Exactly 3 frames: (I0, It, I1)
        triplets.append((str(images[0]), str(images[1]), str(images[2])))
    elif len(images) > 3:
        # Sliding window: (i, i+1, i+2) with stride
        for i in range(0, len(images) - 2, stride):
            triplets.append((str(images[i]), str(images[i+1]), str(images[i+2])))
    
    return triplets


def generate_line_map(it_path: str, output_path: str, canny_low: int, canny_high: int, overwrite: bool = False) -> Optional[str]:
    """
    Generate Canny line map for the middle frame (It).
    
    Args:
        it_path: Path to the middle frame image
        output_path: Path where to save the line map PNG
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        overwrite: Whether to overwrite existing files
        
    Returns:
        Output path if successful, None otherwise
    """
    output_path_obj = Path(output_path)
    
    # Skip if exists and not overwriting
    if output_path_obj.exists() and not overwrite:
        return str(output_path_obj)
    
    try:
        # Read image and convert to grayscale
        img = cv2.imread(it_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Create parent directory if needed
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as 8-bit PNG
        cv2.imwrite(str(output_path_obj), edges)
        
        return str(output_path_obj)
    except Exception as e:
        warnings.warn(f"Failed to generate line map for {it_path}: {e}")
        return None


def validate_image(path: str) -> bool:
    """
    Validate that an image file is readable.
    
    Args:
        path: Path to image file
        
    Returns:
        True if readable, False otherwise
    """
    try:
        img = cv2.imread(path)
        return img is not None
    except Exception:
        return False


def find_repo_root(start_path: Path) -> Path:
    """
    Find repository root by looking for .git directory or other markers.
    
    Args:
        start_path: Starting path to search from
        
    Returns:
        Repository root path
    """
    current = start_path.resolve()
    while current != current.parent:
        if (current / '.git').exists() or (current / 'README.md').exists():
            return current
        current = current.parent
    return Path.cwd()  # Fallback to current working directory


def process_sequence(seq_dir: Path, root_dir: Path, repo_root: Path, lines_dir: Optional[Path],
    exts: List[str],
    min_seq_len: int,
    stride: int,
    gen_lines: bool,
    canny_low: int,
    canny_high: int,
    overwrite_lines: bool,
    relative_paths: bool
) -> List[dict]:
    """
    Process a single sequence directory and return triplet rows.
    
    Args:
        seq_dir: Path to sequence directory
        root_dir: Root directory containing sequences
        repo_root: Repository root for relative path calculation
        lines_dir: Directory for line maps (None if not generating)
        exts: Valid image extensions
        min_seq_len: Minimum sequence length
        stride: Sliding window stride
        gen_lines: Whether to generate line maps
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        overwrite_lines: Whether to overwrite existing line maps
        relative_paths: Whether to use relative paths
        
    Returns:
        List of dictionary rows for CSV
    """
    seq_id = seq_dir.name
    triplets = discover_triplets(seq_dir, exts, min_seq_len, stride)
    
    rows = []
    
    for i, (path_i0, path_it, path_i1) in enumerate(triplets):
        # Validate images
        if not validate_image(path_i0):
            warnings.warn(f"Skipping invalid I0: {path_i0}")
            continue
        if not validate_image(path_it):
            warnings.warn(f"Skipping invalid It: {path_it}")
            continue
        if not validate_image(path_i1):
            warnings.warn(f"Skipping invalid I1: {path_i1}")
            continue
        
        # Store absolute paths for line map generation
        path_i0_abs = Path(path_i0).resolve()
        path_it_abs = Path(path_it).resolve()
        path_i1_abs = Path(path_i1).resolve()
        
        # Convert to relative paths if requested
        if relative_paths:
            path_i0 = str(path_i0_abs.relative_to(repo_root))
            path_it = str(path_it_abs.relative_to(repo_root))
            path_i1 = str(path_i1_abs.relative_to(repo_root))
        else:
            path_i0 = str(path_i0_abs)
            path_it = str(path_it_abs)
            path_i1 = str(path_i1_abs)
        
        # Generate line map if requested
        path_ltau = ""
        if gen_lines and lines_dir:
            # Determine output path: lines_dir/seq_id/middle_basename_ltau.png
            it_basename = Path(path_it_abs).stem
            ltau_filename = f"{it_basename}_ltau.png"
            ltau_output = lines_dir / seq_id / ltau_filename
            
            # Generate line map (use absolute path for input)
            ltau_path = generate_line_map(
                str(path_it_abs),
                str(ltau_output),
                canny_low,
                canny_high,
                overwrite_lines
            )
            
            if ltau_path:
                ltau_path_abs = Path(ltau_path).resolve()
                if relative_paths:
                    path_ltau = str(ltau_path_abs.relative_to(repo_root))
                else:
                    path_ltau = str(ltau_path_abs)
        
        # Create row
        row = {
            'seq_id': seq_id,
            'path_i0': path_i0,
            'path_it': path_it,
            'path_i1': path_i1,
            'tau': 0.5,
            'path_ltau': path_ltau
        }
        rows.append(row)
    
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Generate CSV index for training dataset triplets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_train_csv.py
  python generate_train_csv.py --gen-lines --canny 80,160 --relative-paths
  python generate_train_csv.py --dry-run --limit 50
        """
    )
    
    parser.add_argument(
        '--root',
        type=str,
        default='data/raw/datasets/train_10k',
        help='Directory containing sequence subfolders (default: data/raw/datasets/train_10k)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='data/processed/train_index.csv',
        help='Output CSV path (default: data/processed/train_index.csv)'
    )
    
    parser.add_argument(
        '--exts',
        type=str,
        default='.jpg,.png,.jpeg',
        help='Comma-separated image extensions (default: .jpg,.png,.jpeg)'
    )
    
    parser.add_argument(
        '--gen-lines',
        action='store_true',
        help='Generate Canny line maps for middle frames'
    )
    
    parser.add_argument(
        '--lines-dir',
        type=str,
        default='data/processed/lines',
        help='Directory for line maps (default: data/processed/lines)'
    )
    
    parser.add_argument(
        '--canny',
        type=str,
        default='100,200',
        help='Canny thresholds: low,high (default: 100,200)'
    )
    
    parser.add_argument(
        '--min-seq-len',
        type=int,
        default=3,
        help='Minimum sequence length for triplets (default: 3)'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Sliding window stride for sequences >3 frames (default: 1)'
    )
    
    parser.add_argument(
        '--relative-paths',
        action='store_true',
        help='Store paths relative to repo root'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help=f'Number of parallel workers (default: {cpu_count()})'
    )
    
    parser.add_argument(
        '--overwrite-lines',
        action='store_true',
        help='Re-generate line maps even if they exist'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for deterministic ordering (default: 42)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse and report counts without writing CSV or generating lines'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only first N sequences (for testing)'
    )
    
    args = parser.parse_args()
    
    # Parse extensions
    exts = [ext.strip() for ext in args.exts.split(',')]
    
    # Parse Canny thresholds
    try:
        canny_parts = args.canny.split(',')
        canny_low = int(canny_parts[0])
        canny_high = int(canny_parts[1])
    except (ValueError, IndexError):
        print("ERROR: --canny must be two integers separated by comma (e.g., 100,200)", file=sys.stderr)
        sys.exit(1)
    
    # Setup paths
    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        print(f"ERROR: Root directory does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find repository root for relative paths
    if args.relative_paths:
        repo_root = find_repo_root(root_dir)
    else:
        repo_root = root_dir  # Use root_dir as base for absolute paths
    
    lines_dir = Path(args.lines_dir) if args.gen_lines else None
    
    # Ensure output directory exists
    if not args.dry_run:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all sequence directories, sorted deterministically
    seq_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()], key=lambda p: natural_sort_key(p.name))
    
    if args.limit:
        seq_dirs = seq_dirs[:args.limit]
    
    print(f"Found {len(seq_dirs)} sequence directories")
    if args.relative_paths and repo_root:
        print(f"Repository root: {repo_root}")
    
    # Statistics
    stats = {
        'sequences_scanned': 0,
        'triplets_written': 0,
        'lines_generated': 0,
        'skips': 0
    }
    
    all_rows = []
    
    # Process sequences
    for seq_dir in seq_dirs:
        stats['sequences_scanned'] += 1
        
        try:
            rows = process_sequence(
                seq_dir,
                root_dir,
                repo_root,
                lines_dir,
                exts,
                args.min_seq_len,
                args.stride,
                args.gen_lines,
                canny_low,
                canny_high,
                args.overwrite_lines,
                args.relative_paths
            )
            
            all_rows.extend(rows)
            stats['triplets_written'] += len(rows)
            
            if args.gen_lines:
                stats['lines_generated'] += sum(1 for row in rows if row['path_ltau'])
            
        except Exception as e:
            warnings.warn(f"Error processing {seq_dir}: {e}")
            stats['skips'] += 1
            continue
        
        if stats['sequences_scanned'] % 1000 == 0:
            print(f"Processed {stats['sequences_scanned']} sequences, {stats['triplets_written']} triplets...")
    
    # Write CSV
    if not args.dry_run:
        output_path = Path(args.out)
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['seq_id', 'path_i0', 'path_it', 'path_i1', 'tau', 'path_ltau']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\nCSV written to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequences scanned:  {stats['sequences_scanned']}")
    print(f"Triplets written:   {stats['triplets_written']}")
    if args.gen_lines:
        print(f"Line maps generated: {stats['lines_generated']}")
    print(f"Skipped sequences:   {stats['skips']}")
    print("="*60)
    
    if args.dry_run:
        print("\n[DRY RUN] No files were written.")


if __name__ == '__main__':
    main()

