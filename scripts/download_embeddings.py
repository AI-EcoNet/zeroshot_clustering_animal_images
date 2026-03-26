"""
Download pre-computed embeddings from HuggingFace.

This script downloads pre-computed ViT embeddings from the HUGO-Bench Paper 
Reproducibility repository. Using pre-computed embeddings is much faster than
computing them locally (seconds vs hours).

Available embeddings:
- DINOv3 (1280D) - Best performance
- DINOv2 (1536D) - Second best
- BioCLIP2 (768D)
- BioCLIP 2.5 (768D)
- CLIP (768D)
- SigLIP (768D)

Each model has embeddings for both Aves (birds) and Mammals splits.

Usage:
    python scripts/download_embeddings.py                    # Download all embeddings
    python scripts/download_embeddings.py --model dinov3     # Download only DINOv3
    python scripts/download_embeddings.py --split aves       # Download only Aves
    python scripts/download_embeddings.py --model dinov3 --split aves  # Specific combo
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Constants
REPO_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = REPO_ROOT / "outputs" / "embeddings"

# HuggingFace repository with pre-computed embeddings
HF_REPO = "AI-EcoNet/HUGO-Bench-Paper-Reproducibility"

# Model configurations
MODELS = {
    "dinov3": {
        "name": "DINOv3",
        "hf_config": "embeddings_dinov3_vith16plus",
        "dim": 1280,
        "description": "★★★ BEST - Self-supervised ViT-H+/16"
    },
    "dinov2": {
        "name": "DINOv2", 
        "hf_config": "embeddings_dinov2_vitg14",
        "dim": 1536,
        "description": "★★ SECOND - Self-supervised ViT-G/14"
    },
    "bioclip2": {
        "name": "BioCLIP 2",
        "hf_config": "embeddings_bioclip2_vitl14",
        "dim": 768,
        "description": "Biology domain ViT-L/14"
    },
    "bioclip2_5": {
        "name": "BioCLIP 2.5",
        "hf_config": "embeddings_bioclip2_5_vith14",
        "dim": 768,
        "description": "Biology domain ViT-H/14"
    },
    "clip": {
        "name": "CLIP",
        "hf_config": "embeddings_clip_vitl14",
        "dim": 768,
        "description": "OpenAI CLIP ViT-L/14"
    },
    "siglip": {
        "name": "SigLIP",
        "hf_config": "embeddings_siglip_vitb16",
        "dim": 768,
        "description": "Sigmoid loss ViT-B/16"
    }
}

SPLITS = ["aves", "mammals"]


def check_embedding_exists(model: str, split: str) -> bool:
    """Check if embedding file already exists locally."""
    emb_file = EMBEDDINGS_DIR / f"{split}_{model}_embeddings.pkl"
    return emb_file.exists()


def get_local_embedding_info(model: str, split: str) -> Optional[Dict]:
    """Get info about existing local embedding file."""
    emb_file = EMBEDDINGS_DIR / f"{split}_{model}_embeddings.pkl"
    if not emb_file.exists():
        return None
    
    try:
        with open(emb_file, 'rb') as f:
            data = pickle.load(f)
        
        if hasattr(data, 'embeddings'):
            n_samples = len(data.embeddings)
            n_classes = len(set(data.labels))
        else:
            n_samples = len(data.get('embeddings', []))
            n_classes = len(set(data.get('labels', [])))
        
        size_mb = emb_file.stat().st_size / (1024 * 1024)
        
        return {
            "path": emb_file,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "size_mb": size_mb
        }
    except Exception as e:
        return {"path": emb_file, "error": str(e)}


def download_from_hf_pkl(model: str, split: str, force: bool = False) -> bool:
    """
    Download pre-computed embeddings as PKL file from HuggingFace.
    
    Downloads from the precomputed_embeddings/ folder in the HF repo.
    """
    from huggingface_hub import hf_hub_download
    import shutil
    
    output_file = EMBEDDINGS_DIR / f"{split}_{model}_embeddings.pkl"
    
    if output_file.exists() and not force:
        print(f"      ✓ Already exists: {output_file.name}")
        return True
    
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try downloading from precomputed_embeddings/ folder (our new uploads)
    pkl_path = f"precomputed_embeddings/{split}_{model}_embeddings.pkl"
    
    try:
        print(f"      Downloading: {pkl_path}...")
        local_path = hf_hub_download(
            HF_REPO,
            pkl_path,
            repo_type="dataset",
        )
        
        # Copy to our expected location
        downloaded = Path(local_path)
        if downloaded.exists():
            shutil.copy2(downloaded, output_file)
            
            # Verify the download
            with open(output_file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                n_samples = len(data.get('embeddings', []))
            else:
                n_samples = len(data.embeddings) if hasattr(data, 'embeddings') else 0
            
            print(f"      ✓ Downloaded: {output_file.name} ({n_samples:,} samples)")
            return True
        
    except Exception as e:
        print(f"      [!] Download from precomputed_embeddings failed: {e}")
        print(f"      Trying legacy folders...")
    
    # Try legacy folder (extreme_uneven_embeddings) for dinov3 only
    if model == "dinov3":
        try:
            hf_split = "aves" if split == "aves" else "mammalia"
            pkl_path = f"extreme_uneven_embeddings/{hf_split}_full_dinov3_embeddings.pkl"
            
            print(f"      Trying legacy: {pkl_path}...")
            local_path = hf_hub_download(
                HF_REPO,
                pkl_path,
                repo_type="dataset",
            )
            
            # Load and re-save with our naming convention
            with open(local_path, 'rb') as f:
                data = pickle.load(f)
            
            # Save as dict format
            if isinstance(data, dict):
                embeddings = data.get('embeddings')
                labels = data.get('labels', [])
                paths = data.get('paths', [])
            else:
                embeddings = data.embeddings
                labels = list(data.labels)
                paths = list(data.paths)
            
            # Save with dict format for consistency
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'labels': labels,
                    'paths': paths
                }, f)
            
            print(f"      ✓ Downloaded (legacy): {output_file.name} ({len(labels):,} samples)")
            return True
            
        except Exception as e:
            print(f"      [!] Legacy download failed: {e}")
    
    # Fall back to parquet dataset
    return download_from_hf_parquet(model, split, force)


def download_from_hf_parquet(model: str, split: str, force: bool = False) -> bool:
    """
    Download embeddings from HuggingFace parquet dataset and convert to PKL.
    """
    try:
        from datasets import load_dataset
        import numpy as np
    except ImportError:
        print("[X] Required packages not installed. Run:")
        print("    pip install datasets numpy")
        return False
    
    output_file = EMBEDDINGS_DIR / f"{split}_{model}_embeddings.pkl"
    
    if output_file.exists() and not force:
        print(f"      ✓ Already exists: {output_file.name}")
        return True
    
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_info = MODELS.get(model)
    if not model_info:
        print(f"[X] Unknown model: {model}")
        return False
    
    hf_config = model_info["hf_config"]
    
    try:
        print(f"      Loading from HuggingFace: {hf_config}...")
        ds = load_dataset(HF_REPO, hf_config, split="train")
        
        # Filter by split (aves or mammals/mammalia)
        split_filter = split.lower()
        if split_filter == "mammals":
            split_filter = "mammalia"
        
        print(f"      Filtering for {split}...")
        
        # Check what columns exist
        columns = ds.column_names
        print(f"      Columns: {columns}")
        
        # Filter based on class/species column
        if 'class' in columns:
            # Filter by class column - check first few entries to see format
            sample_classes = [ds[i]['class'] for i in range(min(10, len(ds)))]
            print(f"      Sample classes: {sample_classes[:5]}")
            
            # Determine filter based on class names
            if split == "aves":
                # Aves is typically "Aves" in the class column
                filtered = ds.filter(lambda x: x['class'].lower() == 'aves')
            else:
                # Mammals is "Mammalia"
                filtered = ds.filter(lambda x: x['class'].lower() == 'mammalia')
        else:
            # No class column - use all data
            print(f"      Warning: No 'class' column found, using all data")
            filtered = ds
        
        print(f"      Found {len(filtered):,} samples for {split}")
        
        if len(filtered) == 0:
            print(f"[X] No samples found for {split}")
            return False
        
        # Extract embeddings, labels, paths
        embeddings = np.array([x['embedding'] for x in filtered])
        
        # Try different column names for labels
        if 'species' in columns:
            labels = [x['species'] for x in filtered]
        elif 'label' in columns:
            labels = [x['label'] for x in filtered]
        elif 'species_name' in columns:
            labels = [x['species_name'] for x in filtered]
        else:
            labels = ["unknown"] * len(filtered)
        
        # Try different column names for paths
        if 'image_path' in columns:
            paths = [x['image_path'] for x in filtered]
        elif 'filename' in columns:
            paths = [x['filename'] for x in filtered]
        elif 'path' in columns:
            paths = [x['path'] for x in filtered]
        else:
            paths = [f"image_{i}" for i in range(len(filtered))]
        
        # Create embedding data structure
        from dataclasses import dataclass
        
        @dataclass
        class EmbeddingData:
            embeddings: np.ndarray
            labels: List[str]
            paths: List[str]
        
        emb_data = EmbeddingData(embeddings=embeddings, labels=labels, paths=paths)
        
        # Save
        with open(output_file, 'wb') as f:
            pickle.dump(emb_data, f)
        
        n_classes = len(set(labels))
        size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"      ✓ Saved: {output_file.name}")
        print(f"        {len(labels):,} samples, {n_classes} classes, {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"[X] Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_embeddings() -> None:
    """List all available embeddings (local and remote)."""
    print("\n" + "=" * 70)
    print("  AVAILABLE EMBEDDINGS")
    print("=" * 70)
    
    print("\n  Local embeddings (in outputs/embeddings/):")
    print("-" * 60)
    
    any_local = False
    for model in MODELS:
        for split in SPLITS:
            info = get_local_embedding_info(model, split)
            if info:
                any_local = True
                if "error" in info:
                    print(f"    {split}_{model}: ERROR - {info['error']}")
                else:
                    print(f"    ✓ {split}_{model}: {info['n_samples']:,} samples, "
                          f"{info['n_classes']} classes, {info['size_mb']:.1f} MB")
    
    if not any_local:
        print("    (none)")
    
    print("\n  Remote embeddings (on HuggingFace):")
    print("-" * 60)
    print(f"    Repository: {HF_REPO}")
    print()
    for model, info in MODELS.items():
        print(f"    {model}: {info['description']}")
        print(f"              Config: {info['hf_config']}, Dim: {info['dim']}")
    
    print("\n" + "=" * 70)


def download_embeddings(
    models: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    force: bool = False,
    quiet: bool = False
) -> bool:
    """
    Download pre-computed embeddings.
    
    Args:
        models: List of models to download (None = all)
        splits: List of splits to download (None = all)
        force: Re-download even if exists
        quiet: Suppress progress output
    
    Returns:
        True if all downloads successful
    """
    if models is None:
        models = list(MODELS.keys())
    if splits is None:
        splits = SPLITS.copy()
    
    # Validate inputs
    for model in models:
        if model not in MODELS:
            print(f"[X] Unknown model: {model}")
            print(f"    Available: {', '.join(MODELS.keys())}")
            return False
    
    for split in splits:
        if split not in SPLITS:
            print(f"[X] Unknown split: {split}")
            print(f"    Available: {', '.join(SPLITS)}")
            return False
    
    if not quiet:
        print("\n" + "=" * 70)
        print("  DOWNLOADING PRE-COMPUTED EMBEDDINGS")
        print("=" * 70)
        print(f"  Models: {', '.join(models)}")
        print(f"  Splits: {', '.join(splits)}")
        print(f"  Source: {HF_REPO}")
        print("=" * 70)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for model in models:
        if not quiet:
            print(f"\n  [{model.upper()}] {MODELS[model]['description']}")
        
        for split in splits:
            if check_embedding_exists(model, split) and not force:
                if not quiet:
                    print(f"      ✓ {split}: Already exists (use --force to re-download)")
                skip_count += 1
                continue
            
            if not quiet:
                print(f"      Downloading {split}...")
            
            success = download_from_hf_pkl(model, split, force)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    if not quiet:
        print("\n" + "=" * 70)
        print(f"  Download complete!")
        print(f"    ✓ Downloaded: {success_count}")
        print(f"    ○ Skipped (existing): {skip_count}")
        if fail_count > 0:
            print(f"    ✗ Failed: {fail_count}")
        print("=" * 70)
    
    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-computed embeddings from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  dinov3   - DINOv3 ViT-H+/16 (1280D) ★★★ BEST
  dinov2   - DINOv2 ViT-G/14 (1536D) ★★ SECOND
  bioclip2 - BioCLIP 2 ViT-L/14 (768D)
    bioclip2_5 - BioCLIP 2.5 ViT-H/14 (768D)
  clip     - CLIP ViT-L/14 (768D)
  siglip   - SigLIP ViT-B/16 (768D)

Examples:
  python {Path(__file__).name}                        # Download all
  python {Path(__file__).name} --model dinov3         # Only DINOv3
  python {Path(__file__).name} --split aves           # Only Aves
  python {Path(__file__).name} --model dinov3 dinov2  # Multiple models
  python {Path(__file__).name} --list                 # List available
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        help="Model(s) to download (default: all)"
    )
    
    parser.add_argument(
        "--split", "-s",
        nargs="+",
        choices=SPLITS + ["all", "both"],
        help="Split(s) to download (default: all)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if files exist"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available embeddings (local and remote)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_available_embeddings()
        return
    
    # Parse models
    if args.model is None or "all" in args.model:
        models = None  # All models
    else:
        models = args.model
    
    # Parse splits
    if args.split is None or "all" in args.split or "both" in args.split:
        splits = None  # All splits
    else:
        splits = args.split
    
    # Download
    success = download_embeddings(
        models=models,
        splits=splits,
        force=args.force,
        quiet=args.quiet
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
