"""
Interactive UI Pipeline for Zero-Shot Clustering

This provides a streamlined central UI for running the complete clustering pipeline.
It guides users through all configuration options with sensible defaults based on
the paper: "Vision Transformers for Zero-Shot Clustering of Animal Images"
(Markoff et al., 2026) - https://arxiv.org/abs/2602.03894

Configuration steps:
1. Data Selection (Aves, Mammals, Both, or Custom) - Default: Both
2. Data Distribution (Even, Uneven, All) - Default: Even (200/class)
3. ViT Model Selection - DINOv3 recommended, DINOv2 second best
4. Dimension Reduction - t-SNE recommended, UMAP also recommended
5. Clustering Method with parameters
6. Output Folder Selection
7. Confirmation and "Go" to run

Results are saved in organized folders with descriptive names like:
  results/run_1_dinov3_tsne_hdbscan_15_5/

Usage:
    python ui_pipeline.py
"""

import os
import sys
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

REPO_ROOT = Path(__file__).parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
RESULTS_DIR = REPO_ROOT / "results"
TEST_DATA_DIR = REPO_ROOT / "test_data"
CUSTOM_DATA_DIR = REPO_ROOT / "custom_data"

# HuggingFace dataset with pre-computed embeddings
HUGGINGFACE_REPO = "AI-EcoNet/HUGO-Bench-Paper-Reproducibility"

# Model rankings from paper benchmarking results
VIT_MODELS = {
    "dinov3": {"name": "DINOv3", "dim": 1280, "recommended": True, "order": 1},
    "dinov2": {"name": "DINOv2", "dim": 1536, "recommended": True, "order": 2},
    "bioclip2": {"name": "BioCLIP 2", "dim": 768, "recommended": False, "order": 3},
    "bioclip2_5": {"name": "BioCLIP 2.5", "dim": 768, "recommended": False, "order": 3},
    "clip": {"name": "CLIP", "dim": 768, "recommended": False, "order": 4},
    "siglip": {"name": "SigLIP", "dim": 768, "recommended": False, "order": 5},
}

REDUCTION_METHODS = {
    "tsne": {"name": "t-SNE", "desc": "Best for visualization", "recommended": True, "order": 1},
    "umap": {"name": "UMAP", "desc": "Fast, good structure, fewer outliers", "recommended": True, "order": 2},
    "pca": {"name": "PCA", "desc": "Linear baseline, very fast", "recommended": False, "order": 3},
    "isomap": {"name": "Isomap", "desc": "Preserves geodesic distances", "recommended": False, "order": 4},
    "kernel_pca": {"name": "Kernel PCA", "desc": "Non-linear with RBF kernel", "recommended": False, "order": 5},
}

CLUSTERING_METHODS = {
    "hdbscan": {
        "name": "HDBSCAN",
        "desc": "Density-based, auto cluster count",
        "supervised": False,
        "recommended": True,
        "order": 1
    },
    "dbscan": {
        "name": "DBSCAN",
        "desc": "Density-based, may over-cluster",
        "supervised": False,
        "recommended": False,
        "order": 2
    },
    "hierarchical": {
        "name": "Hierarchical",
        "desc": "Agglomerative, requires K clusters",
        "supervised": True,
        "recommended": True,
        "order": 3
    },
    "gmm": {
        "name": "GMM",
        "desc": "Gaussian Mixture Model, requires K clusters",
        "supervised": True,
        "recommended": False,
        "order": 4
    },
}

HDBSCAN_PRESETS = {
    "small": {"min_cluster_size": 15, "min_samples": 5, "desc": "Even data, <300 samples/class", "default": True},
    "medium": {"min_cluster_size": 100, "min_samples": 30, "desc": "Mixed representation", "default": False},
    "large": {"min_cluster_size": 150, "min_samples": 50, "desc": "Large/uneven data (RECOMMENDED for uneven)", "default": False},
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def print_subheader(title: str, char: str = "-", width: int = 50):
    """Print a formatted subheader."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def find_available_embeddings() -> Dict[str, Dict[str, Path]]:
    """
    Find all available embedding files, organized by split (aves/mammals).
    
    Returns:
        Dict with structure: {split: {model: path}}
        e.g. {"aves": {"dinov3": Path(...), "dinov2": Path(...)}, "mammals": {...}}
    """
    embeddings = {"aves": {}, "mammals": {}}
    
    if EMBEDDINGS_DIR.exists():
        for pkl in EMBEDDINGS_DIR.glob("*.pkl"):
            name = pkl.stem.replace("_embeddings", "")
            
            if name.startswith("mammals_"):
                model = name.replace("mammals_", "")
                embeddings["mammals"][model] = pkl
            elif name.startswith("aves_"):
                model = name.replace("aves_", "")
                embeddings["aves"][model] = pkl
            else:
                # Legacy format without prefix - assume aves
                embeddings["aves"][name] = pkl
    
    # Also check embeddings_mammals folder (legacy location)
    mammals_dir = OUTPUTS_DIR / "embeddings_mammals"
    if mammals_dir.exists():
        for pkl in mammals_dir.glob("*.pkl"):
            model = pkl.stem.replace("_embeddings", "")
            if model not in embeddings["mammals"]:
                embeddings["mammals"][model] = pkl
    
    # Remove empty splits
    embeddings = {k: v for k, v in embeddings.items() if v}
    
    return embeddings


def get_next_run_number() -> int:
    """Get the next run number for results folder."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    existing_runs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not existing_runs:
        return 1
    
    run_numbers = []
    for d in existing_runs:
        try:
            num = int(d.name.split("_")[1])
            run_numbers.append(num)
        except (IndexError, ValueError):
            continue
    
    return max(run_numbers, default=0) + 1


def load_embedding_data(emb_path: Path) -> Dict:
    """Load embedding data from pickle file."""
    with open(emb_path, 'rb') as f:
        return pickle.load(f)


def combine_embeddings(emb_paths: List[Path]) -> Dict:
    """
    Combine embeddings from multiple files into a single dataset.
    Used when 'both' is selected to merge aves and mammals embeddings.
    
    Args:
        emb_paths: List of paths to embedding pickle files
        
    Returns:
        Combined embedding data dict with 'embeddings', 'labels', 'paths'
    """
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    for emb_path in emb_paths:
        with open(emb_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both dict and EmbeddingRecord formats
        if hasattr(data, 'embeddings'):
            emb = data.embeddings
            lbls = list(data.labels)
            paths = list(data.paths)
        else:
            emb = data.get('embeddings', data.get('features', []))
            lbls = list(data.get('labels', []))
            paths = list(data.get('paths', data.get('image_paths', [])))
        
        all_embeddings.append(emb)
        all_labels.extend(lbls)
        all_paths.extend(paths)
    
    # Concatenate embeddings
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"      Combined {len(emb_paths)} files: {len(all_labels):,} samples, {len(set(all_labels))} classes")
    
    return {
        'embeddings': combined_embeddings,
        'labels': all_labels,
        'paths': all_paths
    }


# ============================================================================
# UI PROMPTS
# ============================================================================

def prompt_reduction_selection() -> List[str]:
    """Prompt user to select dimension reduction method(s)."""
    print_header("STEP 4: DIMENSION REDUCTION")
    print()
    print("  Select dimension reduction method(s) for 2D visualization:")
    print()
    
    sorted_methods = sorted(REDUCTION_METHODS.items(), key=lambda x: x[1]["order"])
    
    for i, (key, info) in enumerate(sorted_methods, 1):
        rec = " ★ RECOMMENDED" if info["recommended"] else ""
        print(f"    {i}. {info['name']:12} - {info['desc']}{rec}")
    
    print(f"    {len(sorted_methods) + 1}. Both t-SNE & UMAP (recommended pair)")
    print(f"    {len(sorted_methods) + 2}. ALL methods")
    print()
    print("  ★ t-SNE is marginally better (+0.6-1.1 pp)")
    print("    UMAP has fewer outliers for rare species")
    print()
    
    default = 1  # t-SNE
    
    while True:
        choice = input(f"  Select (1-{len(sorted_methods) + 2}, Enter={default}): ").strip()
        
        if choice == "":
            return ["tsne"]  # Default to t-SNE
        
        if choice == str(len(sorted_methods) + 1):
            return ["tsne", "umap"]
        
        if choice == str(len(sorted_methods) + 2):
            return list(REDUCTION_METHODS.keys())
        
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            methods = []
            for idx in indices:
                if 1 <= idx <= len(sorted_methods):
                    methods.append(sorted_methods[idx - 1][0])
            
            if methods:
                return methods
        except ValueError:
            pass
        
        print("  [!] Invalid choice. Try again.")


def prompt_model_selection(is_custom: bool = False) -> Tuple[str, bool]:
    """
    Prompt user to select ViT model.
    Returns: (model_name, use_precomputed)
    """
    print_header("STEP 3: VIT MODEL SELECTION")
    print()
    print("  Select Vision Transformer model for feature extraction:")
    print()
    
    sorted_models = sorted(VIT_MODELS.items(), key=lambda x: x[1]["order"])
    
    print("  Model               Embed Dim    Recommendation")
    print("  " + "-" * 50)
    
    for i, (key, info) in enumerate(sorted_models, 1):
        rec = "★★★ BEST" if i == 1 else ("★★ RECOMMENDED" if i == 2 else "")
        print(f"    {i}. {info['name']:15} {info['dim']:>5}D        {rec}")
    
    print(f"    {len(sorted_models) + 1}. ALL models (run all and compare)")
    print()
    
    if not is_custom:
        print("  Note: Pre-computed embeddings may be available for download")
        print("        from HuggingFace (faster than computing locally)")
    print()
    
    default = 1  # DINOv3
    
    while True:
        choice = input(f"  Select model (1-{len(sorted_models) + 1}, Enter={default}): ").strip()
        
        if choice == "":
            return "dinov3", not is_custom  # Default to DINOv3 with precomputed if available
        
        if choice == str(len(sorted_models) + 1):
            return "all", False  # All models - must compute
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(sorted_models):
                model = sorted_models[idx - 1][0]
                return model, not is_custom
        except ValueError:
            pass
        
        print("  [!] Invalid choice. Try again.")


def prompt_data_selection() -> Tuple[str, List[Path], bool]:
    """
    Prompt user to select data split (Mammals, Birds, Both, or Custom).
    Returns: (split_name, list_of_paths, is_custom_data)
    """
    print_header("STEP 1: DATA SELECTION")
    print()
    
    # Check available data
    aves_dir = TEST_DATA_DIR / "Aves"
    mammals_dir = TEST_DATA_DIR / "Mammals"
    
    aves_available = aves_dir.exists() and any(aves_dir.iterdir())
    mammals_available = mammals_dir.exists() and any(mammals_dir.iterdir())
    custom_available = CUSTOM_DATA_DIR.exists() and any(CUSTOM_DATA_DIR.iterdir())
    
    print("  Select which data to use for clustering:")
    print()
    
    options = []
    opt_num = 1
    default_opt = None
    
    if aves_available:
        n_classes = sum(1 for d in aves_dir.iterdir() if d.is_dir())
        n_images = sum(1 for _ in aves_dir.rglob("*.jpg")) + sum(1 for _ in aves_dir.rglob("*.png"))
        print(f"    {opt_num}. Aves (Birds)   - {n_classes} species, {n_images:,} images")
        options.append(("aves", [aves_dir], False))
        opt_num += 1
    
    if mammals_available:
        n_classes = sum(1 for d in mammals_dir.iterdir() if d.is_dir())
        n_images = sum(1 for _ in mammals_dir.rglob("*.jpg")) + sum(1 for _ in mammals_dir.rglob("*.png"))
        print(f"    {opt_num}. Mammals        - {n_classes} species, {n_images:,} images")
        options.append(("mammals", [mammals_dir], False))
        opt_num += 1
    
    if aves_available and mammals_available:
        aves_classes = sum(1 for d in aves_dir.iterdir() if d.is_dir())
        mammals_classes = sum(1 for d in mammals_dir.iterdir() if d.is_dir())
        total_classes = aves_classes + mammals_classes
        print(f"    {opt_num}. Both [DEFAULT] - {total_classes} species combined ★")
        options.append(("both", [aves_dir, mammals_dir], False))
        default_opt = opt_num
        opt_num += 1
    
    if custom_available:
        custom_folders = [d for d in CUSTOM_DATA_DIR.iterdir() if d.is_dir()]
        n_classes = len(custom_folders)
        n_images = sum(1 for _ in CUSTOM_DATA_DIR.rglob("*.jpg")) + sum(1 for _ in CUSTOM_DATA_DIR.rglob("*.png"))
        print(f"    {opt_num}. Custom         - {n_classes} folders, {n_images:,} images")
        options.append(("custom", [CUSTOM_DATA_DIR], True))
        if not default_opt:
            default_opt = opt_num
        opt_num += 1
    
    if not options:
        print("  [!] No data found!")
        print()
        print("  To use test data:")
        print("    python scripts/download_dataset.py --split both --yes")
        print()
        print("  To use custom data:")
        print("    1. Create subfolders in custom_data/ for each species")
        print("    2. Place images (.jpg, .png) in respective folders")
        sys.exit(1)
    
    print()
    if default_opt:
        print(f"  ★ = Recommended (press Enter for default)")
    print()
    
    while True:
        prompt = f"  Select data (1-{len(options)}"
        if default_opt:
            prompt += f", Enter={default_opt}"
        prompt += "): "
        
        choice = input(prompt).strip()
        
        # Handle default
        if choice == "" and default_opt:
            return options[default_opt - 1]
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        
        print("  [!] Invalid choice. Try again.")


def prompt_sampling_config() -> Dict:
    """Prompt user for data sampling configuration."""
    print_header("STEP 2: DATA DISTRIBUTION")
    print()
    print("  How should images be sampled from each species?")
    print()
    print("    1. All Data           - Use all available images")
    print("    2. Even [DEFAULT] ★   - Same samples per species (e.g., 200 each)")
    print("    3. Uneven             - Random 20-MAX per species (realistic)")
    print("    4. Custom             - Specify your own min/max per class")
    print()
    print("  ★ Even distribution recommended for fair comparison")
    print("    Uneven simulates real camera trap data")
    print()
    
    config = {
        "distribution": "even",
        "min_per_class": 200,
        "max_per_class": 200,
        "n_classes": None
    }
    
    default = 2  # Even
    
    while True:
        choice = input(f"  Select option (1-4, Enter={default}): ").strip()
        
        if choice == "" or choice == "2":
            # Even distribution (default)
            print()
            try:
                max_val = input("    Samples per class (Enter for 200): ").strip()
                config["max_per_class"] = int(max_val) if max_val else 200
                config["min_per_class"] = config["max_per_class"]
                config["distribution"] = "even"
                
                n_cls = input("    Limit number of classes? (Enter for all): ").strip()
                config["n_classes"] = int(n_cls) if n_cls else None
                
                return config
            except ValueError:
                print("  [!] Invalid number. Using defaults.")
                return config
        
        elif choice == "1":
            config["distribution"] = "all"
            config["min_per_class"] = None
            config["max_per_class"] = None
            return config
        
        elif choice == "3":
            print()
            print("    Note: For uneven data, use HDBSCAN(150,50) preset later")
            try:
                min_val = input("    Min samples per class (Enter for 20): ").strip()
                max_val = input("    Max samples per class (Enter for max): ").strip()
                config["min_per_class"] = int(min_val) if min_val else 20
                config["max_per_class"] = int(max_val) if max_val else None
                config["distribution"] = "uneven"
                
                n_cls = input("    Limit number of classes? (Enter for all): ").strip()
                config["n_classes"] = int(n_cls) if n_cls else None
                
                return config
            except ValueError:
                print("  [!] Invalid number. Try again.")
        
        elif choice == "4":
            print()
            try:
                min_val = input("    Min samples per class (Enter for no minimum): ").strip()
                max_val = input("    Max samples per class (Enter for no limit): ").strip()
                config["min_per_class"] = int(min_val) if min_val else None
                config["max_per_class"] = int(max_val) if max_val else None
                config["distribution"] = "custom"
                
                n_cls = input("    Limit number of classes? (Enter for all): ").strip()
                config["n_classes"] = int(n_cls) if n_cls else None
                
                return config
            except ValueError:
                print("  [!] Invalid number. Try again.")
        
        else:
            print("  [!] Invalid choice.")


def prompt_clustering_selection(n_true_classes: int, is_uneven: bool = False) -> List[str]:
    """Prompt user to select clustering method(s)."""
    print_header("STEP 5: CLUSTERING METHOD")
    print()
    print("  UNSUPERVISED (auto-detect cluster count - when species count unknown):")
    
    sorted_methods = sorted(CLUSTERING_METHODS.items(), key=lambda x: x[1]["order"])
    
    for i, (key, info) in enumerate(sorted_methods, 1):
        rec = ""
        if key == "hdbscan":
            rec = " ★★★ RECOMMENDED"
        elif key == "hierarchical":
            rec = " ★★ (if you know K)"
        sup = " [requires K]" if info["supervised"] else ""
        print(f"    {i}. {info['name']:15} - {info['desc']}{sup}{rec}")
    
    print(f"    {len(sorted_methods) + 1}. ALL methods (run all clustering methods)")
    print()
    
    # Data-specific advice
    print("  Recommendation based on your data:")
    if is_uneven:
        print("    → HDBSCAN with 'large' preset (150,50) for uneven data")
    else:
        print("    → HDBSCAN with 'small' preset (15,5) for even data")
    print(f"    → If you know there are ~{n_true_classes} species, try Hierarchical with K={n_true_classes}")
    print()
    
    default = 1  # HDBSCAN
    
    while True:
        choice = input(f"  Select (1-{len(sorted_methods) + 1}, Enter={default}): ").strip()
        
        if choice == "" or choice == "1":
            return ["hdbscan"]
        
        if choice == str(len(sorted_methods) + 1):
            return list(CLUSTERING_METHODS.keys())
        
        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            methods = []
            for idx in indices:
                if 1 <= idx <= len(sorted_methods):
                    methods.append(sorted_methods[idx - 1][0])
            
            if methods:
                return methods
        except ValueError:
            pass
        
        print("  [!] Invalid choice. Try again.")


def prompt_hdbscan_params(is_uneven: bool = False, sampling_config: Dict = None) -> Dict:
    """
    Prompt for HDBSCAN parameters with smart recommendations based on data distribution.
    
    Recommendations:
    - Even distribution (e.g., 200/class): Use 'small' preset (15, 5)
    - Uneven with 20-MAX range: Use 'medium' preset (100, 30)
    - All data / very uneven: Use 'large' preset (150, 50)
    """
    print_subheader("HDBSCAN PARAMETERS")
    print()
    print("  Presets (based on data characteristics):")
    print()
    
    # Determine recommended preset based on sampling config
    if sampling_config is None:
        sampling_config = {}
    
    distribution = sampling_config.get("distribution", "even")
    min_per_class = sampling_config.get("min_per_class")
    max_per_class = sampling_config.get("max_per_class")
    
    # Smart preset recommendation
    if distribution == "all":
        recommended_preset = "large"
        recommendation_reason = "Using all data → 'large' preset handles variable class sizes"
    elif distribution == "uneven":
        if min_per_class and min_per_class <= 20:
            if max_per_class is None or max_per_class >= 500:
                recommended_preset = "large"
                recommendation_reason = f"Uneven {min_per_class}-MAX → 'large' preset for high variance"
            else:
                recommended_preset = "medium"
                recommendation_reason = f"Uneven {min_per_class}-{max_per_class} → 'medium' preset"
        else:
            recommended_preset = "medium"
            recommendation_reason = f"Uneven distribution → 'medium' preset"
    else:
        # Even or custom with low variance
        if max_per_class and max_per_class <= 100:
            recommended_preset = "small"
            recommendation_reason = f"Even ≤{max_per_class}/class → 'small' preset"
        elif max_per_class and max_per_class <= 300:
            recommended_preset = "small"
            recommendation_reason = f"Even {max_per_class}/class → 'small' preset"
        else:
            recommended_preset = "medium"
            recommendation_reason = f"Even with large samples → 'medium' preset"
    
    preset_defaults = {"small": 1, "medium": 2, "large": 3}
    default = preset_defaults.get(recommended_preset, 1)
    
    for i, (key, info) in enumerate(HDBSCAN_PRESETS.items(), 1):
        rec = " ★ RECOMMENDED" if key == recommended_preset else ""
        print(f"    {i}. {key.upper():8} - min_cluster_size={info['min_cluster_size']}, "
              f"min_samples={info['min_samples']}{rec}")
        print(f"              {info['desc']}")
    
    print("    4. Custom - Specify your own values")
    print()
    print(f"  → {recommendation_reason}")
    print()
    
    while True:
        choice = input(f"  Select preset (1-4, Enter={default}): ").strip()
        
        if choice == "" or choice == str(default):
            return HDBSCAN_PRESETS[recommended_preset].copy()
        
        if choice == "1":
            return HDBSCAN_PRESETS["small"].copy()
        elif choice == "2":
            return HDBSCAN_PRESETS["medium"].copy()
        elif choice == "3":
            return HDBSCAN_PRESETS["large"].copy()
        elif choice == "4":
            try:
                print()
                mcs = input("    min_cluster_size (default 15): ").strip()
                ms = input("    min_samples (default 5): ").strip()
                return {
                    "min_cluster_size": int(mcs) if mcs else 15,
                    "min_samples": int(ms) if ms else 5
                }
            except ValueError:
                print("  [!] Invalid number.")
        else:
            print("  [!] Invalid choice.")


def prompt_supervised_params(method: str, n_true_classes: int) -> Dict:
    """Prompt for supervised clustering parameters (K clusters)."""
    print_subheader(f"{method.upper()} PARAMETERS")
    print()
    print(f"  This method requires specifying the number of clusters (K).")
    print(f"  Your data has approximately {n_true_classes} species.")
    print()
    print(f"  Suggestions:")
    print(f"    K = {n_true_classes}      (optimal for species-level clustering)")
    print(f"    K = {n_true_classes * 2}     (may reveal intra-species variation: sex, age)")
    print(f"    K = {n_true_classes // 2}      (may group similar species together)")
    print()
    
    while True:
        try:
            k = input(f"  Number of clusters K (Enter for {n_true_classes}): ").strip()
            n_clusters = int(k) if k else n_true_classes
            
            if n_clusters < 2:
                print("  [!] K must be at least 2.")
                continue
            
            return {"n_clusters": n_clusters}
        except ValueError:
            print("  [!] Invalid number.")


def prompt_output_folder() -> Path:
    """Prompt for output folder selection."""
    print_header("STEP 6: OUTPUT FOLDER")
    print()
    print(f"  Default output folder: {RESULTS_DIR}")
    print()
    print("    1. Use default (results/)")
    print("    2. Specify custom folder")
    print()
    
    choice = input("  Select (1-2, Enter=1): ").strip()
    
    if choice == "" or choice == "1":
        return RESULTS_DIR
    elif choice == "2":
        custom_path = input("  Enter folder path: ").strip()
        if custom_path:
            path = Path(custom_path)
            path.mkdir(parents=True, exist_ok=True)
            return path
        return RESULTS_DIR
    
    return RESULTS_DIR


def prompt_save_cluster_images() -> bool:
    """Prompt user whether to save images organized by cluster."""
    print_header("STEP 7: SAVE CLUSTER IMAGES")
    print()
    print("  Would you like to save copies of images organized by cluster?")
    print()
    print("  This creates a folder structure like:")
    print("    results/run_X/clusters/")
    print("      cluster_1/   (images in cluster 1)")
    print("      cluster_2/   (images in cluster 2)")
    print("      ...          ")
    print("      rejected/    (outliers/noise)")
    print()
    print("  Note: This copies images and may use significant disk space.")
    print()
    print("    1. Yes - Save cluster images")
    print("    2. No  - Skip (default)")
    print()
    
    choice = input("  Select (1-2, Enter=2): ").strip()
    
    return choice == "1"


def prompt_embedding_selection() -> Tuple[str, List[Path]]:
    """
    Prompt user to select data split and model.
    
    Returns:
        (display_name, list_of_embedding_paths)
    """
    embeddings = find_available_embeddings()
    
    if not embeddings:
        print("  [!] No embeddings found. Run 'python main.py --full' first.")
        sys.exit(1)
    
    # Step 1: Select split (Mammals, Aves, or Both)
    print_header("DATA SPLIT SELECTION")
    print()
    print("  Select which data to use:")
    print()
    
    options = []
    opt_num = 1
    
    if "mammals" in embeddings:
        models = list(embeddings["mammals"].keys())
        print(f"    {opt_num}. Mammals  - {len(models)} models available")
        options.append(("mammals", ["mammals"]))
        opt_num += 1
    
    if "aves" in embeddings:
        models = list(embeddings["aves"].keys())
        print(f"    {opt_num}. Aves (Birds) - {len(models)} models available")
        options.append(("aves", ["aves"]))
        opt_num += 1
    
    if "mammals" in embeddings and "aves" in embeddings:
        print(f"    {opt_num}. Both (Combined)")
        options.append(("both", ["mammals", "aves"]))
        opt_num += 1
    
    print()
    
    while True:
        choice = input(f"  Select data (1-{len(options)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                split_name, splits = options[idx]
                break
        except ValueError:
            pass
        print("  [!] Invalid choice.")
    
    # Step 2: Select model
    # Find common models across selected splits
    if len(splits) == 1:
        available_models = set(embeddings[splits[0]].keys())
    else:
        # For "both", find models available in both splits
        available_models = set(embeddings[splits[0]].keys())
        for split in splits[1:]:
            available_models &= set(embeddings[split].keys())
    
    if not available_models:
        print("  [!] No common models found across selected splits.")
        sys.exit(1)
    
    # Sort models with recommended ones first
    model_order = ["dinov3", "dinov2", "bioclip2", "bioclip2_5", "clip", "siglip"]
    sorted_models = sorted(available_models, key=lambda x: model_order.index(x) if x in model_order else 99)
    
    print_subheader("MODEL SELECTION")
    print()
    print("  Available models:")
    print()
    
    for i, model in enumerate(sorted_models, 1):
        rec = " ★ RECOMMENDED" if model == "dinov3" else (" ★" if model == "dinov2" else "")
        # Get info from first available split
        path = embeddings[splits[0]][model]
        size_mb = path.stat().st_size / (1024 * 1024)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            n_samples = len(data.get('labels', data.get('embeddings', [])))
            n_classes = len(set(data.get('labels', []))) if 'labels' in data else '?'
            print(f"    {i}. {model.upper():10} ({n_samples:,} samples, {n_classes} classes){rec}")
        except:
            print(f"    {i}. {model.upper():10} ({size_mb:.1f}MB){rec}")
    
    print()
    
    while True:
        choice = input(f"  Select model (1-{len(sorted_models)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_models):
                selected_model = sorted_models[idx]
                break
        except ValueError:
            pass
        print("  [!] Invalid choice.")
    
    # Collect paths for selected model across all selected splits
    paths = [embeddings[split][selected_model] for split in splits]
    
    # Create display name
    if split_name == "both":
        display_name = f"{selected_model}_combined"
    else:
        display_name = f"{split_name}_{selected_model}"
    
    return display_name, paths


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def sample_data(
    embeddings: np.ndarray,
    labels: List[str],
    paths: List[str],
    config: Dict,
    seed: int = 42
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Sample data according to configuration."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    if config["distribution"] == "all":
        return embeddings, labels, paths
    
    # Group by class
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[label].append(i)
    
    # Select classes
    classes = list(class_indices.keys())
    if config["n_classes"] and config["n_classes"] < len(classes):
        classes = random.sample(classes, config["n_classes"])
    
    # Sample from each class
    selected_indices = []
    
    for cls in classes:
        indices = class_indices[cls]
        
        min_samples = config.get("min_per_class") or 1
        max_samples = config.get("max_per_class") or len(indices)
        
        if len(indices) < min_samples:
            continue  # Skip classes with too few samples
        
        if config["distribution"] == "even":
            # Take exactly max_per_class samples (or all if fewer)
            n_take = min(max_samples, len(indices))
        elif config["distribution"] == "uneven":
            # Random between min and max
            n_take = random.randint(min_samples, min(max_samples, len(indices)))
        else:  # custom
            n_take = min(max_samples, len(indices))
        
        sampled = random.sample(indices, n_take) if n_take < len(indices) else indices
        selected_indices.extend(sampled)
    
    # Shuffle
    random.shuffle(selected_indices)
    
    # Extract sampled data
    sampled_embeddings = embeddings[selected_indices]
    sampled_labels = [labels[i] for i in selected_indices]
    sampled_paths = [paths[i] for i in selected_indices]
    
    return sampled_embeddings, sampled_labels, sampled_paths


def run_ui_pipeline():
    """Run the interactive UI pipeline with all configuration options."""
    clear_screen()
    
    print_header("ZERO-SHOT CLUSTERING FOR ANIMAL IMAGES", "=", 70)
    print()
    print("  Welcome! This UI will guide you through all configuration options.")
    print()
    print("  Steps:")
    print("    1. Data Selection (Aves, Mammals, Both, Custom)")
    print("    2. Data Distribution (Even, Uneven, All)")
    print("    3. ViT Model Selection (DINOv3 recommended)")
    print("    4. Dimension Reduction (t-SNE recommended)")
    print("    5. Clustering Method (HDBSCAN recommended)")
    print("    6. Output Folder")
    print("    7. Save Cluster Images (optional)")
    print("    8. Confirmation → GO!")
    print()
    input("  Press Enter to start...")
    
    # ========================================================================
    # STEP 1: Data Selection
    # ========================================================================
    split_name, data_paths, is_custom = prompt_data_selection()
    
    print(f"\n  [OK] Selected: {split_name.upper()}")
    
    # Count available data
    total_classes = 0
    total_images = 0
    for path in data_paths:
        for species_dir in path.iterdir():
            if species_dir.is_dir():
                total_classes += 1
                total_images += sum(1 for _ in species_dir.rglob("*.jpg")) + sum(1 for _ in species_dir.rglob("*.png"))
    
    print(f"       {total_classes} species, {total_images:,} images available")
    
    # ========================================================================
    # STEP 2: Data Distribution
    # ========================================================================
    sampling_config = prompt_sampling_config()
    is_uneven = sampling_config["distribution"] == "uneven"
    
    print(f"\n  [OK] Distribution: {sampling_config['distribution']}")
    if sampling_config.get("max_per_class"):
        print(f"       Max per class: {sampling_config['max_per_class']}")
    
    # ========================================================================
    # STEP 3: Model Selection
    # ========================================================================
    selected_model, use_precomputed = prompt_model_selection(is_custom)
    
    print(f"\n  [OK] Model: {selected_model.upper()}")
    
    # ========================================================================
    # STEP 4: Dimension Reduction
    # ========================================================================
    reduction_methods = prompt_reduction_selection()
    
    print(f"\n  [OK] Reduction: {', '.join(m.upper() for m in reduction_methods)}")
    
    # ========================================================================
    # STEP 5: Clustering
    # ========================================================================
    # Estimate true class count
    n_estimated_classes = total_classes
    if sampling_config.get("n_classes"):
        n_estimated_classes = sampling_config["n_classes"]
    
    clustering_methods = prompt_clustering_selection(n_estimated_classes, is_uneven)
    
    print(f"\n  [OK] Clustering: {', '.join(m.upper() for m in clustering_methods)}")
    
    # Get parameters for each clustering method
    clustering_params = {}
    for method in clustering_methods:
        if method == "hdbscan":
            clustering_params[method] = prompt_hdbscan_params(is_uneven, sampling_config)
        elif CLUSTERING_METHODS[method]["supervised"]:
            clustering_params[method] = prompt_supervised_params(method, n_estimated_classes)
        else:
            clustering_params[method] = {}
    
    # ========================================================================
    # STEP 6: Output Folder
    # ========================================================================
    output_dir = prompt_output_folder()
    
    print(f"\n  [OK] Output: {output_dir}")
    
    # ========================================================================
    # STEP 7: Save Cluster Images
    # ========================================================================
    save_cluster_images_flag = prompt_save_cluster_images()
    
    print(f"\n  [OK] Save cluster images: {'Yes' if save_cluster_images_flag else 'No'}")
    
    # ========================================================================
    # STEP 8: CONFIRMATION
    # ========================================================================
    print_header("CONFIGURATION SUMMARY", "=", 70)
    print()
    print(f"  ┌{'─' * 50}┐")
    print(f"  │ {'DATA':12}   {split_name.upper():>35} │")
    print(f"  │ {'Distribution':12}   {sampling_config['distribution'].upper():>35} │")
    if sampling_config.get("max_per_class"):
        print(f"  │ {'Per class':12}   {sampling_config['max_per_class']:>35} │")
    print(f"  │ {'Model':12}   {selected_model.upper():>35} │")
    print(f"  │ {'Reduction':12}   {', '.join(reduction_methods).upper():>35} │")
    print(f"  │ {'Clustering':12}   {', '.join(clustering_methods).upper():>35} │")
    print(f"  │ {'Output':12}   {str(output_dir)[-35:]:>35} │")
    save_images_str = "Yes" if save_cluster_images_flag else "No"
    print(f"  │ {'Save images':12}   {save_images_str:>35} │")
    print(f"  └{'─' * 50}┘")
    print()
    
    # Show clustering parameters
    for method, params in clustering_params.items():
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "desc")
            print(f"  {method.upper()} params: {param_str}")
    
    print()
    print("  " + "=" * 50)
    print("  Ready to run the pipeline!")
    print("  " + "=" * 50)
    print()
    
    confirm = input("  Press Enter to GO, or 'q' to quit: ").strip().lower()
    if confirm == 'q':
        print("\n  [*] Cancelled.")
        return
    
    # ========================================================================
    # EXECUTE PIPELINE
    # ========================================================================
    print_header("RUNNING PIPELINE", "=", 70)
    
    # Import required modules
    from scripts.extract_embeddings import (
        discover_images, 
        extract_and_save as extract_embeddings_func,
        load_embeddings,
        AVAILABLE_MODELS as MODEL_INFO
    )
    from scripts.dimension_reduction import DimensionReducer, run_reduction, save_reduction
    from scripts.clustering import run_clustering, save_cluster_images
    from scripts.visualization import create_cluster_plot, print_cluster_summary
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    
    # ========================================================================
    # Extract or load embeddings
    # ========================================================================
    print(f"\n  [>] Processing embeddings...")
    
    embeddings_dir = OUTPUTS_DIR / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_run = [selected_model] if selected_model != "all" else list(VIT_MODELS.keys())
    
    embedding_files = {}
    
    for model in models_to_run:
        # Handle 'both' split by checking for separate aves/mammals files
        if split_name == "both":
            aves_file = embeddings_dir / f"aves_{model}_embeddings.pkl"
            mammals_file = embeddings_dir / f"mammals_{model}_embeddings.pkl"
            
            if aves_file.exists() and mammals_file.exists():
                print(f"      [*] Combining embeddings: aves + mammals for {model.upper()}")
                embedding_files[model] = [aves_file, mammals_file]  # List indicates combine needed
            else:
                # Check for combined file as fallback
                combined_file = embeddings_dir / f"both_{model}_embeddings.pkl"
                if combined_file.exists():
                    print(f"      [*] Using existing: {combined_file.name}")
                    embedding_files[model] = combined_file
                else:
                    print(f"      [!] Missing embeddings for {model.upper()} (need aves + mammals or both)")
                    print(f"          Looking for: {aves_file.name} + {mammals_file.name}")
                    continue
        else:
            emb_file = embeddings_dir / f"{split_name}_{model}_embeddings.pkl"
            
            if emb_file.exists():
                print(f"      [*] Using existing: {emb_file.name}")
                embedding_files[model] = emb_file
            else:
                print(f"      [>] Extracting {model.upper()} embeddings...")
                
                # Discover images from data paths
                all_image_paths = []
                all_labels = []
                for data_path in data_paths:
                    for species_dir in data_path.iterdir():
                        if species_dir.is_dir():
                            for img_path in species_dir.rglob("*"):
                                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                    all_image_paths.append(img_path)
                                    all_labels.append(species_dir.name)
                
                print(f"          Found {len(all_image_paths):,} images from {len(set(all_labels))} species")
                
                # Extract embeddings
                try:
                    success = extract_embeddings_func(
                        model_name=model,
                        data_path=data_paths[0].parent if len(data_paths) > 1 else data_paths[0],
                        output_path=emb_file,
                        limit_per_class=sampling_config.get("max_per_class"),
                        n_classes=sampling_config.get("n_classes"),
                        random_seed=42
                    )
                    if success:
                        embedding_files[model] = emb_file
                    else:
                        print(f"          [!] Failed to extract {model}")
                except Exception as e:
                    print(f"          [!] Error: {e}")
    
    if not embedding_files:
        print("\n  [X] No embeddings available. Cannot continue.")
        return
    
    # ========================================================================
    # Load embeddings and apply sampling
    # ========================================================================
    print(f"\n  [>] Loading and sampling data...")
    
    for model, emb_source in embedding_files.items():
        print(f"\n  Processing: {model.upper()}")
        
        # Load embeddings - handle both single file and list of files (for combined)
        if isinstance(emb_source, list):
            # Multiple files to combine (aves + mammals for 'both' split)
            combined = combine_embeddings(emb_source)
            embeddings = combined['embeddings']
            labels = combined['labels']
            paths = combined['paths']
        else:
            # Single file
            emb_data = load_embeddings(emb_source)
            embeddings = emb_data.embeddings
            labels = list(emb_data.labels)
            paths = list(emb_data.paths)
        
        # Apply sampling
        embeddings, labels, paths = sample_data(embeddings, labels, paths, sampling_config)
        
        n_samples = len(labels)
        n_classes = len(set(labels))
        print(f"      {n_samples:,} samples from {n_classes} classes")
        
        # ====================================================================
        # Run dimension reduction
        # ====================================================================
        reducer = DimensionReducer(n_components=2, random_state=42)
        
        for red_method in reduction_methods:
            print(f"\n      [>] Running {red_method.upper()} dimension reduction...")
            
            # Run reduction
            if red_method == "tsne":
                red_result = reducer.tsne(embeddings)
            elif red_method == "umap":
                red_result = reducer.umap(embeddings)
            elif red_method == "pca":
                red_result = reducer.pca(embeddings)
            elif red_method == "isomap":
                red_result = reducer.isomap(embeddings)
            elif red_method == "kernel_pca":
                red_result = reducer.kernel_pca(embeddings)
            else:
                print(f"          [!] Unknown method: {red_method}")
                continue
            
            embeddings_2d = red_result.embeddings
            print(f"          Done in {red_result.time_seconds:.1f}s")
            
            # ================================================================
            # Run clustering
            # ================================================================
            for clust_method in clustering_methods:
                params = clustering_params.get(clust_method, {})
                params_copy = {k: v for k, v in params.items() if k != "desc"}
                
                print(f"\n      [>] Running {clust_method.upper()} clustering...")
                
                try:
                    result = run_clustering(embeddings_2d, labels, clust_method, **params_copy)
                except Exception as e:
                    print(f"          [!] Error: {e}")
                    continue
                
                print(f"          Clusters found: {result.n_clusters}")
                print(f"          Outliers: {result.n_outliers}")
                if "v_measure" in result.metrics:
                    print(f"          V-Measure: {result.metrics['v_measure']:.3f}")
                
                # Create run folder
                run_num = get_next_run_number()
                
                # Build folder name with parameters
                param_suffix = ""
                if clust_method == "hdbscan":
                    param_suffix = f"_{params_copy.get('min_cluster_size', 15)}_{params_copy.get('min_samples', 5)}"
                elif clust_method == "dbscan":
                    param_suffix = f"_{params_copy.get('eps_multiplier', 1.0)}_{params_copy.get('min_samples', 5)}"
                elif clust_method in ["hierarchical", "gmm"]:
                    param_suffix = f"_k{params_copy.get('n_clusters', n_classes)}"
                
                folder_name = f"run_{run_num}_{model}_{red_method}_{clust_method}{param_suffix}"
                run_dir = output_dir / folder_name
                run_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\n          Saving to: {folder_name}/")
                
                # Save visualization
                title = f"{split_name.upper()} | {model.upper()} + {red_method.upper()} + {clust_method.upper()}\n"
                title += f"V-measure: {result.metrics.get('v_measure', 0):.3f}, "
                title += f"Clusters: {result.n_clusters}, Samples: {n_samples}"
                
                fig = create_cluster_plot(
                    embeddings_2d=embeddings_2d,
                    cluster_labels=result.labels,
                    labels_true=labels,
                    title=title,
                    show_legend=result.n_clusters <= 30
                )
                
                fig_path = run_dir / "cluster_plot.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Save results
                results_data = {
                    "split": split_name,
                    "model": model,
                    "reduction_method": red_method,
                    "reduction_time": red_result.time_seconds,
                    "clustering_method": clust_method,
                    "clustering_params": params_copy,
                    "clustering_time": result.time_seconds,
                    "n_samples": n_samples,
                    "n_true_classes": n_classes,
                    "n_clusters": result.n_clusters,
                    "n_outliers": result.n_outliers,
                    "metrics": result.metrics,
                    "cluster_labels": result.labels,
                    "true_labels": labels,
                    "embeddings_2d": embeddings_2d,
                    "image_paths": paths,
                }
                
                with open(run_dir / "results.pkl", 'wb') as f:
                    pickle.dump(results_data, f)
                
                # Save summary
                with open(run_dir / "summary.txt", 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write("  CLUSTERING RUN SUMMARY\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Run: {folder_name}\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Data: {split_name}\n")
                    f.write(f"Model: {model}\n")
                    f.write(f"Reduction: {red_method}\n")
                    f.write(f"Clustering: {clust_method}\n")
                    for k, v in params_copy.items():
                        f.write(f"  - {k}: {v}\n")
                    f.write(f"\nSamples: {n_samples}\n")
                    f.write(f"True Classes: {n_classes}\n")
                    f.write(f"Clusters Found: {result.n_clusters}\n")
                    f.write(f"Outliers: {result.n_outliers}\n\n")
                    f.write("Metrics:\n")
                    for k, v in result.metrics.items():
                        if isinstance(v, float):
                            f.write(f"  {k}: {v:.4f}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                
                # Save cluster images if requested
                if save_cluster_images_flag:
                    print(f"\n          Saving cluster images...")
                    saved = save_cluster_images(
                        cluster_labels=result.labels,
                        image_paths=paths,
                        output_dir=run_dir,
                        copy=True
                    )
                    n_saved = sum(len(v) for v in saved.values())
                    print(f"          Saved {n_saved} images to {len(saved)} cluster folders")
    
    # ========================================================================
    # DONE
    # ========================================================================
    print_header("PIPELINE COMPLETE!", "=", 70)
    print()
    print(f"  Results saved to: {output_dir}/")
    print()
    
    # List created runs
    recent_runs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[:5]
    
    if recent_runs:
        print("  Recent runs:")
        for run_dir in recent_runs:
            print(f"    • {run_dir.name}/")
    
    print()
    print("  Each run folder contains:")
    print("    • cluster_plot.png  - Visualization of clusters")
    print("    • results.pkl       - Full results data")
    print("    • summary.txt       - Human-readable summary")
    if save_cluster_images_flag:
        print("    • clusters/         - Images organized by cluster")
    print()


if __name__ == "__main__":
    try:
        run_ui_pipeline()
    except KeyboardInterrupt:
        print("\n\n  [*] Cancelled by user.")
        sys.exit(0)
