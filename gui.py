"""
Native Python GUI for Zero-Shot Clustering Pipeline

This provides a Tkinter-based desktop GUI for running the complete clustering pipeline.
Launch with: python gui.py

Based on: "Vision Transformers for Zero-Shot Clustering of Animal Images"
         Markoff et al. (2026) - https://arxiv.org/abs/2602.03894
Dataset: HUGO-Bench - https://huggingface.co/datasets/AI-EcoNet/HUGO-Bench
"""

import os
import sys
import pickle
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
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

# Model rankings from paper - ordered by performance
VIT_MODELS = [
    ("dinov3", "DINOv3", "★★★ BEST"),
    ("dinov2", "DINOv2", "★★ 2nd"),
    ("bioclip2_5", "BioCLIP 2.5", ""),
    ("bioclip2", "BioCLIP 2", ""),
    ("clip", "CLIP", ""),
    ("siglip", "SigLIP", ""),
]

# Reduction methods - ordered by recommendation
REDUCTION_METHODS = [
    ("tsne", "t-SNE", "★ Recommended"),
    ("umap", "UMAP", "★ Fewer outliers, faster"),
    ("pca", "PCA", "Linear baseline"),
    ("isomap", "Isomap", "Geodesic distances"),
    ("kernel_pca", "Kernel PCA", "Non-linear"),
]

# Clustering methods with data-specific recommendations
# Format: (key, name, requires_k, recommendations_by_split)
CLUSTERING_METHODS = [
    ("hdbscan", "HDBSCAN", False, {
        "aves": ("★★★ BEST", "Auto-detects cluster count"),
        "mammals": ("★★★ BEST", "Auto-detects cluster count"),
        "both": ("★★★ BEST", "Auto-detects cluster count"),
        "custom": ("★★★ RECOMMENDED", "Auto-detects cluster count"),
    }),
    ("hierarchical", "Hierarchical", True, {
        "aves": ("★★ Good", "Set K = expected clusters"),
        "mammals": ("★★ Good", "Set K = expected clusters"),
        "both": ("★★ Good", "Set K = species count"),
        "custom": ("★ Use if K known", "Requires cluster count"),
    }),
    ("gmm", "GMM", True, {
        "aves": ("", "Gaussian assumption"),
        "mammals": ("", "Gaussian assumption"),
        "both": ("", "Gaussian assumption"),
        "custom": ("", "Requires cluster count"),
    }),
    ("dbscan", "DBSCAN", False, {
        "aves": ("", "May over-cluster"),
        "mammals": ("", "May over-cluster"),
        "both": ("", "Sensitive to eps"),
        "custom": ("", "Density-based"),
    }),
]

HDBSCAN_PRESETS = {
    "small": {"min_cluster_size": 15, "min_samples": 5, "desc": "Even data (<300/class)"},
    "medium": {"min_cluster_size": 100, "min_samples": 30, "desc": "Mixed sizes"},
    "large": {"min_cluster_size": 150, "min_samples": 50, "desc": "Uneven/large data"},
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def try_download_embeddings(model: str, split: str, log_func=None) -> Optional[Path]:
    """
    Try to download pre-computed embeddings from HuggingFace.
    
    Returns the path to the downloaded file, or None if download fails.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if log_func:
            log_func("        [!] huggingface-hub not installed, cannot download")
        return None
    
    HF_REPO = "AI-EcoNet/HUGO-Bench-Paper-Reproducibility"
    HF_FOLDER = "precomputed_embeddings"
    
    output_file = EMBEDDINGS_DIR / f"{split}_{model}_embeddings.pkl"
    
    if output_file.exists():
        return output_file
    
    # Try downloading from HuggingFace
    remote_path = f"{HF_FOLDER}/{split}_{model}_embeddings.pkl"
    
    try:
        if log_func:
            log_func(f"        Trying to download from HuggingFace...")
            log_func(f"        {HF_REPO}/{remote_path}")
        
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        local_path = hf_hub_download(
            HF_REPO,
            remote_path,
            repo_type="dataset",
            local_dir=EMBEDDINGS_DIR.parent,
            local_dir_use_symlinks=False
        )
        
        # The downloaded file may be in a subfolder
        downloaded = Path(local_path)
        if downloaded.exists() and downloaded != output_file:
            # Copy to expected location
            import shutil
            shutil.copy2(downloaded, output_file)
        
        if output_file.exists():
            if log_func:
                log_func(f"        ✓ Downloaded successfully!")
            return output_file
        
    except Exception as e:
        if log_func:
            log_func(f"        [!] Download failed: {e}")
    
    return None


def get_available_data() -> Dict:
    """Check what data is available and compute statistics."""
    data = {"aves": None, "mammals": None, "custom": None}
    
    aves_dir = TEST_DATA_DIR / "Aves"
    mammals_dir = TEST_DATA_DIR / "Mammals"
    
    if aves_dir.exists() and any(aves_dir.iterdir()):
        n_classes = sum(1 for d in aves_dir.iterdir() if d.is_dir())
        class_sizes = []
        for d in aves_dir.iterdir():
            if d.is_dir():
                n = sum(1 for _ in d.rglob("*.jpg")) + sum(1 for _ in d.rglob("*.png"))
                class_sizes.append(n)
        n_images = sum(class_sizes)
        max_per_class = max(class_sizes) if class_sizes else 0
        data["aves"] = {"path": aves_dir, "n_classes": n_classes, "n_images": n_images, "max_per_class": max_per_class}
    
    if mammals_dir.exists() and any(mammals_dir.iterdir()):
        n_classes = sum(1 for d in mammals_dir.iterdir() if d.is_dir())
        class_sizes = []
        for d in mammals_dir.iterdir():
            if d.is_dir():
                n = sum(1 for _ in d.rglob("*.jpg")) + sum(1 for _ in d.rglob("*.png"))
                class_sizes.append(n)
        n_images = sum(class_sizes)
        max_per_class = max(class_sizes) if class_sizes else 0
        data["mammals"] = {"path": mammals_dir, "n_classes": n_classes, "n_images": n_images, "max_per_class": max_per_class}
    
    if CUSTOM_DATA_DIR.exists() and any(CUSTOM_DATA_DIR.iterdir()):
        n_classes = sum(1 for d in CUSTOM_DATA_DIR.iterdir() if d.is_dir())
        class_sizes = []
        for d in CUSTOM_DATA_DIR.iterdir():
            if d.is_dir():
                n = sum(1 for _ in d.rglob("*.jpg")) + sum(1 for _ in d.rglob("*.png"))
                class_sizes.append(n)
        n_images = sum(class_sizes)
        max_per_class = max(class_sizes) if class_sizes else 0
        data["custom"] = {"path": CUSTOM_DATA_DIR, "n_classes": n_classes, "n_images": n_images, "max_per_class": max_per_class}
    
    return data


def find_existing_embeddings() -> Dict[str, Dict[str, Path]]:
    """Find all existing embedding files."""
    embeddings = {}
    
    if EMBEDDINGS_DIR.exists():
        for pkl in EMBEDDINGS_DIR.glob("*.pkl"):
            name = pkl.stem.replace("_embeddings", "")
            
            for split in ["aves", "mammals", "both", "custom"]:
                if name.startswith(f"{split}_"):
                    model = name.replace(f"{split}_", "")
                    if split not in embeddings:
                        embeddings[split] = {}
                    embeddings[split][model] = pkl
                    break
            else:
                if "aves" not in embeddings:
                    embeddings["aves"] = {}
                embeddings["aves"][name] = pkl
    
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


def sample_data(
    embeddings: np.ndarray,
    labels: List[str],
    paths: List[str],
    distribution: str,
    samples_per_class: int,
    seed: int = 42
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Sample data according to configuration."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    if distribution == "all":
        return embeddings, labels, paths
    
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, label in enumerate(labels):
        class_indices[label].append(i)
    
    selected_indices = []
    
    for cls, indices in class_indices.items():
        if distribution == "even":
            # Even: take exactly samples_per_class (capped at 500)
            n_take = min(samples_per_class, len(indices), 500)
        elif distribution == "uneven":
            # Uneven: random between 20 and MAX available for that class
            max_available = len(indices)
            min_samples = min(20, max_available)
            n_take = random.randint(min_samples, max_available)
        else:
            n_take = len(indices)
        
        sampled = random.sample(indices, n_take) if n_take < len(indices) else indices
        selected_indices.extend(sampled)
    
    random.shuffle(selected_indices)
    
    sampled_embeddings = embeddings[selected_indices]
    sampled_labels = [labels[i] for i in selected_indices]
    sampled_paths = [paths[i] for i in selected_indices]
    
    return sampled_embeddings, sampled_labels, sampled_paths


def combine_embeddings(emb_paths: List[Path], load_func) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Combine embeddings from multiple files into a single dataset.
    Used when 'both' is selected to merge aves and mammals embeddings.
    
    Args:
        emb_paths: List of paths to embedding pickle files
        load_func: Function to load embeddings (load_embeddings)
        
    Returns:
        Tuple of (combined_embeddings, combined_labels, combined_paths)
    """
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    for emb_path in emb_paths:
        data = load_func(emb_path)
        all_embeddings.append(data.embeddings)
        all_labels.extend(list(data.labels))
        all_paths.extend(list(data.paths))
    
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return combined_embeddings, all_labels, all_paths


# ============================================================================
# GUI APPLICATION
# ============================================================================

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Zero-Shot Clustering for Animal Images")
        self.root.geometry("1000x800")
        self.root.minsize(900, 750)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        self.style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        self.style.configure("Run.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        self.style.configure("Status.TLabel", font=("Segoe UI", 9))
        self.style.configure("Good.TLabel", foreground="green")
        self.style.configure("Warning.TLabel", foreground="orange")
        self.style.configure("Rec.TLabel", foreground="#0066cc", font=("Segoe UI", 9, "italic"))
        
        # Variables
        self.data_var = tk.StringVar()
        self.distribution_var = tk.StringVar(value="even")
        self.samples_var = tk.IntVar(value=200)
        self.model_var = tk.StringVar(value="dinov3")
        self.reduction_vars = {}
        self.clustering_var = tk.StringVar(value="hdbscan")
        self.hdbscan_preset_var = tk.StringVar(value="small")
        self.n_clusters_var = tk.IntVar(value=30)
        self.output_var = tk.StringVar(value=str(RESULTS_DIR))
        self.save_images_var = tk.BooleanVar(value=False)
        
        self.running = False
        self.available_data = get_available_data()
        self.existing_embeddings = find_existing_embeddings()
        
        # Calculate max available per class
        self.max_per_class = 500
        for split_data in self.available_data.values():
            if split_data and split_data.get("max_per_class"):
                self.max_per_class = max(self.max_per_class, split_data["max_per_class"])
        
        self.create_widgets()
        self.update_recommendations()
    
    def create_widgets(self):
        """Create all GUI widgets with two-column layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="Zero-Shot Clustering for Animal Images", style="Title.TLabel").pack()
        ttk.Label(header_frame, text="Markoff et al. (2026) | HUGO-Bench Dataset", style="Subtitle.TLabel").pack()
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 1: Configuration (two columns)
        config_tab = ttk.Frame(notebook, padding="10")
        notebook.add(config_tab, text="⚙️ Configuration")
        
        # Tab 2: Output Log
        log_tab = ttk.Frame(notebook, padding="10")
        notebook.add(log_tab, text="📋 Output Log")
        
        self.notebook = notebook
        
        # === CONFIGURATION TAB - TWO COLUMNS ===
        config_tab.columnconfigure(0, weight=1)
        config_tab.columnconfigure(1, weight=1)
        
        # LEFT COLUMN
        left_col = ttk.Frame(config_tab)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # RIGHT COLUMN
        right_col = ttk.Frame(config_tab)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # === LEFT COLUMN WIDGETS ===
        
        # Step 1: Data Selection
        step1 = ttk.LabelFrame(left_col, text="  1. Data Selection  ", padding="10", style="Section.TLabelframe")
        step1.pack(fill=tk.X, pady=5)
        
        data_choices = self._build_data_choices()
        self.data_combo = ttk.Combobox(step1, textvariable=self.data_var, values=data_choices, 
                                        width=45, state="readonly")
        self.data_combo.pack(fill=tk.X, pady=3)
        self.data_combo.bind("<<ComboboxSelected>>", lambda e: self.update_recommendations())
        
        # Download buttons frame (shown when data is missing)
        self.download_frame = ttk.Frame(step1)
        self.download_frame.pack(fill=tk.X, pady=(5, 0))
        
        has_test_data = self.available_data["aves"] or self.available_data["mammals"]
        
        if not has_test_data:
            # Show download buttons
            ttk.Label(self.download_frame, text="No test data found. Download HUGO-Bench dataset:", 
                      style="Rec.TLabel").pack(anchor=tk.W)
            btn_frame = ttk.Frame(self.download_frame)
            btn_frame.pack(fill=tk.X, pady=3)
            self.download_data_btn = ttk.Button(btn_frame, text="📥 Download Dataset + Embeddings", 
                                                 command=self.download_all_thread)
            self.download_data_btn.pack(side=tk.LEFT, pady=2)
            self.download_status = ttk.Label(btn_frame, text="", style="Status.TLabel")
            self.download_status.pack(side=tk.LEFT, padx=10)
        else:
            # Show refresh button in case user wants to re-check
            self.download_data_btn = None
            self.download_status = None
        
        # Step 2: Data Distribution
        step2 = ttk.LabelFrame(left_col, text="  2. Sampling Strategy  ", padding="10", style="Section.TLabelframe")
        step2.pack(fill=tk.X, pady=5)
        
        dist_frame = ttk.Frame(step2)
        dist_frame.pack(fill=tk.X)
        
        ttk.Radiobutton(dist_frame, text="Even ★", variable=self.distribution_var, 
                        value="even", command=self.update_distribution_ui).pack(anchor=tk.W)
        self.even_desc = ttk.Label(dist_frame, text="Same samples per class (max 500)", style="Rec.TLabel")
        self.even_desc.pack(anchor=tk.W, padx=(20, 0))
        
        self.samples_frame = ttk.Frame(dist_frame)
        self.samples_frame.pack(fill=tk.X, padx=(20, 0), pady=3)
        ttk.Label(self.samples_frame, text="Samples/class:").pack(side=tk.LEFT)
        self.samples_spin = ttk.Spinbox(self.samples_frame, from_=10, to=500, 
                                         textvariable=self.samples_var, width=8)
        self.samples_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(dist_frame, text="Uneven", variable=self.distribution_var, 
                        value="uneven", command=self.update_distribution_ui).pack(anchor=tk.W, pady=(5, 0))
        self.uneven_desc = ttk.Label(dist_frame, text="Random 20 to MAX per class (realistic)", style="Rec.TLabel")
        self.uneven_desc.pack(anchor=tk.W, padx=(20, 0))
        
        ttk.Radiobutton(dist_frame, text="All data", variable=self.distribution_var, 
                        value="all", command=self.update_distribution_ui).pack(anchor=tk.W, pady=(5, 0))
        
        # Step 3: ViT Model
        step3 = ttk.LabelFrame(left_col, text="  3. ViT Model  ", padding="10", style="Section.TLabelframe")
        step3.pack(fill=tk.X, pady=5)
        
        for key, name, rec in VIT_MODELS:
            frame = ttk.Frame(step3)
            frame.pack(fill=tk.X, pady=1)
            
            label_text = name
            if rec:
                label_text += f"  {rec}"
            
            rb = ttk.Radiobutton(frame, text=label_text, variable=self.model_var, value=key,
                                command=self.update_embedding_status)
            rb.pack(side=tk.LEFT)
        
        self.embedding_status = ttk.Label(step3, text="", style="Good.TLabel")
        self.embedding_status.pack(anchor=tk.W, pady=(5, 0))
        
        # === RIGHT COLUMN WIDGETS ===
        
        # Step 4: Dimension Reduction
        step4 = ttk.LabelFrame(right_col, text="  4. Dimension Reduction  ", padding="10", style="Section.TLabelframe")
        step4.pack(fill=tk.X, pady=5)
        
        for key, name, desc in REDUCTION_METHODS:
            var = tk.BooleanVar(value=(key == "tsne"))
            self.reduction_vars[key] = var
            
            frame = ttk.Frame(step4)
            frame.pack(fill=tk.X, pady=1)
            
            cb = ttk.Checkbutton(frame, text=f"{name}", variable=var)
            cb.pack(side=tk.LEFT)
            ttk.Label(frame, text=f"  {desc}", style="Rec.TLabel").pack(side=tk.LEFT)
        
        # Step 5: Clustering
        step5 = ttk.LabelFrame(right_col, text="  5. Clustering Method  ", padding="10", style="Section.TLabelframe")
        step5.pack(fill=tk.X, pady=5)
        
        self.clustering_frames = {}
        for key, name, requires_k, recs in CLUSTERING_METHODS:
            frame = ttk.Frame(step5)
            frame.pack(fill=tk.X, pady=2)
            
            rb = ttk.Radiobutton(frame, text=name, variable=self.clustering_var, value=key,
                                command=self.update_clustering_params)
            rb.pack(side=tk.LEFT)
            
            rec_label = ttk.Label(frame, text="", style="Rec.TLabel")
            rec_label.pack(side=tk.LEFT, padx=(5, 0))
            self.clustering_frames[key] = rec_label
        
        # HDBSCAN params
        self.hdbscan_frame = ttk.Frame(step5)
        self.hdbscan_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        ttk.Label(self.hdbscan_frame, text="Preset:").pack(side=tk.LEFT)
        preset_values = [f"{k.capitalize()} ({v['min_cluster_size']},{v['min_samples']}) - {v['desc']}" 
                        for k, v in HDBSCAN_PRESETS.items()]
        self.hdbscan_combo = ttk.Combobox(self.hdbscan_frame, textvariable=self.hdbscan_preset_var,
                                          values=list(HDBSCAN_PRESETS.keys()), width=8, state="readonly")
        self.hdbscan_combo.set("small")
        self.hdbscan_combo.pack(side=tk.LEFT, padx=5)
        
        self.hdbscan_desc = ttk.Label(self.hdbscan_frame, text="(15,5) - Even data", style="Rec.TLabel")
        self.hdbscan_desc.pack(side=tk.LEFT)
        self.hdbscan_combo.bind("<<ComboboxSelected>>", self.update_hdbscan_desc)
        
        # K clusters frame
        self.k_frame = ttk.Frame(step5)
        
        ttk.Label(self.k_frame, text="K (clusters):").pack(side=tk.LEFT)
        self.k_spin = ttk.Spinbox(self.k_frame, from_=2, to=200, textvariable=self.n_clusters_var, width=6)
        self.k_spin.pack(side=tk.LEFT, padx=5)
        
        self.k_hint = ttk.Label(self.k_frame, text="", style="Rec.TLabel")
        self.k_hint.pack(side=tk.LEFT)
        
        # Step 6: Output
        step6 = ttk.LabelFrame(right_col, text="  6. Output Folder  ", padding="10", style="Section.TLabelframe")
        step6.pack(fill=tk.X, pady=5)
        
        output_frame = ttk.Frame(step6)
        output_frame.pack(fill=tk.X)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_var, width=40)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(output_frame, text="Browse", command=self.browse_output, width=8).pack(side=tk.LEFT, padx=5)
        
        # Step 7: Save Cluster Images
        step7 = ttk.LabelFrame(right_col, text="  7. Save Cluster Images  ", padding="10", style="Section.TLabelframe")
        step7.pack(fill=tk.X, pady=5)
        
        save_frame = ttk.Frame(step7)
        save_frame.pack(fill=tk.X)
        
        self.save_images_check = ttk.Checkbutton(
            save_frame, 
            text="Copy images into cluster folders", 
            variable=self.save_images_var
        )
        self.save_images_check.pack(anchor=tk.W)
        
        save_desc = ttk.Label(
            save_frame, 
            text="Creates cluster_1/, cluster_2/, ... and rejected/ folders with image copies",
            style="Rec.TLabel"
        )
        save_desc.pack(anchor=tk.W, padx=(20, 0))
        
        # === LOG TAB ===
        self.log_text = scrolledtext.ScrolledText(log_tab, height=25, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # === BOTTOM BAR ===
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.run_button = ttk.Button(bottom_frame, text="▶  Run Pipeline", style="Run.TButton",
                                      command=self.run_pipeline_thread)
        self.run_button.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(bottom_frame, text="Ready", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.progress = ttk.Progressbar(bottom_frame, mode="indeterminate", length=200)
        self.progress.pack(side=tk.RIGHT)
    
    def _build_data_choices(self) -> List[str]:
        """Build data selection choices."""
        choices = []
        default = None
        
        if self.available_data["aves"]:
            d = self.available_data["aves"]
            choices.append(f"Aves (Birds) — {d['n_classes']} species, {d['n_images']:,} images")
        
        if self.available_data["mammals"]:
            d = self.available_data["mammals"]
            choices.append(f"Mammals — {d['n_classes']} species, {d['n_images']:,} images")
        
        if self.available_data["aves"] and self.available_data["mammals"]:
            total_cls = self.available_data["aves"]["n_classes"] + self.available_data["mammals"]["n_classes"]
            total_img = self.available_data["aves"]["n_images"] + self.available_data["mammals"]["n_images"]
            choices.append(f"Both ★ — {total_cls} species, {total_img:,} images")
            default = choices[-1]
        
        if self.available_data["custom"]:
            d = self.available_data["custom"]
            choices.append(f"Custom — {d['n_classes']} folders, {d['n_images']:,} images")
        
        if not choices:
            choices = ["No data available — run download first"]
        
        self.data_var.set(default or choices[0])
        return choices
    
    def update_distribution_ui(self):
        """Update UI based on distribution selection and update HDBSCAN preset."""
        dist = self.distribution_var.get()
        if dist == "even":
            self.samples_frame.pack(fill=tk.X, padx=(20, 0), pady=3)
            # Recommend small preset for even distribution
            self.hdbscan_preset_var.set("small")
            self.update_hdbscan_desc()
        elif dist == "uneven":
            self.samples_frame.pack_forget()
            # Recommend medium preset for uneven distribution
            self.hdbscan_preset_var.set("medium")
            self.update_hdbscan_desc()
        else:  # all
            self.samples_frame.pack_forget()
            # Recommend large preset for all data
            self.hdbscan_preset_var.set("large")
            self.update_hdbscan_desc()
    
    def update_hdbscan_desc(self, event=None):
        """Update HDBSCAN preset description."""
        preset = self.hdbscan_preset_var.get()
        if preset in HDBSCAN_PRESETS:
            p = HDBSCAN_PRESETS[preset]
            dist = self.distribution_var.get()
            # Determine if this is the recommended preset for current distribution
            recommended_preset = {"even": "small", "uneven": "medium", "all": "large"}.get(dist, "small")
            rec_marker = " ★ RECOMMENDED" if preset == recommended_preset else ""
            self.hdbscan_desc.config(text=f"({p['min_cluster_size']},{p['min_samples']}) - {p['desc']}{rec_marker}")
    
    def update_recommendations(self):
        """Update all recommendations based on selected data."""
        split = self.get_selected_split()
        
        # Update clustering recommendations
        for key, name, requires_k, recs in CLUSTERING_METHODS:
            if split in recs:
                rec, desc = recs[split]
                self.clustering_frames[key].config(text=f"{rec} {desc}")
            else:
                self.clustering_frames[key].config(text="")
        
        # Update K hint
        n_classes = self._get_n_classes()
        self.k_hint.config(text=f"(True K ≈ {n_classes})")
        self.n_clusters_var.set(n_classes)
        
        # Update embedding status
        self.update_embedding_status()
    
    def update_clustering_params(self):
        """Show/hide clustering parameters."""
        method = self.clustering_var.get()
        
        if method == "hdbscan":
            self.hdbscan_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
            self.k_frame.pack_forget()
        elif method in ["hierarchical", "gmm"]:
            self.hdbscan_frame.pack_forget()
            self.k_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        else:
            self.hdbscan_frame.pack_forget()
            self.k_frame.pack_forget()
    
    def update_embedding_status(self):
        """Update embedding status indicator."""
        split = self.get_selected_split()
        model = self.model_var.get()
        
        possible_files = [
            EMBEDDINGS_DIR / f"{split}_{model}_embeddings.pkl",
            EMBEDDINGS_DIR / f"{model}_embeddings.pkl",
            EMBEDDINGS_DIR / f"mammals_{model}_embeddings.pkl" if split in ["mammals", "both"] else None,
        ]
        
        found_emb = None
        for f in possible_files:
            if f and f.exists():
                found_emb = f
                break
        
        if found_emb:
            self.embedding_status.config(
                text=f"✓ Found: {found_emb.name}",
                style="Good.TLabel"
            )
        else:
            self.embedding_status.config(
                text="⚠ Will compute embeddings (takes several minutes)",
                style="Warning.TLabel"
            )
    
    def get_selected_split(self) -> str:
        """Get selected data split name."""
        choice = self.data_var.get()
        if choice.startswith("Aves"):
            return "aves"
        elif choice.startswith("Mammals"):
            return "mammals"
        elif choice.startswith("Both"):
            return "both"
        elif choice.startswith("Custom"):
            return "custom"
        return ""
    
    def _get_n_classes(self) -> int:
        """Get number of classes for selected data."""
        split = self.get_selected_split()
        if split == "aves" and self.available_data["aves"]:
            return self.available_data["aves"]["n_classes"]
        elif split == "mammals" and self.available_data["mammals"]:
            return self.available_data["mammals"]["n_classes"]
        elif split == "both":
            n = 0
            if self.available_data["aves"]:
                n += self.available_data["aves"]["n_classes"]
            if self.available_data["mammals"]:
                n += self.available_data["mammals"]["n_classes"]
            return n
        elif split == "custom" and self.available_data["custom"]:
            return self.available_data["custom"]["n_classes"]
        return 30
    
    def get_selected_paths(self) -> List[Path]:
        """Get selected data paths."""
        choice = self.data_var.get()
        paths = []
        
        if choice.startswith("Aves") and self.available_data["aves"]:
            paths = [self.available_data["aves"]["path"]]
        elif choice.startswith("Mammals") and self.available_data["mammals"]:
            paths = [self.available_data["mammals"]["path"]]
        elif choice.startswith("Both"):
            if self.available_data["aves"]:
                paths.append(self.available_data["aves"]["path"])
            if self.available_data["mammals"]:
                paths.append(self.available_data["mammals"]["path"])
        elif choice.startswith("Custom") and self.available_data["custom"]:
            paths = [self.available_data["custom"]["path"]]
        
        return paths
    
    def browse_output(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(initialdir=self.output_var.get())
        if folder:
            self.output_var.set(folder)
    
    def log(self, message: str):
        """Add message to log."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def download_all_thread(self):
        """Download dataset and embeddings in a separate thread."""
        if self.running:
            return
        
        self.running = True
        if self.download_data_btn:
            self.download_data_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Downloading...")
        
        self.notebook.select(1)
        self.log_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.download_all)
        thread.daemon = True
        thread.start()
    
    def download_all(self):
        """Download HUGO-Bench dataset and pre-computed embeddings."""
        try:
            self.log("=" * 60)
            self.log("  DOWNLOADING HUGO-BENCH DATASET & EMBEDDINGS")
            self.log("=" * 60)
            self.log("")
            
            # Download dataset
            self.log("[1/2] Downloading HUGO-Bench dataset...")
            self.log("      Source: https://huggingface.co/datasets/AI-EcoNet/HUGO-Bench")
            self.log("")
            
            try:
                from scripts.download_dataset import download_dataset
                success = download_dataset(
                    output_dir=TEST_DATA_DIR,
                    split=None,  # Both aves and mammals
                    skip_confirmation=True
                )
                if success:
                    self.log("[OK] Dataset downloaded successfully!")
                else:
                    self.log("[!] Dataset download had issues - check log above")
            except Exception as e:
                self.log(f"[X] Dataset download failed: {e}")
            
            self.log("")
            
            # Download embeddings for all models
            self.log("[2/2] Downloading pre-computed embeddings (all models)...")
            self.log("      Models: DINOv3, DINOv2, BioCLIP 2, BioCLIP 2.5, CLIP, SigLIP")
            self.log("      Splits: Aves, Mammals")
            self.log("      (12 total embedding files)")
            self.log("")
            
            try:
                from scripts.download_embeddings import download_embeddings
                success = download_embeddings(
                    models=None,  # Download all models
                    splits=None,  # Download all splits
                    force=False,
                    quiet=False
                )
                if success:
                    self.log("[OK] All embeddings downloaded successfully!")
                else:
                    self.log("[!] Some embeddings may not have downloaded")
            except Exception as e:
                self.log(f"[X] Embeddings download failed: {e}")
            
            self.log("")
            self.log("=" * 60)
            self.log("  DOWNLOAD COMPLETE - Refreshing data...")
            self.log("=" * 60)
            
            # Refresh available data
            self.available_data = get_available_data()
            
            # Update UI on main thread
            self.root.after(0, self._refresh_after_download)
            
        except Exception as e:
            self.log(f"\n[X] Download error: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.root.after(0, self._finish_download)
    
    def _refresh_after_download(self):
        """Refresh UI after download completes."""
        # Rebuild data choices
        data_choices = self._build_data_choices()
        self.data_combo.config(values=data_choices)
        
        # Hide download buttons if we now have data
        has_test_data = self.available_data["aves"] or self.available_data["mammals"]
        if has_test_data:
            # Clear download frame content
            for widget in self.download_frame.winfo_children():
                widget.destroy()
            self.log("\n[OK] Data is now available! Switching to Configuration tab...")
            # Switch back to config tab
            self.notebook.select(0)
            # Update embedding status for selected model
            self.update_embedding_status()
    
    def _finish_download(self):
        """Reset UI state after download."""
        self.running = False
        if self.download_data_btn:
            self.download_data_btn.config(state=tk.NORMAL)
        self.progress.stop()
        self.status_label.config(text="Download complete")
    
    def run_pipeline_thread(self):
        """Run pipeline in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Running...")
        
        self.notebook.select(1)
        self.log_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.run_pipeline)
        thread.daemon = True
        thread.start()
    
    def run_pipeline(self):
        """Run the complete clustering pipeline."""
        try:
            split_name = self.get_selected_split()
            data_paths = self.get_selected_paths()
            
            if not split_name or not data_paths:
                self.log("[X] No valid data selected!")
                self.finish_pipeline()
                return
            
            model = self.model_var.get()
            distribution = self.distribution_var.get()
            samples_per_class = self.samples_var.get()
            reduction_methods = [k for k, v in self.reduction_vars.items() if v.get()]
            clustering_method = self.clustering_var.get()
            output_path = Path(self.output_var.get())
            
            if not reduction_methods:
                reduction_methods = ["tsne"]
            
            self.log("=" * 55)
            self.log("  ZERO-SHOT CLUSTERING PIPELINE")
            self.log("=" * 55)
            self.log(f"  Data:         {split_name.upper()}")
            self.log(f"  Distribution: {distribution}" + (f" ({samples_per_class}/class)" if distribution == "even" else " (20-MAX)"))
            self.log(f"  Model:        {model.upper()}")
            self.log(f"  Reduction:    {', '.join(r.upper() for r in reduction_methods)}")
            self.log(f"  Clustering:   {clustering_method.upper()}")
            self.log("=" * 55)
            self.log("")
            
            # Import modules
            self.log("[1/5] Loading modules...")
            
            from scripts.extract_embeddings import (
                extract_and_save as extract_embeddings_func,
                load_embeddings
            )
            from scripts.dimension_reduction import DimensionReducer
            from scripts.clustering import run_clustering, save_cluster_images
            from scripts.visualization import create_cluster_plot
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            output_path.mkdir(parents=True, exist_ok=True)
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Check for existing embeddings
            self.log("\n[2/5] Checking embeddings...")
            
            # For "both" split, we need to combine aves + mammals embeddings
            use_combined = False
            aves_emb_file = None
            mammals_emb_file = None
            
            if split_name == "both":
                aves_emb_file = EMBEDDINGS_DIR / f"aves_{model}_embeddings.pkl"
                mammals_emb_file = EMBEDDINGS_DIR / f"mammals_{model}_embeddings.pkl"
                
                # Try downloading missing embeddings from HuggingFace
                if not aves_emb_file.exists():
                    self.log(f"      ⚙ Aves embeddings not found locally...")
                    downloaded = try_download_embeddings(model, "aves", self.log)
                    if downloaded:
                        aves_emb_file = downloaded
                
                if not mammals_emb_file.exists():
                    self.log(f"      ⚙ Mammals embeddings not found locally...")
                    downloaded = try_download_embeddings(model, "mammals", self.log)
                    if downloaded:
                        mammals_emb_file = downloaded
                
                if aves_emb_file.exists() and mammals_emb_file.exists():
                    self.log(f"      ✓ Found: {aves_emb_file.name}")
                    self.log(f"      ✓ Found: {mammals_emb_file.name}")
                    self.log(f"      → Will combine aves + mammals for analysis")
                    use_combined = True
                else:
                    # Check for a combined file
                    emb_file = EMBEDDINGS_DIR / f"both_{model}_embeddings.pkl"
                    if emb_file.exists():
                        self.log(f"      ✓ Using existing: {emb_file.name}")
                    else:
                        missing = []
                        if not aves_emb_file.exists():
                            missing.append("aves")
                        if not mammals_emb_file.exists():
                            missing.append("mammals")
                        self.log(f"      ✗ Missing embeddings for: {', '.join(missing)}")
                        self.log(f"        Need: {aves_emb_file.name} + {mammals_emb_file.name}")
                        self.log(f"        Run: python scripts/download_embeddings.py")
                        self.log(f"        Or run embedding extraction first (select single split)")
                        self.finish_pipeline()
                        return
            else:
                # Single split - aves, mammals, or custom
                emb_file = EMBEDDINGS_DIR / f"{split_name}_{model}_embeddings.pkl"
                
                alt_emb_files = [
                    EMBEDDINGS_DIR / f"{model}_embeddings.pkl",
                ]
                
                existing_emb = None
                if emb_file.exists():
                    existing_emb = emb_file
                else:
                    for alt in alt_emb_files:
                        if alt and alt.exists():
                            existing_emb = alt
                            break
                
                if existing_emb:
                    self.log(f"      ✓ Using existing: {existing_emb.name}")
                    emb_file = existing_emb
                else:
                    # Try downloading from HuggingFace (only for aves/mammals, not custom)
                    if split_name in ["aves", "mammals"]:
                        self.log(f"      ⚙ Embeddings not found locally...")
                        downloaded = try_download_embeddings(model, split_name, self.log)
                        if downloaded:
                            emb_file = downloaded
                            existing_emb = emb_file
                            self.log(f"      ✓ Using downloaded: {emb_file.name}")
                
                if not existing_emb:
                    # Need to compute embeddings locally
                    self.log(f"      ⚙ Extracting {model.upper()} embeddings...")
                    self.log("        (This may take several minutes...)")
                    
                    success = extract_embeddings_func(
                        model_name=model,
                        data_path=data_paths[0].parent if len(data_paths) > 1 else data_paths[0],
                        output_path=emb_file,
                        limit_per_class=samples_per_class if distribution == "even" else None,
                        random_seed=42
                    )
                    
                    if not success or not emb_file.exists():
                        self.log("      ✗ Failed to extract embeddings")
                        self.finish_pipeline()
                        return
                
                    self.log("      ✓ Embeddings saved")
            
            # Load embeddings
            self.log("\n[3/5] Loading and sampling data...")
            
            if use_combined:
                # Combine aves + mammals embeddings
                self.log(f"      Combining aves + mammals embeddings...")
                embeddings, labels, paths = combine_embeddings(
                    [aves_emb_file, mammals_emb_file],
                    load_embeddings
                )
                self.log(f"      Combined: {len(labels):,} samples from {len(set(labels))} classes")
            else:
                emb_data = load_embeddings(emb_file)
                embeddings = emb_data.embeddings
                labels = list(emb_data.labels)
                paths = list(emb_data.paths)
                self.log(f"      Loaded: {len(labels):,} samples")
            
            # Apply sampling
            embeddings, labels, paths = sample_data(
                embeddings, labels, paths,
                distribution, samples_per_class
            )
            
            n_samples = len(labels)
            n_classes = len(set(labels))
            self.log(f"      After sampling: {n_samples:,} samples, {n_classes} classes")
            
            # Dimension reduction & clustering
            self.log("\n[4/5] Running analysis...")
            reducer = DimensionReducer(n_components=2, random_state=42)
            
            for red_method in reduction_methods:
                self.log(f"\n      → {red_method.upper()} reduction...")
                
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
                    continue
                
                embeddings_2d = red_result.embeddings
                self.log(f"        Done in {red_result.time_seconds:.1f}s")
                
                # Clustering
                self.log(f"      → {clustering_method.upper()} clustering...")
                
                cluster_params = {}
                if clustering_method == "hdbscan":
                    preset = HDBSCAN_PRESETS.get(self.hdbscan_preset_var.get(), HDBSCAN_PRESETS["small"])
                    cluster_params = {
                        "min_cluster_size": preset["min_cluster_size"],
                        "min_samples": preset["min_samples"]
                    }
                elif clustering_method in ["hierarchical", "gmm"]:
                    cluster_params = {"n_clusters": self.n_clusters_var.get()}
                
                try:
                    result = run_clustering(embeddings_2d, labels, clustering_method, **cluster_params)
                except Exception as e:
                    self.log(f"        ✗ Error: {e}")
                    continue
                
                self.log(f"        Clusters: {result.n_clusters}")
                self.log(f"        Outliers: {result.n_outliers}")
                
                # Create visualization & save
                self.log("\n[5/5] Saving results...")
                
                title = f"{split_name.upper()} | {model.upper()} + {red_method.upper()} + {clustering_method.upper()}\n"
                title += f"V-measure: {result.metrics.get('v_measure', 0):.3f}, Clusters={result.n_clusters}, N={n_samples}"
                
                fig = create_cluster_plot(
                    embeddings_2d=embeddings_2d,
                    cluster_labels=result.labels,
                    labels_true=labels,
                    title=title,
                    show_legend=result.n_clusters <= 30
                )
                
                run_num = get_next_run_number()
                
                param_suffix = ""
                if clustering_method == "hdbscan":
                    param_suffix = f"_{cluster_params['min_cluster_size']}_{cluster_params['min_samples']}"
                elif clustering_method in ["hierarchical", "gmm"]:
                    param_suffix = f"_k{cluster_params['n_clusters']}"
                
                folder_name = f"run_{run_num}_{model}_{red_method}_{clustering_method}{param_suffix}"
                run_dir = output_path / folder_name
                run_dir.mkdir(parents=True, exist_ok=True)
                
                fig.savefig(run_dir / "cluster_plot.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                results_data = {
                    "split": split_name,
                    "model": model,
                    "reduction": red_method,
                    "clustering": clustering_method,
                    "params": cluster_params,
                    "n_samples": n_samples,
                    "n_classes": n_classes,
                    "n_clusters": result.n_clusters,
                    "metrics": result.metrics,
                }
                
                with open(run_dir / "results.pkl", 'wb') as f:
                    pickle.dump(results_data, f)
                
                with open(run_dir / "summary.txt", 'w') as f:
                    f.write(f"Run: {folder_name}\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Clusters: {result.n_clusters}\n")
                    f.write(f"Samples: {n_samples}\n")
                    f.write(f"Classes: {n_classes}\n")
                    f.write(f"Outliers: {result.n_outliers}\n")
                
                # Save cluster images if requested
                if self.save_images_var.get():
                    self.log("      Copying images to cluster folders...")
                    saved = save_cluster_images(
                        cluster_labels=result.labels,
                        image_paths=paths,
                        output_dir=run_dir,
                        copy=True
                    )
                    n_clusters_saved = len([k for k in saved.keys() if k != -1])
                    n_rejected = len(saved.get(-1, []))
                    self.log(f"      ✓ Saved {n_clusters_saved} cluster folders + {n_rejected} rejected")
                
                self.log(f"      ✓ {folder_name}/")
            
            self.log("\n" + "=" * 55)
            self.log("  ✓ PIPELINE COMPLETE!")
            self.log("=" * 55)
            self.log(f"\n  Results: {output_path}")
            
            self.root.after(100, lambda: self.show_complete_dialog(str(output_path)))
            
        except Exception as e:
            import traceback
            self.log(f"\n[X] ERROR: {str(e)}")
            self.log(traceback.format_exc())
        
        finally:
            self.root.after(0, self.finish_pipeline)
    
    def show_complete_dialog(self, output_path: str):
        """Show completion dialog."""
        if messagebox.askyesno("Complete", "Pipeline finished!\n\nOpen results folder?"):
            try:
                import subprocess
                p = str(output_path)
                if sys.platform.startswith("win"):
                    os.startfile(p)
                elif sys.platform == "darwin":
                    subprocess.run(["open", p], check=True)
                else:
                    subprocess.run(["xdg-open", p], check=True)
            except Exception as e:
                messagebox.showinfo("Open Folder", f"Could not open folder: {e}")
    
    def finish_pipeline(self):
        """Clean up after pipeline finishes."""
        self.running = False
        self.progress.stop()
        self.run_button.config(state=tk.NORMAL)
        self.status_label.config(text="Ready")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Launch the GUI."""
    root = tk.Tk()
    app = ClusteringApp(root)
    
    app.model_var.trace_add("write", lambda *_: app.update_embedding_status())
    app.data_var.trace_add("write", lambda *_: app.update_recommendations())
    
    root.mainloop()


if __name__ == "__main__":
    main()
