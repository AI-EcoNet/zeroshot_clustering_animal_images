# Zero-Shot Clustering for Animal Images

A pipeline for unsupervised clustering of wildlife images using Vision Transformer (ViT) embeddings. Uses pre-trained models to extract features from images and clusters them without requiring labeled training data.

## Paper & Resources

This implementation is based on the paper:

> **Vision Transformers for Zero-Shot Clustering of Animal Images: A Comparative Benchmarking Study**  
> Hugo Markoff, Stefan Hein Bengtson, Michael Ørsted  
> *Aalborg University, Denmark*  
> 
> **[arXiv Preprint](https://arxiv.org/abs/2602.03894)** | **[Interactive Demo](https://hugomarkoff.github.io/animal_visual_transformer/)**

The paper presents a comprehensive benchmarking framework evaluating 5 ViT models × 5 dimensionality reduction approaches × 4 clustering algorithms across 60 species (30 mammals and 30 birds). Key findings include near-perfect species-level clustering (V-measure: 0.958) using DINOv3 embeddings with t-SNE and supervised hierarchical clustering, while unsupervised HDBSCAN achieves competitive performance (V-measure: 0.943) with only 1.14% outlier rejection.

## Features

- **Multiple ViT Models**: DINOv3 (best), DINOv2, BioCLIP 2, BioCLIP 2.5, CLIP, SigLIP
- **Dimension Reduction**: t-SNE (recommended), UMAP, PCA, Isomap, Kernel PCA
- **Clustering Methods**: HDBSCAN (recommended), DBSCAN, Hierarchical, GMM
- **Custom Data Support**: Works with your own image folders
- **Cluster Image Export**: Save images organized by cluster (cluster_1/, cluster_2/, rejected/)
- **GUI Interface**: Native Tkinter-based desktop application

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python gui.py

# Or run the interactive CLI
python main.py
```

The GUI will guide you through:
1. Data selection (test data or custom data)
2. Data sampling configuration  
3. Model selection
4. Dimension reduction method
5. Clustering configuration
6. Output folder
7. **Save cluster images** (optional - copies images into cluster folders)

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### DINOv3 Access (Optional)
DINOv3 requires HuggingFace authentication:
1. Create an account at [huggingface.co](https://huggingface.co)
2. Request access to [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
3. Login: `huggingface-cli login`

## Using Custom Data

### Folder Structure
Place your images in subfolders within `custom_data/`, where each subfolder represents a species or category:

```
custom_data/
├── species_a/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── species_b/
│   ├── image1.jpg
│   └── ...
└── species_c/
    └── ...
```

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

### Running with Custom Data
```bash
python main.py
# Select "Custom" when prompted for data selection
```

## Command Line Options

```bash
# Interactive mode (default)
python main.py

# Quick test with default settings
python main.py --quick

# Full pipeline: download test data + run all models
python main.py --full

# Specify data path and model
python main.py --data-path custom_data/MyImages --models dinov3

# Use test data with specific configuration
python main.py --use-test-data --models dinov2 --clustering hdbscan --hdbscan-preset small
```

### Available Arguments
| Argument | Description |
|----------|-------------|
| `--data-path PATH` | Path to custom data folder |
| `--use-test-data` | Use test data instead of custom data |
| `--full` | Run full pipeline with all models |
| `--quick` | Quick test with DINOv3 + t-SNE + HDBSCAN |
| `--models MODEL [MODEL ...]` | Models to use (dinov3, dinov2, bioclip2, bioclip2_5, clip, siglip, all) |
| `--reductions METHOD [METHOD ...]` | Reduction methods (tsne, umap, pca, isomap, kernel_pca, all) |
| `--clustering METHOD` | Clustering method (hdbscan, dbscan, hierarchical, gmm) |
| `--hdbscan-preset PRESET` | HDBSCAN preset (small, medium, large) |
| `--n-classes N` | Limit number of classes |
| `--limit-per-class N` | Limit samples per class |
| `--seed N` | Random seed (default: 42) |

## Output Structure

Results are saved in the `results/` folder:

```
results/
└── run_1_dinov3_tsne_hdbscan_15_5/
    ├── cluster_plot.png    # Visualization
    ├── results.pkl         # Full results data
    ├── summary.txt         # Human-readable summary
    └── clusters/           # (Optional) Images by cluster
        ├── cluster_1/
        ├── cluster_2/
        ├── ...
        └── rejected/       # Outliers
```

## Pre-computed Embeddings

For the test dataset, pre-computed embeddings are available:

```bash
# List available embeddings
python scripts/download_embeddings.py --list

# Download all embeddings
python scripts/download_embeddings.py

# Download specific model
python scripts/download_embeddings.py --model dinov3
```

## BioCLIP 2.5 Leaderboard Snapshot

Latest BioCLIP 2.5 benchmark artifacts are published in the Hugging Face dataset:
- `precomputed_embeddings/aves_bioclip2_5_embeddings.pkl`
- `precomputed_embeddings/mammals_bioclip2_5_embeddings.pkl`
- `bioclip2_5_benchmark/benchmark_results_20260326_185250.csv`
- `bioclip2_5_benchmark/benchmark_results_20260326_185250.json`
- `bioclip2_5_benchmark/summary_20260326_185250.md`
- `bioclip2_5_benchmark/CURRENT_LEADERBOARDS.md`

Score definitions used in current leaderboards:
- Official score: `split_balanced_v_mean = (Aves mean V + Mammalia mean V) / 2`
- BioCLIP 2.5 local score: overall V-measure on full Aves+Mammals run
- Top models (mean): arithmetic mean of each model's config-level score

### Combined Top Configurations (Official + BioCLIP 2.5)

| Rank | Source | Model | Reduction | Clustering | Score |
|------|--------|-------|-----------|------------|-------|
| 1 | official_hf | dinov3_vith16plus | TSNE | HIERARCHICAL_K30 | 0.9576 |
| 2 | official_hf | dinov3_vith16plus | TSNE | GMM_K30 | 0.9559 |
| 3 | official_hf | dinov3_vith16plus | UMAP | HIERARCHICAL_K30 | 0.9545 |
| 4 | official_hf | dinov3_vith16plus | UMAP | GMM_K30 | 0.9528 |
| 5 | official_hf | dinov3_vith16plus | TSNE | HDBSCAN | 0.9435 |
| 11 | local_bioclip2_5_full | bioclip2_5_vith14 | TSNE | HDBSCAN | 0.8822 |
| 19 | local_bioclip2_5_full | bioclip2_5_vith14 | UMAP | HDBSCAN | 0.8590 |

### Top Models (Mean Score)

| Rank | Model | Mean Score | #Configs |
|------|-------|------------|----------|
| 1 | dinov3_vith16plus | 0.6759 | 60 |
| 2 | dinov2_vitg14 | 0.6479 | 60 |
| 3 | bioclip2_5_vith14 | 0.5969 | 9 |
| 4 | bioclip2_vitl14 | 0.4849 | 60 |
| 5 | clip_vitl14 | 0.4312 | 60 |
| 6 | siglip_vitb16 | 0.4309 | 60 |

## Model Recommendations

| Model | Embed Dim | Best For |
|-------|-----------|----------|
| **DINOv3** ⭐ | 1280D | Best overall performance |
| **DINOv2** | 1536D | Great alternative, no auth required |
| BioCLIP 2 | 768D | Biology-specific features |
| BioCLIP 2.5 | 768D | Strong large-model biology baseline |
| CLIP | 768D | General purpose |
| SigLIP | 768D | Efficient, smaller footprint |

## Clustering Presets

### HDBSCAN Presets
| Preset | min_cluster_size | min_samples | Use Case |
|--------|------------------|-------------|----------|
| **small** | 15 | 5 | Even distribution, <300 samples/class |
| **medium** | 100 | 30 | Mixed representation |
| **large** | 150 | 50 | Large/uneven data |

## Project Structure

```
├── main.py                 # Main entry point
├── ui_pipeline.py          # Interactive CLI pipeline
├── requirements.txt        # Python dependencies
├── custom_data/            # Your images here
├── test_data/              # Test dataset (Aves/Mammals)
├── outputs/                # Embeddings, reductions, clustering
├── results/                # Final run results
└── scripts/
    ├── extract_embeddings.py    # ViT feature extraction
    ├── dimension_reduction.py   # t-SNE, UMAP, etc.
    ├── clustering.py            # Clustering algorithms
    ├── visualization.py         # Plotting utilities
    ├── download_dataset.py      # Download test data
    └── download_embeddings.py   # Download pre-computed embeddings
```

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{markoff2026vision,
  title={Vision Transformers for Zero-Shot Clustering of Animal Images: A Comparative Benchmarking Study},
  author={Markoff, Hugo and Bengtson, Stefan Hein and {\O}rsted, Michael},
  journal={arXiv preprint arXiv:2602.03894},
  year={2026}
}
```

## License

MIT License
