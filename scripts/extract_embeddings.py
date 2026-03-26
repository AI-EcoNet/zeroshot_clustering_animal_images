"""
Extract embeddings from images using Vision Transformer models.

Supported models:
- DINOv2 (ViT-g/14) - 1536D embeddings
- DINOv3 (ViT-H+/16) - 1280D embeddings (requires HuggingFace auth)
- CLIP (ViT-L/14) - 768D embeddings
- SigLIP (ViT-B/16) - 768D embeddings
- BioCLIP 2 (ViT-L/14) - 768D embeddings
- BioCLIP 2.5 (ViT-H/14) - 768D embeddings
"""

from __future__ import annotations

import os
import sys
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Available models with their configurations
AVAILABLE_MODELS = {
    "dinov2": {
        "full_name": "dinov2_vitg14",
        "description": "DINOv2 ViT-g/14 (1.1B params, 1536D)",
        "embedding_dim": 1536,
        "requires_auth": False
    },
    "dinov3": {
        "full_name": "dinov3_vith16plus",
        "description": "DINOv3 ViT-H+/16 (840M params, 1280D)",
        "embedding_dim": 1280,
        "requires_auth": True
    },
    "clip": {
        "full_name": "clip_vitl14",
        "description": "CLIP ViT-L/14 (304M params, 768D)",
        "embedding_dim": 768,
        "requires_auth": False
    },
    "siglip": {
        "full_name": "siglip_vitb16",
        "description": "SigLIP ViT-B/16 (200M params, 768D)",
        "embedding_dim": 768,
        "requires_auth": False
    },
    "bioclip2": {
        "full_name": "bioclip2_vitl14",
        "description": "BioCLIP 2 ViT-L/14 (304M params, 768D)",
        "embedding_dim": 768,
        "requires_auth": False
    },
    "bioclip2_5": {
        "full_name": "bioclip2_5_vith14",
        "description": "BioCLIP 2.5 ViT-H/14 (1B params, 768D)",
        "embedding_dim": 768,
        "requires_auth": False
    }
}


@dataclass(frozen=True)
class EmbeddingRecord:
    """Container holding a single model's embedding payload."""
    embeddings: np.ndarray
    labels: List[str]
    paths: List[str]

    @property
    def n_samples(self) -> int:
        return len(self.embeddings)

    @property
    def n_features(self) -> int:
        return self.embeddings.shape[1] if self.embeddings.ndim == 2 else 0

    @property
    def n_classes(self) -> int:
        return len(set(self.labels))


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DINOV3_MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"


def check_huggingface_login() -> Tuple[bool, Optional[str]]:
    """
    Check if user is logged into HuggingFace.
    Returns (is_logged_in, username_or_none).
    """
    # Try multiple methods to find a token
    token = None
    
    # Method 1: HfFolder (standard location)
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    except Exception:
        pass
    
    # Method 2: Environment variables
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Method 3: Try huggingface_hub's built-in token getter
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass
    
    # Method 4: Check token file directly
    if not token:
        try:
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                token = token_path.read_text().strip()
        except Exception:
            pass
    
    if token:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user_info = api.whoami(token=token)
            return True, user_info.get("name", user_info.get("fullname", "User"))
        except Exception:
            # Token exists but might be invalid - still return True to try
            return True, "User"
    
    return False, None


def try_load_dinov3_model():
    """
    Try to load DINOv3 model directly. Returns (success, model, processor, error_msg).
    This is the "try first, ask later" approach.
    """
    try:
        from transformers import AutoModel, AutoImageProcessor
        
        print("    Attempting to load DINOv3 from HuggingFace...")
        model = AutoModel.from_pretrained(DINOV3_MODEL_ID)
        processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
        return True, model, processor, None
    except Exception as e:
        error_str = str(e).lower()
        if "401" in error_str or "403" in error_str or "gated" in error_str:
            return False, None, None, "access_denied"
        elif "token" in error_str or "auth" in error_str:
            return False, None, None, "not_logged_in"
        else:
            return False, None, None, str(e)


def check_dinov3_access() -> Tuple[bool, str]:
    """
    Check if user has access to DINOv3 model.
    Returns (has_access, status_message).
    """
    try:
        from huggingface_hub import HfApi, HfFolder
        token = HfFolder.get_token() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            return False, "Not logged in"
        
        api = HfApi()
        # Try to get model info - this will fail if no access
        try:
            model_info = api.model_info(DINOV3_MODEL_ID, token=token)
            return True, "Access granted"
        except Exception as e:
            error_str = str(e).lower()
            if "401" in error_str or "403" in error_str or "gated" in error_str or "access" in error_str:
                return False, "Access not granted - need to request access"
            elif "404" in error_str:
                return False, "Model not found"
            else:
                # Might still work, let's try
                return True, f"Unknown status, will try anyway"
    except Exception as e:
        return False, f"Error checking access: {e}"


def authenticate_huggingface(retry: bool = True) -> bool:
    """
    Authenticate with HuggingFace for DINOv3 access.
    Shows a nice terminal popup with login status and instructions.
    Returns True if authenticated, False if user chooses to skip.
    """
    # First check if already logged in
    is_logged_in, username = check_huggingface_login()
    
    if is_logged_in:
        # Check if we have access to DINOv3
        has_access, access_status = check_dinov3_access()
        
        if has_access:
            print()
            print("╔" + "═" * 68 + "╗")
            print("║" + " " * 20 + "🤗 HUGGINGFACE STATUS" + " " * 27 + "║")
            print("╠" + "═" * 68 + "╣")
            print("║" + f"  ✓ Logged in as: {username}".ljust(68) + "║")
            print("║" + f"  ✓ DINOv3 model access: GRANTED".ljust(68) + "║")
            print("║" + f"  ✓ Model: {DINOV3_MODEL_ID}".ljust(68) + "║")
            print("╚" + "═" * 68 + "╝")
            print()
            return True
        else:
            # Logged in but no access to model
            print()
            print("╔" + "═" * 68 + "╗")
            print("║" + " " * 15 + "⚠️  DINOV3 ACCESS REQUIRED" + " " * 26 + "║")
            print("╠" + "═" * 68 + "╣")
            print("║" + f"  ✓ Logged in as: {username}".ljust(68) + "║")
            print("║" + f"  ✗ DINOv3 access: {access_status}".ljust(68) + "║")
            print("╠" + "═" * 68 + "╣")
            print("║" + " " * 68 + "║")
            print("║" + "  DINOv3 is a GATED MODEL - you must request access first!".ljust(68) + "║")
            print("║" + " " * 68 + "║")
            print("║" + "  📋 STEP 1: REQUEST ACCESS TO THE MODEL".ljust(68) + "║")
            print("║" + "     Go to the model page and click 'Request Access':".ljust(68) + "║")
            print("║" + f"     👉 https://huggingface.co/{DINOV3_MODEL_ID}".ljust(68) + "║")
            print("║" + " " * 68 + "║")
            print("║" + "  📋 STEP 2: CREATE A FINE-GRAINED TOKEN".ljust(68) + "║")
            print("║" + "     Go to: 👉 https://huggingface.co/settings/tokens".ljust(68) + "║")
            print("║" + "     Click 'Create new token' → Select 'Fine-grained'".ljust(68) + "║")
            print("║" + " " * 68 + "║")
            print("║" + "  📋 STEP 3: SET TOKEN PERMISSIONS".ljust(68) + "║")
            print("║" + "     Under 'Repositories permissions':".ljust(68) + "║")
            print("║" + f"     • Select: {DINOV3_MODEL_ID}".ljust(68) + "║")
            print("║" + "     • Permission: 'Read access to contents'".ljust(68) + "║")
            print("║" + " " * 68 + "║")
            print("║" + "  📋 STEP 4: LOGIN WITH NEW TOKEN".ljust(68) + "║")
            print("║" + "     Run: huggingface-cli login".ljust(68) + "║")
            print("║" + "     Paste your new fine-grained token".ljust(68) + "║")
            print("║" + " " * 68 + "║")
            print("╠" + "═" * 68 + "╣")
            print("║" + "  🎯 OPTIONS:".ljust(68) + "║")
            print("║" + "     [1] I've completed the steps - try again".ljust(68) + "║")
            print("║" + "     [2] Skip DINOv3 (continue with other models)".ljust(68) + "║")
            print("║" + "     [3] Exit and set up access later".ljust(68) + "║")
            print("╚" + "═" * 68 + "╝")
            print()
            
            while True:
                choice = input("  Enter choice [1/2/3]: ").strip()
                
                if choice == "1":
                    # Re-check access
                    has_access, access_status = check_dinov3_access()
                    if has_access:
                        print()
                        print("  ✅ Access granted! Proceeding with DINOv3...")
                        print()
                        return True
                    else:
                        print()
                        print(f"  ❌ Still no access: {access_status}")
                        print("  Please complete all steps above and try again.")
                        print()
                        continue
                        
                elif choice == "2":
                    print()
                    print("  ℹ Skipping DINOv3 - will continue with other models")
                    print()
                    return False
                    
                elif choice == "3":
                    print()
                    sys.exit(0)
                    
                else:
                    print("  ⚠ Invalid choice. Please enter 1, 2, or 3")
            
            return False
    
    # Not logged in - show authentication prompt
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "🔐 HUGGINGFACE AUTHENTICATION" + " " * 23 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + "  ⚠ NOT LOGGED IN - DINOv3 requires HuggingFace authentication".ljust(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + " " * 68 + "║")
    print("║" + "  DINOv3 is a GATED MODEL requiring special access!".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  📋 STEP 1: CREATE HUGGINGFACE ACCOUNT".ljust(68) + "║")
    print("║" + "     👉 https://huggingface.co/join".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  📋 STEP 2: REQUEST ACCESS TO DINOV3 MODEL".ljust(68) + "║")
    print("║" + "     Go to the model page and click 'Request Access':".ljust(68) + "║")
    print("║" + f"     👉 https://huggingface.co/{DINOV3_MODEL_ID}".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  📋 STEP 3: CREATE A FINE-GRAINED ACCESS TOKEN".ljust(68) + "║")
    print("║" + "     👉 https://huggingface.co/settings/tokens".ljust(68) + "║")
    print("║" + "     Click 'Create new token' → Select 'Fine-grained'".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  📋 STEP 4: SET TOKEN PERMISSIONS".ljust(68) + "║")
    print("║" + "     Under 'Repositories permissions':".ljust(68) + "║")
    print("║" + f"     • Select repo: {DINOV3_MODEL_ID}".ljust(68) + "║")
    print("║" + "     • Permission: 'Read access to contents'".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  📋 STEP 5: COPY YOUR TOKEN".ljust(68) + "║")
    print("║" + "     Token format: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".ljust(68) + "║")
    print("║" + " " * 68 + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + "  🎯 OPTIONS:".ljust(68) + "║")
    print("║" + "     [1] Enter your HuggingFace token now".ljust(68) + "║")
    print("║" + "     [2] Skip DINOv3 (continue with other models)".ljust(68) + "║")
    print("║" + "     [3] Exit and set up authentication later".ljust(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    while True:
        choice = input("  Enter choice [1/2/3]: ").strip()
        
        if choice == "1":
            print()
            print("  ┌" + "─" * 50 + "┐")
            print("  │ Paste your token below (it won't be displayed): │")
            print("  └" + "─" * 50 + "┘")
            
            # Try to use getpass for hidden input, fall back to regular input
            try:
                import getpass
                token = getpass.getpass("  Token: ").strip()
            except Exception:
                token = input("  Token: ").strip()
            
            if token:
                try:
                    from huggingface_hub import login
                    login(token=token, add_to_git_credential=False)
                    
                    # Verify login worked
                    is_logged_in, username = check_huggingface_login()
                    if is_logged_in:
                        # Check model access
                        has_access, access_status = check_dinov3_access()
                        print()
                        print("  ╔" + "═" * 58 + "╗")
                        print("  ║" + f"  ✅ Logged in as: {username}".ljust(58) + "║")
                        if has_access:
                            print("  ║" + "  ✅ DINOv3 access: GRANTED".ljust(58) + "║")
                        else:
                            print("  ║" + f"  ⚠️  DINOv3 access: {access_status}".ljust(58) + "║")
                        print("  ║" + "  Token saved for future sessions.".ljust(58) + "║")
                        print("  ╚" + "═" * 58 + "╝")
                        print()
                        
                        if not has_access:
                            print("  ⚠️  You're logged in but may not have model access.")
                            print("     Make sure you've requested access at:")
                            print(f"     👉 https://huggingface.co/{DINOV3_MODEL_ID}")
                            print()
                        
                        return True
                    else:
                        print("  ❌ Token saved but verification failed. Trying anyway...")
                        return True
                        
                except Exception as e:
                    print()
                    print(f"  ❌ Authentication failed: {e}")
                    print()
                    if retry:
                        continue
                    return False
            else:
                print("  ⚠ No token entered")
                continue
                
        elif choice == "2":
            print()
            print("  ℹ Skipping DINOv3 - will continue with other models")
            print()
            return False
            
        elif choice == "3":
            print()
            print("  ┌" + "─" * 60 + "┐")
            print("  │ To authenticate later, run:".ljust(61) + "│")
            print("  │   huggingface-cli login".ljust(61) + "│")
            print("  └" + "─" * 60 + "┘")
            print()
            sys.exit(0)
            
        else:
            print("  ⚠ Invalid choice. Please enter 1, 2, or 3")


def load_model(model_name: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, callable]:
    """
    Load a Vision Transformer model and its preprocessing function.
    
    Args:
        model_name: One of the model keys from AVAILABLE_MODELS
        device: Device to load model on (auto-detected if None)
    
    Returns:
        (model, preprocess_fn) tuple
    """
    if device is None:
        device = get_device()
    
    model_info = AVAILABLE_MODELS.get(model_name.lower())
    if not model_info:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    full_name = model_info["full_name"]
    print(f"[>] Loading {model_name.upper()} on {device}...")
    print(f"    {model_info['description']}")
    print(f"    (First download may take several minutes)")
    
    try:
        if full_name == "dinov2_vitg14":
            print("    Downloading from torch.hub (facebookresearch/dinov2)...")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        elif full_name == "dinov3_vith16plus":
            # Try to load directly first (user might already be authenticated)
            success, loaded_model, processor, error = try_load_dinov3_model()
            
            if not success:
                # Failed - show authentication prompt
                print(f"    ⚠ Could not load DINOv3: {error}")
                if not authenticate_huggingface():
                    raise ValueError("DINOv3 requires HuggingFace authentication")
                
                # Try again after authentication
                print("    Retrying DINOv3 download...")
                from transformers import AutoModel, AutoImageProcessor
                loaded_model = AutoModel.from_pretrained(DINOV3_MODEL_ID)
                processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID)
            
            class DINOv3Encoder(torch.nn.Module):
                def __init__(self, model, processor):
                    super().__init__()
                    self.model = model
                    self.processor = processor
                
                def forward(self, images):
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    return outputs.last_hidden_state[:, 0, :]
            
            model = DINOv3Encoder(loaded_model, processor)
            preprocess = lambda x: x
            
        elif full_name == "clip_vitl14":
            print("    Downloading from OpenAI CLIP (ViT-L/14)...")
            try:
                import clip
            except ImportError:
                raise ImportError("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            
            clip_model, preprocess = clip.load("ViT-L/14", device=device)
            
            class CLIPImageEncoder(torch.nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.clip_model = clip_model
                
                def forward(self, x):
                    return self.clip_model.encode_image(x)
            
            model = CLIPImageEncoder(clip_model)
            
        elif full_name == "siglip_vitb16":
            print("    Downloading from HuggingFace (google/siglip-base-patch16-224)...")
            from transformers import AutoModel, AutoImageProcessor
            
            siglip_model = AutoModel.from_pretrained('google/siglip-base-patch16-224')
            processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-224')
            
            class SigLIPImageEncoder(torch.nn.Module):
                def __init__(self, model, processor):
                    super().__init__()
                    self.model = model
                    self.processor = processor
                
                def forward(self, images):
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    outputs = self.model.vision_model(**inputs)
                    return outputs.pooler_output
            
            model = SigLIPImageEncoder(siglip_model, processor)
            preprocess = lambda x: x
            
        elif full_name in ("bioclip2_vitl14", "bioclip2_5_vith14"):
            print("    Downloading BioCLIP from HuggingFace...")
            try:
                import open_clip
            except ImportError:
                raise ImportError("OpenCLIP not installed. Install with: pip install open_clip_torch")
            
            hf_map = {
                "bioclip2_vitl14": "hf-hub:imageomics/bioclip-2",
                "bioclip2_5_vith14": "hf-hub:imageomics/bioclip-2.5-vith14",
            }
            
            hub_id = hf_map.get(full_name)
            model, _, preprocess = open_clip.create_model_and_transforms(hub_id)
            
            class BioCLIPImageEncoder(torch.nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.clip_model = clip_model
                
                def forward(self, x):
                    return self.clip_model.encode_image(x)
            
            model = BioCLIPImageEncoder(model)
        else:
            raise ValueError(f"Unknown model: {full_name}")
        
        model = model.to(device)
        model.eval()
        print(f"[OK] {model_name.upper()} loaded successfully")
        
        return model, preprocess
        
    except ImportError as e:
        print(f"\n[X] Failed to load {model_name}: {e}")
        raise
    except Exception as e:
        print(f"\n[X] Failed to load {model_name}: {e}")
        raise


def discover_images(
    dataset_path: Path, 
    limit_per_class: Optional[int] = None,
    n_classes: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[Path], List[str]]:
    """
    Discover all images in a dataset directory.
    
    Supports two structures:
    1. Labeled: dataset_path/species_name/image.jpg
    2. Unlabeled: dataset_path/image.jpg (labels extracted from filename if possible)
    
    Args:
        dataset_path: Path to dataset directory
        limit_per_class: Limit images per class
        n_classes: Limit number of classes
        random_seed: Random seed for sampling
    
    Returns:
        (image_paths, labels)
    """
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Skip non-species folders
    skip_folders = {'results', 'embeddings', 'metadata', 'outputs', '__pycache__'}
    
    # First check if this is a labeled structure (with subfolders)
    subfolders = [d for d in dataset_path.iterdir() 
                  if d.is_dir() and d.name not in skip_folders and not d.name.startswith('.')]
    
    image_paths = []
    labels = []
    
    if subfolders:
        # Labeled structure: images in species folders
        print(f"[>] Found labeled structure with {len(subfolders)} class folders")
        
        # Optionally limit classes
        if n_classes and n_classes < len(subfolders):
            subfolders = random.sample(subfolders, n_classes)
            print(f"    Randomly selected {n_classes} classes")
        
        for class_folder in sorted(subfolders):
            class_name = class_folder.name
            
            # Find all images
            class_images = []
            for img_path in class_folder.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    class_images.append(img_path)
            
            if not class_images:
                continue
            
            # Limit per class if specified
            if limit_per_class and len(class_images) > limit_per_class:
                class_images = random.sample(class_images, limit_per_class)
            
            for img_path in class_images:
                image_paths.append(img_path)
                labels.append(class_name)
    
    else:
        # Unlabeled structure: images directly in folder
        print(f"[>] Found flat structure (no subfolders)")
        
        for img_path in dataset_path.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                # Try to extract label from filename
                filename = img_path.stem
                
                # Handle format: species-name_0001 or uncertain_species-name_0001
                if filename.startswith("uncertain_"):
                    filename = filename[10:]
                
                parts = filename.rsplit('_', 1)
                if len(parts) >= 2 and parts[1].isdigit():
                    label = parts[0]
                else:
                    label = "unknown"
                
                image_paths.append(img_path)
                labels.append(label)
    
    print(f"[OK] Found {len(image_paths)} images across {len(set(labels))} classes")
    
    return image_paths, labels


def extract_embeddings(
    model: torch.nn.Module,
    preprocess_fn: callable,
    image_paths: List[Path],
    device: Optional[torch.device] = None,
    batch_size: int = 32
) -> np.ndarray:
    """Extract embeddings from images."""
    if device is None:
        device = get_device()
    
    embeddings = []
    
    # Check model type
    model_class = model.__class__.__name__
    needs_pil = 'SigLIP' in model_class or 'DINOv3' in model_class
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            pil_images = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    if needs_pil:
                        pil_images.append(img)
                    else:
                        img_tensor = preprocess_fn(img)
                        batch_images.append(img_tensor)
                except Exception as e:
                    print(f"\n[!] Failed to process {img_path}: {e}")
                    continue
            
            if not batch_images and not pil_images:
                continue
            
            if needs_pil:
                batch_embeddings = model(pil_images)
            else:
                batch_tensor = torch.stack(batch_images).to(device)
                batch_embeddings = model(batch_tensor)
            
            if isinstance(batch_embeddings, dict):
                batch_embeddings = batch_embeddings.get('pooler_output', batch_embeddings.get('last_hidden_state'))
            if batch_embeddings.dim() > 2:
                batch_embeddings = batch_embeddings.mean(dim=1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def save_embeddings(
    embeddings: np.ndarray,
    labels: List[str],
    paths: List[str],
    output_path: Path
) -> None:
    """Save embeddings to a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        'embeddings': embeddings,
        'labels': labels,
        'paths': paths
    }
    
    with output_path.open('wb') as f:
        pickle.dump(payload, f)
    
    print(f"[OK] Saved embeddings to: {output_path}")


def load_embeddings(path: Path) -> EmbeddingRecord:
    """Load embeddings from a pickle file."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    
    with path.open("rb") as f:
        payload = pickle.load(f)
    
    return EmbeddingRecord(
        embeddings=np.asarray(payload['embeddings']),
        labels=list(payload['labels']),
        paths=list(payload['paths'])
    )


def extract_and_save(
    model_name: str,
    data_path: Path,
    output_path: Path,
    limit_per_class: Optional[int] = None,
    n_classes: Optional[int] = None,
    batch_size: int = 32,
    random_seed: int = 42
) -> bool:
    """Extract embeddings for a model and save to disk."""
    try:
        device = get_device()
        model, preprocess = load_model(model_name, device)
        
        print(f"\n[>] Discovering images in {data_path}...")
        image_paths, labels = discover_images(
            data_path, 
            limit_per_class=limit_per_class,
            n_classes=n_classes,
            random_seed=random_seed
        )
        
        if len(image_paths) == 0:
            print(f"[X] No images found in {data_path}")
            return False
        
        print(f"[>] Extracting embeddings with {model_name.upper()}...")
        embeddings = extract_embeddings(model, preprocess, image_paths, device, batch_size)
        
        print(f"[>] Saving embeddings...")
        save_embeddings(embeddings, labels, [str(p) for p in image_paths], output_path)
        
        print(f"[OK] Extracted {len(embeddings)} embeddings for {model_name.upper()}")
        return True
        
    except ValueError as e:
        if "authentication" in str(e).lower():
            print(f"\n[!] Skipping {model_name} - authentication required")
            return False
        raise
    except Exception as e:
        print(f"\n[X] Error processing {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from images using Vision Transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_embeddings.py --data-path test_data/Mammals
  python extract_embeddings.py --models dinov3 dinov2 --data-path test_data/Aves
  python extract_embeddings.py --models all --limit-per-class 200 --n-classes 20
        """
    )
    
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        required=True,
        help="Path to image data folder"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/embeddings",
        help="Output directory for embeddings (default: outputs/embeddings)"
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        type=str,
        default=["all"],
        choices=["all"] + list(AVAILABLE_MODELS.keys()),
        help="Models to use (default: all)"
    )
    
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Limit number of images per class"
    )
    
    parser.add_argument(
        "--n-classes",
        type=int,
        default=None,
        help="Limit number of classes"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if embeddings exist"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    
    if not data_path.exists():
        print(f"[X] Data path not found: {data_path}")
        sys.exit(1)
    
    # Determine which models to use
    if "all" in args.models:
        models = list(AVAILABLE_MODELS.keys())
    else:
        models = args.models
    
    print("\n" + "=" * 70)
    print("  EMBEDDING EXTRACTION")
    print("=" * 70)
    print(f"  Data path: {data_path}")
    print(f"  Models: {', '.join(models)}")
    if args.limit_per_class:
        print(f"  Limit per class: {args.limit_per_class}")
    if args.n_classes:
        print(f"  Number of classes: {args.n_classes}")
    print("=" * 70)
    
    # Extract embeddings for each model
    results = {}
    for model_name in models:
        output_path = output_dir / f"{model_name}_embeddings.pkl"
        
        if output_path.exists() and not args.force:
            print(f"\n[*] Embeddings already exist for {model_name.upper()}")
            print(f"    Use --force to re-extract")
            results[model_name] = True
            continue
        
        success = extract_and_save(
            model_name=model_name,
            data_path=data_path,
            output_path=output_path,
            limit_per_class=args.limit_per_class,
            n_classes=args.n_classes,
            batch_size=args.batch_size,
            random_seed=args.seed
        )
        results[model_name] = success
    
    # Summary
    print("\n" + "=" * 70)
    print("  EXTRACTION SUMMARY")
    print("=" * 70)
    for model_name, success in results.items():
        status = "[OK]" if success else "[SKIPPED]"
        print(f"  {status} {model_name.upper()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
