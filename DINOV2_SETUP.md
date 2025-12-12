# DINOv2 Installation Guide for MonoDINO-DETR

## Quick Summary

**You don't need to install DINOv2 separately!** The DINOv2 architecture is already built into this codebase. You only need to download the **Depth Anything V2 pretrained weights**, which contain the pretrained DINOv2 backbone.

---

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)

```bash
# Run the installation script
bash install_dinov2.sh
```

This will:
- Create the `checkpoints/` directory
- Download the ViT-Base weights (default)
- Optionally download other variants

### Option 2: Manual Installation

```bash
# Create checkpoints directory
mkdir -p checkpoints
cd checkpoints

# Download ViT-Base (recommended, 97.5M params)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

# OR download other variants:

# ViT-Small (fastest, 24.8M params)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# ViT-Large (best performance, 335.3M params)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

# ViT-Giant (highest accuracy, 1.3B params)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth

cd ..
```

---

## 📁 File Structure After Installation

```
MonoDINO-DETR/
├── checkpoints/
│   ├── depth_anything_v2_vitb.pth  # ViT-Base (default)
│   ├── depth_anything_v2_vits.pth  # ViT-Small (optional)
│   ├── depth_anything_v2_vitl.pth  # ViT-Large (optional)
│   └── depth_anything_v2_vitg.pth  # ViT-Giant (optional)
├── configs/
│   └── monodinodetr.yaml
├── lib/
│   └── models/
│       └── monodinodetr/
│           ├── backbone.py          # Loads DINOv2 from Depth Anything V2
│           └── depth_anything_v2/
│               ├── dinov2.py        # DINOv2 architecture (built-in)
│               └── dpt.py           # DPT head for depth
└── ...
```

---

## 🔧 Configuration

### Choosing a Model Variant

Edit `configs/monodinodetr.yaml`:

```yaml
model:
  # Change this to use different DINOv2 variants
  backbone: 'vitb'  # Options: 'vits', 'vitb', 'vitl', 'vitg'
```

### Model Variant Comparison

| Variant | Parameters | Speed | Memory | Accuracy | Use Case |
|---------|-----------|-------|--------|----------|----------|
| ViT-S   | 24.8M     | Fast  | Low    | Good     | Testing, low-resource |
| ViT-B   | 97.5M     | ⭐ Balanced | Medium | Very Good | **Recommended** |
| ViT-L   | 335.3M    | Slow  | High   | Excellent | Best accuracy |
| ViT-G   | 1.3B      | Very Slow | Very High | Best | Research only |

---

## 🧠 How It Works

### Architecture Overview

```
Input Image (RGB)
    ↓
┌──────────────────────────────────┐
│  DepthAnythingV2 Model           │
│  ┌────────────────────────────┐  │
│  │  DINOv2 ViT Backbone       │  │  ← Pretrained weights loaded here
│  │  (Vision Transformer)      │  │
│  │  - Extract features from   │  │
│  │    intermediate layers     │  │
│  │  - Multi-scale features    │  │
│  └────────────────────────────┘  │
│              ↓                    │
│  ┌────────────────────────────┐  │
│  │  DPT Head                  │  │
│  │  (Dense Prediction Trans.) │  │
│  │  - Depth estimation        │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
    ↓                    ↓
Multi-scale          Depth Map
DINO Features        
    ↓                    ↓
┌──────────────────────────────────┐
│  MonoDINO-DETR                   │
│  - Depth-Aware Transformer       │
│  - 3D Object Detection           │
└──────────────────────────────────┘
```

### What Gets Loaded

When you load `depth_anything_v2_vitb.pth`:

1. **DINOv2 Backbone** (`self.backbone = depth_anything.pretrained`)
   - Pretrained Vision Transformer (ViT)
   - Frozen by default (parameters `requires_grad=False`)
   - Extracts rich semantic features

2. **DPT Head** (`self.dpt_head = depth_anything.depth_head`)
   - Dense Prediction Transformer
   - Also frozen by default
   - Produces monocular depth estimates

### Key Code (backbone.py, lines 173-193)

```python
# Initialize DepthAnythingV2 with DINOv2 backbone
depth_anything = DepthAnythingV2(**depthanything_model_configs[name])

# Load pretrained weights (contains both DINOv2 + DPT)
model_weights_path = f'checkpoints/depth_anything_v2_{name}.pth'
depth_anything.load_state_dict(torch.load(model_weights_path))

# Extract DINOv2 backbone
self.backbone = depth_anything.pretrained  # This is DINOv2!

# Freeze most parameters (fine-tuning strategy)
for param in self.backbone.named_parameters():
    param.requires_grad_(False)

# Extract DPT head for depth
self.dpt_head = depth_anything.depth_head
```

---

## ✅ Verification

To verify the installation:

```bash
# Check if weights exist
ls -lh checkpoints/

# Should show:
# depth_anything_v2_vitb.pth  (size: ~372 MB for ViT-Base)
```

To verify in Python:

```python
import torch
from lib.models.monodinodetr.depth_anything_v2.dpt import DepthAnythingV2

# Load model
model = DepthAnythingV2(encoder='vitb')
weights = torch.load('checkpoints/depth_anything_v2_vitb.pth')
model.load_state_dict(weights)

# Check DINOv2 backbone
print(f"DINOv2 embed_dim: {model.pretrained.embed_dim}")  # Should be 768 for ViT-B
print(f"DINOv2 num_blocks: {model.pretrained.n_blocks}")  # Should be 12 for ViT-B
print("✓ DINOv2 loaded successfully!")
```

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: depth_anything_v2_vitb.pth"

**Solution:** Run the installation script or manually download the weights to the `checkpoints/` directory.

### Issue: "CUDA out of memory"

**Solution:** 
- Use a smaller variant: `backbone: 'vits'`
- Reduce batch size in config: `batch_size: 4`

### Issue: "Checkpoint path not found"

**Solution:** The path in `backbone.py` has been updated to:
```python
model_weights_path = '/home/hice1/rrustagi7/scratch/glen/MonoDINO-DETR/checkpoints/depth_anything_v2_{name}.pth'
```

If your directory structure is different, update line 176 in `lib/models/monodinodetr/backbone.py`.

---

## 📚 Additional Resources

- **Depth Anything V2**: https://github.com/DepthAnything/Depth-Anything-V2
- **DINOv2**: https://github.com/facebookresearch/dinov2
- **Hugging Face Hub**: https://huggingface.co/depth-anything

---

## 🎯 Summary

1. ✅ DINOv2 architecture is **already in the code** (no separate installation needed)
2. ✅ Only need to **download Depth Anything V2 weights** (contains pretrained DINOv2)
3. ✅ Weights are automatically loaded in `backbone.py`
4. ✅ Use `bash install_dinov2.sh` for easy setup

**You're ready to train!** 🚀



