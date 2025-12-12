# Quick Start Guide - MonoDINO-DETR with DINOv2

## 🚀 TL;DR - Get Running in 5 Steps

```bash
# 1. Install DINOv2 weights (Depth Anything V2)
bash install_dinov2.sh

# 2. Compile deformable attention
cd lib/models/monodinodetr/ops/
bash make.sh
cd ../../../..

# 3. Create logs directory
mkdir -p logs

# 4. Update data path in configs/monodinodetr.yaml
# Change line 5: root_dir: '/path/to/your/KITTI'

# 5. Train!
bash train.sh configs/monodinodetr.yaml > logs/monodinodetr.log
```

---

## 📦 What You Just Installed

The `install_dinov2.sh` script downloads **Depth Anything V2 weights**, which contain:

- ✅ **Pretrained DINOv2 Vision Transformer** (the backbone)
- ✅ **DPT Head** (for depth estimation)
- ✅ Everything needed for MonoDINO-DETR

**No separate DINOv2 installation required!** The architecture is already in the code.

---

## 🎯 Model Variants

Choose in `configs/monodinodetr.yaml`:

```yaml
model:
  backbone: 'vitb'  # Change this line
```

| Model | Size | Speed | Memory | When to Use |
|-------|------|-------|--------|-------------|
| `vits` | 25M  | ⚡⚡⚡ | 💾 | Quick experiments |
| `vitb` | 98M  | ⚡⚡ | 💾💾 | **Recommended (default)** |
| `vitl` | 335M | ⚡ | 💾💾💾 | Max accuracy |
| `vitg` | 1.3B | 🐌 | 💾💾💾💾 | Research only |

---

## ✅ Verify Installation

```bash
# Check if weights are downloaded
ls -lh checkpoints/

# Should show:
# depth_anything_v2_vitb.pth  (~372 MB)
```

---

## 🏃 Training Commands

**Single GPU:**
```bash
bash train.sh configs/monodinodetr.yaml > logs/train.log
```

**Multi-GPU (4 GPUs, batch=32):**
```bash
bash train.sh configs/monodinodetr.yaml --batch_size 32 --num_gpus 4 > logs/train_multi.log
```

**Resume from checkpoint:**
```yaml
# In configs/monodinodetr.yaml, uncomment:
trainer:
  resume_model: True
  pretrain_model: outputs/checkpoint_epoch_100.pth
```

---

## 📊 Monitor Training

```bash
# Watch training progress
tail -f logs/monodinodetr.log

# Check outputs
ls outputs/
```

---

## 🧪 Testing

```bash
# Test with best checkpoint
bash test.sh configs/monodinodetr.yaml

# Test specific checkpoint (e.g., epoch 150)
# Edit configs/monodinodetr.yaml:
tester:
  checkpoint: 150
```

---

## 🔍 Understanding the Pipeline

```
1. Input Image
   ↓
2. DINOv2 Backbone (from Depth Anything V2 weights)
   ├─→ Multi-scale features (3 levels)
   └─→ Depth map
   ↓
3. Feature Projection
   ↓
4. Depth Predictor Module
   ↓
5. Depth-Aware Transformer
   ├─→ Depth Cross-Attention
   ├─→ Self-Attention
   └─→ Visual Cross-Attention
   ↓
6. Prediction Heads
   ├─→ Class (car/pedestrian/cyclist)
   ├─→ 2D Box
   ├─→ 3D Dimensions
   ├─→ Depth (fusion of 3 sources)
   └─→ Orientation
```

---

## 🎨 Customization

### Change Backbone Variant
```yaml
# configs/monodinodetr.yaml
model:
  backbone: 'vitl'  # Use ViT-Large instead
```

### Adjust Training
```yaml
optimizer:
  lr: 0.0001  # Lower learning rate
  
trainer:
  max_epoch: 300  # Train longer
  gpu_ids: '0,1,2,3,4,5,6,7'  # Use 8 GPUs
```

### Fine-tune DINOv2
```python
# lib/models/monodinodetr/backbone.py, line 184
# Comment out this line to make DINOv2 trainable:
# for name_1, param in self.backbone.named_parameters():
#     param.requires_grad_(False)  # Remove this line
```

---

## ❓ Common Issues

**Q: "FileNotFoundError: depth_anything_v2_vitb.pth"**  
A: Run `bash install_dinov2.sh` first

**Q: "CUDA out of memory"**  
A: Use `backbone: 'vits'` or reduce `batch_size`

**Q: "No module named 'MultiScaleDeformableAttention'"**  
A: Compile deformable attention: `cd lib/models/monodinodetr/ops/ && bash make.sh`

**Q: "Where is DINOv2?"**  
A: It's in `lib/models/monodinodetr/depth_anything_v2/dinov2.py` and loaded via Depth Anything V2 weights

---

## 📖 Full Documentation

- **DINOv2 Setup Details**: See `DINOV2_SETUP.md`
- **Original README**: See `README.md`
- **Paper**: https://arxiv.org/abs/2502.00315

---

## 🎉 You're All Set!

```bash
# Ready to train MonoDINO-DETR with DINOv2! 🚀
bash train.sh configs/monodinodetr.yaml > logs/monodinodetr.log &
```

Monitor with: `tail -f logs/monodinodetr.log`

Good luck! 🍀



