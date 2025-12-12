#!/bin/bash
# Installation script for DINOv2 weights via Depth Anything V2
# For MonoDINO-DETR project

set -e  # Exit on error

echo "============================================"
echo "MonoDINO-DETR: Installing DINOv2 Weights"
echo "============================================"
echo ""

# Create checkpoints directory
echo "Step 1: Creating checkpoints directory..."
mkdir -p checkpoints
cd checkpoints

# Check which backbone is needed from config
echo ""
echo "Step 2: Downloading Depth Anything V2 weights..."
echo "Note: The config uses 'vitb' (ViT-Base) by default"
echo ""

# Function to download with progress
download_model() {
    local model_name=$1
    local url=$2
    local filename="depth_anything_v2_${model_name}.pth"
    
    if [ -f "$filename" ]; then
        echo "✓ $filename already exists, skipping download"
    else
        echo "Downloading $filename..."
        wget --progress=bar:force:noscroll "$url" -O "$filename"
        echo "✓ Downloaded $filename"
    fi
}

# Download ViT-Base (default in config)
echo ""
echo "Downloading ViT-Base (recommended, 97.5M params)..."
download_model "vitb" "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"

echo ""
echo "Would you like to download other model variants? (y/n)"
echo "  - ViT-Small (faster, 24.8M params)"
echo "  - ViT-Large (better accuracy, 335.3M params)"  
echo "  - ViT-Giant (best accuracy, 1.3B params)"
read -p "Download all variants? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Downloading all variants..."
    
    download_model "vits" "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    download_model "vitl" "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    download_model "vitg" "https://huggingface.co/depth-anything/Depth-Anything-V2-Giant/resolve/main/depth_anything_v2_vitg.pth"
fi

cd ..

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "Downloaded weights contain pretrained DINOv2 backbone + DPT head"
echo ""
echo "To use different model variants, update the 'backbone' field in configs/monodinodetr.yaml:"
echo "  - backbone: 'vits'  # ViT-Small"
echo "  - backbone: 'vitb'  # ViT-Base (default)"
echo "  - backbone: 'vitl'  # ViT-Large"
echo "  - backbone: 'vitg'  # ViT-Giant"
echo ""
echo "The weights are loaded automatically from:"
echo "  /home/hice1/rrustagi7/scratch/glen/MonoDINO-DETR/checkpoints/"
echo ""



