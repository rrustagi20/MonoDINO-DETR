#!/usr/bin/env python3
"""
Compute per-class mean [h,w,l] from KITTI-format label files and per-channel image mean/std
for the training split. Prints YAML-ready structures.

Usage: python tools/compute_dataset_stats.py [DATASET_ROOT]
Default DATASET_ROOT: /home/hice1/rrustagi7/scratch/final_ai2_kitti
"""
import sys
import os
import numpy as np
from collections import defaultdict
from PIL import Image


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    cls = parts[0]
    try:
        h = float(parts[8])
        w = float(parts[9])
        l = float(parts[10])
    except Exception:
        return None
    return cls, h, w, l


def compute_class_mean_sizes(label_dir):
    sizes = defaultdict(list)
    files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    for fn in files:
        with open(os.path.join(label_dir, fn), 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                cls, h, w, l = parsed
                sizes[cls].append([h, w, l])

    class_means = {}
    class_stds = {}
    for cls, arr in sizes.items():
        a = np.array(arr, dtype=np.float32)
        mean = a.mean(axis=0).tolist()
        std = a.std(axis=0).tolist()
        class_means[cls] = [float(round(x, 6)) for x in mean]
        class_stds[cls] = [float(round(x, 6)) for x in std]
    return class_means, class_stds


def compute_image_mean_std(image_dir, max_images=None):
    # Running mean and var per channel
    count = 0
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)

    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])
    if max_images:
        files = files[:max_images]

    for fn in files:
        path = os.path.join(image_dir, fn)
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: couldn't open {path}: {e}")
            continue
        arr = np.array(img).astype(np.float64) / 255.0
        # shape H W C
        pixels = arr.reshape(-1, 3)
        sum_c += pixels.sum(axis=0)
        sumsq_c += (pixels ** 2).sum(axis=0)
        count += pixels.shape[0]

    if count == 0:
        raise RuntimeError('No image pixels counted')
    mean = (sum_c / count).tolist()
    var = (sumsq_c / count) - (np.array(mean) ** 2)
    std = np.sqrt(np.maximum(var, 0)).tolist()
    mean = [float(round(x, 6)) for x in mean]
    std = [float(round(x, 6)) for x in std]
    return mean, std


def main():
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else '/home/hice1/rrustagi7/scratch/final_ai2_kitti'
    train_dir = os.path.join(dataset_root, 'training')
    label_dir = os.path.join(train_dir, 'label_2')
    image_dir = os.path.join(train_dir, 'image_2')

    if not os.path.isdir(label_dir):
        print('Label dir not found:', label_dir)
        sys.exit(1)
    if not os.path.isdir(image_dir):
        print('Image dir not found:', image_dir)
        sys.exit(1)

    print('# Computing per-class mean sizes from labels in', label_dir)
    class_means, class_stds = compute_class_mean_sizes(label_dir)
    print('\nclass_mean_sizes:')
    for cls in sorted(class_means.keys()):
        print(f"  '{cls}': {class_means[cls]}")

    print('\nclass_size_stds:')
    for cls in sorted(class_stds.keys()):
        print(f"  '{cls}': {class_stds[cls]}")

    print('\n# Computing image mean/std from training images in', image_dir)
    # Limit images if there are too many (set None to use all)
    mean, std = compute_image_mean_std(image_dir, max_images=None)
    print('\nimage_mean:', mean)
    print('image_std :', std)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import os
import glob
import numpy as np
from PIL import Image

ROOT = '/home/hice1/rrustagi7/scratch/final_ai2_kitti'
LABEL_DIR = os.path.join(ROOT, 'training', 'label_2')
IMAGE_DIR = os.path.join(ROOT, 'training', 'image_2')

# Collect sizes per class
sizes = {}
count = {}
for fn in glob.glob(os.path.join(LABEL_DIR, '*.txt')):
    with open(fn, 'r') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts = line.split()
            cls = parts[0]
            # ensure there are enough fields
            if len(parts) < 14:
                continue
            try:
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
            except:
                continue
            if cls not in sizes:
                sizes[cls] = np.zeros(3, dtype=np.float64)
                count[cls] = 0
            sizes[cls] += np.array([h,w,l], dtype=np.float64)
            count[cls] += 1

class_mean_sizes = {}
for cls, s in sizes.items():
    n = count[cls]
    if n>0:
        class_mean_sizes[cls] = (s / n).tolist()
    else:
        class_mean_sizes[cls] = [0.0,0.0,0.0]

# Compute image mean/std (per-channel) over training images
# We'll compute incrementally to avoid large memory
img_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
if len(img_files) == 0:
    print('# No training images found for computing mean/std')
    channel_mean = [0.485,0.456,0.406]
    channel_std = [0.229,0.224,0.225]
else:
    s1 = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for i, p in enumerate(img_files):
        try:
            im = Image.open(p).convert('RGB')
            arr = np.array(im).astype(np.float32) / 255.0
        except Exception as e:
            print('# Warning: failed reading', p, e)
            continue
        h,w,c = arr.shape
        pixels = h*w
        arr_flat = arr.reshape(-1,3)
        s1 += arr_flat.sum(axis=0)
        s2 += (arr_flat**2).sum(axis=0)
        n_pixels += pixels
        # lightweight progress
        if (i+1) % 500 == 0:
            print(f'# processed {i+1} images')
    channel_mean = (s1 / n_pixels).tolist()
    channel_var = (s2 / n_pixels) - (np.array(channel_mean)**2)
    channel_std = np.sqrt(channel_var).tolist()

# Print YAML-ready mapping
import json
print('class_mean_sizes:')
for cls in sorted(class_mean_sizes.keys()):
    vals = [float(x) for x in class_mean_sizes[cls]]
    print(f"  '{cls}': [{vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}]")

print('\nmean: [' + ', '.join([f'{x:.6f}' for x in channel_mean]) + ']')
print('std:  [' + ', '.join([f'{x:.6f}' for x in channel_std]) + ']')
print('\n# counts per class:')
for cls in sorted(count.keys()):
    print(f"# {cls}: {count[cls]}")
