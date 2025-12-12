# filename: lib/datasets/ai2thor_kitti/ai2thor_kitti_dataset.py

import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import random
import math # Make sure math is imported

ImageFile.LOAD_TRUNCATED_IMAGES = True # Good practice

# Assuming these utils are accessible and work generically with KITTI format data
# If get_objects_from_label is hardcoded for KITTI classes, it might need modification
from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label # ASSUMPTION: Parses format correctly
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
# Evaluation imports - keep for potential future use, but modify eval function
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
from lib.datasets.kitti.kitti_eval_python.kitti_common import get_label_annos
import copy
try:
    # Try importing PhotometricDistort if it exists in your structure
    from .pd import PhotometricDistort
except ImportError:
    # Provide a dummy class if it's missing, or handle its absence
    print("Warning: PhotometricDistort class not found. Photometric augmentation disabled.")
    class PhotometricDistort:
        def __call__(self, img):
            return img

class AI2THOR_KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):
        print(f"Initializing AI2THOR_KITTI_Dataset for split: {split}")

        # --- Basic Configuration ---
        self.root_dir = cfg.get('root_dir') # Path to the KITTI-formatted AI2THOR data
        self.split = split
        assert self.split in ['train', 'val', 'trainval', 'test', 'mini_train', 'mini_val'], f"Invalid split: {split}"

        # --- Class Configuration (CRITICAL CHANGES) ---
        # Get list of classes to train on from config (REQUIRED)
        # Example cfg entry: class_names: ['Mug', 'Laptop', 'Chair', 'Apple', ...]
        self.class_name = cfg.get('class_names')
        if not self.class_name:
            raise ValueError("'class_names' not provided in dataset config (cfg). Please list all desired object classes.")

        self.num_classes = len(self.class_name)
        self.cls2id = {name: i for i, name in enumerate(self.class_name)}
        print(f"  Using {self.num_classes} classes: {self.class_name}")
        print(f"  Class to ID mapping: {self.cls2id}")

        # `writelist` should contain the classes we actually process from label files
        # It should usually match self.class_name unless you want to ignore some during loading
        self.writelist = cfg.get('writelist', self.class_name) # Default to using all specified classes
        print(f"  Processing objects of types: {self.writelist}")

        self.use_dontcare = cfg.get('use_dontcare', True) # Handle 'DontCare' labels?
        if self.use_dontcare:
            if 'DontCare' not in self.writelist: # Ensure DontCare is handled if needed
                 self.writelist.append('DontCare')

        self.max_objs = cfg.get('max_objs', 50) # Max objects per image to process

        # --- Model Input/Output Configuration ---
        # Get target resolution (model input size) from config (REQUIRED)
        # Example cfg entry: input_resolution: [640, 480] # W, H
        self.input_resolution = np.array(cfg.get('input_resolution'))
        if self.input_resolution is None:
             raise ValueError("'input_resolution' [W, H] not provided in dataset config (cfg).")

        # Get model output downsampling ratio from config (REQUIRED)
        # Example cfg entry: output_downsample: 4 # Or 8, 16, 32 depending on model
        self.downsample = cfg.get('output_downsample')
        if self.downsample is None:
             raise ValueError("'output_downsample' ratio not provided in dataset config (cfg).")

        self.output_resolution = self.input_resolution // self.downsample # Resolution of heatmap/output features

        # --- Paths Configuration ---
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # --- Data Split Loading ---
        # User MUST create these files (e.g., train.txt, val.txt) in root_dir/ImageSets/
        # containing the 6-digit frame IDs (one per line) for the split.
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}. Please create it.")
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]
        print(f"  Loaded {len(self.idx_list)} indices from {self.split_file}")

        # --- Data Augmentation Configuration ---
        self.data_augmentation = True if split in ['train', 'trainval', 'mini_train'] else False
        print(f"  Data augmentation enabled: {self.data_augmentation}")

        self.aug_pd = cfg.get('aug_pd', True) and self.data_augmentation # Photometric Distortions
        self.aug_crop = cfg.get('aug_crop', True) and self.data_augmentation # Random Crop/Scale/Shift
        self.aug_calib = cfg.get('aug_calib', False) # Augment calibration? Less common for synthetic data

        self.random_flip = cfg.get('random_flip', 0.5) if self.data_augmentation else 0.0
        self.random_crop = cfg.get('random_crop', 0.5) if self.data_augmentation else 0.0 # Prob of doing crop aug
        self.scale = cfg.get('scale', 0.2) # Scale variation for cropping
        self.shift = cfg.get('shift', 0.1) # Center shift variation for cropping

        self.depth_scale = cfg.get('depth_scale', 'normal') # How depth is affected by cropping

        # --- Statistics ---
        # ImageNet mean/std are usually a safe default, especially with pre-trained backbones
        self.mean = np.array(cfg.get('mean', [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(cfg.get('std', [0.229, 0.224, 0.225]), dtype=np.float32)

        # --- Mean Shape (CRITICAL CHANGE) ---
        # Option 1: Provide pre-calculated mean sizes in config (RECOMMENDED)
        # Example cfg entry: class_mean_sizes: {'Mug': [h,w,l], 'Chair': [h,w,l], ...}
        # Option 2: Set meanshape=False and use zeros
        self.meanshape = cfg.get('meanshape', False) # True to subtract mean shape
        class_mean_sizes_dict = cfg.get('class_mean_sizes', None)
        self.cls_mean_size = np.zeros((self.num_classes, 3), dtype=np.float32)
        if self.meanshape:
            if class_mean_sizes_dict:
                print("  Loading class mean sizes from config.")
                for i, name in enumerate(self.class_name):
                    if name in class_mean_sizes_dict:
                        self.cls_mean_size[i] = class_mean_sizes_dict[name]
                    else:
                        print(f"  Warning: Mean size not provided for class '{name}'. Using [0,0,0].")
            else:
                 print("  Warning: 'meanshape' is True, but 'class_mean_sizes' not provided in config. Using [0,0,0] for all.")
                 self.meanshape = False # Disable if not provided

        # --- Other Settings ---
        self.use_3d_center = cfg.get('use_3d_center', True) # Project 3D center or use 2D center?
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno') # Use label box ('anno') or projected 3d box ('proj')?
        assert self.bbox2d_type in ['anno', 'proj']
        self.clip_2d = cfg.get('clip_2d', False) # Clip 2d boxes tightly to image bounds after aug?

        # Initialize photometric distortion augmentation if enabled
        self.pd = PhotometricDistort() if self.aug_pd else lambda img: img

    def get_image(self, idx):
        """Loads image file."""
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        if not os.path.exists(img_file):
             raise FileNotFoundError(f"Image file not found: {img_file}")
        try:
            return Image.open(img_file).convert('RGB') # Ensure RGB
        except Exception as e:
             raise IOError(f"Error opening image file {img_file}: {e}")

    def get_label(self, idx):
        """Loads label file and parses objects."""
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        if not os.path.exists(label_file):
             # Return empty list if label file doesn't exist (e.g., for test set or errors)
             # print(f"Warning: Label file not found: {label_file}. Returning empty list.")
             return []
        try:
            return get_objects_from_label(label_file)
        except Exception as e:
            print(f"Error parsing label file {label_file}: {e}")
            return [] # Return empty list on parsing error

    def get_calib(self, idx):
        """Loads calibration file."""
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        try:
            return Calibration(calib_file) # Assumes kitti_utils.Calibration works
        except Exception as e:
            raise IOError(f"Error reading calibration file {calib_file}: {e}")

    def eval(self, results_dir, logger):
        """
        Evaluation for custom classes.
        This function is adapted from the original KITTI evaluation but is made generic
        for the classes specified in the dataset config.
        """
        logger.info("==> Loading detections and ground truth annotations...")
        img_ids = [int(id) for id in self.idx_list]
        
        # Load detection results (dt_annos) from the folder where model outputs are saved
        try:
            dt_annos = get_label_annos(results_dir)
        except Exception as e:
            logger.error(f"Error loading detection results from {results_dir}: {e}")
            return 0.0

        # Load ground truth (gt_annos) from the dataset's label directory
        try:
            gt_annos = get_label_annos(self.label_dir, img_ids)
        except Exception as e:
            logger.error(f"Error loading ground truth labels from {self.label_dir}: {e}")
            return 0.0

        # Dynamically create the class mapping from the config
        # class_to_id = self.cls2id
        
        # logger.info('==> Evaluating...')
        # all_class_aps = []

        # # Iterate over all classes defined in the dataset config
        # for class_name in self.class_name:
        #     if class_name not in class_to_id:
        #         logger.warning(f"Class '{class_name}' is in `class_name` but not in `cls2id` mapping. Skipping.")
        #         continue
            
        #     class_id = class_to_id[class_name]
            
        #     # Run the official KITTI evaluation metric for the current class
        #     # This function compares ground truth and detections for a specific class ID
        #     results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, class_id)
            
        #     logger.info(results_str)
        #     all_class_aps.append(mAP3d_R40)

        # # Calculate the mean AP across all classes
        # if not all_class_aps:
        #     logger.warning("No classes were evaluated. Returning 0.0.")
        #     return 0.0
            
        # mean_ap = np.mean(all_class_aps)
        # logger.info(f"==> Mean AP@R40 across all classes: {mean_ap:.4f}")
        
        # # Return the mean AP as the primary metric
        # return mean_ap
        return 0.0 # Placeholder return until eval is implemented
        
    def __len__(self):
        """Returns the number of samples in the split."""
        return len(self.idx_list)

    def __getitem__(self, item):
        """Gets one sample (image, calibration, targets, info)."""
        index = int(self.idx_list[item])  # Get the 6-digit file index

        #  --- Get Inputs ---
        img = self.get_image(index)
        img_size = np.array(img.size, dtype=np.int32) # Original W, H
        calib = self.get_calib(index)
        P2 = calib.P2.copy() # Get camera projection matrix

        # --- Image Augmentation ---
        center = np.array(img_size) / 2.0
        crop_size = img_size
        crop_scale = 1.0
        random_flip_flag = False

        if self.data_augmentation:
            # 1. Photometric Distortions (Color jitter, etc.)
            if self.aug_pd:
                img_np = np.array(img).astype(np.float32)
                img_np = self.pd(img_np) # Apply augmentations
                img = Image.fromarray(img_np.astype(np.uint8)) # Convert back to PIL

            # 2. Random Flip
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                P2[0, 2] = img_size[0] - P2[0, 2] # Adjust principal point x
                P2[0, 3] = -P2[0, 3] # Adjust Tx (assuming Tx = -fx * cx_shifted_by_flip)

            # 3. Random Crop, Scale, Shift
            if self.aug_crop and np.random.random() < self.random_crop:
                crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                crop_size = img_size * crop_scale
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # --- Affine Transformation (Crop/Resize to Model Input Size) ---
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.input_resolution, inv=1) # Use inv=1 to get both transforms
        img = img.transform(tuple(self.input_resolution.tolist()), # Target size (W, H)
                            method=Image.AFFINE,
                            data=tuple(trans.reshape(-1).tolist()), # Apply forward transform matrix
                            resample=Image.BILINEAR)

        # --- Scale Camera Calibration Matrix ---
        # P2 is a 3x4 projection matrix: [fx, 0, cx, tx; 0, fy, cy, ty; 0, 0, 1, 0]
        # When the image is resized, intrinsics must be scaled.
        # The affine transform `trans` maps from original image coords to input image coords.
        # We can find the overall scaling by seeing how it maps the top-left (0,0) and a point (1,1).
        p1 = affine_transform(np.array([0,0], dtype=np.float32), trans)
        p2 = affine_transform(np.array([1,1], dtype=np.float32), trans)
        scale_x = p2[0] - p1[0]
        scale_y = p2[1] - p1[1]

        P2[0, :] *= scale_x  # Scale fx, cx, and tx by width ratio
        P2[1, :] *= scale_y  # Scale fy, cy, and ty by height ratio

        # --- Image Normalization ---
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # Convert to C * H * W for PyTorch

        # --- Info dictionary ---
        info = {'img_id': index,
                'img_size': img_size, # Original size before augmentation
                'img_size_input': self.input_resolution, # Model input size
               }

        # --- Handle Test Split (No Labels) ---
        if self.split == 'test':
             # For test, return image, original P2, transformed image, and info
             # Note: You might need original P2 if evaluation requires projecting back
             return img, P2, img, info # Check what your model's forward pass expects for test mode


        # --- Get Labels ---
        objects = self.get_label(index)

        # --- Label Augmentation (Random Flip affects labels) ---
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size) # Flip internal calib state if needed by label processing funcs
            for obj in objects:
                # Flip 2D box
                x1, _, x2, _ = obj.box2d
                obj.box2d[0] = img_size[0] - x2 - 1 # Adjust for 0-based index
                obj.box2d[2] = img_size[0] - x1 - 1
                # Flip 3D position x
                if self.aug_calib: # If calib object itself was flipped
                     obj.pos[0] *= -1
                # Flip angles alpha and ry
                obj.alpha = math.pi - obj.alpha
                obj.ry = math.pi - obj.ry
                # Normalize angles
                while obj.alpha > math.pi: obj.alpha -= 2 * math.pi
                while obj.alpha <= -math.pi: obj.alpha += 2 * math.pi
                while obj.ry > math.pi: obj.ry -= 2 * math.pi
                while obj.ry <= -math.pi: obj.ry += 2 * math.pi


        # --- Prepare Target Tensors ---
        # Initialize empty targets (using torch tensors might be better eventually)
        heatmap = np.zeros((self.num_classes, self.output_resolution[1], self.output_resolution[0]), dtype=np.float32) # C x H x W
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32) # Calibration matrix for each object
        offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32) # Residual size (after mean subtraction)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32) # Original 3D size (before mean subtraction)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        labels = np.zeros((self.max_objs), dtype=np.int8) # Class labels for each object
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32) # 2D boxes: [cx_norm, cy_norm, w_norm, h_norm]
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32) # 3D boxes: [cx_norm, cy_norm, l, r, t, b]
        occlusion = np.zeros((self.max_objs), dtype=np.int64) # Occlusion level for each object
        # For tracking which entries are valid
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_target = np.zeros((self.max_objs), dtype=np.bool_) # Mask for valid objects used in loss
        mask_2d = np.zeros((self.max_objs), dtype=np.bool_) # Mask for 2D detection quality (based on truncation/occlusion)

        # Track number of objects added
        obj_count = 0

        # --- Process Objects ---
        for i, obj in enumerate(objects):
            if obj_count >= self.max_objs: break # Stop if max_objs reached

            # 1. Filter by Class
            if obj.cls_type not in self.writelist: continue
            if obj.cls_type.lower() == 'dontcare' and not self.use_dontcare: continue

            # 2. Filter by KITTI Difficulty/Depth (adjust thresholds for AI2THOR if needed)
            # Example KITTI filtering:
            # if obj.level_str == 'UnKnown' or obj.pos[2] < 2: continue # pos[2] is Z (depth)
            # threshold = 65
            # if obj.pos[2] > threshold: continue
            # Adapt these based on your analysis script results if necessary

            # 3. Get Class ID
            cls_id = self.cls2id.get(obj.cls_type, -1) # Use .get for safety
            if cls_id == -1: continue # Skip if class name not in our defined list

            # 4. Process 2D Bbox
            bbox_2d = obj.box2d.copy()
            # Apply affine transformation to 2D box corners
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            # Clip box to target image dimensions
            bbox_2d[[0, 2]] = np.clip(bbox_2d[[0, 2]], 0, self.input_resolution[0] - 1)
            bbox_2d[[1, 3]] = np.clip(bbox_2d[[1, 3]], 0, self.input_resolution[1] - 1)
            bbox_h, bbox_w = bbox_2d[3] - bbox_2d[1], bbox_2d[2] - bbox_2d[0]

            # Skip if box is too small after transform/clipping
            if bbox_h <= 1 or bbox_w <= 1: continue

            # 5. Calculate Center Point (for heatmap)
            # Always calculate 2D box center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2.0, (bbox_2d[1] + bbox_2d[3]) / 2.0], dtype=np.float32)
            
            # Option 1: Use projected 3D center (if use_3d_center=True)
            if self.use_3d_center:
                center_3d_cam = obj.pos + [0, -obj.h / 2.0, 0] # Camera coords, bottom center adjusted to object center
                center_3d_img, _ = calib.rect_to_img(center_3d_cam.reshape(1, 3)) # Project to image plane
                center_3d_img = center_3d_img[0]
                # Apply flip adjustment if necessary (already done if obj.pos[0] was flipped)
                # Apply affine transformation to the projected 3D center
                center_proj = affine_transform(center_3d_img, trans)
                # Check if projected center is within bounds
                if not (0 <= center_proj[0] < self.input_resolution[0] and \
                        0 <= center_proj[1] < self.input_resolution[1]):
                    continue # Skip if projected 3D center is outside
                center_heatmap = center_proj / self.downsample

            # Option 2: Use 2D box center (if use_3d_center=False)
            else:
                center_proj = center_2d  # Use 2D center as proj center when not using 3D center
                center_heatmap = center_2d / self.downsample


            # 6. Generate Heatmap
            # Convert center coords to integer for heatmap indexing
            center_heatmap_int = center_heatmap.astype(np.int32)
            # Check bounds again after downsampling
            if not (0 <= center_heatmap_int[0] < self.output_resolution[0] and \
                    0 <= center_heatmap_int[1] < self.output_resolution[1]):
                continue # Skip if downsampled center is outside

            # Calculate adaptive gaussian radius (optional, from CenterNet)
            # h_heatmap, w_heatmap = bbox_h / self.downsample, bbox_w / self.downsample
            # radius = gaussian_radius((math.ceil(h_heatmap), math.ceil(w_heatmap)))
            # radius = max(0, int(radius))
            radius = 2 # Or use a fixed radius

            # Draw gaussian heatmap
            draw_umich_gaussian(heatmap[cls_id], center_heatmap_int, radius)


            # --- Encode Other Targets ---

            # 7. Store Class Label
            labels[obj_count] = cls_id
            
            # 8. Regression Offsets (deviation from discrete center on heatmap)
            offset_2d[obj_count] = center_2d / self.downsample - center_heatmap_int # Offset for 2D center
            offset_3d[obj_count] = center_proj / self.downsample - center_heatmap_int # Offset for projected 3D center

            # 9. Depth (Z in camera coords) - Adjust with crop scale if applicable
            obj_depth = obj.pos[2]
            if self.depth_scale == 'normal' and self.data_augmentation and self.aug_crop :
                obj_depth *= crop_scale
            elif self.depth_scale == 'inverse' and self.data_augmentation and self.aug_crop:
                 obj_depth /= crop_scale # Check if this logic is correct for inverse scaling
            depth[obj_count] = obj_depth

            # 10. Heading Angle (bin and residual)
            # Note: KITTI's alpha is observation angle, ry is orientation relative to camera
            # We use ry for heading bin/res calculation usually
            heading_angle = obj.ry # Use the (potentially flipped) ry
            heading_bin[obj_count], heading_res[obj_count] = angle2class(heading_angle) # Util function assumed

            # 11. Dimensions (Height, Width, Length)
            obj_dims = np.array([obj.h, obj.w, obj.l])
            src_size_3d[obj_count] = obj_dims # Store original size
            if self.meanshape:
                 size_3d[obj_count] = obj_dims - self.cls_mean_size[cls_id] # Store residual
            else:
                 size_3d[obj_count] = obj_dims # Store absolute dims if not using meanshape
            
            # Store calibration matrix for this object
            calibs[obj_count] = P2
            
            # Store occlusion level (if available)
            if hasattr(obj, 'occlusion'):
                occlusion[obj_count] = obj.occlusion
            else:
                occlusion[obj_count] = 0 # Default to no occlusion

            # Store 2D size on the output heatmap resolution
            size_2d[obj_count] = bbox_w / self.downsample, bbox_h / self.downsample

            # 12. Compute normalized 2D boxes and boxes_3d
            # Normalize centers and boxes by input resolution
            center_2d_norm = center_2d / self.input_resolution  # Normalize to [0, 1]
            center_proj_norm = center_proj / self.input_resolution  # Normalize to [0, 1]
            bbox_2d_norm = bbox_2d.copy()
            bbox_2d_norm[[0, 2]] = bbox_2d[[0, 2]] / self.input_resolution[0]  # Normalize x
            bbox_2d_norm[[1, 3]] = bbox_2d[[1, 3]] / self.input_resolution[1]  # Normalize y
            
            # 2D boxes: [center_x_norm, center_y_norm, width_norm, height_norm]
            size_2d_norm = np.array([bbox_w / self.input_resolution[0], bbox_h / self.input_resolution[1]])
            boxes[obj_count] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            
            # boxes_3d: [normalized_3d_center_x, normalized_3d_center_y, l, r, t, b]
            # Compute distances from projected 3D center to 2D box edges
            l = float(center_proj_norm[0] - bbox_2d_norm[0])  # left
            r = float(bbox_2d_norm[2] - center_proj_norm[0])  # right
            t = float(center_proj_norm[1] - bbox_2d_norm[1])  # top
            b = float(bbox_2d_norm[3] - center_proj_norm[1])  # bottom
            
            # Always ensure all values are positive (3D center should be inside or near 2D box)
            min_val = 1e-3  # Minimum valid value
            l = max(l, min_val)
            r = max(r, min_val)
            t = max(t, min_val)
            b = max(b, min_val)
            
            # Additional safety check: ensure resulting box is valid
            # Box coords will be: x1 = cx - l, x2 = cx + r, y1 = cy - t, y2 = cy + b
            # We need x2 > x1 (i.e., l + r > 0) and y2 > y1 (i.e., t + b > 0)
            if (l + r) <= 2 * min_val or (t + b) <= 2 * min_val:
                continue  # Skip objects with degenerate boxes
            
            # Final validation: ensure computed box coordinates are valid
            x1 = center_proj_norm[0] - l
            x2 = center_proj_norm[0] + r  
            y1 = center_proj_norm[1] - t
            y2 = center_proj_norm[1] + b
            if x2 <= x1 or y2 <= y1:
                continue  # Skip if box is still invalid
            
            boxes_3d[obj_count] = center_proj_norm[0], center_proj_norm[1], l, r, t, b

            # --- Store Index and Mask ---
            indices[obj_count] = center_heatmap_int[1] * self.output_resolution[0] + center_heatmap_int[0] # Flattened index
            mask_target[obj_count] = 1 # Mark this object as valid for loss calculation
            
            # Set mask_2d based on truncation and occlusion (for 2D detection quality)
            # In AI2-THOR, objects are typically not truncated/occluded much, so we can be lenient
            # Original KITTI uses: truncation <= 0.5 and occlusion <= 2
            if hasattr(obj, 'trucation') and hasattr(obj, 'occlusion'):
                if obj.trucation <= 0.5 and obj.occlusion <= 2:
                    mask_2d[obj_count] = 1
            else:
                # If truncation/occlusion not available, assume good quality
                mask_2d[obj_count] = 1

            obj_count += 1 # Increment count of valid objects processed


        # --- Collect Return Data ---
        targets = {
            'heatmap': heatmap,       # Target heatmap (C, H_out, W_out)
            'labels': labels,         # Class labels (max_objs,) - needed by matcher
            'boxes': boxes,           # 2D boxes: [cx_norm, cy_norm, w_norm, h_norm] (max_objs, 4)
            'boxes_3d': boxes_3d,     # 3D boxes: [cx_norm, cy_norm, l, r, t, b] (max_objs, 6)
            'offset_2d': offset_2d,   # 2D center offset (max_objs, 2)
            'size_2d': size_2d,       # 2D size on heatmap (max_objs, 2)
            'offset_3d': offset_3d,   # 3D center offset (max_objs, 2)
            'depth': depth,           # Depth (max_objs, 1)
            'size_3d': size_3d,       # 3D size residual/absolute (max_objs, 3)
            'heading_bin': heading_bin, # Heading angle bin (max_objs, 1)
            'heading_res': heading_res, # Heading angle residual (max_objs, 1)
            'indices': indices,       # Index of center on heatmap (max_objs,)
            'mask_target': mask_target, # Mask for valid objects (max_objs,)
            'mask_2d': mask_2d,       # Mask for 2D detection quality (max_objs,)
            'calib_P2': P2,           # Store P2 for potential use in model/loss
            'img_size': img_size,     # Original image size (needed by trainer)
        }

        # Return image, P2 (often needed by models), targets dict, and info dict
        return img, P2, targets, info


# --- Example Usage (for testing the dataset class) ---
if __name__ == '__main__':
    import sys
    # Add project root to sys.path if lib is not directly accessible
    # sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from torch.utils.data import DataLoader

    # --- Define Example Configuration (Mimic your YAML config) ---
    # !! MUST BE PROVIDED !!
    all_ai2thor_classes = [
        'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Book',
        'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'ButterKnife', 'Cabinet', 'Candle', 'CellPhone', 'Chair',
        'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'Desktop', 'DiningTable',
        'DishSponge', 'DogBed', 'Drawer', 'Dresser', 'Dumbbell', 'Egg', 'Faucet', 'FloorLamp', 'Footstool', 'Fork',
        'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain',
        'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce', 'Microwave', 'Mirror', 'Mug', 'Newspaper',
        'Ottoman', 'Painting', 'Pan', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate',
        'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'RoomDecor', 'Safe', 'SaltShaker', 'ScrubBrush',
        'Shelf', 'ShelvingUnit', 'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead', 'SideTable', 'Sink',
        'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'Stool',
        'StoveBurner', 'StoveKnob', 'TVStand', 'TableTopDecor', 'TeddyBear', 'Television', 'TennisRacket',
        'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato', 'Towel', 'TowelHolder',
        'Vase', 'Watch', 'WateringCan', 'WineBottle'
    ] # List all unique classes from your analysis script output (excluding DontCare)

    # Example: Load pre-calculated mean sizes (replace with your actual values)
    # These are illustrative, GET THEM FROM YOUR analysis_kitti_dataset.py output!
    example_mean_sizes = {
         'Mug': [0.101, 0.117, 0.100],
         'Laptop': [0.313, 0.469, 0.428],
         'Chair': [0.954, 0.589, 0.594],
         'Apple': [0.116, 0.117, 0.120],
         # ... Add ALL classes from all_ai2thor_classes with their means
    }

    # Example configuration dictionary
    cfg = {
        'root_dir': '/home/hice1/rrustagi7/scratch/final_ai2_kitti', # <<< YOUR KITTI dataset path
        'class_names': all_ai2thor_classes,
        'input_resolution': [1280, 384], # Match your data generation image size for simplicity first
        'output_downsample': 4, # Example downsampling for heatmap (adjust based on model)
        'max_objs': 50,
        'meanshape': True, # Use mean shape subtraction? Set to False if not providing sizes
        'class_mean_sizes': example_mean_sizes, # Provide calculated means if meanshape=True
        'writelist': all_ai2thor_classes, # Process all defined classes
        'use_dontcare': True,
        # Augmentation settings (can tune these)
        'aug_pd': True,
        'aug_crop': True,
        'random_flip': 0.5,
        'random_crop': 0.5,
        'scale': 0.2,
        'shift': 0.1,
        'use_3d_center': True, # Or False
    }

    # Create dataset instance (e.g., for training split)
    # Ensure you have 'train.txt' in '/home/rahul/gt/fa25/glen/kitti_format_dataset/ImageSets/'
    try:
        dataset = AI2THOR_KITTI_Dataset('train', cfg) # Or 'val'
        print(f"\nDataset created successfully. Number of samples: {len(dataset)}")

        # Test loading one sample
        if len(dataset) > 0:
            print("\nAttempting to load first sample...")
            inputs, P2, targets, info = dataset[0]
            print("Sample loaded successfully!")
            print("  Input image shape:", inputs.shape)
            print("  P2 shape:", P2.shape)
            print("  Targets keys:", targets.keys())
            print("  Info:", info)
            # You can add more checks here, e.g., print target shapes
            # print("  Target heatmap shape:", targets['heatmap'].shape)
            # print("  Valid object mask:", targets['mask_target'])
        else:
            print("Dataset is empty, cannot load a sample.")

    except FileNotFoundError as e:
        print(f"\nError creating dataset: {e}")
        print("Please ensure the root directory is correct and the 'ImageSets/train.txt' file exists.")
    except ValueError as e:
         print(f"\nError creating dataset: {e}")
         print("Please check the configuration values provided in 'cfg'.")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()