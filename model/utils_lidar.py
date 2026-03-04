"""
Enhanced DataLoader for PoLaRIS Dataset with:
1. 16-bit Infrared Image Support
2. LiDAR Point Cloud Loading
3. Soft Label Support (Oracle Masks with float values)
"""

from PIL import Image, ImageOps, ImageFilter
import os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch

def get_img_ids_from_dir(images_dir, suffix='.png'):
    """
    Auto-scan images directory and return list of image IDs (filenames without suffix)
    
    Args:
        images_dir: Path to images directory
        suffix: Image file suffix (default: '.png')
    
    Returns:
        List of image IDs sorted alphabetically
    """
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    img_files = [f for f in os.listdir(images_dir) if f.endswith(suffix)]
    img_ids = [f.replace(suffix, '') for f in img_files]
    return sorted(img_ids)


def bin_to_depth_image(lidar_points, img_size):
    """
    Project pre-aligned lidar_roi .bin points to a 2D depth image.

    Fallback path when depth_maps/*.npy does not exist.
    'lidar_roi' points are projected onto the camera image plane beforehand
    (using calibration), so (x, y) are already pixel coordinates.

    Filtering (mirrors data_tools.py filter_lidar_points):
      - Remove NaN / Inf values (fixes RuntimeWarning on cast)
      - Keep points with positive depth (z > 0)
      - Keep points with sufficient intensity (intensity > 5.0)

    Args:
        lidar_points: (N, 4) numpy array  [x=col, y=row, z=depth, intensity]
                      Empty array → all-zero image.
        img_size:     PIL image size tuple (width, height)

    Returns:
        PIL Image mode 'F' (float32), depth in original z units (meters).
    """
    W, H = img_size
    depth_map = np.zeros((H, W), dtype=np.float32)

    if len(lidar_points) == 0:
        return Image.fromarray(depth_map, mode='F')

    # Step 1: Remove NaN / Inf (root cause of RuntimeWarning)
    finite_mask = np.isfinite(lidar_points).all(axis=1)
    lidar_points = lidar_points[finite_mask]

    if len(lidar_points) == 0:
        return Image.fromarray(depth_map, mode='F')

    # Step 2: Intensity filter (mirrors data_tools.py min_intensity=5.0)
    intensity = lidar_points[:, 3]
    int_mask = intensity > 5.0
    lidar_points = lidar_points[int_mask]

    if len(lidar_points) == 0:
        return Image.fromarray(depth_map, mode='F')

    cols   = np.round(lidar_points[:, 0]).astype(np.int32)
    rows   = np.round(lidar_points[:, 1]).astype(np.int32)
    depths = lidar_points[:, 2].astype(np.float32)

    # Step 3: Keep in-bounds, positive-depth points
    valid = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H) & (depths > 0)
    cols, rows, depths = cols[valid], rows[valid], depths[valid]

    if len(cols) > 0:
        # MaxPool semantics: keep maximum depth for overlapping pixels
        # (consistent with LiDARDownsampler design choice)
        np.maximum.at(depth_map, (rows, cols), depths)

    return Image.fromarray(depth_map, mode='F')

class PoLaRISTrainLoader(Dataset):
    """
    Training Dataset Loader for PoLaRIS with LiDAR fusion
    
    Key Features:
    - 16-bit infrared image support with proper normalization
    - LiDAR point cloud loading (.bin files)
    - Soft Label oracle masks (preserves float values 0.0-1.0)
    - Data augmentation (flip, scale, crop, blur)
    """
    
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id=None, base_size=512, crop_size=480,
                 transform=None, suffix='.png', normalize_16bit=True, normalize_mode='minmax',
                 in_channels=1, image_folder='images', oracle_masks_folder='oracle_masks',
                 depth_maps_dir=None):
        """
        Args:
            normalize_16bit: bool or str, whether to normalize 16-bit images (True/False or 'True'/'False')
            normalize_mode: str, normalization strategy for 16-bit images:
                - 'minmax': Min-Max normalization (default, adaptive per image)
                - 'global': Global scaling with fixed range
                - 'percentile': Percentile-based clipping (reduces outlier impact)
                - 'clahe': CLAHE (Contrast Limited Adaptive Histogram Equalization)
            depth_maps_dir: optional override path for pre-projected depth maps (.npy).
                            If None, defaults to {dataset_dir}/depth_maps.
        """
        super(PoLaRISTrainLoader, self).__init__()

        self.transform = transform
        self.images = os.path.join(dataset_dir, image_folder)

        # Auto-scan images directory if img_id not provided
        if img_id is None:
            img_id = get_img_ids_from_dir(self.images, suffix)
            print(f"[PoLaRISTrainLoader] Auto-scanned {len(img_id)} images from {self.images}")

        # Normalize all item ids to strings to avoid type errors when building paths
        self._items = [str(i) for i in img_id]
        self.masks = os.path.join(dataset_dir, 'masks')
        self.lidar_roi = os.path.join(dataset_dir, 'lidar_roi')
        self.oracle_masks = os.path.join(dataset_dir, oracle_masks_folder)
        # Allow external depth_maps directory (e.g. from a different dataset folder)
        self.depth_maps = depth_maps_dir if depth_maps_dir is not None else os.path.join(dataset_dir, 'depth_maps')
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

        # Handle normalize_16bit as bool or string
        if isinstance(normalize_16bit, str):
            self.normalize_16bit = (normalize_16bit.lower() == 'true')
        else:
            self.normalize_16bit = normalize_16bit

        self.normalize_mode = normalize_mode
        self.in_channels = in_channels

        # [NEW 2026-02-20] Load category mapping for scene-weighted loss
        self.category_map = {}
        category_file = os.path.join(dataset_dir, 'selection_summary_new.txt')
        if not os.path.exists(category_file):
            category_file = os.path.join(dataset_dir, 'select-view', 'selection_summary_new.txt')

        if os.path.exists(category_file):
            with open(category_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    if len(parts) >= 2:
                        img_name = parts[0].strip().replace('.png', '')
                        category = int(parts[1].strip())
                        self.category_map[img_name] = category
            print(f"  - Loaded {len(self.category_map)} category mappings from {category_file}")
        else:
            print(f"  ⚠️  Category file not found: {category_file}")
            print(f"  → All samples will default to Cat0 (unclassified)")

        print(f"[PoLaRISTrainLoader] Initialized with {len(self._items)} samples")
        print(f"  - Images: {self.images}")
        print(f"  - LiDAR: {self.lidar_roi}")
        print(f"  - Oracle Masks: {self.oracle_masks}")
        print(f"  - Depth Maps: {self.depth_maps}")
        print(f"  - 16-bit normalization: {self.normalize_16bit}")
        print(f"  - Normalization mode: {self.normalize_mode}")
        print(f"  - Input channels: {in_channels}")

    def _load_image(self, img_path):
        """
        Load infrared image with 16-bit support
        
        Returns:
            img: PIL Image
            is_16bit: bool flag
        """
        img = Image.open(img_path)
        
        # Check if 16-bit
        is_16bit = False
        if img.mode == 'I;16':
            is_16bit = True
            # Convert to float32 array for processing
            img_array = np.array(img, dtype=np.float32)
            
            if self.normalize_16bit:
                if self.normalize_mode == 'minmax':
                    # Min-Max normalization (adaptive per image)
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_array = (img_array - img_min) / (img_max - img_min) * 255.0
                    else:
                        img_array = np.zeros_like(img_array)
                
                elif self.normalize_mode == 'global':
                    # Global scaling with fixed range
                    img_array = img_array / 65535.0 * 255.0
                
                elif self.normalize_mode == 'percentile':
                    # Percentile-based clipping (reduces outlier impact)
                    p_low = np.percentile(img_array, 1)
                    p_high = np.percentile(img_array, 99)
                    img_array = np.clip(img_array, p_low, p_high)
                    if p_high > p_low:
                        img_array = (img_array - p_low) / (p_high - p_low) * 255.0
                    else:
                        img_array = np.zeros_like(img_array)
                
                elif self.normalize_mode == 'clahe':
                    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    # First normalize to 0-255 uint8
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_norm = ((img_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
                    else:
                        img_norm = np.zeros_like(img_array, dtype=np.uint8)
                    # Apply CLAHE
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_array = clahe.apply(img_norm).astype(np.float32)
                
                else:
                    # Fallback to minmax
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_array = (img_array - img_min) / (img_max - img_min) * 255.0
                    else:
                        img_array = np.zeros_like(img_array)
            else:
                # Simple scaling (may lose detail in dark regions)
                img_array = img_array / 65535.0 * 255.0
            
            # Convert back to uint8 PIL Image for augmentation
            img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        else:
            # 8-bit image, convert to grayscale
            img = img.convert('L')
        
        return img, is_16bit

    def _load_lidar(self, lidar_path):
        """
        Load LiDAR point cloud from .bin file

        Returns:
            points: numpy array (N, 4) [x, y, z, intensity]
                    Returns zeros if file not found
        """
        if os.path.exists(lidar_path):
            try:
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
                return points
            except Exception as e:
                print(f"Warning: Failed to load LiDAR {lidar_path}: {e}")
                return np.zeros((0, 4), dtype=np.float32)
        else:
            # Return empty array if no LiDAR data
            return np.zeros((0, 4), dtype=np.float32)

    def _load_depth(self, depth_path):
        """
        Load depth map from .npy file

        Returns:
            depth: PIL Image in 'F' mode (float32)
        """
        if os.path.exists(depth_path):
            try:
                depth_arr = np.load(depth_path)
                return Image.fromarray(depth_arr, mode='F')
            except Exception as e:
                print(f"Warning: Failed to load depth map {depth_path}: {e}")
                # Return zeros with default size (will be resized later)
                return None
        else:
            return None

    def _sync_transform(self, img, mask, oracle_mask, depth=None, lidar_points=None):
        """
        Synchronized data augmentation

        Note on LiDAR flipping:
        - LiDAR points in this implementation are NOT flipped because they're
          already converted to depth maps (stored in depth_maps/*.npy)
        - If you need to augment raw LiDAR points, you would need to:
          1. Check if points are in image coordinates (x, y, z)
          2. Flip x coordinate: x_flipped = image_width - x
        - Current implementation uses pre-generated depth maps, so LiDAR
          points (.bin files) are only loaded for reference, not for training
        """
        # Random horizontal flip
        flip_flag = random.random() < 0.5
        if flip_flag:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            oracle_mask = oracle_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if depth is not None:
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

            # LiDAR points: Currently not used for training (depth maps are used instead)
            # If you need to flip LiDAR points in image coordinates, uncomment:
            # if lidar_points is not None and len(lidar_points) > 0:
            #     img_width = img.size[0]
            #     lidar_points[:, 0] = img_width - lidar_points[:, 0]  # Flip x coordinate
        
        crop_size = self.crop_size
        
        # Random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        oracle_mask = oracle_mask.resize((ow, oh), Image.BILINEAR)  # Use BILINEAR to preserve gradients
        if depth is not None:
            depth = depth.resize((ow, oh), Image.BILINEAR)

        # Pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            oracle_mask = ImageOps.expand(oracle_mask, border=(0, 0, padw, padh), fill=0)
            if depth is not None:
                # Use paste method for float images (mode 'F')
                new_depth = Image.new('F', (ow + padw, oh + padh), 0.0)
                new_depth.paste(depth, (0, 0))
                depth = new_depth

        # Random crop
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        oracle_mask = oracle_mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if depth is not None:
            depth = depth.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # Gaussian blur (only on image, not masks or depth)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # Convert to numpy arrays
        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        oracle_mask = np.array(oracle_mask, dtype=np.float32)
        if depth is not None:
            depth = np.array(depth, dtype=np.float32)

        return img, mask, oracle_mask, depth, lidar_points

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = os.path.join(self.images, img_id + self.suffix)
        label_path = os.path.join(self.masks, img_id + self.suffix)
        oracle_path = os.path.join(self.oracle_masks, img_id + self.suffix)
        lidar_path = os.path.join(self.lidar_roi, img_id + '.bin')
        depth_path = os.path.join(self.depth_maps, img_id + '.npy')

        # Load infrared image (with 16-bit support)
        img, is_16bit = self._load_image(img_path)

        # Load depth map if in_channels == 2
        depth = None
        if self.in_channels == 2:
            depth = self._load_depth(depth_path)
            if depth is None:
                # Fallback: project lidar_roi .bin → 2D depth image
                # Triggered when depth_maps/*.npy absent (e.g. Pohang-Canal-3k)
                lidar_for_depth = self._load_lidar(lidar_path)
                depth = bin_to_depth_image(lidar_for_depth, img.size)
                if np.max(np.array(depth)) == 0:
                    # No valid points either → genuine zero
                    depth = Image.new('F', img.size, 0.0)

        # Load GT mask
        mask = Image.open(label_path).convert('L')

        # Load Oracle Mask (Soft Label)
        if os.path.exists(oracle_path):
            oracle_mask = Image.open(oracle_path).convert('L')
        else:
            # Fallback to zeros if not generated yet
            oracle_mask = Image.new('L', mask.size, 0)
            print(f"Warning: Oracle mask not found for {img_id}, using zeros")

        # Load LiDAR point cloud
        lidar_points = self._load_lidar(lidar_path)

        # Synchronized augmentation
        img, mask, oracle_mask, depth, lidar_points = self._sync_transform(
            img, mask, oracle_mask, depth, lidar_points
        )

        # Normalize and convert to tensor
        if self.in_channels == 2:
            # 2-channel mode: stack IR + depth
            img = img / 255.0  # Normalize IR to [0, 1]
            # [FIX 2026-03-04] Pohang-Canal lidar_roi z实测范围 0~911m，
            # 原 /80.0 会导致归一化值远超[0,1]，改为 /1000.0 覆盖常见海面探测距离
            depth = depth / 1000.0  # Normalize depth (max ~1000m for maritime LiDAR)
            combined = np.stack([img, depth], axis=0)  # (2, H, W)
            img_tensor = torch.from_numpy(combined).float()
        elif self.in_channels == 3:
            # 3-channel mode: pseudo-RGB (灰度复制3次，匹配DNANet)
            img = img / 255.0  # Normalize to [0, 1]
            # 将单通道灰度图复制成3通道
            combined = np.stack([img, img, img], axis=0)  # (3, H, W)
            img_tensor = torch.from_numpy(combined).float()
        else:
            # Single channel mode (in_channels=1)
            img = img / 255.0
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, H, W)

        # Normalize masks to [0, 1] and preserve float values
        # CRITICAL: Do NOT binarize oracle_mask - preserve soft labels!
        mask_hard = mask / 255.0  # Binary GT: {0.0, 1.0}
        mask_soft = oracle_mask / 255.0  # Soft: [0.0, 1.0], e.g., 0.6 for no-LiDAR targets

        # Add channel dimension
        mask_hard = np.expand_dims(mask_hard, axis=0)  # (1, H, W)
        mask_soft = np.expand_dims(mask_soft, axis=0)  # (1, H, W)

        # Convert LiDAR to tensor
        lidar_tensor = torch.from_numpy(lidar_points).float()  # (N, 4)

        # [NEW 2026-02-20] Get category for scene-weighted loss
        category = self.category_map.get(img_id, 0)  # Default to Cat0 if not found

        return {
            'image': img_tensor,  # (1, H, W) or (2, H, W) depending on in_channels
            'mask': torch.from_numpy(mask_hard).float(),  # (1, H, W) - Hard GT for evaluation
            'mask_hard': torch.from_numpy(mask_hard).float(),  # (1, H, W) - Hard Label for seg loss
            'mask_soft': torch.from_numpy(mask_soft).float(),  # (1, H, W) - Soft Label for conf loss
            'oracle_mask': torch.from_numpy(mask_soft).float(),  # (1, H, W) - Backward compatibility
            'lidar': lidar_tensor,  # (N, 4)
            'img_id': img_id,
            'is_16bit': is_16bit,
            'category': category  # [NEW] Scene category for weighted loss
        }

    def __len__(self):
        return len(self._items)


class PoLaRISTestLoader(Dataset):
    """
    Test Dataset Loader for PoLaRIS with LiDAR fusion
    """
    
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id=None, base_size=512, crop_size=480,
                 transform=None, suffix='.png', normalize_16bit=True, normalize_mode='minmax',
                 in_channels=1, image_folder='images', depth_maps_dir=None):
        super(PoLaRISTestLoader, self).__init__()

        self.transform = transform
        self.images = os.path.join(dataset_dir, image_folder)

        # Auto-scan images directory if img_id not provided
        if img_id is None:
            img_id = get_img_ids_from_dir(self.images, suffix)
            print(f"[PoLaRISTestLoader] Auto-scanned {len(img_id)} images from {self.images}")

        # Normalize all item ids to strings to avoid type errors when building paths
        self._items = [str(i) for i in img_id]
        self.masks = os.path.join(dataset_dir, 'masks')
        self.lidar_roi = os.path.join(dataset_dir, 'lidar_roi')
        # Allow external depth_maps directory (e.g. from a different dataset folder)
        self.depth_maps = depth_maps_dir if depth_maps_dir is not None else os.path.join(dataset_dir, 'depth_maps')
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

        # Handle normalize_16bit as bool or string
        if isinstance(normalize_16bit, str):
            self.normalize_16bit = (normalize_16bit.lower() == 'true')
        else:
            self.normalize_16bit = normalize_16bit

        self.normalize_mode = normalize_mode
        self.in_channels = in_channels

        # [NEW 2026-02-20] Load category mapping for per-category evaluation
        self.category_map = {}
        category_file = os.path.join(dataset_dir, 'selection_summary_new.txt')
        if not os.path.exists(category_file):
            category_file = os.path.join(dataset_dir, 'select-view', 'selection_summary_new.txt')

        if os.path.exists(category_file):
            with open(category_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    if len(parts) >= 2:
                        img_name = parts[0].strip().replace('.png', '')
                        category = int(parts[1].strip())
                        self.category_map[img_name] = category
            print(f"  - Loaded {len(self.category_map)} category mappings")
        else:
            print(f"  ⚠️  Category file not found, per-category metrics unavailable")

        print(f"[PoLaRISTestLoader] Initialized with {len(self._items)} samples")
        print(f"  - Input channels: {in_channels}")
        print(f"  - Normalization mode: {normalize_mode}")

    def _load_image(self, img_path):
        """Load infrared image with 16-bit support"""
        img = Image.open(img_path)
        
        is_16bit = False
        if img.mode == 'I;16':
            is_16bit = True
            img_array = np.array(img, dtype=np.float32)
            
            if self.normalize_16bit:
                if self.normalize_mode == 'minmax':
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_array = (img_array - img_min) / (img_max - img_min) * 255.0
                    else:
                        img_array = np.zeros_like(img_array)
                
                elif self.normalize_mode == 'global':
                    img_array = img_array / 65535.0 * 255.0
                
                elif self.normalize_mode == 'percentile':
                    p_low = np.percentile(img_array, 1)
                    p_high = np.percentile(img_array, 99)
                    img_array = np.clip(img_array, p_low, p_high)
                    if p_high > p_low:
                        img_array = (img_array - p_low) / (p_high - p_low) * 255.0
                    else:
                        img_array = np.zeros_like(img_array)
                
                elif self.normalize_mode == 'clahe':
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_norm = ((img_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
                    else:
                        img_norm = np.zeros_like(img_array, dtype=np.uint8)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_array = clahe.apply(img_norm).astype(np.float32)
                
                else:
                    # Fallback to minmax
                    img_min = img_array.min()
                    img_max = img_array.max()
                    if img_max > img_min:
                        img_array = (img_array - img_min) / (img_max - img_min) * 255.0
                    else:
                        img_array = np.zeros_like(img_array)
            else:
                img_array = img_array / 65535.0 * 255.0
            
            img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        else:
            img = img.convert('L')
        
        return img, is_16bit

    def _load_lidar(self, lidar_path):
        """Load LiDAR point cloud"""
        if os.path.exists(lidar_path):
            try:
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
                return points
            except:
                return np.zeros((0, 4), dtype=np.float32)
        else:
            return np.zeros((0, 4), dtype=np.float32)

    def _load_depth(self, depth_path):
        """Load depth map from .npy file"""
        if os.path.exists(depth_path):
            try:
                depth_arr = np.load(depth_path)
                return Image.fromarray(depth_arr, mode='F')
            except:
                return None
        else:
            return None

    def _testval_sync_transform(self, img, mask, depth=None):
        """Resize to base_size for testing"""
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        if depth is not None:
            depth = depth.resize((base_size, base_size), Image.BILINEAR)

        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        if depth is not None:
            depth = np.array(depth, dtype=np.float32)

        return img, mask, depth

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = os.path.join(self.images, img_id + self.suffix)
        label_path = os.path.join(self.masks, img_id + self.suffix)
        lidar_path = os.path.join(self.lidar_roi, img_id + '.bin')
        depth_path = os.path.join(self.depth_maps, img_id + '.npy')

        # Load data
        img, is_16bit = self._load_image(img_path)
        mask = Image.open(label_path).convert('L')
        lidar_points = self._load_lidar(lidar_path)

        # Load depth map if in_channels == 2
        depth = None
        if self.in_channels == 2:
            depth = self._load_depth(depth_path)
            if depth is None:
                # Fallback: project lidar_roi .bin → 2D depth image
                lidar_for_depth = self._load_lidar(lidar_path)
                depth = bin_to_depth_image(lidar_for_depth, img.size)
                if np.max(np.array(depth)) == 0:
                    depth = Image.new('F', img.size, 0.0)

        # Transform
        img, mask, depth = self._testval_sync_transform(img, mask, depth)

        # Normalize and convert to tensor
        if self.in_channels == 2:
            # 2-channel mode: stack IR + depth
            img = img / 255.0
            depth = depth / 1000.0  # [FIX 2026-03-04] maritime LiDAR max ~1000m
            combined = np.stack([img, depth], axis=0)
            img_tensor = torch.from_numpy(combined).float()
        elif self.in_channels == 3:
            # 3-channel mode: pseudo-RGB (灰度复制3次，匹配DNANet)
            img = img / 255.0
            combined = np.stack([img, img, img], axis=0)  # (3, H, W)
            img_tensor = torch.from_numpy(combined).float()
        else:
            # Single channel mode (in_channels=1)
            img = img / 255.0
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                img_tensor = torch.from_numpy(img).unsqueeze(0).float()

        mask = np.expand_dims(mask / 255.0, axis=0)
        lidar_tensor = torch.from_numpy(lidar_points).float()

        # [NEW 2026-02-20] Get category for per-category evaluation
        category = self.category_map.get(img_id, 0)  # Default to Cat0 if not found

        return {
            'image': img_tensor,
            'mask': torch.from_numpy(mask).float(),
            'lidar': lidar_tensor,
            'img_id': img_id,
            'is_16bit': is_16bit,
            'category': category  # [NEW] Scene category for per-category metrics
        }

    def __len__(self):
        return len(self._items)


# ==========================================
# Loss Functions for Soft Labels
# ==========================================

class SoftIoULoss(torch.nn.Module):
    """
    Soft IoU Loss that supports float targets (soft labels)
    
    Works with:
    - Binary targets: {0, 1}
    - Soft targets: [0.0, 1.0], e.g., 0.6 for uncertain regions
    """
    def __init__(self, smooth=1e-6):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - predicted logits or probabilities
            target: (B, 1, H, W) - soft labels [0.0, 1.0]
        """
        # Apply sigmoid if pred is logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Soft IoU
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class SoftBCELoss(torch.nn.Module):
    """
    Binary Cross Entropy Loss with support for soft labels
    """
    def __init__(self, pos_weight=None):
        super(SoftBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - predicted logits
            target: (B, 1, H, W) - soft labels [0.0, 1.0]
        """
        return self.bce(pred, target)


class CombinedSoftLoss(torch.nn.Module):
    """
    Combined loss: BCE + IoU for better convergence
    """
    def __init__(self, bce_weight=0.5, iou_weight=0.5, pos_weight=None):
        super(CombinedSoftLoss, self).__init__()
        self.bce_loss = SoftBCELoss(pos_weight=pos_weight)
        self.iou_loss = SoftIoULoss()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)

        return self.bce_weight * bce + self.iou_weight * iou


def polaris_collate_fn(batch):
    """
    Custom collate function for PoLaRIS DataLoader.

    Handles variable-length LiDAR point clouds by keeping them as a list
    instead of trying to stack them into a single tensor.

    Works for both training (with oracle_mask) and testing (without oracle_mask).

    Args:
        batch: List of dicts from __getitem__

    Returns:
        Collated batch with:
        - image: (B, C, H, W) - stacked tensor
        - mask: (B, 1, H, W) - stacked tensor
        - oracle_mask: (B, 1, H, W) - stacked tensor (only for training)
        - lidar: List[Tensor] - variable-length point clouds
        - img_id: List[str] - image IDs
        - is_16bit: List[bool] - 16-bit flags
        - category: List[int] - scene categories (Cat0/1/2/3)
    """
    # Stack fixed-size tensors
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])

    # Keep variable-size data as lists
    lidar = [item['lidar'] for item in batch]
    img_id = [item['img_id'] for item in batch]
    is_16bit = [item['is_16bit'] for item in batch]

    # [NEW 2026-02-20] Add category support for scene-weighted loss
    category = [item.get('category', 0) for item in batch]  # Default to Cat0 if not present

    # Build return dict
    result = {
        'image': images,
        'mask': masks,
        'lidar': lidar,
        'img_id': img_id,
        'is_16bit': is_16bit,
        'category': category  # [NEW] Scene categories
    }

    # Add optional fields if present (training mode)
    if 'oracle_mask' in batch[0]:
        oracle_masks = torch.stack([item['oracle_mask'] for item in batch])
        result['oracle_mask'] = oracle_masks

    # Add dual-supervision labels if present (for training with hard+soft labels)
    if 'mask_hard' in batch[0]:
        mask_hard = torch.stack([item['mask_hard'] for item in batch])
        result['mask_hard'] = mask_hard

    if 'mask_soft' in batch[0]:
        mask_soft = torch.stack([item['mask_soft'] for item in batch])
        result['mask_soft'] = mask_soft

    return result
