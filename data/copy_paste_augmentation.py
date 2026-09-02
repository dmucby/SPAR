"""Copy-paste augmentation for dynamic-region supervision."""

import os
import json
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from pycocotools.coco import COCO
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


COCO_DYNAMIC_CATEGORIES = {
    # People
    'person': 1,
    # Animals
    'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20,
    'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25,
    # Vehicles
    'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6,
    'train': 7, 'truck': 8, 'boat': 9,
}


class ObjectLibrary:
    """Load segmented objects from RGBA files or COCO annotations."""
    
    def __init__(
        self,
        library_path: Optional[str] = None,
        coco_annotation_path: Optional[str] = None,
        coco_image_dir: Optional[str] = None,
        max_objects: int = 5000,
        min_object_size: int = 64,
    ):
        self.objects: List[Dict] = []  # List of {'image': np.ndarray [H,W,3], 'mask': np.ndarray [H,W]}
        self.max_objects = max_objects
        self.min_object_size = min_object_size
        
        if library_path and os.path.isdir(library_path):
            self._load_from_directory(library_path)
        elif coco_annotation_path and coco_image_dir:
            self._load_from_coco(coco_annotation_path, coco_image_dir)
        else:
            print("[CopyPaste] No object library found. Using synthetic rectangles as fallback.")
            self._generate_synthetic_objects()
    
    def _load_from_directory(self, library_path: str):
        lib_path = Path(library_path)
        files = sorted(lib_path.glob("*.png"))[:self.max_objects]
        
        for f in files:
            try:
                img = np.array(Image.open(f).convert("RGBA"))
                if img.shape[0] < self.min_object_size or img.shape[1] < self.min_object_size:
                    continue
                rgb = img[:, :, :3]
                mask = (img[:, :, 3] > 128).astype(np.float32)
                self.objects.append({'image': rgb, 'mask': mask})
            except Exception:
                continue
        
        print(f"[CopyPaste] Loaded {len(self.objects)} objects from {library_path}")
    
    def _load_from_coco(self, annotation_path: str, image_dir: str):
        if not HAS_COCO:
            print("[CopyPaste] pycocotools not found, using synthetic objects.")
            self._generate_synthetic_objects()
            return
        
        coco = COCO(annotation_path)
        target_cat_ids = list(COCO_DYNAMIC_CATEGORIES.values())
        
        img_ids = set()
        for cat_id in target_cat_ids:
            img_ids.update(coco.getImgIds(catIds=[cat_id]))
        img_ids = list(img_ids)[:self.max_objects * 2]
        
        for img_id in img_ids:
            if len(self.objects) >= self.max_objects:
                break
            
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
            
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=target_cat_ids)
            anns = coco.loadAnns(ann_ids)
            
            for ann in anns:
                if len(self.objects) >= self.max_objects:
                    break
                if ann.get('iscrowd', 0):
                    continue
                
                mask = coco.annToMask(ann)
                bbox = ann['bbox']  # [x, y, w, h]
                x, y, w, h = [int(v) for v in bbox]
                
                if w < self.min_object_size or h < self.min_object_size:
                    continue
                
                # Crop object
                obj_img = img[y:y+h, x:x+w].copy()
                obj_mask = mask[y:y+h, x:x+w].astype(np.float32)
                
                self.objects.append({'image': obj_img, 'mask': obj_mask})
        
        print(f"[CopyPaste] Extracted {len(self.objects)} objects from COCO.")
    
    def _generate_synthetic_objects(self, n=200):
        for _ in range(n):
            h = random.randint(64, 256)
            w = random.randint(64, 256)
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            obj_img = np.ones((h, w, 3), dtype=np.uint8) * color
            
            mask = np.zeros((h, w), dtype=np.float32)
            if HAS_CV2:
                cv2.ellipse(
                    mask,
                    center=(w // 2, h // 2),
                    axes=(w // 2 - 5, h // 2 - 5),
                    angle=0, startAngle=0, endAngle=360,
                    color=1.0, thickness=-1
                )
            else:
                yy, xx = np.ogrid[:h, :w]
                mask[((xx - w/2)**2 / (w/2)**2 + (yy - h/2)**2 / (h/2)**2) <= 1] = 1.0
            
            self.objects.append({'image': obj_img, 'mask': mask})
        
        print(f"[CopyPaste] Generated {len(self.objects)} synthetic objects (fallback).")
    
    def sample(self, n: int = 1) -> List[Dict]:
        if len(self.objects) == 0:
            return []
        return random.choices(self.objects, k=n)
    
    def __len__(self):
        return len(self.objects)


class CopyPasteAugmentor:
    """Paste foreground objects across views and return binary supervision masks."""
    
    def __init__(self, config):
        self.config = config
        cp_config = config.training.get('copy_paste', {})
        
        self.num_objects_range = cp_config.get('num_objects_range', [1, 2])
        self.size_ratio_range = cp_config.get('size_ratio_range', [0.25, 0.35])
        self.margin_ratio = cp_config.get('margin_ratio', 0.15)
        self.gaussian_sigma = cp_config.get('gaussian_sigma', 3)
        self.same_object_prob = cp_config.get('same_object_prob', 0.20)
        self.jitter_ratio = cp_config.get('jitter_ratio', 0.05)
        
        self.object_library = ObjectLibrary(
            library_path=cp_config.get('object_library_path', None),
            coco_annotation_path=cp_config.get('coco_annotation_path', None),
            coco_image_dir=cp_config.get('coco_image_dir', None),
            max_objects=cp_config.get('max_objects', 5000),
        )
    
    def _resize_object(
        self,
        obj_img: np.ndarray,
        obj_mask: np.ndarray,
        target_h: int,
        target_w: int,
        size_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img_area = target_h * target_w
        target_area = img_area * size_ratio
        
        obj_h, obj_w = obj_img.shape[:2]
        obj_aspect = obj_w / obj_h
        
        new_h = int(np.sqrt(target_area / obj_aspect))
        new_w = int(new_h * obj_aspect)
        
        margin_h = int(target_h * self.margin_ratio)
        margin_w = int(target_w * self.margin_ratio)
        max_h = target_h - 2 * margin_h
        max_w = target_w - 2 * margin_w
        
        if new_h > max_h or new_w > max_w:
            scale = min(max_h / new_h, max_w / new_w)
            new_h = int(new_h * scale)
            new_w = int(new_w * scale)
        
        new_h = max(new_h, 8)
        new_w = max(new_w, 8)
        
        resized_img = np.array(
            Image.fromarray(obj_img).resize((new_w, new_h), Image.LANCZOS)
        )
        resized_mask = np.array(
            Image.fromarray((obj_mask * 255).astype(np.uint8)).resize(
                (new_w, new_h), Image.LANCZOS
            )
        ).astype(np.float32) / 255.0
        
        return resized_img, resized_mask
    
    def _get_random_position(
        self,
        obj_h: int,
        obj_w: int,
        img_h: int,
        img_w: int,
    ) -> Tuple[int, int]:
        margin_h = int(img_h * self.margin_ratio)
        margin_w = int(img_w * self.margin_ratio)
        
        max_y = img_h - obj_h - margin_h
        max_x = img_w - obj_w - margin_w
        
        y = random.randint(max(margin_h, 0), max(max_y, margin_h))
        x = random.randint(max(margin_w, 0), max(max_x, margin_w))
        
        return y, x
    
    def _gaussian_blend_mask(self, mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        """Smooth object boundaries before alpha compositing."""
        if not HAS_CV2:
            return mask
        
        ksize = int(sigma * 6) | 1
        blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigma)
        return blurred
    
    def _paste_object_on_image(
        self,
        scene_img: np.ndarray,
        obj_img: np.ndarray,
        obj_mask: np.ndarray,
        position: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Composite an object and return the image and binary paste mask."""
        y, x = position
        obj_h, obj_w = obj_img.shape[:2]
        img_h, img_w = scene_img.shape[:2]
        
        paste_h = min(obj_h, img_h - y)
        paste_w = min(obj_w, img_w - x)
        
        if paste_h <= 0 or paste_w <= 0:
            return scene_img.copy(), np.zeros((img_h, img_w), dtype=np.float32)
        
        obj_cropped = obj_img[:paste_h, :paste_w]
        mask_cropped = obj_mask[:paste_h, :paste_w]
        
        # Gaussian smoothing on mask boundary
        mask_blended = self._gaussian_blend_mask(mask_cropped, sigma=self.gaussian_sigma)
        mask_blended = mask_blended[:, :, np.newaxis] if mask_blended.ndim == 2 else mask_blended
        
        result = scene_img.copy()
        region = result[y:y+paste_h, x:x+paste_w]
        obj_float = obj_cropped.astype(np.float32) / 255.0 if obj_cropped.max() > 1 else obj_cropped.astype(np.float32)
        region_float = region.astype(np.float32) if region.max() <= 1 else region.astype(np.float32) / 255.0
        
        blended = region_float * (1.0 - mask_blended) + obj_float * mask_blended
        result[y:y+paste_h, x:x+paste_w] = blended
        
        paste_mask = np.zeros((img_h, img_w), dtype=np.float32)
        paste_mask[y:y+paste_h, x:x+paste_w] = (mask_cropped > 0.5).astype(np.float32)
        
        return result, paste_mask
    
    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
        input_idx: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Augment a multi-view batch and return per-view paste masks."""
        images = batch['image']  # [B, V, C, H, W]
        B, V, C, H, W = images.shape
        device = images.device
        original_dtype = images.dtype

        augmented_images = images.clone()
        paste_masks = torch.zeros(B, V, 1, H, W, device=device)

        jitter_max_y = max(1, int(H * self.jitter_ratio))
        jitter_max_x = max(1, int(W * self.jitter_ratio))

        for b in range(B):
            if random.random() > 0.5:
                continue

            num_objects = random.randint(*self.num_objects_range)
            
            is_same_object_across_views = random.random() < self.same_object_prob

            for obj_idx in range(num_objects):
                if is_same_object_across_views:
                    obj_data = self.object_library.sample(1)[0]
                    size_ratio = random.uniform(*self.size_ratio_range)
                    resized_img, resized_mask = self._resize_object(
                        obj_data['image'], obj_data['mask'], H, W, size_ratio
                    )
                    obj_h, obj_w = resized_img.shape[:2]
                    base_y, base_x = self._get_random_position(obj_h, obj_w, H, W)

                    for v in range(V):
                        jy = random.randint(-jitter_max_y, jitter_max_y)
                        jx = random.randint(-jitter_max_x, jitter_max_x)
                        paste_y = max(0, min(base_y + jy, H - obj_h))
                        paste_x = max(0, min(base_x + jx, W - obj_w))

                        scene_np = augmented_images[b, v].permute(1, 2, 0).cpu().numpy()
                        aug_img, p_mask = self._paste_object_on_image(
                            scene_np, resized_img, resized_mask, (paste_y, paste_x)
                        )
                        augmented_images[b, v] = torch.from_numpy(aug_img).permute(2, 0, 1).to(device)
                        paste_masks[b, v, 0] = torch.maximum(
                            paste_masks[b, v, 0],
                            torch.from_numpy(p_mask).to(device)
                        )
                else:
                    for v in range(V):
                        obj_data = self.object_library.sample(1)[0]
                        size_ratio = random.uniform(*self.size_ratio_range)
                        resized_img, resized_mask = self._resize_object(
                            obj_data['image'], obj_data['mask'], H, W, size_ratio
                        )
                        obj_h, obj_w = resized_img.shape[:2]
                        paste_y, paste_x = self._get_random_position(obj_h, obj_w, H, W)

                        scene_np = augmented_images[b, v].permute(1, 2, 0).cpu().numpy()
                        aug_img, p_mask = self._paste_object_on_image(
                            scene_np, resized_img, resized_mask, (paste_y, paste_x)
                        )
                        augmented_images[b, v] = torch.from_numpy(aug_img).permute(2, 0, 1).to(device)
                        paste_masks[b, v, 0] = torch.maximum(
                            paste_masks[b, v, 0],
                            torch.from_numpy(p_mask).to(device)
                        )

        augmented_batch = {}
        for k, v in batch.items():
            if k == 'image':
                augmented_batch['image'] = augmented_images.clamp(0, 1).to(original_dtype)
            else:
                augmented_batch[k] = v

        return augmented_batch, paste_masks
