"""D-RE10K dataset loader for SPAR training."""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage
from typing import List, Dict, Optional

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("[DRE10K] WARNING: PyAV not installed. Install via: pip install av")


class DRE10KDataset(Dataset):
    """Load multi-view D-RE10K scenes and their camera parameters."""

    def __init__(
        self,
        config,
        split: str = 'train',
        metadata_dir: str = None,
        video_dir: str = None,
    ):
        super().__init__()
        self.config = config

        dataset_root = config.training.get('dre10k_root', './datasets/Dynamic-RE10K')
        self.dataset_root = dataset_root
        if split == 'train':
            sub_dir = 'train_zip'
        else:
            sub_dir = 'test_zip'

        if metadata_dir is None:
            metadata_dir = os.path.join(dataset_root, sub_dir, 'metadata')
        if video_dir is None:
            video_dir = os.path.join(dataset_root, sub_dir, 'videos')

        self.metadata_dir = metadata_dir
        self.video_dir = video_dir
        self.image_root = config.training.get('dre10k_image_root', '')
        self.prefer_image_path = config.training.get('dre10k_prefer_image_path', True)

        self.scene_ids = sorted([
            f.replace('.json', '')
            for f in os.listdir(metadata_dir)
            if f.endswith('.json')
        ])
        print(f"[DRE10K] Loaded {len(self.scene_ids)} scenes from {metadata_dir}")

        self._metadata_cache: Dict[str, dict] = {}

        self.resize_h = config.model.image_tokenizer.image_size  # 256
        self.patch_size = config.model.image_tokenizer.patch_size  # 16
        self.square_crop = config.training.get('square_crop', True)
        self.num_views = config.training.get('num_views', 8)
        global_vs_cfg = config.training.get('view_selector', {})
        dre10k_vs_cfg = config.training.get('dre10k_view_selector', {})
        
        self.min_frame_dist = dre10k_vs_cfg.get('min_frame_dist', global_vs_cfg.get('min_frame_dist', 1))
        self.max_frame_dist = dre10k_vs_cfg.get('max_frame_dist', global_vs_cfg.get('max_frame_dist', 15))
        self.use_curriculum = dre10k_vs_cfg.get('use_curriculum', global_vs_cfg.get('use_curriculum', False))
        self.curriculum_iter = dre10k_vs_cfg.get('curriculum_iter', global_vs_cfg.get('curriculum_iter', 30000))
        self.curriculum_start_min_frame_dist = dre10k_vs_cfg.get(
            'curriculum_start_min_frame_dist', global_vs_cfg.get('curriculum_start_min_frame_dist', 1))
        self.curriculum_start_max_frame_dist = dre10k_vs_cfg.get(
            'curriculum_start_max_frame_dist', global_vs_cfg.get('curriculum_start_max_frame_dist', 8))
        
        self.current_iteration = 0
        self.cur_min_frame_dist = self.curriculum_start_min_frame_dist if self.use_curriculum else self.min_frame_dist
        self.cur_max_frame_dist = self.curriculum_start_max_frame_dist if self.use_curriculum else self.max_frame_dist

        # Video frame cache (per-scene in-memory cache)
        self._frame_cache: Dict[str, np.ndarray] = {}
        self.max_cache_scenes = config.training.get('max_cache_scenes', 50)

    def __len__(self):
        return len(self.scene_ids)

    def update_iteration(self, iteration: int):
        """Update current iteration for curriculum learning."""
        self.current_iteration = iteration
        if not self.use_curriculum:
            return
        
        progress = min(iteration / self.curriculum_iter, 1.0)
        self.cur_min_frame_dist = int(
            self.curriculum_start_min_frame_dist + 
            (self.min_frame_dist - self.curriculum_start_min_frame_dist) * progress
        )
        self.cur_max_frame_dist = int(
            self.curriculum_start_max_frame_dist + 
            (self.max_frame_dist - self.curriculum_start_max_frame_dist) * progress
        )
        print(f'[DRE10K] Curriculum progress: {progress:.3f}, min_dist: {self.cur_min_frame_dist}, max_dist: {self.cur_max_frame_dist}')

    def _load_metadata(self, scene_id: str) -> dict:
        if scene_id not in self._metadata_cache:
            json_path = os.path.join(self.metadata_dir, f'{scene_id}.json')
            with open(json_path, 'r') as f:
                self._metadata_cache[scene_id] = json.load(f)
        return self._metadata_cache[scene_id]

    def _decode_video_frames(self, scene_id: str) -> np.ndarray:
        """Decode all RGB frames from a scene video and cache the result."""
        if scene_id in self._frame_cache:
            return self._frame_cache[scene_id]

        video_path = os.path.join(self.video_dir, f'{scene_id}.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        frames = None

        if HAS_AV:
            frame_list = []
            container = av.open(video_path)
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                frame_list.append(img)
            container.close()
            if frame_list:
                frames = np.stack(frame_list)

        # Fallback: imageio
        if frames is None:
            import imageio
            reader = imageio.get_reader(video_path, 'ffmpeg')
            frames = np.stack([f for f in reader])
            reader.close()

        if len(self._frame_cache) >= self.max_cache_scenes:
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]
        self._frame_cache[scene_id] = frames

        return frames

    def _resolve_image_path(self, scene_id: str, frame_info: dict, fallback_frame_idx: int) -> Optional[str]:
        """
        Resolve frame image path from metadata.
        Supports:
          1) absolute image_path
          2) relative image_path (relative to metadata_dir / dataset_root / cwd)
          3) placeholder "$Your Data Path$" + configured dre10k_image_root
          4) heuristic default image roots
        """
        raw_path = frame_info.get('image_path', None)
        if not raw_path:
            raw_path = ''
        raw_path = str(raw_path).strip()

        frame_idx = int(frame_info.get('frame_idx', fallback_frame_idx))
        candidates = []

        if self.image_root:
            candidates.append(os.path.join(self.image_root, scene_id, f'{frame_idx:05d}.png'))

        # Heuristic candidates for preprocessed image layout.
        heuristic_roots = [
            os.path.join(self.dataset_root, 'train_official', 'images'),
            os.path.join(self.dataset_root, 'test_official', 'images'),
            os.path.join(self.dataset_root, 'images'),
            os.path.join(os.path.dirname(self.metadata_dir), 'images'),
        ]
        for root in heuristic_roots:
            p = os.path.join(root, scene_id, f'{frame_idx:05d}.png')
            if p not in candidates:
                candidates.append(p)

        if raw_path:
            if '$Your Data Path$' in raw_path and self.image_root:
                # For metadata paths like "$Your Data Path$/train/images/<scene>/<idx>.png".
                candidates.append(raw_path.replace('$Your Data Path$', self.image_root))
            if '$Your Data Path$' not in raw_path:
                if os.path.isabs(raw_path):
                    candidates.append(raw_path)
                else:
                    candidates.append(os.path.join(self.metadata_dir, raw_path))
                    candidates.append(os.path.join(self.dataset_root, raw_path.lstrip('./')))
                    candidates.append(os.path.abspath(raw_path))

        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _load_frame_from_image_path(
        self, scene_id: str, frame_info: dict, fallback_frame_idx: int
    ) -> Optional[np.ndarray]:
        image_path = self._resolve_image_path(scene_id, frame_info, fallback_frame_idx)
        if image_path is None:
            return None
        try:
            with PILImage.open(image_path) as image:
                return np.array(image.convert('RGB'))
        except Exception:
            return None

    def _select_frame_indices(self, total_frames: int) -> List[int]:
        """Sample a temporally bounded set of input and target views."""
        if total_frames <= self.num_views:
            indices = list(range(total_frames))
            while len(indices) < self.num_views:
                indices.append(random.choice(range(total_frames)))
            return sorted(indices)
        
        min_dist = self.cur_min_frame_dist
        max_dist = min(self.cur_max_frame_dist, total_frames - 1)
        
        if max_dist <= min_dist:
            max_dist = min_dist + 1
        if max_dist >= total_frames:
            max_dist = total_frames - 1
        if min_dist >= max_dist:
            min_dist = max(1, max_dist - 1)
        
        frame_dist = random.randint(min_dist, max_dist)
        
        if total_frames <= frame_dist:
            frame_dist = total_frames - 1
        
        start_frame = random.randint(0, total_frames - frame_dist - 1)
        end_frame = start_frame + frame_dist
        
        middle_range = list(range(start_frame + 1, end_frame))
        middle_need = self.num_views - 2
        
        if len(middle_range) >= middle_need:
            middle_frames = sorted(random.sample(middle_range, middle_need))
        else:
            middle_frames = middle_range[:]
            while len(middle_frames) < middle_need:
                if middle_range:
                    middle_frames.append(random.choice(middle_range))
                else:
                    middle_frames.append(start_frame)
            middle_frames = sorted(middle_frames)
        
        return [start_frame] + middle_frames + [end_frame]

    def _process_frame(self, frame_np: np.ndarray, fxfycxcy: list):
        """

        Args:
            frame_np: [H, W, 3] uint8
            fxfycxcy: [fx, fy, cx, cy]

        Returns:
            image: [3, resize_h, resize_h] float32, range [0, 1]
            fxfycxcy_new: [4] float32
        """
        image = PILImage.fromarray(frame_np)
        original_w, original_h = image.size
        target_size = self.resize_h  # 256

        fx, fy, cx, cy = fxfycxcy

        if self.square_crop:
            scale = max(target_size / original_h, target_size / original_w)
            resize_h = int(round(original_h * scale / self.patch_size) * self.patch_size)
            resize_w = int(round(original_w * scale / self.patch_size) * self.patch_size)
            resize_h = max(resize_h, target_size)
            resize_w = max(resize_w, target_size)

            image = image.resize((resize_w, resize_h), resample=PILImage.LANCZOS)

            resize_ratio_x = resize_w / original_w
            resize_ratio_y = resize_h / original_h
            fx_new = fx * resize_ratio_x
            fy_new = fy * resize_ratio_y
            cx_new = cx * resize_ratio_x
            cy_new = cy * resize_ratio_y

            start_h = (resize_h - target_size) // 2
            start_w = (resize_w - target_size) // 2
            image = image.crop((start_w, start_h, start_w + target_size, start_h + target_size))
            cx_new -= start_w
            cy_new -= start_h
        else:
            resize_w = int(target_size / original_h * original_w)
            resize_w = int(round(resize_w / self.patch_size) * self.patch_size)
            resize_h = target_size

            image = image.resize((resize_w, resize_h), resample=PILImage.LANCZOS)

            resize_ratio_x = resize_w / original_w
            resize_ratio_y = resize_h / original_h
            fx_new = fx * resize_ratio_x
            fy_new = fy * resize_ratio_y
            cx_new = cx * resize_ratio_x
            cy_new = cy * resize_ratio_y

        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        fxfycxcy_new = torch.tensor([fx_new, fy_new, cx_new, cy_new], dtype=torch.float32)

        return image, fxfycxcy_new

    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]

        try:
            metadata = self._load_metadata(scene_id)
            frames_data = metadata['frames']
            total_frames = len(frames_data)

            selected_indices = self._select_frame_indices(total_frames)

            images = []
            fxfycxcys = []
            c2ws = []
            all_frames_np = None  # lazy video decode fallback

            for frame_idx in selected_indices:
                frame_info = frames_data[frame_idx]
                actual_idx = frame_info.get('frame_idx', frame_idx)

                frame_np = None
                if self.prefer_image_path:
                    frame_np = self._load_frame_from_image_path(scene_id, frame_info, frame_idx)
                if frame_np is None:
                    if all_frames_np is None:
                        all_frames_np = self._decode_video_frames(scene_id)
                    if actual_idx < len(all_frames_np):
                        frame_np = all_frames_np[actual_idx]
                    else:
                        frame_np = all_frames_np[min(frame_idx, len(all_frames_np) - 1)]

                fxfycxcy = frame_info['fxfycxcy']
                w2c = np.array(frame_info['w2c'], dtype=np.float32)
                # w2c → c2w
                try:
                    c2w = np.linalg.inv(w2c)
                except np.linalg.LinAlgError:
                    c2w = np.eye(4, dtype=np.float32)

                image, fxfycxcy_new = self._process_frame(frame_np, fxfycxcy)

                images.append(image)
                fxfycxcys.append(fxfycxcy_new)
                c2ws.append(torch.from_numpy(c2w).float())

            images = torch.stack(images)         # [V, 3, H, W]
            fxfycxcys = torch.stack(fxfycxcys)   # [V, 4]
            c2ws = torch.stack(c2ws)             # [V, 4, 4]
            return {
                'image': images,
                'fxfycxcy': fxfycxcys,
                'c2w': c2ws,
                'scene_name': scene_id,
                'frame_indices': torch.tensor(selected_indices, dtype=torch.long),  # [V]
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[DRE10K] Error loading scene {scene_id}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
