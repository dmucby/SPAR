# Copyright (c) 2025 WildRayZer implementation.
# D-RE10K test dataset loader (decoded-frame only for evaluation).

import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset


class DRE10KTestDataset(Dataset):
    """
    D-RE10K-Mask test dataset (decoded-frame only).

    Directory layout:
        test_root/
            images/<scene_id>/00000.png, ...
            binary_masks/<scene_id>/00000.png, ...
            metadata/<scene_id>.json
    """

    def __init__(
        self,
        test_root: str,
        image_size: int = 256,
        patch_size: int = 16,
        num_input_views: int = 2,
        num_target_views: int = 6,
        square_crop: bool = True,
        fix_total_views: int = 0,
        view_idx_dict: dict = None,
        image_root: str = "",
        prefer_metadata_image_path: bool = True,
        load_pseudo_labels: bool = False,
    ):
        super().__init__()
        self.test_root = test_root
        self.metadata_dir = os.path.join(test_root, "metadata")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_input_views = num_input_views
        self.square_crop = square_crop
        self.view_idx_dict = view_idx_dict or {}
        self.image_root = image_root or ""
        self.prefer_metadata_image_path = prefer_metadata_image_path
        self.load_pseudo_labels = load_pseudo_labels

        if fix_total_views > 0 and fix_total_views > num_input_views:
            self.num_target_views = fix_total_views - num_input_views
            self.num_views = fix_total_views
            print(
                f"[DRE10KTest] fix_total_views={fix_total_views}: "
                f"input={num_input_views}, target={self.num_target_views}"
            )
        else:
            self.num_target_views = num_target_views
            self.num_views = num_input_views + num_target_views

        all_scene_ids = sorted(
            [f.replace(".json", "") for f in os.listdir(self.metadata_dir) if f.endswith(".json")]
        )

        if self.view_idx_dict:
            self.scene_ids = [s for s in all_scene_ids if s in self.view_idx_dict]
            print(
                f"[DRE10KTest] Filtered to {len(self.scene_ids)}/{len(all_scene_ids)} "
                "scenes using view_idx_file"
            )
        else:
            self.scene_ids = all_scene_ids
        print(f"[DRE10KTest] Loaded {len(self.scene_ids)} test scenes from {self.metadata_dir}")

        self._metadata_cache = {}

    def __len__(self):
        return len(self.scene_ids)

    def _load_metadata(self, scene_id: str) -> dict:
        if scene_id not in self._metadata_cache:
            json_path = os.path.join(self.metadata_dir, f"{scene_id}.json")
            with open(json_path, "r") as f:
                self._metadata_cache[scene_id] = json.load(f)
        return self._metadata_cache[scene_id]

    def _resolve_image_path(self, scene_id: str, frame_info: dict, fallback_frame_idx: int) -> Optional[str]:
        raw_path = str(frame_info.get("image_path", "")).strip()
        actual_idx = int(frame_info.get("frame_idx", fallback_frame_idx))
        candidates = []

        if self.image_root:
            candidates.extend(
                [
                    os.path.join(self.image_root, scene_id, f"{actual_idx:05d}.png"),
                    os.path.join(self.image_root, scene_id, f"{fallback_frame_idx:05d}.png"),
                ]
            )
        candidates.extend(
            [
                os.path.join(self.test_root, "images", scene_id, f"{actual_idx:05d}.png"),
                os.path.join(self.test_root, "images", scene_id, f"{fallback_frame_idx:05d}.png"),
                os.path.join(os.path.dirname(self.metadata_dir), "images", scene_id, f"{actual_idx:05d}.png"),
                os.path.join(
                    os.path.dirname(self.metadata_dir), "images", scene_id, f"{fallback_frame_idx:05d}.png"
                ),
            ]
        )

        if self.prefer_metadata_image_path and raw_path:
            if "$Your Data Path$" in raw_path and self.image_root:
                candidates.insert(0, raw_path.replace("$Your Data Path$", self.image_root))
            if "$Your Data Path$" not in raw_path:
                if os.path.isabs(raw_path):
                    candidates.insert(0, raw_path)
                else:
                    candidates.insert(0, os.path.join(self.metadata_dir, raw_path))
                    candidates.insert(0, os.path.join(self.test_root, raw_path.lstrip("./")))
                    candidates.insert(0, os.path.abspath(raw_path))

        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _select_view_indices(self, total_frames: int):
        num_views = self.num_views
        if total_frames < num_views:
            indices = list(range(total_frames))
            while len(indices) < num_views:
                indices.append(indices[-1])
        else:
            step = (total_frames - 1) / (num_views - 1)
            indices = [int(round(i * step)) for i in range(num_views)]

        all_sorted = sorted(set(indices))
        if len(all_sorted) < num_views:
            candidates = [i for i in range(total_frames) if i not in all_sorted]
            while len(all_sorted) < num_views and candidates:
                all_sorted.append(candidates.pop(0))
                all_sorted = sorted(all_sorted)

        all_sorted = all_sorted[:num_views]
        input_indices = [all_sorted[0], all_sorted[-1]]
        remaining = [idx for idx in all_sorted[1:-1]]

        need_more = self.num_input_views - 2
        if need_more > 0 and remaining:
            step_r = max(1, len(remaining) / (need_more + 1))
            for i in range(need_more):
                pick_idx = min(int(round((i + 1) * step_r)), len(remaining) - 1)
                input_indices.append(remaining[pick_idx])

        input_indices = sorted(input_indices)
        target_indices = sorted([idx for idx in all_sorted if idx not in input_indices])

        while len(target_indices) < self.num_target_views:
            for idx in range(total_frames):
                if idx not in input_indices and idx not in target_indices:
                    target_indices.append(idx)
                    target_indices = sorted(target_indices)
                    if len(target_indices) >= self.num_target_views:
                        break
            else:
                break
        target_indices = target_indices[: self.num_target_views]
        return input_indices, target_indices

    def _process_frame(self, frame_np: np.ndarray, fxfycxcy: list):
        image = PILImage.fromarray(frame_np)
        original_w, original_h = image.size
        target_size = self.image_size

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
            start_h = 0
            start_w = 0

        image_np = np.array(image) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float()
        fxfycxcy_new = torch.tensor([fx_new, fy_new, cx_new, cy_new], dtype=torch.float32)
        return image_t, fxfycxcy_new, start_h, start_w, resize_ratio_x, resize_ratio_y

    def _process_mask(self, mask_np: np.ndarray, resize_ratio_x, resize_ratio_y, start_h, start_w):
        original_h, original_w = mask_np.shape[:2]
        target_size = self.image_size
        mask_pil = PILImage.fromarray(mask_np)

        if self.square_crop:
            resize_h = int(round(original_h * resize_ratio_y / self.patch_size) * self.patch_size)
            resize_w = int(round(original_w * resize_ratio_x / self.patch_size) * self.patch_size)
            resize_h = max(resize_h, target_size)
            resize_w = max(resize_w, target_size)
            mask_pil = mask_pil.resize((resize_w, resize_h), resample=PILImage.NEAREST)
            mask_pil = mask_pil.crop((start_w, start_h, start_w + target_size, start_h + target_size))
        else:
            resize_w = int(target_size / original_h * original_w)
            resize_w = int(round(resize_w / self.patch_size) * self.patch_size)
            resize_h = target_size
            mask_pil = mask_pil.resize((resize_w, resize_h), resample=PILImage.NEAREST)

        mask_np_out = np.array(mask_pil).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np_out).float()

    def _process_label_map(self, label_np: np.ndarray, resize_ratio_x, resize_ratio_y, start_h, start_w):
        original_h, original_w = label_np.shape[:2]
        target_size = self.image_size
        label_pil = PILImage.fromarray(label_np.astype(np.int32), mode="I")

        if self.square_crop:
            resize_h = int(round(original_h * resize_ratio_y / self.patch_size) * self.patch_size)
            resize_w = int(round(original_w * resize_ratio_x / self.patch_size) * self.patch_size)
            resize_h = max(resize_h, target_size)
            resize_w = max(resize_w, target_size)
            label_pil = label_pil.resize((resize_w, resize_h), resample=PILImage.NEAREST)
            label_pil = label_pil.crop((start_w, start_h, start_w + target_size, start_h + target_size))
        else:
            resize_w = int(target_size / original_h * original_w)
            resize_w = int(round(resize_w / self.patch_size) * self.patch_size)
            resize_h = target_size
            label_pil = label_pil.resize((resize_w, resize_h), resample=PILImage.NEAREST)

        label_np_out = np.array(label_pil).astype(np.int64)
        return torch.from_numpy(label_np_out).long()

    def __getitem__(self, idx):
        scene_id = self.scene_ids[idx]
        metadata = self._load_metadata(scene_id)
        frames_data = metadata["frames"]
        total_frames = len(frames_data)

        if scene_id in self.view_idx_dict:
            view_spec = self.view_idx_dict[scene_id]
            input_indices = list(view_spec["context"])
            target_indices = list(view_spec["target"])
        else:
            input_indices, target_indices = self._select_view_indices(total_frames)
        all_indices = input_indices + target_indices
        all_indices_sorted = sorted(all_indices)

        images = []
        fxfycxcys = []
        c2ws = []
        masks = []
        pseudo_labels = []
        scene_labelset = None
        mask_dir = os.path.join(self.test_root, "binary_masks", scene_id)
        pseudo_label_dir = os.path.join(self.test_root, "pseudo_labelmaps", scene_id)

        if self.load_pseudo_labels:
            labelset_path = os.path.join(pseudo_label_dir, "labelset.json")
            if os.path.exists(labelset_path):
                with open(labelset_path, "r") as f:
                    scene_labelset = json.load(f)

        for frame_idx in all_indices_sorted:
            frame_info = frames_data[frame_idx]
            fxfycxcy = frame_info["fxfycxcy"]

            w2c = np.array(frame_info["w2c"], dtype=np.float32)
            try:
                c2w = np.linalg.inv(w2c)
            except np.linalg.LinAlgError:
                c2w = np.eye(4, dtype=np.float32)

            img_path = self._resolve_image_path(scene_id, frame_info, frame_idx)
            if img_path is None:
                raise FileNotFoundError(
                    f"[DRE10KTest] decoded frame not found for scene={scene_id}, "
                    f"frame_idx={frame_idx}, metadata_frame_idx={frame_info.get('frame_idx', frame_idx)}. "
                    "Please extract frames first (e.g. scripts/extract_dre10k_decoded_frames.py)."
                )
            frame_np = np.array(PILImage.open(img_path).convert("RGB"))
            image_t, fxfycxcy_new, sh, sw, rx, ry = self._process_frame(frame_np, fxfycxcy)

            mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, f"{frame_info.get('frame_idx', frame_idx):05d}.png")
            if os.path.exists(mask_path):
                mask_np = np.array(PILImage.open(mask_path).convert("L"))
                mask_t = self._process_mask(mask_np, rx, ry, sh, sw)
            else:
                mask_t = torch.zeros(self.image_size, self.image_size)

            if self.load_pseudo_labels:
                pseudo_label_path = os.path.join(pseudo_label_dir, f"{frame_idx:05d}_labelmap.npy")
                if not os.path.exists(pseudo_label_path):
                    pseudo_label_path = os.path.join(pseudo_label_dir, f"{frame_idx:05d}.png")
                if not os.path.exists(pseudo_label_path):
                    pseudo_label_path = os.path.join(
                        pseudo_label_dir, f"{frame_info.get('frame_idx', frame_idx):05d}_labelmap.npy"
                    )
                if not os.path.exists(pseudo_label_path):
                    pseudo_label_path = os.path.join(
                        pseudo_label_dir, f"{frame_info.get('frame_idx', frame_idx):05d}.png"
                    )

                if os.path.exists(pseudo_label_path):
                    if pseudo_label_path.endswith(".npy"):
                        pseudo_label_np = np.load(pseudo_label_path)
                    else:
                        pseudo_label_np = np.array(PILImage.open(pseudo_label_path))
                    pseudo_label_t = self._process_label_map(pseudo_label_np, rx, ry, sh, sw)
                    pseudo_labels.append(pseudo_label_t)
                else:
                    pseudo_labels.append(torch.zeros(self.image_size, self.image_size, dtype=torch.long))

            images.append(image_t)
            fxfycxcys.append(fxfycxcy_new)
            c2ws.append(torch.from_numpy(c2w).float())
            masks.append(mask_t)

        images = torch.stack(images)
        fxfycxcys = torch.stack(fxfycxcys)
        c2ws = torch.stack(c2ws)
        masks = torch.stack(masks)

        ctx_positions = torch.tensor([all_indices_sorted.index(i) for i in input_indices], dtype=torch.long)
        tgt_positions = torch.tensor([all_indices_sorted.index(i) for i in target_indices], dtype=torch.long)

        result = {
            "image": images,
            "fxfycxcy": fxfycxcys,
            "c2w": c2ws,
            "binary_mask": masks,
            "scene_name": scene_id,
            "context_indices": ctx_positions,
            "target_indices": tgt_positions,
            "frame_indices": torch.tensor(all_indices_sorted, dtype=torch.long),
        }
        if self.load_pseudo_labels and pseudo_labels:
            result["pseudo_labels"] = torch.stack(pseudo_labels)
            if scene_labelset is not None:
                result["scene_labelset"] = scene_labelset
        return result
