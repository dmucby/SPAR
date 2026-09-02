# Third-Party Notices

SPAR builds on research code and pretrained components from several upstream projects. Third-party files remain under their original licenses; the repository-level `LICENSE.md` does not replace those terms.

## DINOv3

- Upstream: <https://github.com/facebookresearch/dinov3>
- Runtime code: `model/dinov3/`
- License: DINOv3 License
- Included license: `model/dinov3/LICENSE.md`

DINOv3 model weights are not distributed in this repository. Users must obtain access from Meta and comply with the DINOv3 License.

## Segment Anything 2 (SAM2)

- Upstream: <https://github.com/facebookresearch/sam2>
- Runtime code: `model/sam2/`
- Main license: Apache License 2.0
- Optional connected-components code: BSD 3-Clause
- Included licenses: `model/sam2/LICENSE` and `model/sam2/LICENSE_cctorch`

SAM2 is used only for optional test-time mask refinement in the paper-aligned release path. SAM2 checkpoints are not distributed here.

## LSeg / Language-Driven Semantic Segmentation

- Upstream: <https://github.com/isl-org/lang-seg>
- Included as the pinned `submodules/lang_seg` Git submodule
- License: MIT

LSeg checkpoints are not distributed here.

## OpenAI CLIP

- Upstream: <https://github.com/openai/CLIP>
- Installed as a Python dependency for the LSeg text/image encoder
- License: MIT

## LVSM, RayZer, and WildRayZer lineage

Parts of this repository retain or adapt code from the LVSM, RayZer, and WildRayZer research implementations, as indicated by copyright headers and source comments in individual files. Users must review and comply with the corresponding upstream terms when redistributing derived files.

The perceptual-loss implementation in `model/loss.py` also records its source adaptations in the file header. The VGG weight file is downloaded separately and is not redistributed.

## Datasets and weights

RealEstate, D-RE10K, ScanNet, COCO, pretrained model weights, and trained SPAR checkpoints are not part of this source release. Each remains governed by its provider's license or terms of use.
