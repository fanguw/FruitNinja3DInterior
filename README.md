# FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting

**CVPR 2025 Paper**  
📄 [FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_FruitNinja_3D_Object_Interior_Texture_Generation_with_Gaussian_Splatting_CVPR_2025_paper.html)

This repository hosts the **trained 3D Gaussian Splatting (3DGS) models with internal textures** for the paper *FruitNinja*, accepted at CVPR 2025.

Our method generates high-quality internal textures in 3DGS representations, enabling **real-time slicing** and view rendering under arbitrary geometric transformations — a novel capability not supported by previous 3DGS techniques.

> ⚠️ **Note**: The code will be released soon. For now, we are releasing the trained 3DGS models for early exploration and research comparison.

---

## 🔍 Overview

**FruitNinja** addresses a major limitation in existing 3DGS frameworks: lack of realistic internal textures when an object is cut or sliced.

- We introduce a method for synthesizing internal textures from a small number of cross-sectional views.
- The trained models support arbitrary slicing and rendering without any post-optimization.
- A diffusion model is used to guide cross-sectional inpainting, and a voxel-smoothing mechanism ensures texture consistency throughout the object volume.
- We also introduce **OpaqueAtomGS**, a training strategy that improves convergence and fidelity by enforcing atomic-scale, high-opacity Gaussian particles.

For more technical details, please refer to the full [paper PDF](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_FruitNinja_3D_Object_Interior_Texture_Generation_with_Gaussian_Splatting_CVPR_2025_paper.html).

---

## 📦 Contents

This repo currently includes:

- Trained `.ply` files with internal textures (opaque atomic Gaussians)
- 3DGS models for several example objects (e.g., watermelon, apple, bread, pomegranate, cake)
- Each model is trained to support real-time slicing and view rendering of both surface and internal structure

---

## 📁 Download

All models are hosted [here](https://drive.google.com/file/d/1cS9okuSILWpZMubL8oH1cLK2tktqMPwr).

The folder includes:
- `*.ply` files for each object used in experiment (watermelon, apple, orange, red velvet cake, pomegranate, bread)

---

## 🧪 Inference & Code (Coming Soon)

We are finalizing the open-source release of the training and rendering pipeline, including:
- Cross-section-conditioned SDS optimization
- Voxel-based smoothing
- OpaqueAtomGS training utilities

Please stay tuned for updates!

---

## 📄 Citation

If you use our models in your work, please cite:

```bibtex
@inproceedings{wu2025fruitninja,
  title     = {FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting},
  author    = {Fangyu Wu and Yuhao Chen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
