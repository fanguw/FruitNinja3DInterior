# FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting

**CVPR 2025 Paper**  
[FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_FruitNinja_3D_Object_Interior_Texture_Generation_with_Gaussian_Splatting_CVPR_2025_paper.html)

This repository provides trained 3D Gaussian Splatting (3DGS) models with internal textures for the paper *FruitNinja*, accepted at CVPR 2025.

Our method generates realistic internal textures in 3DGS representations, enabling real-time slicing and view rendering under arbitrary geometric transformations — a capability not supported by previous 3DGS methods.

## Project Page

For more details, examples, and visual results, visit the official project page:  
[https://fanguw.github.io/FruitNinja3D/](https://fanguw.github.io/FruitNinja3D/)


---

## Overview

FruitNinja addresses a major limitation in existing 3DGS frameworks: lack of realistic internal textures when an object is cut or sliced.

- Synthesizes internal textures from a small number of cross-sectional views.
- Supports arbitrary slicing and rendering without post-optimization.
- Uses a diffusion model for cross-sectional inpainting, combined with voxel-based smoothing for consistent textures throughout the volume.
- Introduces **OpaqueAtomGS**, a training strategy that improves convergence and fidelity by enforcing high-opacity, atomic-scale Gaussians.

For technical details, please refer to the [paper PDF](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_FruitNinja_3D_Object_Interior_Texture_Generation_with_Gaussian_Splatting_CVPR_2025_paper.html).

---

## Contents

This repository includes:

- Trained `.ply` files with internal textures.
- 3DGS models for several example objects (watermelon, apple, bread, pomegranate, cake).
- A raw 3DGS model without internal filling for demonstration.
- Scripts for internal texture filling and re-training.

---

## Downloads

**Trained models:**  
[Download all pre-filled models](https://drive.google.com/file/d/1cS9okuSILWpZMubL8oH1cLK2tktqMPwr)

**Raw model (no internal texture) for quick start:**  
[orange_raw.ply](https://drive.google.com/file/d/1lDegRQ8y6MdeobUQmeDcBV4GqbAGrwuA/view?usp=sharing)

---

## Quick Start

Example workflow to reproduce the internal filling and re-training process.

### 1. Clone with submodule

```bash
git clone --recurse-submodules <your-repo-url>
cd <your-repo-folder>
```


If already cloned:

```
git submodule update --init --recursive
```

### 2. Set up Python environment
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Download the example raw model
Download orange_raw.ply and place it in the project root or a desired folder.

### 4. Run internal filling
This generates a filled .ply with internal textures.
```
python internal_filling.py \
  --model_path ./orange_raw.ply \
  --physics_config ./config/orange_physics.json \
  --output_path ./outputs \
  --white_bg \
  --debug
```
This will produce gs_fill.ply in the specified output directory.


### 5. Re-train with the filled model
Use the generated gs_fill.ply for re-training:
```
python train_orange_demo.py \
  --config ./config/orange_train.json \
  --model_path ./outputs \
  --physics_config ./config/orange_physics.json \
  --output_path ./outputs \
  --debug \
  --gs_path ./outputs/gs_fill.ply \
  --gs_ori_path ./orange_raw.ply
```

## Fine-tuning Stable Diffusion Depth Model

To generate realistic internal cross-sectional textures, we fine-tune the **Stable Diffusion 2 Depth model** using object-specific cross-sectional images.

### Steps to fine-tune:

1. **Clone the Diffusers repository**

   ```bash
   git clone https://github.com/huggingface/diffusers
   cd diffusers

2. **Copy the training script**

Copy scripts/train_dreambooth_depth.py from
Dreambooth-Anything https://github.com/francislabountyjr/Dreambooth-Anything
into your local diffusers/examples/dreambooth directory:

```
# From your diffusers root folder
mkdir -p examples/dreambooth
cp <path-to-Dreambooth-Anything>/scripts/train_dreambooth_depth.py examples/dreambooth/
```

3. **Use the provided training images**

This repository includes sample images for fine-tuning, covering canonical slicing angles:

```
./data_finetune_images/<object>
  ├── vertical/
  └── horizontal/
```
These images help the model learn realistic interior details.

4. **Run the fine-tuning script**

Follow the Diffusers Dreambooth guide to fine-tune the depth model with your data. Adjust hyperparameters and paths as needed for your system.

5. **Update your configuration**

Once fine-tuning is complete, you can specify your new model ID in your config file. For example, in /config/orange_trainfer.config:

```
  "vertical_sd_model": "stabilityai/stable-diffusion-2-depth",
  "horizontal_sd_model": "stabilityai/stable-diffusion-2-depth"
```
Replace these values with your own fine-tuned model IDs if you have uploaded them to the Hugging Face Hub or a local path.

## Acknowledgement

This repository is based on [PhysGaussian](https://github.com/XPandora/PhysGaussian).  
We thank and appreciate the authors for their inspiring work and open-source contribution, which laid the foundation for this project.


## Citation
If you use this work, please cite:

```
@inproceedings{wu2025fruitninja,
  title     = {FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting},
  author    = {Fangyu Wu and Yuhao Chen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
