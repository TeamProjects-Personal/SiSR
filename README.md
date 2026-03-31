# SiSR
Single image Super Resolution 

## Overview

This folder contains the work for **Member 1**:

- Download/organize datasets (primary: **Set5, Set14, Urban100**)
- Generate **LR–HR pairs** for **×2, ×3, ×4** (bicubic degradation)
- Implement baseline interpolation upscaling:
    - Nearest Neighbor
    - Bilinear
    - Bicubic (main baseline)
- Save all generated outputs to `results/basic/`
- Share the two `.py` files + generated results

---

## Datasets

Primary datasets (required):

- Set5
- Set14
- Urban100

Optional datasets (only if time permits):

- BSD100
- Manga109
- etc.

---

## Requirements

### System

- Python **3.12** (recommended)

## Setup

### 1) Clone the repository

Replace `<REPO_LINK_HERE>` with your repo link:

- `git clone <REPO_LINK_HERE>`
- `cd <REPO_FOLDER_NAME>`

---

### 2) Create and activate a virtual environment (recommended)

Create:

- Windows:
    - `python -m venv .venv`
- macOS/Linux:
    - `python3 -m venv .venv`

Activate:

- Windows (PowerShell/Vscode Terminal):
    - `.\.venv\Scripts\Activate.ps1`
- Windows (CMD):
    - `.venv\Scripts\activate.bat`
- macOS/Linux:
    - `source .venv/bin/activate`

Upgrade pip:

- `python -m pip install --upgrade pip`

---

### 3) Install dependencies

- `pip install pillow opencv-python numpy`


---

## Project structure (expected)

Create (or verify) this structure:

- `classical/`
    - `data/`
        - `HR/` (original high-resolution images)
        - `LR/` (generated low-resolution images)
    - `methods/`
        - `basic_interpolation.py`
    - `results/`
        - `basic/`
    - `generate_lr_hr_pairs.py`

---

## Dataset setup (Set5, Set14, Urban100)

Put datasets inside `classical/data/HR/` like this:

- `classical/data/HR/Set5/`
- `classical/data/HR/Set14/`
- `classical/data/HR/Urban100/`

Image formats can be `.png`, etc.

---

## How LR–HR pairs are generated

`generate_lr_hr_pairs.py` should:

1. Read HR images from:
    - `classical/data/HR/Set5/`
    - `classical/data/HR/Set14/`
    - `classical/data/HR/Urban100/`
      
2. Downsample HR → LR using **bicubic**:
    - OpenCV `cv.resize(..., interpolation=cv.INTER_CUBIC)`
      
3. Save LR images for **×2, ×3, ×4** into (recommended):
    - `classical/data/LR/x2/<dataset_name>/...`
    - `classical/data/LR/x3/<dataset_name>/...`
    - `classical/data/LR/x4/<dataset_name>/...`

---

## Basic interpolation methods

`methods/basic_interpolation.py`
Has 3 different function for each interpolation method and dataset  

Supported methods:

- `"nearest"` → `cv.INTER_NEAREST`
- `"bilinear"` → `cv.INTER_LINEAR`
- `"bicubic"` → `cv.INTER_CUBIC` (main baseline)

---

## Running the scripts

### A) Generate LR–HR pairs

From the repo root:

- `python classical/generate_lr_hr_pairs.py`

Or from inside `classical/`:

- `python generate_lr_hr_pairs.py`

---

### B) Generate baseline upscaling results

After LR images exist, run your baseline inference script (depends on how you structure it). The expectation is:

- Read LR images from:
    - `classical/data/LR/x2/`, `x3/`, `x4/`
      
- Produce outputs into:
    - `classical/results/basic/<method>/x2/`
    - `classical/results/basic/<method>/x3/`
    - `classical/results/basic/<method>/x4/`

Example:

- `classical/results/basic/nearest/x2/Set5/...`
- `classical/results/basic/bilinear/x3/Set14/...`
- `classical/results/basic/bicubic/x4/Urban100/...`

---

## Deliverables to share with the team

Share/upload:

1. `classical/generate_lr_hr_pairs.py`
2. `classical/methods/basic_interpolation.py`
3. Generated outputs (for Set5, Set14, Urban100):
    - `classical/data/LR/` (all scales)
    - `classical/results/basic/`

That’s all for Member 1’s scope.
