# Progressive Tempering Sampler with Diffusion (PTSD)

**Official implementation of "Progressive Tempering Sampler with Diffusion" (ICML 2025)**  

> 🔬 This repository contains the code for PT simulation, PT+DM training, and PTSD training.

🚧 **Coming Soon!** Full documentation will be released shortly.

- [Installation](#installation)
- [Running Experiments](#running-experiments)
  - [1. Simulate Parallel Tempering (PT)](#1-simulate-parallel-tempering-pt)
  - [2. Train PT+DM (Diffusion Matching)](#2-train-ptdm-diffusion-matching)
  - [3. Train PTSD (Progressive Tempering Sampler with Diffusion)](#3-train-ptsd-progressive-tempering-sampler-with-diffusion)
- [Tasks](#tasks)
- [Citation](#citation)
- [Contact](#contact)

---

## Installation

Create the conda environment and install the required Python dependencies:

```bash
conda env create -f environment.yaml
pip install -r requirements.txt
```

---

## Running Experiments

We provide an example on the GMM task. This includes three main stages:

### 1. Simulate PT

Run parallel tempering to generate initial samples, saved to `data/pt/pt_gmm.pt`:

```bash
python main.py --config-name=pt_gmm
```

### 2. Train PT+DM

Train a diffusion model using the PT samples:

```bash
python main.py --config-name=gmm prefix="ptdm"
```

### 3. Train PTSD
Train the full PTSD framework using both PT and DM:

```bash
python main.py --config-name=gmm
```

---

## Tasks
We provide codes for following tasks
 - gmm: Mixture of Gaussian with 40 modes (d=2)
 - mw32: Many Well potential (d=32)
 - lj55: Lennard-Jones potential with 55 particles (d=165)
 - aldp: Alanine Dipeptide in internal coordinate (d=60)
 - aldp_cart: Alanine Dipeptide in Cartesian coordinate(d=66)

---

## Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{YourICMLPaper2025,
  title     = {Progressive Tempering Sampler with Diffusion},
  author    = {Your Name and Co-authors},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025}
}
```

---

## Contact

For questions or feedback, feel free to open an issue or contact the corresponding authors.

---