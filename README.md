# Progressive Tempering Sampler with Diffusion (PTSD)

[![Paper](https://img.shields.io/badge/paper-arxiv.2410.12456-B31B1B.svg)](https://www.arxiv.org/abs/2506.05231)

**Official implementation of "Progressive Tempering Sampler with Diffusion" (ICML 2025)**  

> 🔬 This repository contains the code for PT simulation, PT+DM training, and PTSD training.

🚧 **Coming Soon!** Full documentation will be released shortly.

- [Installation](#installation)
- [Running Experiments](#running-experiments)
  - [1. Simulate PT](#1-simulate-parallel-tempering-pt)
  - [2. Train PT+DM](#2-train-ptdm-diffusion-matching)
  - [3. Train PTSD](#3-train-ptsd-progressive-tempering-sampler-with-diffusion)
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
@misc{rissanen2025progressivetemperingsamplerdiffusion,
      title={Progressive Tempering Sampler with Diffusion}, 
      author={Severi Rissanen and RuiKang OuYang and Jiajun He and Wenlin Chen and Markus Heinonen and Arno Solin and José Miguel Hernández-Lobato},
      year={2025},
      eprint={2506.05231},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.05231}, 
}
```

---

## Contact

For questions or feedback, feel free to open an issue or contact the corresponding authors.

---