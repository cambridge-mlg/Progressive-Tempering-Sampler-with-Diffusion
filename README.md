# Progressive Tempering Sampler with Diffusion (PTSD)

**Official implementation of "Progressive Tempering Sampler with Diffusion" (ICML 2025)**  

> 🔬 This repository contains the code for PT simulation, PT+DM training, and PTSD training.

🚧 **Coming Soon!** Full codebase and documentation will be released shortly.

---

## Installation

Create the conda environment and install the required Python dependencies:

```bash
conda env create -f environment.yaml
pip install -r requirements.txt
```

---

## Running Experiments

We provide an example on the GMM (Gaussian Mixture Model) target distribution. This includes three main stages:

### 1. Simulate Parallel Tempering (PT)

Run parallel tempering to generate initial samples, saved to `data/pt/pt_gmm.pt`:

```bash
python run.py --config-name=pt_gmm
```

### 2. Train PT+DM (Diffusion Matching)

Train a diffusion model using the PT samples:

```bash
python run.py --config-name=gmm prefix="ptdm"
```

### 3. Train PTSD (Progressive Tempering Sampler with Diffusion)

Train the full PTSD framework using both PT and DM:

```bash
python run.py --config-name=gmm
```

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

For questions or feedback, feel free to open an issue or contact the authors.

---