# Progressive-Tempering-Sampler-with-Diffusion
Official implementation of Progressive Tempering Sampler with Diffusion (ICML 2025)

Coming Soon!


## Install
'''
conda create -f environment.yaml
pip install -r requirement.txt
'''

## Experiments
To run experiments, including PT simulation, PT+DM training, and PTSD training, we provide an example on GMM as follows:
1. Simulate PT to get data saved in "data/pt/pt_gmm.pt"
'''
python --config-name=pt_gmm
'''
2. Train PT+DM
'''
python --config-name=gmm prefix="ptdm"
'''
3. Train PTSD
'''
python --config-name=gmm
'''