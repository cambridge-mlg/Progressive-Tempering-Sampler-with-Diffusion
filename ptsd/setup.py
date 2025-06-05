import yaml
import wandb
from pathlib import Path
from omegaconf import OmegaConf
import os
import torch
import numpy as np
import datetime


class HParams:
    def __init__(self, **kwargs):
        self._data = {}  # Store attributes in a dictionary
        for key, value in kwargs.items():
            self[key] = value

    def __setattr__(self, key, value):
        if key == '_data':  # Allow setting the _data dictionary
            super().__setattr__(key, value)
        else:
            if isinstance(value, dict):
                for k, v in value.items():
                    self._data[k] = v
            else:
                self._data[key] = value

    def __getattr__(self, key):
        try:
            return self._data[str(key)]
        except KeyError:
            raise AttributeError(f"'HParams' object has no attribute '{key}'")

    def __getitem__(self, key):
        return self._data[str(key)]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self._data[k] = v
        else:
            self._data[str(key)] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return f"HParams({self._data})"

    def __str__(self):
        return str(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __deepcopy__(self, memo):
        return HParams(**self._data)


def setup(
    config_path: str, device: str, use_wandb: bool = True, experiment_group: str = None
) -> HParams:
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    config["device"] = device
    if config["temp_schedule"] == 'geom':
        all_temps = (
            torch.from_numpy(
                np.geomspace(
                    config["temp_low"], config["temp_high"], config["total_n_temp"]
                )
            )
            .float()
            .to(device)
        )
    elif config["temp_schedule"] == 'linear':
        all_temps = (
            torch.linspace(
                config["temp_low"], config["temp_high"], config["total_n_temp"]
            )
            .float()
            .to(device)
        )
    else:
        raise ValueError("Unknown temperature schedule")
    config["all_temps"] = all_temps
    config["LG_step_size"] = torch.tensor(
        [config["LG_step_size"]] * config["total_n_temp"]
    ).to(device)
    if "num_samples_to_generate_per_batch" not in config:
        config["num_samples_to_generate_per_batch"] = config["BUFFER_MAX_SIZE"]

    hparams = HParams(**config)
    os.makedirs(hparams.plot_fold, exist_ok=True)
    os.makedirs(f"{hparams.plot_fold}/training", exist_ok=True)
    os.makedirs(hparams.model_save_path, exist_ok=True)

    if use_wandb:
        with open('configs/wandb.yaml', 'r') as file:
            wandb_config = yaml.safe_load(file)

        wandb_config['tags'] = ['PTSD', config["name"]]
        wandb_config['group'] = config["name"]
        wandb.init(
            entity=wandb_config['entity'],
            project=wandb_config['project'],
            mode="online" if not wandb_config['offline'] else "offline",
            group=wandb_config['group'],
            tags=wandb_config['tags'],
            config=config,
            name=(
                f"{config['name']}_{experiment_group}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if experiment_group is not None
                else None
            ),
        )

    return hparams


from ptsd.targets.lennard_jones import LennardJonesPotential
from ptsd.targets.many_well import ManyWellPotential
from ptsd.targets.gmm import GMM
from ptsd.targets.aldp import AldpPotential, get_aldp_potential


def get_target(name, device):
    if name == 'lj13':
        n_particles = 13
        n_dimensions = 3
        dim = n_particles * n_dimensions
        return (
            LennardJonesPotential(
                dim=dim,
                n_particles=n_particles,
                eps=1.0,
                rm=1.0,
                oscillator=True,
                oscillator_scale=1.0,
                two_event_dims=False,
                energy_factor=1.0,
                device=device,
            ),
            dim,
            n_particles,
            n_dimensions,
        )
    elif name == 'lj55':
        n_particles = 55
        n_dimensions = 3
        dim = n_particles * n_dimensions
        return (
            LennardJonesPotential(
                dim=dim,
                n_particles=n_particles,
                eps=1.0,
                rm=1.0,
                oscillator=True,
                oscillator_scale=1.0,
                two_event_dims=False,
                energy_factor=1.0,
                device=device,
            ),
            dim,
            n_particles,
            n_dimensions,
        )
    elif name == 'mw32':
        dim = 32
        return ManyWellPotential(dim=dim, device=device), dim, 1, dim
    elif name == 'gmm':
        dim = 2
        return (
            GMM(
                dim=dim,
                n_mixes=40,
                loc_scaling=40.0,
                log_var_scaling=1.0,
                seed=0,
                n_test_set_samples=1000,
                device=device,
            ),
            dim,
            1,
            dim,
        )
    elif name == 'aldp':
        aldp_config_path = "configs/aldp/aldp_config.yaml"
        dim = 60
        return get_aldp_potential(aldp_config_path, device=device), dim, 1, dim
    else:
        raise ValueError(f"Unknown target: {name}")


def initialise_buffer(name, device, model_save_path, buffer_max_size, load_init_buffer):
    init_buffer = None
    if load_init_buffer:
        init_buffer = torch.load(
            f"{model_save_path}/init_buffer.pt", map_location=device
        )
        if name in ['lj13', 'lj55']:
            init_buffer = init_buffer[:, :buffer_max_size, :]
    return init_buffer


def _safe_convert_value(value: str):
    """Safely convert string value to appropriate type."""
    # Try converting to numeric types first
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Handle boolean values
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False

    # Handle None
    if value.lower() == 'none':
        return None

    # Keep as string for all other cases
    return value


def setup_config_with_overrides(
    config_path: str, device: str, overrides: dict, experiment_group: str
) -> HParams:
    """Setup configuration with command line overrides."""
    # Get base config
    base_config = OmegaConf.load(config_path)

    # Parse any config overrides from command line
    if len(overrides) > 0:
        override_pairs = []
        for key, value in overrides.items():
            override_pairs.append(f"{key}={value}")

        if override_pairs:
            overrides = OmegaConf.from_dotlist(override_pairs)
            # Merge preserving the types and nested structure
            config = OmegaConf.merge(base_config, overrides)
        else:
            config = base_config
    else:
        config = base_config

    # Convert merged config back to a string path and use existing setup
    temp_config_path = 'temp_config.yaml'
    OmegaConf.save(config, temp_config_path)
    try:
        hparams = setup(temp_config_path, device, config.use_wandb, experiment_group)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    hparams.experiment_group = experiment_group

    return hparams


def setup_config_with_overrides_simple(
    config_path: str, device: str, overrides: dict, experiment_group: str
):
    base_config = OmegaConf.load(config_path)

    config["device"] = device
    # Parse any config overrides from command line
    if len(overrides) > 0:
        override_pairs = []
        for key, value in overrides.items():
            override_pairs.append(f"{key}={value}")

        if override_pairs:
            overrides = OmegaConf.from_dotlist(override_pairs)
            # Merge preserving the types and nested structure
            config = OmegaConf.merge(base_config, overrides)
        else:
            config = base_config
    else:
        config = base_config

    config.experiment_group = experiment_group

    return config


from ptsd.net.model import EGNN, MLP


def get_model(
    name, dim, n_particles, n_dimensions, hidden_dims, t_embed_dims, n_layers, device
):
    if name in ['lj13', 'lj55']:
        return EGNN(
            in_dim=dim,
            n_particles=n_particles,
            n_dimensions=n_dimensions,
            hidden_dims=hidden_dims,
            t_embed_dims=t_embed_dims,
            n_layers=n_layers,
            skip_connect=True,
        ).to(device)
    elif name in ['mw32', 'gmm']:
        return MLP(dim, [256 for _ in range(5)], 128, skip_connect=True).to(device)
    elif name == 'aldp':
        return MLP(dim, [1024 for _ in range(5)], 256, skip_connect=True).to(device)
    else:
        raise ValueError(f"Unknown target: {name}")
