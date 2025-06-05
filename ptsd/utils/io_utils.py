import os
import yaml

# Map of parameter names to their shorthand versions
PARAM_SHORTHAND = {
    'experiment_group': 'exp',
    'hidden_dims': 'hd',
    't_embed_dims': 'ted',
    'n_layers': 'nl',
    'learning_rate': 'lr',
    'bsz': 'bs',
    'max_grad_norm': 'mgn',
    'seed': 's',
    'temp_low': 'tl',
    'temp_high': 'th',
    'total_n_temp': 'tnt',
    'temp_schedule': 'ts',
    'BUFFER_MAX_SIZE': 'bs',
    'swap_buffer_interval': 'si',
    'init_LG_steps': 'ils',
    'init_num_chains': 'inc',
    'init_mcmc_abundance': 'ima',
    'init_mcmc_pick_interval': 'impi',
    'init_num_abundance': 'ina',
    'init_LG_pick_interval': 'ilpi',
    'LG_steps': 'ls',
    'num_abundance': 'na',
    'LG_pick_interval': 'lpi',
    'LG_step_size': 'lss',
    'importance_resample': 'ir',
    'is_integration_steps': 'iis',
    'is_weight_max_quantile': 'iwmq',
    'tmin': 'tmin',
    'tmax': 'tmax',
    'num_plot_samples': 'nps',
    'min_inter_dist': 'mid',
    'max_inter_dist': 'mxid',
    'use_wandb': 'uw',
    'num_samples_to_generate_per_batch': 'nsgpb',
}

SLURM_PARAMS = [
    'time',
    'mem',
    'cpus',
    'partition',
    'gpu-type',
    'gpu_type',
    'platform',
    'script_to_run',
    'gpus',
]
OTHER_NON_CONFIG_PARAMS = ['config_path', 'experiment_group']


def remove_config_prefix(kwargs):
    return {k.replace('config.', ''): v for k, v in kwargs.items()}


def get_run_dir(kwargs, click=None):
    # Base directory structure
    base_dir = os.path.join('runs', kwargs['experiment_group'])

    # Create run name from explicitly set parameters
    param_components = []

    # Add parameters that were explicitly set by user
    for param in kwargs:
        if (
            '=' in param
        ):  # take into account args that were passed as --param=value intead of --param value
            param, value = param.split('=')
        else:
            value = kwargs[param]
        if click:
            if (
                param in click.get_current_context().params
                and not param in OTHER_NON_CONFIG_PARAMS
            ):
                if (
                    click.get_current_context().get_parameter_source(param)
                    == click.core.ParameterSource.COMMANDLINE
                ):
                    shorthand = PARAM_SHORTHAND.get(
                        param, param
                    )  # Use original if no shorthand exists
                    param_components.append(f"{shorthand}_{value}")
        else:
            if param not in SLURM_PARAMS and param not in OTHER_NON_CONFIG_PARAMS:
                shorthand = PARAM_SHORTHAND.get(
                    param, param
                )  # Use original if no shorthand exists
                param_components.append(f"{shorthand}_{value}")

    run_name = "_".join(param_components)
    if run_name == '':
        run_name = 'default'

    run_dir = os.path.join(base_dir, run_name)
    plot_dir = os.path.join(run_dir, 'plots')
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    # Save hyperparameters as yaml
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(kwargs, f, default_flow_style=False)

    return run_dir, plot_dir, checkpoint_dir
