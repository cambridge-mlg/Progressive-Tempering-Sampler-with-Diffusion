import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import yaml
import os
import datetime
import torch
import numpy as np

from ptsd.train import train, train_ptdm, train_gtdm
from ptsd.eval import eval
from ptsd.pt_sim import parallel_tempering_simulation
from ptsd.utils.metrics import (
    MMD_energy,
    TVD_energy,
    Wasserstain2_energy,
    Wasserstain2_data,
)
from ptsd.utils.diffusion_utils import Eular_solve
from ptsd.utils.diffusion_utils import (
    Eular_solve,
    Euler_solve_wrapper,
    Euler_solve_with_log_pdf,
    importance_sample_with_reweighting,
)
from ptsd.utils.data_parallel_utils import (
    maybe_wrap_data_parallel,
    unwrap_data_parallel,
    save_model,
    load_model,
)
from ptsd.utils.logging_utils import setup_logging


def setup_ptsd(cfg: DictConfig):
    if cfg["temp_schedule"] == 'geom':
        all_temps = (
            torch.from_numpy(
                np.geomspace(cfg["temp_low"], cfg["temp_high"], cfg["total_n_temp"])
            )
            .float()
            .tolist()
        )
    elif cfg["temp_schedule"] == 'linear':
        all_temps = (
            torch.linspace(cfg["temp_low"], cfg["temp_high"], cfg["total_n_temp"])
            .float()
            .to(cfg.device)
            .tolist()
        )
    else:
        raise ValueError("Unknown temperature schedule")
    cfg.all_temps = all_temps
    cfg["LG_step_size"] = [cfg["LG_step_size"]] * cfg["total_n_temp"]
    cfg["num_samples_to_generate_per_batch"] = (
        cfg["num_samples_to_generate_per_batch"]
        if cfg["num_samples_to_generate_per_batch"]
        else cfg["BUFFER_MAX_SIZE"]
    )
    return cfg


def setup(cfg: DictConfig):
    os.makedirs(cfg.plot_fold, exist_ok=True)
    os.makedirs(f"{cfg.plot_fold}/training", exist_ok=True)
    os.makedirs(cfg.model_save_path, exist_ok=True)
    cfg["adaptive_data_sigma"] = cfg.get("adaptive_data_sigma", False)
    # Set default for data parallel if not present
    cfg["use_data_parallel"] = cfg.get("use_data_parallel", False)

    if cfg.logger:
        model_type = cfg["prefix"].upper() if "prefix" in cfg else "PTSD"
        cfg.logger.tags = [model_type, cfg["name"]]
        cfg.logger.group = cfg["name"]
        saved_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            entity=cfg.logger.entity,
            project=cfg.logger.project,
            mode="online" if not cfg.logger.offline else "offline",
            group=cfg.logger.group,
            tags=cfg.logger.tags,
            config=saved_config,
            name=(
                f"{model_type}_{cfg['name']}_{cfg.logger.experiment_group}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if cfg.logger.experiment_group is not None
                else None
            ),
        )
    return cfg


def setup_pt(cfg: DictConfig):
    cfg.plot_path = f"{cfg['plot_fold']}/{cfg['name']}_{cfg['dim']}"
    os.makedirs(cfg.plot_path, exist_ok=True)
    os.makedirs(cfg.save_fold, exist_ok=True)
    if cfg["temp_schedule"] == 'geom':
        all_temps = (
            torch.from_numpy(
                np.geomspace(cfg["temp_low"], cfg["temp_high"], cfg["total_n_temp"])
            )
            .float()
            .tolist()
        )
    elif cfg["temp_schedule"] == 'linear':
        all_temps = (
            torch.linspace(cfg["temp_low"], cfg["temp_high"], cfg["total_n_temp"])
            .float()
            .tolist()
        )
    else:
        raise ValueError("Unknown temperature schedule")
    cfg.all_temps = all_temps
    return cfg


def main_ptsd(target, cfg: DictConfig):
    cfg = setup_ptsd(cfg)
    cfg = setup(cfg)
    # Create models and optionally wrap with DataParallel
    models = []
    for _ in range(cfg.total_n_temp):
        model = hydra.utils.instantiate(cfg.net).to(cfg.device)
        model = maybe_wrap_data_parallel(model, cfg)
        models.append(model)

    opts = []
    for model in models:
        if hasattr(cfg, "weight_decay") and float(cfg.weight_decay) > 0:
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=float(cfg.weight_decay),
            )
        else:
            opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        opts.append(opt)

    if cfg.load_init_buffer and os.path.exists(
        f"{cfg.init_samples_path}/init_samples.pt"
    ):
        init_buffer_ckpt = torch.load(
            f"{cfg.init_samples_path}/init_samples.pt", map_location=cfg.device
        )
        if not isinstance(init_buffer_ckpt, dict):
            init_buffer = None
            print(
                "The format of loaded init buffer is outdated! A new one will be generated later..."
            )
        else:
            init_buffer, init_buffer_config = (
                init_buffer_ckpt["data"],
                init_buffer_ckpt["config"],
            )
            for key in init_buffer_config.keys():
                if init_buffer_config[key] != cfg[key]:
                    init_buffer = None
                    print(
                        "The loaded init buffer is invalid! A new one will be generated later..."
                    )
                    break
            if init_buffer.shape[1] < cfg["BUFFER_MAX_SIZE"]:
                init_buffer = None
                print("Init buffer is under-size! A new one will be generated later...")
            elif init_buffer.shape[1] > cfg["BUFFER_MAX_SIZE"]:
                # Randomly subsample the buffer to match the maximum size
                indices = torch.randperm(init_buffer.shape[1])[: cfg["BUFFER_MAX_SIZE"]]
                init_buffer = init_buffer[:, indices, :].clone().detach()
                print("Init buffer is over-size. Thinned.")
    else:
        init_buffer = None
        print("No init buffer. Or init buffer NOT FOUND! It will be generated later...")

    _, _, BUFFER = train(target, models, opts, cfg, init_buffer=init_buffer)
    os.makedirs(cfg.model_save_path, exist_ok=True)
    torch.save(BUFFER, os.path.join(cfg.model_save_path, 'ptsd_buffer.pt'))

    """baseline 2: PT"""
    # TODO: add config check
    if os.path.exists(f"data/pt/pt_{cfg.name}.pt"):
        # shape of pt_samples: [num_temps, num_chains, num_steps, dim] or [num_chains, num_steps, dim]
        pt_samples = torch.load(f"data/pt/pt_{cfg.name}.pt", map_location=cfg.device)
        if pt_samples.ndim == 4:
            pt_samples = pt_samples[0]
        pt_samples = pt_samples[
            :, -cfg.num_eval_samples // pt_samples.shape[0] :, :
        ].reshape(-1, cfg.dim)
    else:
        # placeholder
        # TODO: Simulate PT if no sample found
        pt_samples = target.sample([cfg.num_eval_samples])

    t1_model = hydra.utils.instantiate(cfg.net).to(cfg.device)
    ckpt = load_model(t1_model, f"{cfg.model_save_path}/model_1.0.pt", cfg.device)
    t1_model = maybe_wrap_data_parallel(t1_model, cfg)

    with torch.no_grad():
        sample_func = Euler_solve_wrapper(
            Eular_solve, num_samples_per_batch=cfg.num_samples_to_generate_per_batch
        )
        t1_samples = (
            sample_func(
                target,
                t1_model,
                torch.randn([cfg.num_eval_samples, cfg.dim], device=cfg.device)
                * cfg.tmax,
            )
            * cfg.normalization
        )

    """Evaluation"""
    eval(
        gt_samples=target.sample([cfg.num_eval_samples]),
        sample_dict={
            "ptsd": t1_samples.unsqueeze(0),
            "pt": pt_samples.unsqueeze(0),
        },
        metric_list=[
            MMD_energy(target, kernel_num=5, fix_sigma=1),
            TVD_energy(target),
            Wasserstain2_energy(target),
            Wasserstain2_data(target),
        ],
        save_path=cfg.plot_fold,
    )


def main_ptdm(target, cfg: DictConfig):
    cfg = setup(cfg)
    cfg["num_samples_to_generate_per_batch"] = (
        cfg["num_samples_to_generate_per_batch"]
        if cfg["num_samples_to_generate_per_batch"]
        else cfg["num_eval_samples"]
    )
    cfg["check_interval"] = (
        cfg["check_interval"] if cfg["check_interval"] else cfg["Epochs"]
    )

    model = hydra.utils.instantiate(cfg.net).to(cfg.device)
    model = maybe_wrap_data_parallel(model, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    pt_samples = torch.load(cfg.data_path, map_location=cfg.device).to(torch.float32)
    if pt_samples.ndim == 3:
        # In this case, the first dimension refers to the temperature
        pt_samples = pt_samples[0]
    if cfg.max_samples_to_consider < pt_samples.shape[0]:
        pt_samples = pt_samples[cfg.warmup_steps : cfg.max_samples_to_consider]
    else:
        # Get samples after warmup
        pt_samples = pt_samples[cfg.warmup_steps :]
        # If we need more samples than available, do sampling with replacement
        indices = torch.randint(
            0,
            pt_samples.shape[0],
            (cfg.max_samples_to_consider,),
            device=pt_samples.device,
        )
        pt_samples = pt_samples[indices]
    metric_list = [
        MMD_energy(target, kernel_num=5, fix_sigma=1),
        TVD_energy(target),
        Wasserstain2_energy(target),
        Wasserstain2_data(target),
    ]
    train_ptdm(target, model, opt, pt_samples, metric_list, cfg)


def main_gtdm(target, cfg: DictConfig):
    cfg = setup(cfg)
    cfg["num_samples_to_generate_per_batch"] = (
        cfg["num_samples_to_generate_per_batch"]
        if cfg["num_samples_to_generate_per_batch"]
        else cfg["num_eval_samples"]
    )

    model = hydra.utils.instantiate(cfg.net).to(cfg.device)
    model = maybe_wrap_data_parallel(model, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    metric_list = [
        MMD_energy(target, kernel_num=5, fix_sigma=1),
        TVD_energy(target),
        Wasserstain2_energy(target),
        Wasserstain2_data(target),
    ]
    train_gtdm(target, model, opt, metric_list, cfg)


def main_pt(target, cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = setup(cfg)
    config = setup_pt(cfg)
    OmegaConf.set_struct(cfg, True)
    # aldp_config_path = "configs/aldp.yaml"
    if config["name"] == "aldp":
        init_position_file = (
            yaml.safe_load(open(config["aldp_config_path"], "r"))
            .get("data", {})
            .get("transform")
        )
        x_init = torch.load(init_position_file, map_location=config["device"])
        x_init = target.coordinate_transform.inverse(x_init)[0]
        x_init = x_init.unsqueeze(0).repeat(
            config["total_n_temp"], config["num_chains"], 1
        )
    elif config["name"] == "aldp_cart":
        init_position_file = (
            yaml.safe_load(open(config['aldp_config_path'], "r"))
            .get("data", {})
            .get("transform")
        )
        x_init = torch.load(init_position_file, map_location=config["device"])
        x_init = x_init.unsqueeze(0).repeat(
            config["total_n_temp"], config["num_chains"], 1
        )
    else:
        x_init = (
            torch.randn(
                (config["total_n_temp"], config["num_chains"], target.dim),
                device=config["device"],
            )
            * 0.01
        )
    parallel_tempering_simulation(target, x_init=x_init, config=cfg)


def main_sample(target, cfg: DictConfig):
    """
    Loads a pre-trained model, generates samples, evaluates them against ground truth,
    and saves the samples, visualizations, and evaluation metrics.

    Results are saved in a subdirectory within `cfg.model_save_path` named
    `new_samples/samples_{cfg.num_eval_samples}_is{cfg.importance_resample}_w{cfg.is_weight_max_quantile}_seed{cfg.seed}`.

    Saved results include:
    - Generated samples: `samples.pt`
    - Visualizations:
        - For AldpPotential or AldpPotentialCart: Plots logged to Weights & Biases (wandb).
        - For GMM: A scatter plot `samples.png`.
    - Evaluation metrics:
        - `eval_metrics.txt` (and `eval_metrics_w_norm.txt` for GMM if normalization is used)
          containing MMD, TVD, Wasserstein-2 distances.
        - Effective Sample Size (ESS) ratio is saved in `eval_metrics.txt` and logged to wandb
          if importance resampling is used.

    Key configuration parameters in `cfg`:
    - `model_save_path`: Base directory where models and results are stored. The specific
                         checkpoint is loaded from `model_save_path/checkpoints/model_1.0.pt`
                         or `model_save_path/model.pt`.
    - `num_eval_samples`: Total number of samples to generate for evaluation.
    - `num_samples_to_generate_per_batch`: Batch size for the sample generation process.
    - `net`: Configuration for instantiating the model.
    - `device`: The device to run computations on (e.g., 'cuda', 'cpu').
    - `importance_resample`: Boolean flag to enable/disable importance resampling.
    - `is_integration_steps`: (Used if `importance_resample` is True) Number of integration
                              steps for the SDE solver during importance sampling.
    - `tmax`: Maximum integration time for the SDE solver.
    - `normalization`: Factor used to scale the generated samples.
    - `is_weight_max_quantile`: (Used if `importance_resample` is True) Quantile for capping
                                 the importance sampling weights.
    - `seed`: Random seed for reproducibility.
    - `dim`: Dimensionality of the data.

    Args:
        target: The target distribution object (e.g., GMM, AldpPotential).
        cfg: A DictConfig object containing the configuration parameters.
    """
    cfg = setup(cfg)
    cfg["num_samples_to_generate_per_batch"] = (
        cfg["num_samples_to_generate_per_batch"]
        if cfg["num_samples_to_generate_per_batch"]
        else cfg["num_eval_samples"]
    )

    # Important params here:
    # cfg.num_eval_samples
    # cfg.num_samples_to_generate_per_batch
    # cfg.model_save_path
    # (the neural net params need to be input correctly in the config file directly)

    model = hydra.utils.instantiate(cfg.net).to(cfg.device)

    # Load the model from the specified checkpoint path
    checkpoint_path = os.path.join(cfg.model_save_path, 'checkpoints', 'model_1.0.pt')
    if os.path.exists(checkpoint_path):
        load_model(model, checkpoint_path, cfg.device)
        print(f"Model loaded from {checkpoint_path}")
    else:
        checkpoint_path = os.path.join(cfg.model_save_path, 'model.pt')
        if os.path.exists(checkpoint_path):
            load_model(model, checkpoint_path, cfg.device)
            print(f"Model loaded from {checkpoint_path}")
        else:
            raise ValueError(f"Model not found at {checkpoint_path}")

    model = maybe_wrap_data_parallel(model, cfg)

    from ptsd.targets.aldp_cartesian import AldpPotentialCart
    from ptsd.targets.aldp import AldpPotential
    from ptsd.targets.gmm import GMM

    with torch.no_grad():
        if cfg.importance_resample:
            sample_func = Euler_solve_wrapper(
                Euler_solve_with_log_pdf,
                num_samples_per_batch=cfg.num_samples_to_generate_per_batch,
            )
            samples, log_probs = sample_func(
                target,
                model,
                torch.randn([cfg.num_eval_samples, cfg.dim], device=cfg.device)
                * cfg.tmax,
                num_steps=cfg.is_integration_steps,
            )
            samples = samples * cfg.normalization
            target_log_probs = target.log_prob(samples)
            # Cap the maximum IS weight
            log_prob_threshold = torch.quantile(
                target_log_probs - log_probs, cfg.is_weight_max_quantile
            )
            log_prob_diff = target_log_probs - log_probs
            capped_diff = torch.minimum(log_prob_diff, log_prob_threshold)
            log_probs = target_log_probs - capped_diff
            generate_log_probs = target_log_probs - capped_diff
            samples, ess_ratio, weights = importance_sample_with_reweighting(
                samples=samples,
                model_log_probs=log_probs,
                target_log_probs=target_log_probs,
            )
        else:
            sample_func = Euler_solve_wrapper(
                Eular_solve, num_samples_per_batch=cfg.num_samples_to_generate_per_batch
            )
            samples = (
                sample_func(
                    target,
                    model,
                    torch.randn([cfg.num_eval_samples, cfg.dim], device=cfg.device)
                    * cfg.tmax,
                )
                * cfg.normalization
            )
            ess_ratio = None
        if isinstance(target, AldpPotentialCart):
            samples = target.reflect_d_to_l_cartesian(samples)

    save_path = os.path.join(
        cfg.model_save_path,
        'new_samples',
        f'samples_{cfg.num_eval_samples}_is{cfg.importance_resample}_w{cfg.is_weight_max_quantile}_seed{cfg.seed}',
    )
    os.makedirs(save_path, exist_ok=True)

    if isinstance(target, AldpPotential) or isinstance(target, AldpPotentialCart):
        # Get samples for visualization
        images_dict = target.plot_samples(
            samples.to(cfg.device),
            target.sample([cfg.num_eval_samples]).to(cfg.device),
            iter=0,
            metric_dir=save_path,
            plot_dir=save_path,
            batch_size=1000000,
        )
        for key, value in images_dict.items():
            wandb.log({f"samples/{key}": wandb.Image(value)})

    elif isinstance(target, GMM):
        # Get samples for visualization
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.scatter(
            samples[:, 0].cpu().numpy(),
            samples[:, 1].cpu().numpy(),
            label='samples',
            s=1,
        )
        gt_samples = target.sample([cfg.num_eval_samples])
        plt.scatter(
            gt_samples[:, 0].cpu().numpy(),
            gt_samples[:, 1].cpu().numpy(),
            label='gt samples',
            s=1,
        )
        plt.legend()
        plt.savefig(os.path.join(save_path, 'samples.png'))
        plt.close()

    # Save ESS to evaluation metrics file if available
    if ess_ratio is not None:
        eval_metrics_path = os.path.join(save_path, 'eval_metrics.txt')
        with open(eval_metrics_path, 'w') as f:
            f.write(f"ESS: {ess_ratio:.6f}\n")
        wandb.log({"samples/ess_ratio": ess_ratio})

    torch.save(samples, os.path.join(save_path, 'samples.pt'))

    if not (
        isinstance(target, AldpPotential)
        or isinstance(target, AldpPotentialCart)
        or isinstance(target, GMM)
    ):
        eval(
            gt_samples=target.sample([cfg.num_eval_samples]),
            sample_dict={
                "ptsd": samples.unsqueeze(0),
                "gt": target.sample([cfg.num_eval_samples]).unsqueeze(0),
            },
            metric_list=[
                MMD_energy(target, kernel_num=5, fix_sigma=1),
                TVD_energy(target),
                Wasserstain2_energy(target),
                Wasserstain2_data(target),
            ],
            save_path=save_path,
        )
    if isinstance(target, GMM):
        print("Evaluating GMM with normalization...")
        eval(
            gt_samples=target.sample([cfg.num_eval_samples]),
            sample_dict={
                "ptsd": samples.unsqueeze(0) / cfg.normalization,
                "gt": target.sample([cfg.num_eval_samples]).unsqueeze(0)
                / cfg.normalization,
            },
            metric_list=[
                MMD_energy(target, kernel_num=5, fix_sigma=1),
                TVD_energy(target),
                Wasserstain2_energy(target),
                Wasserstain2_data(target),
            ],
            save_path=save_path,
            file_name="eval_metrics_w_norm.txt",
        )


@hydra.main(version_base="1.3", config_path="configs/", config_name="gmm.yaml")
def main(cfg: DictConfig):
    # Set NCCL environment variables for stability
    os.environ["NCCL_DEBUG"] = "INFO"  # Provides more debugging information
    # AMD-specific optimizations for ROCm
    if torch.version.hip is not None:  # Check if running on ROCm/AMD
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"
        # Additional ROCm-specific settings
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["HSA_ENABLE_SDMA"] = "0"
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    target = hydra.utils.instantiate(cfg.target)
    os.makedirs(cfg.model_save_path, exist_ok=True)
    logger = setup_logging(cfg.model_save_path)
    if cfg.TRAINING_SET_SIZE is None:
        cfg.TRAINING_SET_SIZE = cfg.BUFFER_MAX_SIZE
    if "prefix" not in cfg:
        main_ptsd(target, cfg)
    else:
        if cfg.prefix == "ptdm":
            main_ptdm(target, cfg)
        elif cfg.prefix == "gtdm":
            main_gtdm(target, cfg)
        elif cfg.prefix == "sample":
            main_sample(target, cfg)
        elif cfg.prefix == "pt":
            main_pt(target, cfg)
        else:
            raise ValueError("prefix NOT found!")


if __name__ == "__main__":
    main()
