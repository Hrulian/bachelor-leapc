import datetime
from pathlib import Path

import torch
from sac_fop import SacFopTrainer, SacTrainerConfig

from leap_c.examples.cartpole.controller import CartPoleController, CartPoleControllerConfig
from leap_c.examples.cartpole.env import CartPoleEnv
from leap_c.run import init_run


def create_cfg(seed: int) -> SacTrainerConfig:
    # ---- Configuration ----
    cfg = SacTrainerConfig()

    # ---- Section: cfg.trainer ----
    cfg.seed = seed
    cfg.train_steps = 200_000
    cfg.val_freq = 10_000
    cfg.val_num_render_rollouts = 1
    cfg.val_render_mode = "rgb_array"
    cfg.ckpt_modus = "all"
    cfg.buffer_size = 1_000_000
    cfg.gamma = 0.99
    cfg.tau = 0.005
    cfg.soft_update_freq = 1
    cfg.lr_q = 0.001
    cfg.lr_pi = 0.001
    cfg.init_alpha = 0.02
    cfg.report_loss_freq = 100
    cfg.update_freq = 4

    # ---- Section: cfg.trainer.log ----
    cfg.log.verbose = True
    cfg.log.interval = 1_000
    cfg.log.window = 10_000
    cfg.log.csv_logger = True
    cfg.log.tensorboard_logger = True
    cfg.log.wandb_logger = False
    cfg.log.wandb_init_kwargs = {}

    return cfg


def run_sac_fop(
    cfg: SacTrainerConfig,
    use_checkpoint: bool = True,
    device: str = "cuda",
) -> float:
    """
    Args:
        cfg: The configuration for running the controller.
        output_path: The path to save outputs to.
            If it already exists, the run will continue from the last checkpoint.
        device: The device to use.
    """
    if use_checkpoint:
        output_path = Path("output") / f"sac_fop_cartpole_seed_{seed}"
    else:
        now = datetime.datetime.now()
        date = now.strftime("%Y_%m_%d")
        time = now.strftime("%H_%M_%S")
        output_path = Path("output") / f"sac_fop_cartpole_seed_{seed}_{date}_{time}"
    controller_cfg = CartPoleControllerConfig()
    controller = CartPoleController(controller_cfg)
    
    trainer = SacFopTrainer(
        val_env=CartPoleEnv(render_mode="rgb_array"),
        train_env=CartPoleEnv(),
        controller=controller,
        output_path=output_path,
        device=device,
        cfg=cfg,
    )
    if use_checkpoint:
        # Quickfix
        trainer.state.step = 20000
    init_run(trainer, cfg, output_path)

    return trainer.run()


if __name__ == "__main__":
    seed = 1337
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    cfg = create_cfg(seed)

    run_sac_fop(cfg=cfg, use_checkpoint=True, device=device)
