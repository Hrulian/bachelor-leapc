"""Provides a trainer for a Soft Actor-Critic algorithm that uses a differentiable MPC
layer in the policy network."""

from pathlib import Path
from typing import Any, Generator, NamedTuple

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from critic import SacCritic
from mlp import Mlp
from tanh_gaussian import SquashedGaussian

from leap_c.ocp.acados.controller import AcadosController
from leap_c.torch.rl.buffer import ReplayBuffer
from leap_c.torch.rl.sac import SacTrainerConfig
from leap_c.torch.rl.utils import soft_target_update
from leap_c.torch.utils.seed import mk_seed
from leap_c.trainer import Trainer
from leap_c.utils.gym import seed_env, wrap_env


class SacFopActorOutput(NamedTuple):
    """Output of the SAC-FOP actor.

    Attributes:
        param: The predicted parameters (which have been input into the controller).
        log_prob: The log-probability of the parameter distribution.
        stats: A dictionary containing several statistics of internal modules.
        action: The action output by the controller.
        status: The status of the MPC solver (`0` if successful).
        ctx: The context object containing information about the MPC solve.
    """

    param: torch.Tensor
    log_prob: torch.Tensor
    stats: dict[str, float]
    action: torch.Tensor
    status: torch.Tensor
    ctx: Any | None

    def select(self, mask: torch.Tensor) -> "SacFopActorOutput":
        """Select a subset of the output based on the given mask. Discards stats and ctx."""
        return SacFopActorOutput(
            self.param[mask],
            self.log_prob[mask],
            None,  # type:ignore
            self.action[mask],
            self.status[mask],
            None,
        )


class MpcRlActor(nn.Module):
    """An actor module for SAC-FOP, containing a differentiable MPC layer and injecting noise in the
    parameter space.

    Attributes:
        controller: The differentiable parameterized controller used to compute actions from
            parameters.
        mlp: The MLP used to predict the parameters of the controller from the observations.
        squashed_gaussian: The bounded distribution used to sample parameters.
    """

    controller: AcadosController
    mlp: Mlp
    squashed_gaussian: SquashedGaussian

    def __init__(
        self,
        controller: AcadosController,
        observation_space: spaces.Box,
    ) -> None:
        """Initializes the FOP actor.

        Args:
            controller: The differentiable parameterized controller used to compute actions from
                parameters.
            observation_space: The observation space of the environment.
        """
        super().__init__()
        self.controller = controller
        param_space = controller.param_space  # type: ignore
        param_dim = param_space.shape[0]  # type: ignore
        s_dim = observation_space.shape[0]
        self.squashed_gaussian = SquashedGaussian(space=param_space) # type: ignore
        self.mlp = Mlp(
            input_sizes=s_dim,
            output_sizes=[param_dim, param_dim], #type:ignore
        )

    def forward(
        self, obs: np.ndarray, ctx: Any | None = None, deterministic: bool = False
    ) -> SacFopActorOutput:
        """The given observations are passed to MLP, which is
        used to predict a bounded distribution in the (learnable) parameter space of the
        controller using the MLP. Afterwards, this parameters are sampled from this distribution,
        and passed to the controller, which then computes the final actions.

        Args:
            obs: The observations to compute the actions for.
            ctx: The optional context object containing information about the previous controller
                solve. Can be used, e.g., to warm-start the solver.
            deterministic: If `True`, use the mode of the distribution instead of sampling.
        """
        # TODO: Take a look at the forward pass.
        # It uses self.mlp to predict the mean and log-std of a Gaussian distribution
        # Then those are passed to the SquashedGaussian to get samples of a distribution
        # in the parameter space
        # The samples are then passed to the controller to get the actions
        mean, log_std = self.mlp(obs)
        param, log_prob, dist_stats = self.squashed_gaussian(mean, log_std, deterministic)
        ctx, action = self.controller(obs, param, ctx)

        return SacFopActorOutput(
            param,
            log_prob,
            {**dist_stats, **ctx.log}, #type:ignore
            action,
            ctx.status, #type:ignore
            ctx,
        )

class SacFopTrainer(Trainer[SacTrainerConfig]):
    """A trainer implementing Soft Actor-Critic (SAC) that uses a differentiable controller layer in
    the policy network (SAC-FOP).
    Injects parameter noise and uses an action critic.

    Attributes:
        train_env: The training environment.
        q: The Q-function approximator (critic).
        q_target: The target Q-function approximator.
        q_optim: The optimizer for the Q-function.
        pi: The policy network containing the parameterized controller (the actor).
        pi_optim: The optimizer for the policy network.
        log_alpha: The logarithm of the temperature parameter.
        target_entropy: The target entropy for the policy.
            If `None`, the temperature is fixed.
        buffer: The replay buffer used to store transitions.
    """

    train_env: gym.Env
    q: SacCritic
    q_target: SacCritic
    q_optim: torch.optim.Optimizer
    pi: MpcRlActor
    pi_optim: torch.optim.Optimizer
    log_alpha: nn.Parameter
    buffer: ReplayBuffer

    def __init__(
        self,
        cfg: SacTrainerConfig,
        val_env: gym.Env,
        output_path: str | Path,
        device: str,
        train_env: gym.Env,
        controller: AcadosController,
    ) -> None:
        """Initializes the SAC-FOP trainer.

        Args:
            cfg: The configuration for the trainer.
            val_env: The validation environment.
            output_path: The path to the output directory.
            device: The device on which the trainer is running.
            train_env: The training environment.
            controller: The controller to use in the policy.
        """
        super().__init__(cfg, val_env, output_path, device)

        observation_space = train_env.observation_space

        self.train_env = wrap_env(train_env)

        self.q = SacCritic(
            train_env.action_space, #type: ignore
            observation_space,
            cfg.num_critics,
        )
        self.q_target = SacCritic(
            train_env.action_space, #type: ignore
            observation_space,
            cfg.num_critics,
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = MpcRlActor(
            controller,
            observation_space #type:ignore
        )

        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.log_alpha = nn.Parameter(torch.tensor(cfg.init_alpha).log())  # type: ignore

        self.buffer = ReplayBuffer(
            cfg.buffer_size, device=device, collate_fn_map=controller.collate_fn_map
        )

    def train_loop(self) -> Generator[int, None, None]:
        is_terminated = is_truncated = True
        policy_state = None
        obs = None

        while True:
            # Execute a step ==================================================
            if is_terminated or is_truncated:
                obs, _ = seed_env(self.train_env, mk_seed(self.rng), {"mode": "train"})
                policy_state = None
                is_terminated = is_truncated = False

            obs_batched = self.buffer.collate([obs])

            with torch.no_grad():
                pi_output: SacFopActorOutput = self.pi(
                    obs_batched, policy_state, deterministic=False
                )
                action = pi_output.action.cpu().numpy()[0]
                param = pi_output.param.cpu().numpy()[0]

            obs_prime, reward, is_terminated, is_truncated, info = self.train_env.step(action)

            # Logging
            self.report_stats("train_trajectory", {"param": param, "action": action}, verbose=True)
            self.report_stats("train_policy_rollout", pi_output.stats, verbose=True)  # type: ignore
            if "episode" in info or "task" in info:
                self.report_stats("train", {**info.get("episode", {}), **info.get("task", {})})

            self.buffer.put(
                (
                    obs,
                    action,
                    reward,
                    obs_prime,
                    is_terminated,
                    pi_output.ctx,  # Store additional information that may optionally be used later
                )
            )

            obs = obs_prime
            policy_state = pi_output.ctx
            # End of Step ===============================================
            
            # Update networks ===========================================
            if (
                self.state.step >= self.cfg.train_start
                and len(self.buffer) >= self.cfg.batch_size
                and self.state.step % self.cfg.update_freq == 0
            ):
                # sample batch
                o, a, r, o_prime, te, ps_sol = self.buffer.sample(self.cfg.batch_size)

                # sample action
                pi_o = self.pi(o, ps_sol)
                with torch.no_grad():
                    pi_o_prime = self.pi(o_prime, ps_sol)

                pi_o_stats = pi_o.stats

                # Only use samples where the MPC solver was successful for both
                # current and next action.
                # Since we are differentiating through the MPC solver, 
                # this is necessary to avoid faulty gradients.
                mask_status = (pi_o.status == 0) & (pi_o_prime.status == 0)
                o = o[mask_status]
                a = a[mask_status]
                r = r[mask_status]
                o_prime = o_prime[mask_status]
                te = te[mask_status]
                pi_o = pi_o.select(mask_status)
                pi_o_prime = pi_o_prime.select(mask_status)

                # update critic
                alpha = self.log_alpha.exp().item()
                with torch.no_grad():
                    # NOTE: The SACCritic actually outputs a list of Q-values,
                    # one for each critic, which we concatenate here to take the minimum over.
                    # This is used to combat overestimation bias.
                    q_prime = torch.cat(self.q_target(o_prime, pi_o_prime.action), dim=1)
                    q_min = torch.min(q_prime, dim=1, keepdim=True).values
                    r = r[:, None]
                    te = te[:, None]
                    log_p_prime = pi_o_prime.log_prob
                    gamma = self.cfg.gamma

                    target = r + gamma * (1 - te) * (q_min - alpha * log_p_prime)
                    
                q = torch.cat(self.q(o, a), dim=1)
                q_loss = torch.mean((q - target).pow(2))

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()

                # update actor
                log_p = pi_o.log_prob
                q_pi = torch.cat(self.q(o, pi_o.action), dim=1)
                min_q_pi = torch.min(q_pi, dim=1).values
                pi_loss = (alpha * log_p - min_q_pi).mean()

                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()

                # soft updates
                soft_target_update(self.q, self.q_target, self.cfg.tau)

                # Logging
                loss_stats = {
                    "q_loss": q_loss.item(),
                    "pi_loss": pi_loss.item(),
                    "alpha": alpha,
                    "q": q.mean().item(),
                    "q_target": target.mean().item(),
                    "masked_samples_perc": 1 - float(mask_status.mean().item()),
                    "entropy": -log_p.mean().item(),
                }
                self.report_stats("loss", loss_stats)
                self.report_stats("train_policy_update", pi_o_stats, verbose=True)

            yield 1

    def act(
        self, obs: np.ndarray, deterministic: bool = False, state: Any | None = None
    ) -> tuple[np.ndarray, Any, dict[str, float]]:
        obs = self.buffer.collate([obs])

        with torch.no_grad():
            pi_output: SacFopActorOutput = self.pi(obs, state, deterministic)

        action = pi_output.action.cpu().numpy()[0]

        return action, pi_output.ctx, pi_output.stats

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        optimizers = [self.q_optim, self.pi_optim]
        return optimizers

    def periodic_ckpt_modules(self) -> list[str]:
        return ["q", "pi", "q_target", "log_alpha"]

    def singleton_ckpt_modules(self) -> list[str]:
        return ["buffer"]
