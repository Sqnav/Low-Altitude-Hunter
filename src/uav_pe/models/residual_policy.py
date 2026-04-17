#!/usr/bin/env python3

from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import Schedule


class ResidualActionHeadPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        base_action_head: nn.Module,
        residual_scale: float = 0.1,
        residual_head_arch: Optional[List[int]] = None,
        value_head_arch: Optional[List[int]] = None,
        log_std_init: float = -3.0,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        core_obs_dim: Optional[int] = None,
        append_prev_action_dim: int = 4,
        **kwargs: Any,
    ):
        value_head_arch = value_head_arch or [256, 256]
        residual_head_arch = residual_head_arch or [256, 256]

        object.__setattr__(self, "_base_action_head", base_action_head)
        self._residual_scale = float(residual_scale)
        self._value_head_arch = value_head_arch
        self._residual_head_arch = residual_head_arch
        self._log_std_init = float(log_std_init)
        self._core_obs_dim = int(core_obs_dim) if core_obs_dim is not None else None
        self._append_prev_action_dim = int(append_prev_action_dim)

        net_arch = net_arch or dict(pi=[256, 256], vf=value_head_arch)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            log_std_init=log_std_init,
            **kwargs,
        )

        for p in self._base_action_head.parameters():
            p.requires_grad = False

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        action_dim = get_action_dim(self.action_space)

        layers: List[nn.Module] = []
        residual_input_dim = latent_dim_pi + self._append_prev_action_dim + action_dim
        last_dim = residual_input_dim
        for h in self._residual_head_arch:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.residual_head = nn.Sequential(*layers)

        self.value_net = nn.Linear(latent_dim_vf, 1)

        log_std_init_vec = th.ones(action_dim, device=self.device) * self._log_std_init
        self.log_std = nn.Parameter(log_std_init_vec)

        self.action_dist = make_proba_distribution(self.action_space)

        if self.ortho_init:
            for module in [self.value_net] + [m for m in self.residual_head if isinstance(m, nn.Linear)]:
                module.apply(partial(self.init_weights, gain=1))

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = self.optimizer_class(
            trainable_params, lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _compute_means(self, obs: th.Tensor, latent_pi: th.Tensor) -> th.Tensor:
        head_dtype = next(self._base_action_head.parameters()).dtype

        if obs.dim() > 2:
            obs_flat = obs.view(obs.shape[0], -1)
        else:
            obs_flat = obs

        if self._core_obs_dim is None:
            raise RuntimeError("ResidualActionHeadPolicy message core_obs_dim，message obs。")

        expected_last_dim = self._core_obs_dim + self._append_prev_action_dim
        if obs_flat.shape[-1] != expected_last_dim:
            raise RuntimeError(
                f"[ResidualPolicy] obs message: "
                f"obs_flat.shape={tuple(obs_flat.shape)}, expected_last_dim={expected_last_dim}"
            )

        core_obs = obs_flat[:, : self._core_obs_dim]
        prev_action = obs_flat[
            :, self._core_obs_dim : self._core_obs_dim + self._append_prev_action_dim
        ]

        head_in_dim = None
        for m in self._base_action_head.modules():
            if isinstance(m, nn.Linear):
                head_in_dim = m.in_features
                break

        if head_in_dim is None:
            raise RuntimeError("message base_action_head message")

        if core_obs.shape[-1] != head_in_dim:
            raise RuntimeError(
                f"[ResidualPolicy] base_action_head message: "
                f"core_obs.shape={tuple(core_obs.shape)}, "
                f"head_in_dim={head_in_dim}. "
                f"message obs message。"
            )

        with th.no_grad():
            base_action = th.tanh(self._base_action_head(core_obs.to(head_dtype))).float()

        residual_input = th.cat([latent_pi, prev_action.float(), base_action.float()], dim=-1)
        residual_raw = self.residual_head(residual_input)
        residual_action = th.tanh(residual_raw)

        mean_actions = base_action + self._residual_scale * residual_action
        mean_actions = th.clamp(mean_actions, -1.0, 1.0)
        return mean_actions

    def _get_action_dist(self, obs: th.Tensor, latent_pi: th.Tensor):
        mean_actions = self._compute_means(obs, latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def get_distribution(self, obs: th.Tensor):
        features = self.extract_features(obs, self.features_extractor)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist(obs, latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(
            obs,
            self.vf_features_extractor if not self.share_features_extractor else self.features_extractor,
        )
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs, self.features_extractor)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features)
            latent_vf = self.mlp_extractor.forward_critic(features)

        distribution = self._get_action_dist(obs, latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs, self.features_extractor)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features)
            latent_vf = self.mlp_extractor.forward_critic(features)

        distribution = self._get_action_dist(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                residual_scale=self._residual_scale,
                residual_head_arch=self._residual_head_arch,
                value_head_arch=self._value_head_arch,
                log_std_init=self._log_std_init,
                core_obs_dim=self._core_obs_dim,
                append_prev_action_dim=self._append_prev_action_dim,
            )
        )
        return data

    def get_residual_head_state_dict(self) -> Dict[str, th.Tensor]:
        return self.residual_head.state_dict()