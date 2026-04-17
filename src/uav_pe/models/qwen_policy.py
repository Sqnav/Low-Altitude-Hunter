#!/usr/bin/env python

import torch
import torch.nn as nn
import warnings
from dataclasses import dataclass, field
from transformers import (
    AutoModel,
    AutoProcessor,
)
from transformers.utils import ModelOutput

from Train_qwen.core.action_mapping import norm_action_to_physical
warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The tokenizer has new PAD/BOS/EOS tokens.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*model config and generation config.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Updated tokens.*", category=UserWarning)


class ActionL1Loss(nn.Module):
    def __init__(self, reduction: str = "mean", yaw_weight: float = 1.0, beta: float = 1.0):
        super(ActionL1Loss, self).__init__()
        self.reduction = reduction
        self.yaw_weight = float(yaw_weight)
        self.beta = float(beta)

    def forward(self, pred, gt, reduction=None):
        if reduction is None:
            reduction = self.reduction

        per_dim = torch.nn.functional.smooth_l1_loss(
            pred, gt, reduction="none", beta=self.beta
        )

        weights = per_dim.new_tensor([1.0, 1.0, 1.0, self.yaw_weight])
        per_frame_loss = (per_dim * weights).sum(dim=-1)

        if reduction == "none":
            return per_frame_loss
        elif reduction == "mean":
            return per_frame_loss.mean()
        elif reduction == "sum":
            return per_frame_loss.sum()
        else:
            raise ValueError(f"Unknown reduction={reduction}")


def trajectory_balanced_mean(per_frame_loss: torch.Tensor, traj_ids: torch.Tensor):
    assert per_frame_loss.shape == traj_ids.shape, \
        f"per_frame_loss shape {per_frame_loss.shape} != traj_ids shape {traj_ids.shape}"
    assert len(per_frame_loss.shape) == 1, f"per_frame_loss should be 1D, got {per_frame_loss.shape}"
    
    if per_frame_loss.numel() == 0:
        return torch.tensor(0.0, device=per_frame_loss.device, dtype=per_frame_loss.dtype)
    
    unique_traj_ids = torch.unique(traj_ids)
    
    traj_losses = []
    for traj_id in unique_traj_ids:
        mask = (traj_ids == traj_id)
        traj_loss = per_frame_loss[mask].mean()
        traj_losses.append(traj_loss)
    
    if len(traj_losses) > 0:
        return torch.stack(traj_losses).mean()
    else:
        return torch.tensor(0.0, device=per_frame_loss.device, dtype=per_frame_loss.dtype)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-2B-Instruct")
    model_max_length: int = field(default=2048)
    freeze_backbone: bool = field(default=False)
    use_numeric_encoder: bool = field(default=True, metadata={"help": "Enable numeric encoder (always enabled in this template)."})
    use_backbone: bool = field(default=True, metadata={"help": "Enable backbone (always enabled in this template)."})

class UAVQwen3VLModel(nn.Module):
    def __init__(self, model_name_or_path, model_max_length=2048, freeze_backbone=False, use_device_map_auto=None, use_numeric_encoder=True, use_backbone=True):
        super().__init__()
        import os
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        is_distributed = local_rank >= 0 or world_size > 1
        self.use_backbone = use_backbone
        
        load_kwargs = {
            "trust_remote_code": True,
            "output_hidden_states": True,
        }
        if hasattr(torch, "float16"):
            load_kwargs["torch_dtype"] = torch.float16
        try:
            import inspect
            from transformers.modeling_utils import PreTrainedModel
            if "dtype" in inspect.signature(PreTrainedModel.from_pretrained).parameters:
                load_kwargs["dtype"] = load_kwargs.pop("torch_dtype", torch.float16)
        except Exception:
            pass
        if use_device_map_auto is None:
            use_device_map_auto = not is_distributed
        if use_device_map_auto:
            load_kwargs["device_map"] = "auto"
        if isinstance(model_name_or_path, str) and (model_name_or_path.startswith("/") or os.path.sep in model_name_or_path):
            load_kwargs["local_files_only"] = True

        import logging
        tlogger = logging.getLogger("transformers.modeling_utils")
        troot = logging.getLogger("transformers")
        old_level, old_root = tlogger.level, troot.level
        tlogger.setLevel(logging.ERROR)
        troot.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized from the model checkpoint.*")
            warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*were not used when.*")
            try:
                load_kwargs_keymap = dict(load_kwargs)
                load_kwargs_keymap["key_mapping"] = {r"^model\.(.*)": r"language_model.\1"}
                self.backbone = AutoModel.from_pretrained(model_name_or_path, **load_kwargs_keymap)
            except TypeError:
                self.backbone = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
        tlogger.setLevel(old_level)
        troot.setLevel(old_root)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*The tokenizer has new PAD/BOS/EOS tokens.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*model config and generation config.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*Updated tokens.*", category=UserWarning)
        processor_kwargs = {"trust_remote_code": True}
        if load_kwargs.get("local_files_only"):
            processor_kwargs["local_files_only"] = True
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        config = self.backbone.config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            hidden_size = 2048
        
        self.use_numeric_encoder = use_numeric_encoder
        if use_numeric_encoder:
            self.num_dim = 3
            self.num_hidden_dim = 128
            self.numeric_encoder = nn.Sequential(
                nn.Linear(self.num_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.num_hidden_dim),
                nn.GELU(),
            )
            fused_dim = (hidden_size + self.num_hidden_dim) if use_backbone else self.num_hidden_dim
            self.action_head = nn.Sequential(
                nn.Linear(fused_dim, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 4),
            )
        else:
            self.action_head = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 4),
            )
        
        if is_distributed:
            pass
        else:
            try:
                first_param = next(self.backbone.parameters())
                backbone_device = first_param.device
                backbone_dtype = first_param.dtype
            except StopIteration:
                backbone_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                backbone_dtype = torch.float16
            self.action_head = self.action_head.to(device=backbone_device, dtype=backbone_dtype)
            if use_numeric_encoder:
                self.numeric_encoder = self.numeric_encoder.to(device=backbone_device, dtype=backbone_dtype)
        
        yaw_w = getattr(self, "yaw_loss_weight", 1.0)
        self.loss_fn = ActionL1Loss(reduction="mean", yaw_weight=yaw_w, beta=1.0)
        self.sign_loss_weight = 0.0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self, 'backbone') and hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        if hasattr(self.backbone, 'prepare_inputs_for_generation'):
            return self.backbone.prepare_inputs_for_generation(*args, **kwargs)
        else:
            return kwargs
    
    def _reorder_cache(self, past, beam_idx):
        if hasattr(self.backbone, '_reorder_cache'):
            return self.backbone._reorder_cache(past, beam_idx)
        else:
            return past
    
    def generate(self, *args, **kwargs):
        if hasattr(self.backbone, 'generate'):
            return self.backbone.generate(*args, **kwargs)
        else:
            raise NotImplementedError("generate() is not supported by this model.")

    def forward(
        self,
        input_ids=None,
        num_state=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        action=None,
        traj_id=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if not getattr(self, "use_backbone", True):
            if not getattr(self, "use_numeric_encoder", False):
                raise ValueError("[UAVQwen3VLModel.forward] use_backbone=False requires use_numeric_encoder=True.")
            if num_state is None:
                raise ValueError("[UAVQwen3VLModel.forward] use_backbone=False requires num_state.")
            if num_state.dim() != 2 or num_state.shape[1] != self.num_dim:
                raise ValueError(
                    f"[UAVQwen3VLModel.forward] num_state shape should be [B, {self.num_dim}], "
                    f"but got {tuple(num_state.shape)}"
                )
            num_feat = num_state
            try:
                first_param = next(self.backbone.parameters())
                target_device = first_param.device
                target_dtype = first_param.dtype
            except StopIteration:
                target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                target_dtype = torch.float16
            if num_feat.device != target_device:
                num_feat = num_feat.to(device=target_device)
            if num_feat.dtype != target_dtype:
                num_feat = num_feat.to(dtype=target_dtype)
            num_feat = self.numeric_encoder(num_feat)
            raw_action = self.action_head(num_feat)
        else:
            if (input_ids is None) and (inputs_embeds is None):
                raise ValueError(
                    "[UAVQwen3VLModel.forward] Must provide input_ids or inputs_embeds; both are None."
                )
            if (input_ids is not None) and (inputs_embeds is not None):
                raise ValueError(
                    "[UAVQwen3VLModel.forward] input_ids and inputs_embeds cannot be provided together."
                )

            backbone_kwargs = {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if input_ids is not None:
                backbone_kwargs["input_ids"] = input_ids
            else:
                backbone_kwargs["inputs_embeds"] = inputs_embeds

            outputs = self.backbone(**backbone_kwargs)
            
            hidden = outputs.hidden_states[-1]
            
            seq_lengths = attention_mask.sum(dim=1) - 1
            seq_lengths = torch.clamp(seq_lengths, min=0, max=hidden.shape[1] - 1)
            state = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lengths]

            if getattr(self, "use_numeric_encoder", False):
                if num_state is None:
                    raise ValueError("[UAVQwen3VLModel.forward] num_state is required but got None.")
                if num_state.dim() != 2 or num_state.shape[1] != self.num_dim:
                    raise ValueError(
                        f"[UAVQwen3VLModel.forward] num_state shape should be [B, {self.num_dim}], "
                        f"but got {tuple(num_state.shape)}"
                    )
                if num_state.device != state.device:
                    num_state = num_state.to(device=state.device)
                if num_state.dtype != state.dtype:
                    num_state = num_state.to(dtype=state.dtype)
                num_feat = self.numeric_encoder(num_state)
                fused_state = torch.cat([state, num_feat], dim=-1)
                raw_action = self.action_head(fused_state)
            else:
                raw_action = self.action_head(state)
        norm_action = torch.tanh(raw_action)

        max_speed = getattr(self, "max_speed", 5.0)
        max_yaw_rate = getattr(self, "max_yaw_rate", 45.0)
        pred_action = norm_action_to_physical(norm_action, max_speed, max_yaw_rate)
        
        loss = None
        main_loss = None
        sign_loss = None
        if action is not None:
            if action.shape != pred_action.shape:
                raise ValueError(
                    f"Action shape mismatch: pred_action.shape={pred_action.shape}, action.shape={action.shape}"
                )
            if action.device != pred_action.device:
                action = action.to(device=pred_action.device)
            if action.dtype != pred_action.dtype:
                action = action.to(dtype=pred_action.dtype)
            
            scale = pred_action.new_tensor([max_speed, max_speed, max_speed, max_yaw_rate])
            pred_norm = pred_action / scale
            action_norm = action / scale
            per_frame_loss = self.loss_fn(pred_norm, action_norm, reduction="none")

            if traj_id is not None:
                if traj_id.device != per_frame_loss.device:
                    traj_id = traj_id.to(device=per_frame_loss.device)
                main_loss = trajectory_balanced_mean(per_frame_loss, traj_id)
            else:
                main_loss = per_frame_loss.mean()
            loss = main_loss
            sign_w = getattr(self, "sign_loss_weight", 0.0)
            if sign_w > 0:
                eps = getattr(self, "sign_loss_eps", 1e-3)
                with torch.no_grad():
                    mask = (action.abs() > eps).to(pred_action.dtype)
                sign_term = torch.nn.functional.softplus(-pred_action * action)
                masked = sign_term * mask
                if mask.sum() > 0:
                    sign_loss = masked.sum() / mask.sum()
                    loss = loss + sign_w * sign_loss
            if loss.dim() > 0:
                loss = loss.sum()
        
        return ModelOutput(
            loss=loss,
            action=norm_action,
        )
