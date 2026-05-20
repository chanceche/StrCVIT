# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
import json
import os
import atexit
from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)

from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class StrLoRAConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_Sample`]
    """
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)
    topk: int = field(default=0)
    subtopk: int = field(default=0)
    attn_proj_dim: int = field(default=0)

    sample_router_distill_lambda: float = field(default=0.0)
    sample_router_distill_beta: float = field(default=0.99)
    sample_router_distill_tau: float = field(default=1.0)
    freeze_qk: bool = field(default=False)
    freeze_expo: bool = field(default=False)

    def __post_init__(self):
        self.peft_type = PeftType.STRLORA


class StrLoRAModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])
        
        # Register auto-save at exit
        # atexit.register(self.save_router_stats)

    def save_router_stats(self, path=None):
        # Check for distributed setting
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = 0
        if is_distributed:
            rank = torch.distributed.get_rank()

        if path is None:
            if is_distributed:
                path = "router_stats_global.json"
            else:
                pid = os.getpid()
                path = f"router_stats_pid{pid}.json"
        
        # Check if file exists to avoid overwriting (only needed on rank 0)
        if rank == 0:
            if os.path.exists(path):
                base, ext = os.path.splitext(path)
                i = 1
                while os.path.exists(f"{base}_{i}{ext}"):
                    i += 1
                path = f"{base}_{i}{ext}"
        
        stats = {}
        # Iterate over all modules to find StrLoRALinear
        for name, module in self.model.named_modules():
            if isinstance(module, StrLoRALinear):
                if hasattr(module, 'expert_counts'):
                    # Clone counts to avoid modifying the buffer in-place during reduction if that's a concern
                    counts = module.expert_counts.clone()
                    
                    if is_distributed:
                        # Sum counts across all processes
                        # This must be called on all processes to work
                        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
                    
                    # Only rank 0 prepares the stats for saving
                    if rank == 0:
                        counts = counts.cpu()
                        total = counts.sum()
                        # Calculate distribution (percentage)
                        distribution = (counts / total).tolist() if total > 0 else torch.zeros_like(counts).tolist()
                        
                        stats[name] = {
                            "counts": counts.tolist(),
                            "distribution": distribution
                        }
        
        if rank == 0:
            if stats:
                try:
                    with open(path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"[StrLoRA] Router statistics saved to {path}")
                except Exception as e:
                    print(f"[StrLoRA] Failed to save router statistics: {e}")
            else:
                print(f"[StrLoRA] No router statistics collected. File {path} was not created.")

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_moelora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def clear_router_distill_loss(self):
        for module in self.model.modules():
            if isinstance(module, StrLoRALinear):
                module._last_router_distill_loss = None

    def get_router_distill_loss(self):
        distill_losses = []
        for module in self.model.modules():
            if isinstance(module, StrLoRALinear) and module._last_router_distill_loss is not None:
                distill_losses.append(module._last_router_distill_loss)
        if not distill_losses:
            return None
        return torch.stack(distill_losses).mean()

    @torch.no_grad()
    def update_router_distill_ema(self):
        for module in self.model.modules():
            if isinstance(module, StrLoRALinear):
                module.update_router_distill_ema()

    def get_router_distill_ema_state(self):
        state = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, StrLoRALinear):
                continue
            adapter_name = module.active_adapter
            module_state = {}
            if getattr(module, 'attn_proj_dim', 0) > 0:
                if adapter_name in module.lora_ema_attn_wq_weight:
                    module_state['attn_wq_weight'] = (
                        module.lora_ema_attn_wq_weight[adapter_name].detach().cpu().clone())
                if adapter_name in module.lora_ema_attn_wk_weight:
                    module_state['attn_wk_weight'] = (
                        module.lora_ema_attn_wk_weight[adapter_name].detach().cpu().clone())
                if not getattr(module, 'freeze_expo', False) and adapter_name in module.lora_ema_attn_expert_proto:
                    module_state['attn_expert_proto'] = (
                        module.lora_ema_attn_expert_proto[adapter_name].detach().cpu().clone())
            elif getattr(module, 'subtopk', 0) > 0 and hasattr(module, 'lora_ema_sub_router_weight'):
                if adapter_name in module.lora_ema_sub_router_weight:
                    module_state['sub_router_weight'] = (
                        module.lora_ema_sub_router_weight[adapter_name].detach().cpu().clone())
            if module_state:
                state[name] = module_state
        return state

    def load_router_distill_ema_state(self, state):
        loaded = 0
        for name, module in self.model.named_modules():
            if not isinstance(module, StrLoRALinear):
                continue
            module_state = state.get(name)
            if not module_state:
                continue
            adapter_name = module.active_adapter
            if 'attn_wq_weight' in module_state:
                module._set_ema_parameter(
                    module.lora_ema_attn_wq_weight,
                    adapter_name,
                    module_state['attn_wq_weight'],
                    reference_tensor=module.lora_attn_wq[adapter_name].weight,
                )
            if 'attn_wk_weight' in module_state:
                module._set_ema_parameter(
                    module.lora_ema_attn_wk_weight,
                    adapter_name,
                    module_state['attn_wk_weight'],
                    reference_tensor=module.lora_attn_wk[adapter_name].weight,
                )
            if 'attn_expert_proto' in module_state and not getattr(module, 'freeze_expo', False):
                module._set_ema_parameter(
                    module.lora_ema_attn_expert_proto,
                    adapter_name,
                    module_state['attn_expert_proto'],
                    reference_tensor=module.lora_attn_expert_proto[adapter_name],
                )
                loaded += 1
            elif 'attn_wq_weight' in module_state or 'attn_wk_weight' in module_state:
                loaded += 1
            elif 'sub_router_weight' in module_state:
                if not hasattr(module, 'lora_ema_sub_router_weight'):
                    module.lora_ema_sub_router_weight = nn.ParameterDict({})
                module._set_ema_parameter(
                    module.lora_ema_sub_router_weight,
                    adapter_name,
                    module_state['sub_router_weight'],
                    reference_tensor=module.lora_sub_router[adapter_name].weight,
                )
                loaded += 1
        return loaded


    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
            "topk": getattr(lora_config, "topk", 0),
            "subtopk": getattr(lora_config, "subtopk", 0),
            "attn_proj_dim": getattr(lora_config, "attn_proj_dim", 0),
            "sample_router_distill_lambda": getattr(lora_config, "sample_router_distill_lambda", 0.0),
            "sample_router_distill_beta": getattr(lora_config, "sample_router_distill_beta", 0.99),
            "sample_router_distill_tau": getattr(lora_config, "sample_router_distill_tau", 1.0),
            "freeze_qk": getattr(lora_config, "freeze_qk", False),
            "freeze_expo": getattr(lora_config, "freeze_expo", False),
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            if self.model is not None:
                kwargs["root_model"] = self
            new_module = StrLoRALinear(adapter_name, in_features, out_features, 
                                                    bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_moelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

class StrLoRALayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num

    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: SampleMOELinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: SampleMOELinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)

class StrLoRALinear(nn.Linear, StrLoRALayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.topk = kwargs.pop("topk", 0)
        self.subtopk = kwargs.pop("subtopk", 0)
        root_model = kwargs.pop("root_model", None)
        self.attn_proj_dim = kwargs.pop("attn_proj_dim", 0)
        self.sample_router_distill_lambda = kwargs.pop("sample_router_distill_lambda", 0.0)
        self.sample_router_distill_beta = kwargs.pop("sample_router_distill_beta", 0.99)
        self.sample_router_distill_tau = kwargs.pop("sample_router_distill_tau", 1.0)
        self.freeze_qk = kwargs.pop("freeze_qk", False)
        self.freeze_expo = kwargs.pop("freeze_expo", False)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        object.__setattr__(self, "root_model", root_model)

        StrLoRALayer.__init__(self, in_features=in_features, 
                               out_features=out_features, 
                               expert_num=self.expert_num)

        #32 64 128
        self.lora_attn_wq = nn.ModuleDict({})
        self.lora_attn_wk = nn.ModuleDict({})
        self.lora_attn_expert_proto = nn.ParameterDict({})
        self.lora_ema_attn_wq_weight = nn.ParameterDict({})
        self.lora_ema_attn_wk_weight = nn.ParameterDict({})
        self.lora_ema_attn_expert_proto = nn.ParameterDict({})

        if self.attn_proj_dim > 0:
            self.lora_attn_wq.update(
                nn.ModuleDict({adapter_name: torch.nn.Linear(in_features, self.attn_proj_dim, bias=False)})
            )

            self.lora_attn_wk.update(
                nn.ModuleDict({adapter_name: torch.nn.Linear(self.te_dim, self.attn_proj_dim, bias=False)})
            )

            self.lora_attn_expert_proto.update(
                nn.ParameterDict({
                    adapter_name: torch.nn.Parameter(torch.empty(self.expert_num, self.attn_proj_dim))
                })
            )


            nn.init.xavier_uniform_(self.lora_attn_wq[adapter_name].weight)
            nn.init.xavier_uniform_(self.lora_attn_wk[adapter_name].weight)
            nn.init.normal_(self.lora_attn_expert_proto[adapter_name], mean=0.0, std=self.attn_proj_dim ** -0.5)
            with torch.no_grad():
                self.lora_attn_expert_proto[adapter_name].copy_(
                    F.normalize(self.lora_attn_expert_proto[adapter_name], dim=-1)
                )
            if self._use_router_distill():
                if not self.freeze_qk:
                    self._set_ema_parameter(
                        self.lora_ema_attn_wq_weight,
                        adapter_name,
                        self.lora_attn_wq[adapter_name].weight,
                        reference_tensor=self.lora_attn_wq[adapter_name].weight,
                    )
                    self._set_ema_parameter(
                        self.lora_ema_attn_wk_weight,
                        adapter_name,
                        self.lora_attn_wk[adapter_name].weight,
                        reference_tensor=self.lora_attn_wk[adapter_name].weight,
                    )
                if not self.freeze_expo:
                    self._set_ema_parameter(
                        self.lora_ema_attn_expert_proto,
                        adapter_name,
                        self.lora_attn_expert_proto[adapter_name],
                        reference_tensor=self.lora_attn_expert_proto[adapter_name],
                    )


        if self.subtopk > 0:
            self.lora_sub_router = nn.ModuleDict({})
            self.lora_sub_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}))
            if self._use_router_distill() and self.attn_proj_dim <= 0:
                self.lora_ema_sub_router_weight = nn.ParameterDict({})
                self._set_ema_parameter(
                    self.lora_ema_sub_router_weight,
                    adapter_name,
                    self.lora_sub_router[adapter_name].weight,
                    reference_tensor=self.lora_sub_router[adapter_name].weight,
                )

    
        # Initialize expert counts
        # Use a regular attribute instead of a buffer to avoid DDP synchronization overwriting local counts
        self.expert_counts = torch.zeros(self.expert_num)
        # init the Gate network
        self.lora_router = nn.ModuleDict({})
        self.lora_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.te_dim, self.expert_num, bias=False)}))
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self._last_router_distill_loss = None

    def _use_router_distill(self):
        return self.sample_router_distill_lambda > 0 and self.subtopk > 0

    def _clone_ema_tensor(
        self,
        tensor: torch.Tensor,
        reference_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cloned = tensor.detach().clone()
        if reference_tensor is not None:
            return cloned.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        return cloned

    def _set_ema_parameter(
        self,
        parameter_dict: nn.ParameterDict,
        adapter_name: str,
        tensor: torch.Tensor,
        reference_tensor: torch.Tensor | None = None,
    ) -> None:
        parameter_dict[adapter_name] = nn.Parameter(
            self._clone_ema_tensor(tensor, reference_tensor=reference_tensor), requires_grad=False)

    @staticmethod
    def _get_masked_probs(logits: torch.Tensor, expert_mask: torch.Tensor | None = None, tau: float = 1.0):
        if tau <= 0:
            raise ValueError(f"sample_router_distill_tau must be positive, got {tau}")
        logits = logits.float() / tau
        if expert_mask is None:
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs.exp(), log_probs

        mask = expert_mask.bool()
        if mask.dim() == 3 and mask.size(1) == 1 and logits.size(1) != 1:
            mask = mask.expand(logits.size(0), logits.size(1), logits.size(2))
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        valid_mask = mask.any(dim=-1, keepdim=True)
        masked_logits = torch.where(valid_mask, masked_logits, logits)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        return log_probs.exp(), log_probs

    @staticmethod
    def _straight_through_topk(probs: torch.Tensor, k: int):
        if k <= 0 or k >= probs.size(-1):
            return probs
        top_val, top_idx = torch.topk(probs, k=k, dim=-1)
        hard = torch.zeros_like(probs).scatter_(-1, top_idx, top_val)
        hard = hard / (hard.sum(dim=-1, keepdim=True) + 1e-9)
        # Forward uses sparse top-k routing; backward follows dense probabilities.
        return hard.detach() - probs.detach() + probs

    def _compute_router_distill_loss(self, student_logits, teacher_logits, expert_mask=None):
        token_kl = self._compute_router_kl_per_token(student_logits, teacher_logits, expert_mask)
        return token_kl.mean() * self.sample_router_distill_lambda

    def _compute_router_kl_per_token(self, student_logits, teacher_logits, expert_mask=None):
        teacher_probs, teacher_log_probs = self._get_masked_probs(
            teacher_logits, expert_mask, tau=self.sample_router_distill_tau)
        _, student_log_probs = self._get_masked_probs(
            student_logits, expert_mask, tau=self.sample_router_distill_tau)
        if expert_mask is None:
            valid_mask = teacher_probs > 0
        else:
            valid_mask = expert_mask.bool()
            if valid_mask.dim() == 3 and valid_mask.size(1) == 1 and teacher_probs.size(1) != 1:
                valid_mask = valid_mask.expand(teacher_probs.size(0), teacher_probs.size(1), teacher_probs.size(2))
            valid_mask = valid_mask & (teacher_probs > 0)
        kl = torch.zeros_like(student_log_probs)
        kl[valid_mask] = teacher_probs.detach()[valid_mask] * (
            teacher_log_probs.detach()[valid_mask] - student_log_probs[valid_mask])
        return kl.sum(dim=-1)

    def _get_distill_chunk_size(self, seq_len: int) -> int:
        return min(seq_len, 128)

    @staticmethod
    @torch.no_grad()
    def _ema_update_weight(ema_weight: torch.Tensor, weight: torch.Tensor, beta: float):
        ema_weight.mul_(beta).add_(
            weight.data.detach().to(device=ema_weight.device, dtype=ema_weight.dtype), alpha=1 - beta)

    @torch.no_grad()
    def update_router_distill_ema(self):
        if not self._use_router_distill():
            return
        beta = self.sample_router_distill_beta
        adapter_name = self.active_adapter
        if self.attn_proj_dim > 0:
            if not self.freeze_qk:
                self._ema_update_weight(
                    self.lora_ema_attn_wq_weight[adapter_name], self.lora_attn_wq[adapter_name].weight, beta)
                self._ema_update_weight(
                    self.lora_ema_attn_wk_weight[adapter_name], self.lora_attn_wk[adapter_name].weight, beta)
            if not self.freeze_expo and adapter_name in self.lora_ema_attn_expert_proto:
                ema_proto = self.lora_ema_attn_expert_proto[adapter_name]
                proto = self.lora_attn_expert_proto[adapter_name]
                ema_proto.mul_(beta).add_(
                    proto.data.detach().to(device=ema_proto.device, dtype=ema_proto.dtype),
                    alpha=1 - beta)

        if self.subtopk > 0 and self.attn_proj_dim <= 0:
            self._ema_update_weight(
                self.lora_ema_sub_router_weight[adapter_name], self.lora_sub_router[adapter_name].weight, beta)

    def _attn_token_router(
        self,
        x: torch.Tensor,            # [B, L, H] local visual tokens h_i
        text_vec: torch.Tensor,     # [B, te_dim] pooled text embedding
        expert_mask: torch.Tensor | None = None,  # [B, L, E] or [B, 1, E] or None
        use_ema: bool = False,
    ):
        """
        Return:
            router: [B, L, E]
            logits: [B, L, E] (optional for debug)
        """
        # Use the original routing path when the attention router is disabled.
        if getattr(self, "attn_proj_dim", 0) <= 0:
            return None, None
        text_vec = F.layer_norm(text_vec, (text_vec.size(-1),))
        B, L, H = x.shape
        E = self.expert_num
        D = self.attn_proj_dim

        if use_ema:
            lora_attn_wq_weight = self.lora_ema_attn_wq_weight[self.active_adapter].to(device=x.device, dtype=x.dtype)
            lora_attn_wk_weight = self.lora_ema_attn_wk_weight[self.active_adapter].to(device=x.device, dtype=x.dtype)
            if self.freeze_expo:
                expert_proto = self.lora_attn_expert_proto[self.active_adapter]
            else:
                expert_proto = self.lora_ema_attn_expert_proto[self.active_adapter].to(device=x.device, dtype=x.dtype)
        else:
            lora_attn_wq = self.lora_attn_wq[self.active_adapter].to(device=x.device, dtype=x.dtype)
            lora_attn_wk = self.lora_attn_wk[self.active_adapter].to(device=x.device, dtype=x.dtype)
            expert_proto = self.lora_attn_expert_proto[self.active_adapter]
        if use_ema:
            q = F.linear(x, lora_attn_wq_weight, bias=None)
        else:
            q = lora_attn_wq(x)

        text_vec = text_vec.to(device=x.device, dtype=x.dtype)
        if use_ema:
            k = F.linear(text_vec, lora_attn_wk_weight, bias=None)
        else:
            k = lora_attn_wk(text_vec)

        p = F.normalize(expert_proto, dim=-1)
        kp = k[:, None, :] * p[None, :, :]
        logits = (q.unsqueeze(2) * kp.unsqueeze(1)).sum(dim=-1)
        logits = logits * (D ** -0.5)

        if expert_mask is not None:
            if expert_mask.dim() == 3 and expert_mask.size(1) == 1 and L != 1:
                expert_mask = expert_mask.expand(B, L, E)
            logits = logits.masked_fill(~expert_mask.bool(), float("-inf"))

        router = torch.softmax(logits, dim=-1)
        router = self._straight_through_topk(router, getattr(self, "subtopk", 0))

        return router, logits

    def _compute_attn_router_distill_loss_chunked(self, x: torch.Tensor, text_vec: torch.Tensor,
                                                  expert_mask: torch.Tensor):
        x_distill = x.detach()
        text_vec_distill = text_vec.detach()
        text_vec_distill = F.layer_norm(text_vec_distill, (text_vec_distill.size(-1),))
        B, L, _ = x_distill.shape
        D = self.attn_proj_dim
        if self.freeze_qk:
            ema_wq_weight = self.lora_attn_wq[self.active_adapter].weight.to(device=x_distill.device, dtype=x_distill.dtype)
            ema_wk_weight = self.lora_attn_wk[self.active_adapter].weight.to(device=x_distill.device, dtype=x_distill.dtype)
        else:
            ema_wq_weight = self.lora_ema_attn_wq_weight[self.active_adapter].to(device=x_distill.device, dtype=x_distill.dtype)
            ema_wk_weight = self.lora_ema_attn_wk_weight[self.active_adapter].to(device=x_distill.device, dtype=x_distill.dtype)
        if self.freeze_expo:
            expert_proto = self.lora_attn_expert_proto[self.active_adapter]
        else:
            expert_proto = self.lora_ema_attn_expert_proto[self.active_adapter].to(device=x_distill.device, dtype=x_distill.dtype)
        k = F.linear(text_vec_distill.to(device=x_distill.device, dtype=x_distill.dtype), ema_wk_weight, bias=None)
        p = F.normalize(expert_proto, dim=-1)
        kp = k[:, None, :] * p[None, :, :]
        with torch.no_grad():
            q_teacher = F.linear(x_distill, ema_wq_weight, bias=None)
            teacher_logits = (q_teacher.unsqueeze(2) * kp.unsqueeze(1)).sum(dim=-1)
            teacher_logits = teacher_logits * (D ** -0.5)

        lora_attn_wq = self.lora_attn_wq[self.active_adapter].to(device=x_distill.device, dtype=x_distill.dtype)
        lora_attn_wk = self.lora_attn_wk[self.active_adapter].to(device=x_distill.device, dtype=x_distill.dtype)
        student_proto = self.lora_attn_expert_proto[self.active_adapter]
        q_student = lora_attn_wq(x_distill)
        k_student = lora_attn_wk(text_vec_distill.to(device=x_distill.device, dtype=x_distill.dtype))
        p_student = F.normalize(student_proto, dim=-1)
        kp_student = k_student[:, None, :] * p_student[None, :, :]
        student_logits = (q_student.unsqueeze(2) * kp_student.unsqueeze(1)).sum(dim=-1)
        student_logits = student_logits * (D ** -0.5)

        return self._compute_router_kl_per_token(
            student_logits, teacher_logits, expert_mask).mean() * self.sample_router_distill_lambda

    def _compute_sub_router_distill_loss_chunked(self, x: torch.Tensor, expert_mask: torch.Tensor,
                                                 student_logits: torch.Tensor):
        B, L, _ = x.shape
        ema_sub_router_weight = self.lora_ema_sub_router_weight[self.active_adapter].to(device=x.device, dtype=x.dtype)
        chunk_size = self._get_distill_chunk_size(L)
        token_kl_sum = student_logits.new_zeros(())
        token_count = 0
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            with torch.no_grad():
                teacher_logits = F.linear(x[:, start:end, :], ema_sub_router_weight, bias=None)
            expert_mask_chunk = expert_mask[:, start:end, :]
            student_logits_chunk = student_logits[:, start:end, :]
            token_kl = self._compute_router_kl_per_token(
                student_logits_chunk, teacher_logits, expert_mask_chunk)
            token_kl_sum = token_kl_sum + token_kl.sum()
            token_count += token_kl.numel()
        return token_kl_sum / max(token_count, 1) * self.sample_router_distill_lambda
    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data += (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data -= (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = False



    def forward(self, x: torch.Tensor, **kwargs):

        previous_dtype = x.dtype
        self._last_router_distill_loss = None
        

        text_embeds = None
        rm = getattr(self, "root_model", None)
        if rm is not None:
            text_embeds = getattr(rm, "_routing_text_embeds", None)
        if text_embeds is None:
            text_embeds = getattr(self, "_routing_text_embeds", None)


        if self.active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.r[self.active_adapter] > 0 and self.merged: # merge the adapter to linear
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)        
        elif self.r[self.active_adapter] > 0:   # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            use_text_routing = text_embeds is not None
            if use_text_routing:
                te = text_embeds
                self.lora_router = self.lora_router.to(x.device)
                # te: [B, Lt, H] or [Lt, H]  -> make it [B, Lt, H]
                if isinstance(te, (list, tuple)):
                    te = torch.stack(te, dim=0)

                if te.dim() == 2 and x.shape[0] != 1:
                    raise RuntimeError(f"text_embeds missing batch dim: te={te.shape}, x={x.shape}")

                # Match the router device and dtype.
                te = te.to(device=x.device, dtype=x.dtype)

                # mean pooling: [B, H]
                text_vec = te.mean(dim=1)

                # # optional layer norm
                # text_vec = F.layer_norm(text_vec, (text_vec.size(-1),))

                # [B, E]
                router_logits = self.lora_router[self.active_adapter](text_vec)

                if self.topk > 0:
                    router_probs = torch.softmax(router_logits, dim=-1)              # [B, E]
                    router = self._straight_through_topk(router_probs, self.topk)     # [B, E]
                else:
                    router = torch.softmax(router_logits, dim=-1)                    # [B, E]

                # broadcast to token-level: [B, L, E]
                router = router[:, None, :].expand(x.shape[0], x.shape[1], self.expert_num)
            else:
                raise RuntimeError("text_embeds is required for routing but not found in root_model._routing_text_embeds")
            
            if self.subtopk > 0:
                expert_mask = (router > 0).to(router.dtype)  # [B, L, E]
                if self.attn_proj_dim > 0:
                    attn_router, _ = self._attn_token_router(
                        x=x,
                        text_vec=text_vec,
                        expert_mask=expert_mask,
                    )
                    router = attn_router
                    if self.training and self._use_router_distill():
                        self._last_router_distill_loss = self._compute_attn_router_distill_loss_chunked(
                            x=x, text_vec=text_vec, expert_mask=expert_mask)


                else:
                    self.lora_sub_router = self.lora_sub_router.to(x.device)
                    # token-level router logits/probs: [B, L, E]
                    sub_router_logits = self.lora_sub_router[self.active_adapter](x)
                    sub_router_probs = torch.softmax(sub_router_logits, dim=-1)

                    # Route only within the experts selected by the batch-level router.
                    # router: [B, L, E], with batch-level top-k or dense weights broadcast to tokens.
                    
                    masked = sub_router_probs * expert_mask

                    # Fall back to dense token routing if a mask is empty.
                    denom = masked.sum(dim=-1, keepdim=True)
                    masked = torch.where(denom > 0, masked / (denom + 1e-9), sub_router_probs)

                    # Optional token-level top-k within the allowed expert set.
                    if getattr(self, "subtopk", 1) and self.subtopk > 0 and self.subtopk < self.expert_num:
                        masked = self._straight_through_topk(masked, self.subtopk)

                    # Final token-level router.
                    router = masked
                    if self.training and self._use_router_distill():
                        self._last_router_distill_loss = self._compute_sub_router_distill_loss_chunked(
                            x=x, expert_mask=expert_mask, student_logits=sub_router_logits)


            # Update expert usage statistics
            # Training: record only if topk > 0. Inference: record all.
            if (not self.training) or (self.topk > 0):
                with torch.no_grad():
                    flat_router = router.reshape(-1, self.expert_num)
                    usage = flat_router.sum(dim=0)

                    # Ensure expert_counts is on the correct device
                    if self.expert_counts.device != usage.device:
                        self.expert_counts = self.expert_counts.to(usage.device)
                    self.expert_counts += usage

            if self.topk > 0:
                # Apply dropout ONCE before splitting to ensure consistency

                for i in range(self.expert_num):
                    # Skip experts with zero weight for all tokens.
                    if router[..., i].amax().item() == 0:
                        continue

                    result += (
                        self.lora_B[self.active_adapter].loraB[i](
                            self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                        )
                        * self.scaling[self.active_adapter]
                        * router[:, :, i].unsqueeze(-1)
                    )
            else:
                # Original dense execution for Soft Routing
                for i in range(self.expert_num):
                    result += ( # lora process
                        self.lora_B[self.active_adapter].loraB[i](
                            self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                        )
                        * self.scaling[self.active_adapter]
                        * router[:,:,i].unsqueeze(-1)
                    )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result
    


class SampleMOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraA.append(SampleMOEExpert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs
    
class SampleMOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraB.append(SampleMOEExpert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs



class SampleMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y



class SampleMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class SampleMOERouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: StrLoRAConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits
