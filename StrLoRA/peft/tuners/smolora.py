# -*- encoding: utf-8 -*-
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    ModulesToSaveWrapper,
    PeftType,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import (
    Conv2d,
    Embedding,
    Linear4bit,
    Linear8bitLt,
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class SMoLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.SMOLORA`].
    """

    expert_num: int = field(default=4)
    ins_type: int = field(default=0)
    ins_emb: list = field(default=None)

    def __post_init__(self):
        self.peft_type = PeftType.SMOLORA


class SMoLoraModel(LoraModel):
    """
    Create SMoLoRA model from a pretrained transformers model.
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_smolora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "SMoLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
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
                "Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "expert_num": lora_config.expert_num,
            "ins_type": lora_config.ins_type,
            "ins_emb": lora_config.ins_emb,
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
                    "Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            if self.model is not None:
                kwargs["root_model"] = self
            new_module = SMoLoraLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_smolora_config(peft_config, model_config):
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
            raise ValueError("Cannot merge LoRA layers when the model is loaded in 8-bit mode")

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

            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


class SMoLoraLayer(LoraLayer):
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
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: SMoLoraLinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: SMoLoraLinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class SMoLoraLinear(nn.Linear, SMoLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.ins_type = kwargs.pop("ins_type", True)
        self.ins_emb = kwargs.pop("ins_emb", True)
        root_model = kwargs.pop("root_model", None)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SMoLoraLayer.__init__(self, in_features=in_features, out_features=out_features, expert_num=self.expert_num)
        object.__setattr__(self, "root_model", root_model)
        object.__setattr__(self, "_cached_ins_emb", None)

        self.lora_vu_gate = nn.ModuleDict({})
        self.lora_vu_gate.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num // 2, bias=False)}))

        self.lora_ins_gate = nn.ModuleDict({})
        self.lora_ins_gate.update(nn.ModuleDict({adapter_name: nn.Linear(384, self.expert_num // 2, bias=False)}))
        self.lora_fc_A = nn.ModuleDict({})
        self.lora_fc_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.out_features, 1, bias=False)}))
        self.lora_fc_B = nn.ModuleDict({})
        self.lora_fc_B.update(nn.ModuleDict({adapter_name: nn.Linear(self.out_features, 1, bias=False)}))

        self.weight.requires_grad = False
        for param in self.lora_vu_gate.parameters():
            param.requires_grad = self.ins_type <= 0
        for param in self.lora_ins_gate.parameters():
            param.requires_grad = self.ins_type <= 0

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = False

    def _get_ins_emb_table(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = getattr(self, "_cached_ins_emb", None)
        if cached is None or cached.device != device or cached.dtype != dtype:
            cached = torch.as_tensor(self.ins_emb, device=device, dtype=dtype)
            object.__setattr__(self, "_cached_ins_emb", cached)
        return cached

    def _resolve_batch_ins_idx(self, batch_size: int, device: torch.device):
        root_model = getattr(self, "root_model", None)
        batch_ins_idx = None
        if root_model is not None:
            batch_ins_idx = getattr(root_model, "_smolora_batch_ins_idx", None)
        if batch_ins_idx is None:
            batch_ins_idx = getattr(self, "_smolora_batch_ins_idx", None)
        if batch_ins_idx is None:
            return None

        if not isinstance(batch_ins_idx, torch.Tensor):
            batch_ins_idx = torch.as_tensor(batch_ins_idx, device=device, dtype=torch.long)
        else:
            batch_ins_idx = batch_ins_idx.to(device=device, dtype=torch.long)

        if batch_ins_idx.ndim == 0:
            batch_ins_idx = batch_ins_idx.unsqueeze(0)
        batch_ins_idx = batch_ins_idx.reshape(-1)

        if batch_ins_idx.numel() == 1 and batch_size > 1:
            batch_ins_idx = batch_ins_idx.expand(batch_size)
        elif batch_ins_idx.numel() != batch_size:
            raise ValueError(
                f"smolora batch size mismatch: got {batch_ins_idx.numel()} indices for batch size {batch_size}."
            )
        return batch_ins_idx

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            self.lora_vu_gate = self.lora_vu_gate.to(x.device)
            self.lora_ins_gate = self.lora_ins_gate.to(x.device)
            current_ins_emb = self._get_ins_emb_table(x.device, x.dtype)

            x_emb = torch.mean(x, dim=1, keepdim=True)
            vu_router = self.lora_vu_gate[self.active_adapter](x_emb)
            top1_vu_router = torch.zeros_like(vu_router)
            _, max_indices = torch.max(vu_router, dim=-1, keepdim=True)
            top1_vu_router.scatter_(-1, max_indices, 1.0)

            batch_ins_idx = self._resolve_batch_ins_idx(top1_vu_router.size(0), x.device)
            if batch_ins_idx is None:
                selected_ins_emb = current_ins_emb[self.ins_type]
            else:
                selected_ins_emb = current_ins_emb.index_select(0, batch_ins_idx)

            if_router = self.lora_ins_gate[self.active_adapter](selected_ins_emb)
            if if_router.ndim == 1:
                if_router = if_router.unsqueeze(0)
            top1_if_router = torch.zeros_like(if_router)
            _, max_indices = torch.max(if_router, dim=-1, keepdim=True)
            top1_if_router.scatter_(-1, max_indices, 1.0)
            if top1_if_router.size(0) == 1 and top1_vu_router.size(0) > 1:
                top1_if_router = top1_if_router.expand(top1_vu_router.size(0), -1)
            elif top1_if_router.size(0) != top1_vu_router.size(0):
                raise ValueError(
                    f"smolora router batch mismatch: vu={top1_vu_router.size(0)}, if={top1_if_router.size(0)}."
                )

            final_router = torch.cat((top1_vu_router, top1_if_router.unsqueeze(1)), dim=-1)
            vu_result = 0.0
            if_result = 0.0
            for i in range(self.expert_num // 2):
                vu_result += (
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                    * final_router[:, :, i].unsqueeze(-1)
                )

            for i in range(self.expert_num // 2, self.expert_num):
                if_result += (
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                    * final_router[:, :, i].unsqueeze(-1)
                )

            score_A = self.lora_fc_A[self.active_adapter](vu_result)
            score_B = self.lora_fc_B[self.active_adapter](if_result)

            scores = torch.cat([score_A, score_B], dim=-1)
            attention_weights = F.softmax(scores, dim=-1)

            alpha_A = attention_weights[:, :, 0].unsqueeze(-1)
            alpha_B = attention_weights[:, :, 1].unsqueeze(-1)

            result += (alpha_A * vu_result) + (alpha_B * if_result)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result


class SMoLoraLinearA(nn.Module):
    def __init__(self, in_features, out_features, expert_num) -> None:
        super().__init__()
        self.expert_num = expert_num
        self.in_features = in_features
        self.out_features = out_features
        self.loraA = nn.ModuleList([])

        assert self.out_features % self.expert_num == 0
        self.r = self.out_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraA.append(StrLoRAExpert(self.in_features, self.r))

    def forward(self, x):
        return [self.loraA[i](x) for i in range(self.expert_num)]


class SMoLoraLinearB(nn.Module):
    def __init__(self, in_features, out_features, expert_num) -> None:
        super().__init__()
        self.expert_num = expert_num
        self.in_features = in_features
        self.out_features = out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraB.append(StrLoRAExpert(self.r, self.out_features))

    def forward(self, x):
        return [self.loraB[i](x[i]) for i in range(self.expert_num)]


class StrLoRAExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        return self.mlp(x)
