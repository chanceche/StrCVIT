# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction, TrainerCallback
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from swift.utils import is_master
from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp
import logging
def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k and "lora_ema_" not in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if ("lora_" in k and "lora_ema_" not in k) or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k and "lora_ema_" not in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


logger = get_logger()


def get_local_param(param):
    if hasattr(param, 'ds_id') and getattr(param, 'ds_tensor', None) is not None:
        return param.ds_tensor
    return param.data


def clone_local_param(param, grad: bool = False):
    tensor = param.grad if grad else get_local_param(param)
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


class EWCGradientCallback(TrainerCallback):

    def __init__(self, model: nn.Module, ewc_lambda: float):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher = {}
        self.previous_weights = {}
        self.trainable_params = set()
        self.debug_enabled = os.environ.get('EWC_DEBUG', '0').lower() in {'1', 'true', 'yes', 'on'}
        self.debug_param_preview = int(os.environ.get('EWC_DEBUG_PARAM_PREVIEW', '12'))
        self._use_pre_step_fisher_update = True
        self._use_gradient_hooks = False
        self._hook_handles = []
        self._hook_corrected_params = 0
        self._hook_updated_params = 0
        self._hook_correction_abs_sum = 0.0
        self._hook_drift_abs_sum = 0.0
        self._hook_fisher_abs_sum = 0.0
        self._hook_grad_abs_sum = 0.0
        self._fisher_update_denominator = 1
        self._is_gemma_model = self._detect_gemma_model()
        self._init_fisher()

    def _detect_gemma_model(self) -> bool:
        model_type = getattr(getattr(self.model, 'config', None), 'model_type', '') or ''
        architectures = getattr(getattr(self.model, 'config', None), 'architectures', None) or []
        class_names = {
            self.model.__class__.__name__,
            getattr(getattr(self.model, 'base_model', None), '__class__', type('', (), {})).__name__,
            getattr(getattr(getattr(self.model, 'base_model', None), 'model', None), '__class__', type('', (), {})).__name__,
        }
        haystacks = [str(model_type).lower()] + [str(x).lower() for x in architectures] + [x.lower() for x in class_names if x]
        return any('gemma' in text for text in haystacks)

    def _init_fisher(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            local_param = clone_local_param(param)
            if local_param is None:
                continue
            self.fisher[name] = torch.ones_like(local_param)
            self.trainable_params.add(name)

    def _should_log(self, args, state):
        if not self.debug_enabled:
            return False
        logging_steps = max(int(getattr(args, 'logging_steps', 0) or 0), 1)
        global_step = int(getattr(state, 'global_step', 0) or 0)
        return global_step <= 3 or global_step % logging_steps == 0

    def _get_debug_stats(self):
        tracked_names = sorted(self.trainable_params)
        lora_names = [name for name in tracked_names if 'lora_' in name]
        non_lora_names = [name for name in tracked_names if 'lora_' not in name]
        preview = tracked_names[:self.debug_param_preview]
        return {
            'tracked': len(tracked_names),
            'lora': len(lora_names),
            'non_lora': len(non_lora_names),
            'preview': preview,
        }

    def on_train_begin(self, args, state, control, **kwargs):
        self.previous_weights = {
            name: clone_local_param(param)
            for name, param in self.model.named_parameters() if name in self.trainable_params
        }
        self._use_pre_step_fisher_update = True
        self._use_gradient_hooks = False
        self._fisher_update_denominator = max(int(getattr(args, 'gradient_accumulation_steps', 1) or 1), 1)
        if self._use_gradient_hooks:
            self._register_gradient_hooks()
        if self.debug_enabled:
            stats = self._get_debug_stats()
            logger.info(
                'EWC DEBUG train_begin: tracked_params=%s lora_params=%s non_lora_params=%s preview=%s',
                stats['tracked'],
                stats['lora'],
                stats['non_lora'],
                stats['preview'],
            )
            logger.info(
                'EWC DEBUG mode: use_pre_step_fisher_update=%s use_gradient_hooks=%s',
                self._use_pre_step_fisher_update,
                self._use_gradient_hooks,
            )

    def _register_gradient_hooks(self):
        self._remove_gradient_hooks()
        for name, param in self.model.named_parameters():
            if name not in self.trainable_params or not param.requires_grad:
                continue
            local_param = get_local_param(param)
            previous_weights = self.previous_weights.get(name)
            fisher = self.fisher.get(name)
            if local_param is None or previous_weights is None or fisher is None:
                continue
            if not (local_param.shape == previous_weights.shape == fisher.shape):
                continue
            self.previous_weights[name] = previous_weights.to(device=local_param.device, dtype=local_param.dtype)
            self.fisher[name] = fisher.to(device=local_param.device, dtype=local_param.dtype)
            self._hook_handles.append(param.register_hook(self._make_gradient_hook(name, param)))

    def _remove_gradient_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _reset_hook_debug_stats(self):
        self._hook_corrected_params = 0
        self._hook_updated_params = 0
        self._hook_correction_abs_sum = 0.0
        self._hook_drift_abs_sum = 0.0
        self._hook_fisher_abs_sum = 0.0
        self._hook_grad_abs_sum = 0.0

    def _make_gradient_hook(self, name, param):
        def hook(grad):
            fisher = self.fisher.get(name)
            previous_weights = self.previous_weights.get(name)
            current_param = get_local_param(param)
            if grad is None or fisher is None or previous_weights is None or current_param is None:
                return grad
            if not (grad.shape == current_param.shape == fisher.shape == previous_weights.shape):
                return grad
            with torch.no_grad():
                current_param = current_param.to(device=grad.device, dtype=grad.dtype)
                fisher = fisher.to(device=grad.device, dtype=grad.dtype)
                previous_weights = previous_weights.to(device=grad.device, dtype=grad.dtype)
                drift = current_param - previous_weights
                correction = self.ewc_lambda * fisher * drift
                adjusted_grad = grad + correction
                if self.debug_enabled:
                    self._hook_corrected_params += 1
                    self._hook_correction_abs_sum += correction.abs().sum().item()
                    self._hook_drift_abs_sum += drift.abs().sum().item()
                    self._hook_fisher_abs_sum += fisher.abs().sum().item()
                    self._hook_grad_abs_sum += adjusted_grad.abs().sum().item()
                self.fisher[name].add_(adjusted_grad.detach().to(self.fisher[name].dtype).pow(2)
                                       / self._fisher_update_denominator)
                self._hook_updated_params += 1
            return adjusted_grad

        return hook

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.ewc_lambda <= 0:
            return
        if self._use_gradient_hooks:
            if self._should_log(args, state):
                logger.info(
                    'EWC DEBUG grad_hook: step=%s corrected_params=%s correction_abs_sum=%.6e '
                    'drift_abs_sum=%.6e fisher_abs_sum=%.6e updated_params=%s grad_abs_sum=%.6e',
                    state.global_step,
                    self._hook_corrected_params,
                    self._hook_correction_abs_sum,
                    self._hook_drift_abs_sum,
                    self._hook_fisher_abs_sum,
                    self._hook_updated_params,
                    self._hook_grad_abs_sum,
                )
            self._reset_hook_debug_stats()
            return
        matched_params = 0
        corrected_params = 0
        correction_abs_sum = 0.0
        drift_abs_sum = 0.0
        fisher_abs_sum = 0.0
        updated_params = 0
        grad_abs_sum = 0.0
        for name, param in self.model.named_parameters():
            if name not in self.trainable_params or param.grad is None:
                continue
            matched_params += 1
            fisher = self.fisher.get(name)
            previous_weights = self.previous_weights.get(name)
            current_param = get_local_param(param)
            if fisher is None or previous_weights is None or current_param is None:
                continue
            if not (param.grad.shape == current_param.shape == fisher.shape == previous_weights.shape):
                continue
            current_param = current_param.to(device=param.grad.device, dtype=param.grad.dtype)
            fisher = fisher.to(device=param.grad.device, dtype=param.grad.dtype)
            previous_weights = previous_weights.to(device=param.grad.device, dtype=param.grad.dtype)
            correction = self.ewc_lambda * fisher * (current_param - previous_weights)
            if self.debug_enabled:
                corrected_params += 1
                correction_abs_sum += correction.abs().sum().item()
                drift_abs_sum += (current_param - previous_weights).abs().sum().item()
                fisher_abs_sum += fisher.abs().sum().item()
            if self._use_pre_step_fisher_update:
                grad = clone_local_param(param, grad=True)
                if grad is not None and self.fisher[name].shape == grad.shape:
                    self.fisher[name] += grad.pow(2) / max(state.global_step, 1)
                    if self.debug_enabled:
                        updated_params += 1
                        grad_abs_sum += grad.abs().sum().item()
        if self._should_log(args, state):
            logger.info(
                'EWC DEBUG pre_optim: step=%s matched_params=%s corrected_params=%s '
                'correction_abs_sum=%.6e drift_abs_sum=%.6e fisher_abs_sum=%.6e '
                'updated_params=%s grad_abs_sum=%.6e',
                state.global_step,
                matched_params,
                corrected_params,
                correction_abs_sum,
                drift_abs_sum,
                fisher_abs_sum,
                updated_params,
                grad_abs_sum,
            )

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._use_gradient_hooks:
            return
        if self._use_pre_step_fisher_update:
            return
        global_step = max(state.global_step, 1)
        updated_params = 0
        grad_abs_sum = 0.0
        fisher_abs_sum = 0.0
        for name, param in self.model.named_parameters():
            if name not in self.trainable_params:
                continue
            grad = clone_local_param(param, grad=True)
            if grad is not None and self.fisher[name].shape == grad.shape:
                self.fisher[name] += grad.pow(2) / global_step
                if self.debug_enabled:
                    updated_params += 1
                    grad_abs_sum += grad.abs().sum().item()
                    fisher_abs_sum += self.fisher[name].abs().sum().item()
        if self._should_log(args, state):
            logger.info(
                'EWC DEBUG post_optim: step=%s updated_params=%s grad_abs_sum=%.6e fisher_abs_sum=%.6e',
                state.global_step,
                updated_params,
                grad_abs_sum,
                fisher_abs_sum,
            )

    def on_train_end(self, args, state, control, **kwargs):
        self._remove_gradient_hooks()

    def compute_ewc_loss(self, model, device):
        if not self.previous_weights:
            return None
        loss = None
        for name, param in model.named_parameters():
            if name not in self.trainable_params:
                continue
            current_param = get_local_param(param)
            previous_weights = self.previous_weights.get(name)
            fisher = self.fisher.get(name)
            if current_param is None or previous_weights is None or fisher is None:
                continue
            if not (current_param.shape == previous_weights.shape == fisher.shape):
                continue
            current_param = current_param.to(device=device, dtype=torch.float32)
            previous_weights = previous_weights.to(device=device, dtype=torch.float32)
            fisher = fisher.to(device=device, dtype=torch.float32)
            term = (fisher * (current_param - previous_weights).pow(2)).sum()
            loss = term if loss is None else loss + term
        if loss is None:
            return None
        return 0.5 * self.ewc_lambda * loss


class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


def gather_for_unpadded_tensors(input_data, use_gather_object=False):
    from accelerate.utils import gather_object
    input_data = gather_object(input_data)
    output = []
    for _data in input_data:
        if len(_data.shape) == 0:
            _data = _data.unsqueeze(0)
        _data = _data.cpu()
        output.append(_data)
    if len(output[0].shape) == 1 and output[0].shape[0] > 1:
        data = torch.stack(output, dim=0)
    else:
        data = torch.concat(output, dim=0)
    return data


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']
        self.gather_function = gather_for_unpadded_tensors

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import calculate_paired_metrics, calculate_infonce_metrics
        args = self.args
        if args.loss_type == 'infonce':
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.include_for_metrics = ['inputs']
        self.compute_metrics = self.calculate_metric
        self.label_names = ['labels']

        # Set up preprocess_logits_for_metrics to reduce memory usage for generative reranker
        if self.args.loss_type in {'generative_reranker', 'listwise_generative_reranker'}:
            self.preprocess_logits_for_metrics = self._preprocess_generative_reranker_logits
        else:
            self.preprocess_logits_for_metrics = None
        self.gather_function = gather_for_unpadded_tensors

    def _preprocess_generative_reranker_logits(self, logits, labels):
        """
        Preprocess logits for generative reranker to reduce memory usage.
        Extract only the yes/no token logits at the last valid (non -100) timestep
        for each sample, avoiding padded timesteps created by multi-GPU gather.
        """

        # Get token IDs for positive and negative tokens
        positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
        negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

        tokenizer = getattr(self, 'processing_class', None)
        if tokenizer is None:
            # Fallback: return full logits if tokenizer not available
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # Fallback: return full logits if token conversion fails
            return logits

        # Extract only the yes/no token logits from the last non -100 position per sample
        # Shapes: logits [batch, seq_len, vocab]
        if len(logits.shape) == 3:
            positive_logits = logits[:, :, positive_token_id]
            negative_logits = logits[:, :, negative_token_id]
            logits = positive_logits - negative_logits
            return logits
        else:
            # Unexpected shape, return as-is
            return logits

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        import numpy as np
        from swift.plugin.loss import calculate_reranker_metrics
        input_ids = eval_prediction.inputs
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if logits.ndim == 2 and logits.shape[1] > 1:
            pad_token_id = self.tokenizer.pad_token_id
            valid_mask = (input_ids != pad_token_id) & (input_ids != -100)
            last_valid_indices = valid_mask[:, ::-1].argmax(axis=1)
            last_valid_indices = input_ids.shape[1] - 1 - last_valid_indices
            logits = logits[np.arange(logits.shape[0]), last_valid_indices]
        return calculate_reranker_metrics(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            # Get labels and compute outputs
            labels = inputs.get('labels')
            if labels is not None:
                labels = inputs.pop('labels')

            outputs = model(**inputs)

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(
                    outputs,
                    labels,
                    num_items_in_batch=num_items_in_batch,
                    trainer=self,
                    attention_mask=inputs['attention_mask'])
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels, attention_mask=inputs.get('attention_mask'))

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments
    ROUTER_DISTILL_EMA_NAME = 'router_distill_ema.bin'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        self.ewc_lambda = float(getattr(self.args, 'ewc_lambda', 0.) or 0.)
        self.ewc_callback = None
        if self.ewc_lambda > 0 and getattr(self.args, 'enable_ewc', False):
            self.ewc_callback = EWCGradientCallback(self.model, self.ewc_lambda)
            self.add_callback(self.ewc_callback)
            logger.info(f'Enabling EWC regularization, ewc_lambda: {self.ewc_lambda}')
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    def save_trained_model(self, training_args):
        if training_args.lora_rank:
            self.accelerator.wait_for_everyone()
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
            # if self.args.local_rank == 0 or self.args.local_rank == -1:
            if is_master():
                self.model.config.save_pretrained(training_args.output_dir)
                self.model.save_pretrained(training_args.output_dir, state_dict=state_dict, safe_serialization=False)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if (getattr(training_args, 'sample_router_distill_lambda', 0.) > 0
                        and hasattr(unwrapped_model, 'get_router_distill_ema_state')):
                    ema_state = unwrapped_model.get_router_distill_ema_state()
                    if ema_state:
                        torch.save(ema_state, os.path.join(training_args.output_dir, self.ROUTER_DISTILL_EMA_NAME))
            self.accelerator.wait_for_everyone()


    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
                template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        from swift.llm import HfConfigFactory
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        if self.model.model_info.is_moe_model and 'output_router_logits' in inspect.signature(
                base_model.forward).parameters:
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        outputs = model(**inputs)
        unwrapped_model = self.accelerator.unwrap_model(model)
        distill_loss = getattr(unwrapped_model, '_last_distill_loss', None)
        if getattr(outputs, 'aux_loss', None) is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        if distill_loss is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['distill_loss'].update(distill_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            if distill_loss is not None:
                loss = loss + distill_loss.to(loss.device)
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss and channels is not None:
                    mode = 'train' if self.model.training else 'eval'
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)
            if distill_loss is not None:
                loss = loss + distill_loss.to(loss.device)

        if self.ewc_callback is not None:
            ewc_loss = self.ewc_callback.compute_ewc_loss(self.model, loss.device)
            if ewc_loss is not None:
                mode = 'train' if self.model.training else 'eval'
                self.custom_metrics[mode]['ewc_loss'].update(ewc_loss.detach())
                loss = loss + ewc_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)
