from functools import partial
from typing import Dict, Type

import gradio as gr
from packaging import version

from swift.llm.argument.base_args.base_args import get_supported_tuners
from swift.ui.base import BaseUI
from swift.ui.llm_grpo.advanced import GRPOAdvanced
from swift.ui.llm_grpo.dataset import GRPODataset
from swift.ui.llm_grpo.external_rollout import LLMRollout
from swift.ui.llm_grpo.grpo_advanced import GrpoAdvanced
from swift.ui.llm_grpo.hyper import GRPOHyper
from swift.ui.llm_grpo.model import GRPOModel
from swift.ui.llm_grpo.optimizer import GRPOOptimizer
from swift.ui.llm_grpo.quantization import GRPOQuantization
from swift.ui.llm_grpo.report_to import GRPOReportTo
from swift.ui.llm_grpo.reward import Reward
from swift.ui.llm_grpo.rollout import Rollout
from swift.ui.llm_grpo.runtime import GRPORuntime
from swift.ui.llm_grpo.save import GRPOSave
from swift.ui.llm_grpo.tuner import GRPOTuner
from swift.ui.llm_train.llm_train import LLMTrain
from swift.ui.llm_train.runtime import Runtime
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMGRPO(LLMTrain):
    group = 'llm_grpo'

    sub_ui = [
        GRPOModel, GRPODataset, Reward, GRPORuntime, Rollout, GRPOSave, GRPOTuner, GRPOOptimizer, GRPOHyper,
        GRPOQuantization, GRPOAdvanced, GrpoAdvanced, GRPOReportTo, LLMRollout
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_grpo': {
            'label': {
                'zh': 'LLM GRPO',
                'en': 'LLM GRPO',
            }
        },
        'external_alert': {
            'value': {
                'zh': 'Err: {} \nRollout模型部署未完成，请检查日志，稍后开始训练！',
                'en': 'Err: {} \nRollout model deployment is incomplete, '
                'please check the logs and start training later!'
            }
        },
        'submit_alert': {
            'value': {
                'zh':
                '任务已开始，请查看tensorboard或日志记录，请勿关闭终端，否则训练过程将被打断',
                'en':
                'Task started, please check the tensorboard or log file, '
                'do not close the terminal, otherwise the training process will be interrupted'
            }
        },
        'dataset_alert': {
            'value': {
                'zh': '请选择或填入一个数据集',
                'en': 'Please input or select a dataset'
            }
        },
        'submit': {
            'value': {
                'zh': '🚀 开始训练',
                'en': '🚀 Begin'
            }
        },
        'dry_run': {
            'label': {
                'zh': '仅生成运行命令',
                'en': 'Dry-run'
            },
            'info': {
                'zh': '仅生成运行命令，开发者自行运行',
                'en': 'Generate run command only, for manually running'
            }
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to train'
            }
        },
        'train_type': {
            'label': {
                'zh': '训练方式',
                'en': 'Train type'
            },
            'info': {
                'zh': '选择训练的方式',
                'en': 'Select the training type'
            }
        },
        'seed': {
            'label': {
                'zh': '随机数种子',
                'en': 'Seed'
            },
            'info': {
                'zh': '选择随机数种子',
                'en': 'Select a random seed'
            }
        },
        'torch_dtype': {
            'label': {
                'zh': '训练精度',
                'en': 'Training Precision'
            },
            'info': {
                'zh': '选择训练精度',
                'en': 'Select the training precision'
            }
        },
        'envs': {
            'label': {
                'zh': '环境变量',
                'en': 'Extra env vars'
            },
        },
        'use_ddp': {
            'label': {
                'zh': '使用DDP',
                'en': 'Use DDP'
            },
            'info': {
                'zh': '是否使用数据并行训练',
                'en': 'Use Distributed Data Parallel to train'
            }
        },
        'ddp_num': {
            'label': {
                'zh': 'DDP分片数量',
                'en': 'Number of DDP sharding'
            },
            'info': {
                'zh': '启用多少进程的数据并行',
                'en': 'The data parallel size of DDP'
            }
        },
        'use_liger_kernel': {
            'label': {
                'zh': '使用Liger kernel',
                'en': 'Use Liger kernel'
            },
            'info': {
                'zh': 'Liger kernel可以有效降低显存使用',
                'en': 'Liger kernel can reduce memory usage'
            }
        },
        'sequence_parallel_size': {
            'label': {
                'zh': '序列并行大小',
                'en': 'Sequence parallel size',
            },
            'info': {
                'zh': '当前支持CPT/SFT/DPO/GRPO',
                'en': 'Currently supports CPT/SFT/DPO/GRPO',
            }
        },
        'deepspeed': {
            'label': {
                'zh': 'DeepSpeed',
                'en': 'DeepSpeed',
            },
            'info': {
                'zh': '可以选择下拉列表，也支持传入路径',
                'en': 'Choose from the dropbox or fill in a valid path',
            }
        },
        'resume_checkpoint_alert': {
            'value': {
                'zh': '检测到"args.json"在{}中，将从此检查点开始断点续训',
                'en': 'Detected that "args.json" is in {}, will start breakpoint resume training from this checkpoint'
            }
        },
        'resume_only_model_alert': {
            'value': {
                'zh':
                '检测到"args.json"在{}中，但未检测到优化器参数，将仅加载模型参数开始断点续训',
                'en':
                '"args.json" is detected in {}, but optimizer parameters are not detected. '
                'Only model parameters will be loaded to start breakpoint continuation training'
            }
        },
        'more_params': {
            'label': {
                'zh': '其他高级参数',
                'en': 'Other params'
            },
            'info': {
                'zh': '以json格式或--xxx xxx命令行格式填入',
                'en': 'Fill in with json format or --xxx xxx cmd format'
            }
        },
        'extra_params': {
            'label': {
                'zh': '其他参数设置',
                'en': 'Extra settings'
            },
        },
        'train_param': {
            'label': {
                'zh': '训练参数设置',
                'en': 'Train settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_grpo', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                GRPOModel.build_ui(base_tab)
                GRPODataset.build_ui(base_tab)
                Reward.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='train_type', scale=4, choices=list(get_supported_tuners()))
                        gr.Textbox(elem_id='seed', scale=4)
                        gr.Dropdown(elem_id='torch_dtype', scale=4)
                        gr.Checkbox(elem_id='use_liger_kernel', scale=4)
                        gr.Textbox(elem_id='sequence_parallel_size', lines=1, scale=4)
                    with gr.Row():
                        gr.Dropdown(
                            elem_id='gpu_id',
                            multiselect=True,
                            choices=[str(i) for i in range(device_count)] + ['cpu'],
                            value=default_device,
                            scale=8)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='1', scale=4)
                        gr.Dropdown(
                            elem_id='deepspeed',
                            scale=4,
                            allow_custom_value=True,
                            value=None,
                            choices=['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'])
                GRPOHyper.build_ui(base_tab)
                GRPORuntime.build_ui(base_tab)
                with gr.Row(equal_height=True):
                    gr.Textbox(elem_id='envs', scale=12)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                Rollout.build_ui(base_tab)
                LLMRollout.set_lang(cls.lang)
                LLMRollout.build_ui(LLMRollout)
                GRPOTuner.build_ui(base_tab)
                with gr.Accordion(elem_id='extra_params', open=False):
                    with gr.Tabs():
                        GrpoAdvanced.build_ui(base_tab)
                        GRPOAdvanced.build_ui(base_tab)
                        GRPOQuantization.build_ui(base_tab)
                        GRPOSave.build_ui(base_tab)
                        GRPOReportTo.build_ui(base_tab)
                    with gr.Row():
                        gr.Textbox(elem_id='more_params', lines=4, scale=20)

                cls.element('train_type').change(
                    GRPOHyper.update_lr,
                    inputs=[base_tab.element('train_type')],
                    outputs=[cls.element('learning_rate')])

                submit.click(
                    cls.train_local,
                    list(cls.valid_elements().values()), [
                        cls.element('running_cmd'),
                        cls.element('logging_dir'),
                        cls.element('runtime_tab'),
                        cls.element('running_tasks'),
                        cls.element('train_record'),
                    ],
                    queue=True)
                Rollout.element('vllm_mode').change(LLMRollout.external_rollout_display, Rollout.element('vllm_mode'),
                                                    LLMRollout.element('llm_rollout'))
                LLMRollout.element('rollout').click(
                    LLMRollout.rollout_model,
                    list(LLMRollout.valid_elements().values())
                    + [cls.element('model'), cls.element('model_type'),
                       cls.element('template')],
                    [LLMRollout.element('rollout_runtime_tab'),
                     LLMRollout.element('rollout_running_tasks')])

                GRPORuntime.element('kill_task').click(
                    GRPORuntime.kill_task,
                    [GRPORuntime.element('running_tasks')],
                    [GRPORuntime.element('running_tasks')] + [GRPORuntime.element('log')] + GRPORuntime.all_plots,
                ).then(GRPORuntime.reset, [], [GRPORuntime.element('logging_dir')] + [GRPOHyper.element('output_dir')])

                base_tab.element('gpu_id').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('use_ddp').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('ddp_num').change(Rollout.update_num_gen, [
                    GRPOHyper.element('per_device_train_batch_size'),
                    GRPOHyper.element('gradient_accumulation_steps'),
                    cls.element('ddp_num')
                ], [Rollout.element('num_generations')])
                GRPOHyper.element('gradient_accumulation_steps').change(Rollout.update_num_gen, [
                    GRPOHyper.element('per_device_train_batch_size'),
                    GRPOHyper.element('gradient_accumulation_steps'),
                    cls.element('ddp_num')
                ], [Rollout.element('num_generations')])
                GRPOHyper.element('per_device_train_batch_size').change(Rollout.update_num_gen, [
                    GRPOHyper.element('per_device_train_batch_size'),
                    GRPOHyper.element('gradient_accumulation_steps'),
                    cls.element('ddp_num')
                ], [Rollout.element('num_generations')])

    @classmethod
    def prepare_sub_to_filter(cls):
        tabs_relation_dict = {
            key: val
            for key, val in zip(['train_type', 'optimizer', 'vllm_mode'],
                                [GRPOTuner.tabs_to_filter, GRPOOptimizer.tabs_to_filter, Rollout.tabs_to_filter])
        }
        return tabs_relation_dict
