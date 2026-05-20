from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Save(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'save_tab': {
            'label': {
                'zh': '存储参数设置',
                'en': 'Saving settings'
            },
        },
        'push_to_hub': {
            'label': {
                'zh': 'Push to hub',
                'en': 'Push to hub',
            },
            'info': {
                'zh': 'Whether push the output model to the hub',
                'en': 'Whether push the output model to the hub',
            }
        },
        'hub_model_id': {
            'label': {
                'zh': 'Hub model id',
                'en': 'Hub model id',
            },
            'info': {
                'zh': 'Set the hub model id',
                'en': 'Set the hub model id',
            }
        },
        'hub_private_repo': {
            'label': {
                'zh': '设置仓库私有',
                'en': 'Model is private',
            },
            'info': {
                'zh': 'Set the model as private',
                'en': 'Set the model as private',
            }
        },
        'hub_strategy': {
            'label': {
                'zh': '推送策略',
                'en': 'Push strategy',
            },
            'info': {
                'zh': '设置模型推送策略',
                'en': 'Set the push strategy',
            }
        },
        'hub_token': {
            'label': {
                'zh': '仓库token',
                'en': 'The hub token',
            },
            'info': {
                'zh': 'Set the hub token',
                'en': 'Set the hub token',
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='save_tab'):
            with gr.Blocks():
                with gr.Row():
                    gr.Checkbox(elem_id='push_to_hub', scale=20)
                    gr.Textbox(elem_id='hub_model_id', lines=1, scale=20)
                    gr.Checkbox(elem_id='hub_private_repo', scale=20)
                    gr.Dropdown(
                        elem_id='hub_strategy',
                        scale=20,
                        choices=['end', 'every_save', 'checkpoint', 'all_checkpoints'])
                    gr.Textbox(elem_id='hub_token', lines=1, scale=20)
