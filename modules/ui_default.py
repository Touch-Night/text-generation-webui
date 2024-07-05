import gradio as gr

from modules import logits, shared, ui, utils
from modules.prompts import count_tokens, load_prompt
from modules.text_generation import (
    generate_reply_wrapper,
    get_token_ids,
    stop_everything_event
)
from modules.utils import gradio

inputs = ('textbox-default', 'interface_state')
outputs = ('output_textbox', 'html-default')


def create_ui():
    mu = shared.args.multi_user
    with gr.Tab('默认', elem_id='default-tab'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['textbox-default'] = gr.Textbox(value='', lines=27, label='输入', elem_classes=['textbox_default', 'add_scrollbar'])
                    shared.gradio['token-counter-default'] = gr.HTML(value="<span>0</span>", elem_classes=["token-counter", "default-token-counter"])

                with gr.Row():
                    shared.gradio['Generate-default'] = gr.Button('生成', variant='primary')
                    shared.gradio['Stop-default'] = gr.Button('停止', elem_id='stop')
                    shared.gradio['Continue-default'] = gr.Button('继续')

                with gr.Row():
                    shared.gradio['prompt_menu-default'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='提示', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['prompt_menu-default'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, 'refresh-button', interactive=not mu)
                    shared.gradio['save_prompt-default'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_prompt-default'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)

            with gr.Column():
                with gr.Tab('原始'):
                    shared.gradio['output_textbox'] = gr.Textbox(lines=27, label='输出', elem_id='textbox-default', elem_classes=['textbox_default_output', 'add_scrollbar'])

                with gr.Tab('Markdown'):
                    shared.gradio['markdown_render-default'] = gr.Button('渲染')
                    shared.gradio['markdown-default'] = gr.Markdown()

                with gr.Tab('HTML'):
                    shared.gradio['html-default'] = gr.HTML()

                with gr.Tab('Logits'):
                    with gr.Row():
                        with gr.Column(scale=10):
                            shared.gradio['get_logits-default'] = gr.Button('获取下一个词符概率')
                        with gr.Column(scale=1):
                            shared.gradio['use_samplers-default'] = gr.Checkbox(label='使用采样器', value=True, elem_classes=['no-background'])

                    with gr.Row():
                        shared.gradio['logits-default'] = gr.Textbox(lines=23, label='输出', elem_classes=['textbox_logits', 'add_scrollbar'])
                        shared.gradio['logits-default-previous'] = gr.Textbox(lines=23, label='先前输出', elem_classes=['textbox_logits', 'add_scrollbar'])

                with gr.Tab('词符'):
                    shared.gradio['get_tokens-default'] = gr.Button('获取输入的词符ID')
                    shared.gradio['tokens-default'] = gr.Textbox(lines=23, label='词符', elem_classes=['textbox_logits', 'add_scrollbar', 'monospace'])


def create_event_handlers():
    shared.gradio['Generate-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate_reply_wrapper, gradio(inputs), gradio(outputs), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox-default'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate_reply_wrapper, gradio(inputs), gradio(outputs), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['markdown_render-default'].click(lambda x: x, gradio('output_textbox'), gradio('markdown-default'), queue=False)
    shared.gradio['Continue-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate_reply_wrapper, [shared.gradio['output_textbox']] + gradio(inputs)[1:], gradio(outputs), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Stop-default'].click(stop_everything_event, None, None, queue=False)
    shared.gradio['prompt_menu-default'].change(load_prompt, gradio('prompt_menu-default'), gradio('textbox-default'), show_progress=False)
    shared.gradio['save_prompt-default'].click(
        lambda x: x, gradio('textbox-default'), gradio('save_contents')).then(
        lambda: 'prompts/', None, gradio('save_root')).then(
        lambda: utils.current_time() + '.txt', None, gradio('save_filename')).then(
        lambda: gr.update(visible=True), None, gradio('file_saver'))

    shared.gradio['delete_prompt-default'].click(
        lambda: 'prompts/', None, gradio('delete_root')).then(
        lambda x: x + '.txt', gradio('prompt_menu-default'), gradio('delete_filename')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))

    shared.gradio['textbox-default'].change(lambda x: f"<span>{count_tokens(x)}</span>", gradio('textbox-default'), gradio('token-counter-default'), show_progress=False)
    shared.gradio['get_logits-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        logits.get_next_logits, gradio('textbox-default', 'interface_state', 'use_samplers-default', 'logits-default'), gradio('logits-default', 'logits-default-previous'), show_progress=False)

    shared.gradio['get_tokens-default'].click(get_token_ids, gradio('textbox-default'), gradio('tokens-default'), show_progress=False)
