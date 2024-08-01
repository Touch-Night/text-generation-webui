import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image

from modules import chat, shared, ui, utils
from modules.html_generator import chat_html_wrapper
from modules.text_generation import stop_everything_event
from modules.utils import gradio

inputs = ('Chat input', 'interface_state')
reload_arr = ('history', 'name1', 'name2', 'mode', 'chat_style', 'character_menu')
clear_arr = ('delete_chat-confirm', 'delete_chat', 'delete_chat-cancel')


def create_ui():
    mu = shared.args.multi_user

    shared.gradio['Chat input'] = gr.State()
    shared.gradio['history'] = gr.JSON({'internal': [], 'visible': []}, visible=False)

    with gr.Tab('聊天', elem_id='chat-tab', elem_classes=("old-ui" if shared.args.chat_buttons else None)):
        with gr.Row():
            with gr.Column(elem_id='chat-col'):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}, '', '', 'chat', 'cai-chat', ''))

                with gr.Row(elem_id="chat-input-row"):
                    with gr.Column(scale=1, elem_id='gr-hover-container'):
                        gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                    with gr.Column(scale=10, elem_id='chat-input-container'):
                        shared.gradio['textbox'] = gr.Textbox(label='', placeholder='发送消息', elem_id='chat-input', elem_classes=['add_scrollbar'])
                        shared.gradio['show_controls'] = gr.Checkbox(value=shared.settings['show_controls'], label='显示控件 (Ctrl+S)', elem_id='show-controls')
                        shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='打字中', elem_id='typing-container')

                    with gr.Column(scale=1, elem_id='generate-stop-container'):
                        with gr.Row():
                            shared.gradio['Stop'] = gr.Button('停止', elem_id='stop', visible=False)
                            shared.gradio['Generate'] = gr.Button('生成', elem_id='Generate', variant='primary')

        # Hover menu buttons
        with gr.Column(elem_id='chat-buttons'):
            with gr.Row():
                shared.gradio['Regenerate'] = gr.Button('重新生成 (Ctrl + Enter)', elem_id='Regenerate')
                shared.gradio['Continue'] = gr.Button('继续 (Alt + Enter)', elem_id='Continue')
                shared.gradio['Remove last'] = gr.Button('删除上一条 (Ctrl + Shift + Backspace)', elem_id='Remove-last')

            with gr.Row():
                shared.gradio['Replace last reply'] = gr.Button('替换上一条回复 (Ctrl + Shift + L)', elem_id='Replace-last')
                shared.gradio['Copy last reply'] = gr.Button('复制上一条回复 (Ctrl + Shift + K)', elem_id='Copy-last')
                shared.gradio['Impersonate'] = gr.Button('AI帮答 (Ctrl + Shift + M)', elem_id='Impersonate')

            with gr.Row():
                shared.gradio['Send dummy message'] = gr.Button('发送假消息')
                shared.gradio['Send dummy reply'] = gr.Button('触发假回复')

            with gr.Row():
                shared.gradio['send-chat-to-default'] = gr.Button('发送至默认')
                shared.gradio['send-chat-to-notebook'] = gr.Button('发送至笔记本')

        with gr.Row(elem_id='past-chats-row', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['rename_chat'] = gr.Button('重命名', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_chat'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_chat-confirm'] = gr.Button('确认', variant='stop', visible=False, elem_classes=['refresh-button', 'focus-on-chat-input'])
                    shared.gradio['delete_chat-cancel'] = gr.Button('取消', visible=False, elem_classes=['refresh-button', 'focus-on-chat-input'])
                    shared.gradio['Start new chat'] = gr.Button('新建聊天', elem_classes=['refresh-button', 'focus-on-chat-input'])

                with gr.Row(elem_id='rename-row'):
                    shared.gradio['rename_to'] = gr.Textbox(label='重命名为：', placeholder='新名称', visible=False, elem_classes=['no-background'])
                    with gr.Row():
                        shared.gradio['rename_to-confirm'] = gr.Button('确认', visible=False, elem_classes=['refresh-button', 'focus-on-chat-input'])
                        shared.gradio['rename_to-cancel'] = gr.Button('取消', visible=False, elem_classes=['refresh-button', 'focus-on-chat-input'])

                gr.Markdown("过往聊天")
                with gr.Row():
                    shared.gradio['unique_id'] = gr.Radio(label="", elem_classes=['slim-dropdown', 'pretty_scrollbar'], interactive=not mu, elem_id='past-chats')

        with gr.Row(elem_id='chat-controls', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['start_with'] = gr.Textbox(label='回复开头', placeholder='当然可以！', value=shared.settings['start_with'], elem_classes=['add_scrollbar'])

                with gr.Row():
                    shared.gradio['mode'] = gr.Radio(choices=['chat', 'chat-instruct', 'instruct'], value=shared.settings['mode'] if shared.settings['mode'] in ['chat', 'chat-instruct'] else None, label='模式', info='定义如何生成聊天提示。在 instruct 和 chat-instruct 模式下，默认使用 参数 > 指令模板下 选择的指令模板。', elem_id='chat-mode')

                with gr.Row():
                    shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='聊天界面风格', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')

                with gr.Row():
                    shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=12, label='chat-instruct模式下的指令', info='“<|character|>”和“<|prompt|>”分别会被替换成机器人名称和常规聊天提示词。', visible=shared.settings['mode'] == 'chat-instruct', elem_classes=['add_scrollbar'])


def create_chat_settings_ui():
    mu = shared.args.multi_user
    with gr.Tab('聊天'):
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Tab("角色"):
                    with gr.Row():
                        shared.gradio['character_menu'] = gr.Dropdown(value=None, choices=utils.get_available_characters(), label='角色', elem_id='character-menu', info='用在chat和chat-instruct模式下。', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button', elem_id="save-character", interactive=not mu)
                        shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)

                    shared.gradio['name2'] = gr.Textbox(value='', lines=1, label='角色的名字')
                    shared.gradio['context'] = gr.Textbox(value='', lines=10, label='背景', elem_classes=['add_scrollbar'])
                    shared.gradio['greeting'] = gr.Textbox(value='', lines=5, label='问候', elem_classes=['add_scrollbar'])

                with gr.Tab("用户"):
                    shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='名字')
                    shared.gradio['user_bio'] = gr.Textbox(value=shared.settings['user_bio'], lines=10, label='描述', info='您可以在这里写下有关您自己的描述。', placeholder='{{user}}\'s personality: ...', elem_classes=['add_scrollbar'])

                with gr.Tab('聊天记录'):
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['save_chat_history'] = gr.Button(value='保存历史记录')

                        with gr.Column():
                            shared.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='上传历史记录JSON文件')

                with gr.Tab('上传角色'):
                    with gr.Tab('YAML 或 JSON'):
                        with gr.Row():
                            shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='JSON 或 YAML 文件', interactive=not mu)
                            shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='头像（可选）', interactive=not mu)

                        shared.gradio['Submit character'] = gr.Button(value='提交', interactive=False)

                    with gr.Tab('TavernAI 角色卡'):
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['upload_img_tavern'] = gr.Image(type='pil', label='TavernAI PNG 文件', elem_id='upload_img_tavern', interactive=not mu)
                                shared.gradio['tavern_json'] = gr.State()
                            with gr.Column():
                                shared.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='名字', interactive=False)
                                shared.gradio['tavern_desc'] = gr.Textbox(value='', lines=10, label='描述', interactive=False, elem_classes=['add_scrollbar'])

                        shared.gradio['Submit tavern character'] = gr.Button(value='提交', interactive=False)

            with gr.Column(scale=1):
                shared.gradio['character_picture'] = gr.Image(label='角色头像', type='pil', interactive=not mu)
                shared.gradio['your_picture'] = gr.Image(label='您的头像', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None, interactive=not mu)

    with gr.Tab('指令模板'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='已保存的指令模板', info="选择模板后，点击“加载”来加载并应用它。", value='None', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)
                    shared.gradio['load_template'] = gr.Button("加载", elem_classes='refresh-button')
                    shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

            with gr.Column():
                pass

        with gr.Row():
            with gr.Column():
                shared.gradio['custom_system_message'] = gr.Textbox(value=shared.settings['custom_system_message'], lines=2, label='自定义系统消息', info='如果不为空，将代替默认消息使用。', elem_classes=['add_scrollbar'])
                shared.gradio['instruction_template_str'] = gr.Textbox(value='', label='指令模板', lines=24, info='根据您正在使用的模型/LoRA进行更改。在指令和聊天指令模式下使用。', elem_classes=['add_scrollbar', 'monospace'])
                with gr.Row():
                    shared.gradio['send_instruction_to_default'] = gr.Button('发送至默认', elem_classes=['small-button'])
                    shared.gradio['send_instruction_to_notebook'] = gr.Button('发送至笔记本', elem_classes=['small-button'])
                    shared.gradio['send_instruction_to_negative_prompt'] = gr.Button('发送至负面提示', elem_classes=['small-button'])

            with gr.Column():
                shared.gradio['chat_template_str'] = gr.Textbox(value=shared.settings['chat_template_str'], label='聊天模板', lines=22, elem_classes=['add_scrollbar', 'monospace'])


def create_event_handlers():

    # Obsolete variables, kept for compatibility with old extensions
    shared.input_params = gradio(inputs)
    shared.reload_inputs = gradio(reload_arr)

    shared.gradio['Generate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Regenerate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, regenerate=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, _continue=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Impersonate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x, gradio('textbox'), gradio('Chat input'), show_progress=False).then(
        chat.impersonate_wrapper, gradio(inputs), gradio('textbox', 'display'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Replace last reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_replace_last_reply_click, gradio('textbox', 'interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Send dummy message'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_dummy_message_click, gradio('textbox', 'interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Send dummy reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_dummy_reply_click, gradio('textbox', 'interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Remove last'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_remove_last_click, gradio('interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Stop'].click(
        stop_everything_event, None, None, queue=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display'), show_progress=False)

    if not shared.args.multi_user:
        shared.gradio['unique_id'].select(
            ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
            chat.handle_unique_id_select, gradio('interface_state'), gradio('history', 'display'), show_progress=False)

    shared.gradio['Start new chat'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_start_new_chat_click, gradio('interface_state'), gradio('history', 'display', 'unique_id'), show_progress=False)

    shared.gradio['delete_chat'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, gradio(clear_arr))
    shared.gradio['delete_chat-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, gradio(clear_arr))
    shared.gradio['delete_chat-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_delete_chat_confirm_click, gradio('interface_state'), gradio('history', 'display', 'unique_id') + gradio(clear_arr), show_progress=False)

    shared.gradio['rename_chat'].click(chat.handle_rename_chat_click, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)
    shared.gradio['rename_to-cancel'].click(lambda: [gr.update(visible=False)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)
    shared.gradio['rename_to-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_rename_chat_confirm, gradio('rename_to', 'interface_state'), gradio('unique_id', 'rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)

    shared.gradio['rename_to'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_rename_chat_confirm, gradio('rename_to', 'interface_state'), gradio('unique_id', 'rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)

    shared.gradio['load_chat_history'].upload(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_upload_chat_history, gradio('load_chat_history', 'interface_state'), gradio('history', 'display', 'unique_id'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_chat()}}')

    shared.gradio['character_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_character_menu_change, gradio('interface_state'), gradio('history', 'display', 'name1', 'name2', 'character_picture', 'greeting', 'context', 'unique_id'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.update_big_picture_js}; updateBigPicture()}}')

    shared.gradio['mode'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_mode_change, gradio('interface_state'), gradio('history', 'display', 'chat_style', 'chat-instruct_command', 'unique_id'), show_progress=False).then(
        None, gradio('mode'), None, js="(mode) => {mode === 'instruct' ? document.getElementById('character-menu').parentNode.parentNode.style.display = 'none' : document.getElementById('character-menu').parentNode.parentNode.style.display = ''}")

    shared.gradio['chat_style'].change(chat.redraw_html, gradio(reload_arr), gradio('display'), show_progress=False)
    shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, gradio('history'), gradio('textbox'), show_progress=False)

    # Save/delete a character
    shared.gradio['save_character'].click(chat.handle_save_character_click, gradio('name2'), gradio('save_character_filename', 'character_saver'), show_progress=False)
    shared.gradio['delete_character'].click(lambda: gr.update(visible=True), None, gradio('character_deleter'), show_progress=False)
    shared.gradio['load_template'].click(chat.handle_load_template_click, gradio('instruction_template'), gradio('instruction_template_str', 'instruction_template'), show_progress=False)
    shared.gradio['save_template'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_save_template_click, gradio('instruction_template_str'), gradio('save_filename', 'save_root', 'save_contents', 'file_saver'), show_progress=False)

    shared.gradio['delete_template'].click(chat.handle_delete_template_click, gradio('instruction_template'), gradio('delete_filename', 'delete_root', 'file_deleter'), show_progress=False)
    shared.gradio['save_chat_history'].click(
        lambda x: json.dumps(x, indent=4), gradio('history'), gradio('temporary_text')).then(
        None, gradio('temporary_text', 'character_menu', 'mode'), None, js=f'(hist, char, mode) => {{{ui.save_files_js}; saveHistory(hist, char, mode)}}')

    shared.gradio['Submit character'].click(
        chat.upload_character, gradio('upload_json', 'upload_img_bot'), gradio('character_menu'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['Submit tavern character'].click(
        chat.upload_tavern_character, gradio('upload_img_tavern', 'tavern_json'), gradio('character_menu'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['upload_json'].upload(lambda: gr.update(interactive=True), None, gradio('Submit character'))
    shared.gradio['upload_json'].clear(lambda: gr.update(interactive=False), None, gradio('Submit character'))
    shared.gradio['upload_img_tavern'].upload(chat.check_tavern_character, gradio('upload_img_tavern'), gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['upload_img_tavern'].clear(lambda: (None, None, None, gr.update(interactive=False)), None, gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['your_picture'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_your_picture_change, gradio('your_picture', 'interface_state'), gradio('display'), show_progress=False)

    shared.gradio['send_instruction_to_default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_instruction_click, gradio('interface_state'), gradio('textbox-default'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send_instruction_to_notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_instruction_click, gradio('interface_state'), gradio('textbox-notebook'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['send_instruction_to_negative_prompt'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_instruction_click, gradio('interface_state'), gradio('negative_prompt'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_generation_parameters()}}')

    shared.gradio['send-chat-to-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_chat_click, gradio('interface_state'), gradio('textbox-default'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send-chat-to-notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_chat_click, gradio('interface_state'), gradio('textbox-notebook'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['show_controls'].change(None, gradio('show_controls'), None, js=f'(x) => {{{ui.show_controls_js}; toggle_controls(x)}}')
