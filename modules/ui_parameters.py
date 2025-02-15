from pathlib import Path

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui(default_preset):
    mu = shared.args.multi_user
    generate_params = presets.load_preset(default_preset)
    with gr.Tab("参数", elem_id="parameters"):
        with gr.Tab("生成"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset, label='预设', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['random_preset'] = gr.Button('🎲', elem_classes='refresh-button')

                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(label="按加载器过滤", choices=["All"] + list(loaders.loaders_and_params.keys()), value="All", elem_classes='slim-dropdown')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('## 曲线形状')
                            shared.gradio['temperature'] = gr.Slider(0.01, 5, value=generate_params['temperature'], step=0.01, label='采样温度')
                            shared.gradio['dynatemp_low'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_low'], step=0.01, label='最低动态温度', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_high'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_high'], step=0.01, label='最高动态温度', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_exponent'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_exponent'], step=0.01, label='动态温度指数', visible=generate_params['dynamic_temperature'])
                            shared.gradio['smoothing_factor'] = gr.Slider(0.0, 10.0, value=generate_params['smoothing_factor'], step=0.01, label='平滑因子', info='激活二次采样。')
                            shared.gradio['smoothing_curve'] = gr.Slider(1.0, 10.0, value=generate_params['smoothing_curve'], step=0.01, label='平滑曲线', info='调整二次采样的下降曲线。')

                            gr.Markdown('## 曲线截断')
                            shared.gradio['min_p'] = gr.Slider(0.0, 1.0, value=generate_params['min_p'], step=0.01, label='Min P')
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='Top P')
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='Top K')
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='Typical P')
                            shared.gradio['xtc_threshold'] = gr.Slider(0, 0.5, value=generate_params['xtc_threshold'], step=0.01, label='XTC阈值', info='以一定的可能性移除除概率最小的词符以外的概率超过此阈值的词符。')
                            shared.gradio['xtc_probability'] = gr.Slider(0, 1, value=generate_params['xtc_probability'], step=0.01, label='XTC可能性', info='被纳入移除考虑的词符实际被移除的可能性。值为0表示此采样器不生效，1表示移除总是发生。')
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='ε截断')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='η截断')
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='无尾采样超参数')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='Top A')

                            gr.Markdown('## 重复抑制')
                            shared.gradio['dry_multiplier'] = gr.Slider(0, 5, value=generate_params['dry_multiplier'], step=0.01, label='DRY因子', info='将值设为大于零以启用DRY。推荐值：0.8。')
                            shared.gradio['dry_allowed_length'] = gr.Slider(1, 20, value=generate_params['dry_allowed_length'], step=1, label='DRY允许重复的序列长度', info='可免于被惩罚的最长重复序列。')
                            shared.gradio['dry_base'] = gr.Slider(1, 4, value=generate_params['dry_base'], step=0.01, label='DRY基数', info='控制随着重复的序列的长度增长，惩罚的增长有多快。')
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='重复度惩罚因子')
                            shared.gradio['frequency_penalty'] = gr.Slider(0, 2, value=generate_params['frequency_penalty'], step=0.05, label='按出现频率的重复度惩罚因子')
                            shared.gradio['presence_penalty'] = gr.Slider(0, 2, value=generate_params['presence_penalty'], step=0.05, label='按是否存在的重复度惩罚加数')
                            shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='编码器重复惩罚')
                            shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='禁止重复的N元语法元数')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='用于重复度惩罚计算的词符范围')

                        with gr.Column():
                            gr.Markdown('## 可选采样方法')
                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='惩罚系数α', info='用于对比搜索，必须取消勾选“使用采样算法”')
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='指导比例', info='用于CFG，1.5是个不错的值。')
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat模式', info='模式1仅适用于llama.cpp。')
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat参数τ')
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat参数η')

                            gr.Markdown('## 其他选项')
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], value=shared.settings['max_new_tokens'], step=1, label='最大新词符数', info='⚠️此值设置得太高可能会导致提示词被截断。')
                            shared.gradio['prompt_lookup_num_tokens'] = gr.Slider(value=shared.settings['prompt_lookup_num_tokens'], minimum=0, maximum=10, step=1, label='提示词查找解码词符数', info='启用提示词查找解码。')
                            shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='每秒最多词符数', info='为了文本实时可读。')
                            shared.gradio['max_updates_second'] = gr.Slider(value=shared.settings['max_updates_second'], minimum=0, maximum=24, step=1, label='每秒最大UI刷新次数', info='如果你在流式输出时感到UI卡顿，可以调整此设置。')

                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='使用采样算法')
                            shared.gradio['dynamic_temperature'] = gr.Checkbox(value=generate_params['dynamic_temperature'], label='动态温度')
                            shared.gradio['temperature_last'] = gr.Checkbox(value=generate_params['temperature_last'], label='温度采样放最后', info='将温度/动态温度/二次采样移至采样器堆栈的末端，忽略它们在“采样器优先级”中的位置。')
                            shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label='自动确定最大新词符数', info='将最大新词符数扩展到可用的上下文长度。')
                            shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='禁用序列终止符', info='强制模型永不过早结束生成。')
                            shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='在提示词开头添加序列起始符', info='禁用此项可以使回复更加具有创造性。')
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='跳过特殊词符', info='有些特定的模型需要取消这个设置。')
                            shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label='激活文本流式输出')
                            shared.gradio['static_cache'] = gr.Checkbox(value=shared.settings['static_cache'], label='静态 KV 缓存', info='使用静态缓存来提升性能。')

                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Number(precision=0, step=256, value=get_truncation_length(), label='将提示词截断至此长度', info='如果提示词超出这个长度，最左边的词符将被移除。大多数模型要求这个长度最多为2048。')
                            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='种子（-1表示随机）')

                            shared.gradio['sampler_priority'] = gr.Textbox(value=generate_params['sampler_priority'], lines=12, label='采样器优先级', info='参数名用新行或逗号分隔。', elem_classes=['add_scrollbar'])
                            shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=2, value=shared.settings["custom_stopping_strings"] or None, label='自定义停止字符串', info='用英文半角逗号分隔，用""包裹。', placeholder='"\\n", "\\nYou:"')
                            shared.gradio['custom_token_bans'] = gr.Textbox(value=shared.settings['custom_token_bans'] or None, label='禁用词符', info='填入要禁用的词符ID，用英文半角逗号分隔。你可以在默认或笔记本标签页获得词符的ID。')
                            shared.gradio['show_after'] = gr.Textbox(value=shared.settings['show_after'] or None, label='在此后显示', info='在出现此处指定的文本之前隐藏回复。', placeholder="</think>")
                            shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label='反向提示词', info='用于CFG。只有在指导比例不为1时使用。', lines=3, elem_classes=['add_scrollbar'])
                            shared.gradio['dry_sequence_breakers'] = gr.Textbox(value=generate_params['dry_sequence_breakers'], label='DRY序列匹配中断符', info='这些词符会打断并分隔序列的匹配。该参数以逗号分隔的引号字符串列表形式指定。')
                            with gr.Row() as shared.gradio['grammar_file_row']:
                                shared.gradio['grammar_file'] = gr.Dropdown(value='None', choices=utils.get_available_grammars(), label='从文件加载语法（.gbnf）', elem_classes='slim-dropdown')
                                ui.create_refresh_button(shared.gradio['grammar_file'], lambda: None, lambda: {'choices': utils.get_available_grammars()}, 'refresh-button', interactive=not mu)
                                shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                                shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

                            shared.gradio['grammar_string'] = gr.Textbox(value='', label='语法', lines=16, elem_classes=['add_scrollbar', 'monospace'])

        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(loaders.blacklist_samplers, gradio('filter_by_loader', 'dynamic_temperature'), gradio(loaders.list_all_samplers()), show_progress=False)
    shared.gradio['preset_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.load_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()), show_progress=False)

    shared.gradio['random_preset'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.random_preset, gradio('interface_state'), gradio('interface_state') + gradio(presets.presets_params()), show_progress=False)

    shared.gradio['grammar_file'].change(load_grammar, gradio('grammar_file'), gradio('grammar_string'), show_progress=False)
    shared.gradio['dynamic_temperature'].change(lambda x: [gr.update(visible=x)] * 3, gradio('dynamic_temperature'), gradio('dynatemp_low', 'dynatemp_high', 'dynatemp_exponent'), show_progress=False)


def get_truncation_length():
    if 'max_seq_len' in shared.provided_arguments or shared.args.max_seq_len != shared.args_defaults.max_seq_len:
        return shared.args.max_seq_len
    elif 'n_ctx' in shared.provided_arguments or shared.args.n_ctx != shared.args_defaults.n_ctx:
        return shared.args.n_ctx
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r', encoding='utf-8').read()
    else:
        return ''
