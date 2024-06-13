import gc
import os
import pprint
import re
import time
import traceback
from pathlib import Path

import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import (
    is_ccl_available,
    is_npu_available,
    is_xpu_available
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig
)

import modules.shared as shared
from modules import RoPE, sampler_hijack
from modules.logging_colors import logger
from modules.models_settings import get_model_metadata

transformers.logging.set_verbosity_error()

local_rank = None
if shared.args.deepspeed:
    import deepspeed
    from transformers.deepspeed import (
        HfDeepSpeedConfig,
        is_deepspeed_zero3_enabled
    )

    from modules.deepspeed_parameters import generate_ds_config

    # Distributed setup
    local_rank = shared.args.local_rank if shared.args.local_rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if is_xpu_available() and is_ccl_available():
        torch.xpu.set_device(local_rank)
        deepspeed.init_distributed(backend="ccl")
    elif is_npu_available():
        torch.npu.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="hccl")
    else:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()
    ds_config = generate_ds_config(shared.args.bf16, 1 * world_size, shared.args.nvme_offload_dir)
    dschf = HfDeepSpeedConfig(ds_config)  # Keep this object alive for the Transformers integration

sampler_hijack.hijack_samplers()


last_generation_time = time.time()


def load_model(model_name, loader=None):
    logger.info(f"正在加载\"{model_name}\"")
    t0 = time.time()

    shared.is_seq2seq = False
    shared.model_name = model_name
    load_func_map = {
        'Transformers': huggingface_loader,
        'AutoGPTQ': AutoGPTQ_loader,
        'llama.cpp': llamacpp_loader,
        'llamacpp_HF': llamacpp_HF_loader,
        'ExLlamav2': ExLlamav2_loader,
        'ExLlamav2_HF': ExLlamav2_HF_loader,
        'AutoAWQ': AutoAWQ_loader,
        'HQQ': HQQ_loader,
    }

    metadata = get_model_metadata(model_name)
    if loader is None:
        if shared.args.loader is not None:
            loader = shared.args.loader
        else:
            loader = metadata['loader']
            if loader is None:
                logger.error('模型的路径不存在。正在退出。')
                raise ValueError

    shared.args.loader = loader
    output = load_func_map[loader](model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)

    shared.settings.update({k: v for k, v in metadata.items() if k in shared.settings})
    if loader.lower().startswith('exllama'):
        shared.settings['truncation_length'] = shared.args.max_seq_len
    elif loader in ['llama.cpp', 'llamacpp_HF']:
        shared.settings['truncation_length'] = shared.args.n_ctx

    logger.info(f"在{(time.time()-t0):.2f}秒内加载了\"{model_name}\"。")
    logger.info(f"加载器：\"{loader}\"")
    logger.info(f"截断长度：{shared.settings['truncation_length']}")
    logger.info(f"指令模板：\"{metadata['instruction_template']}\"")
    return model, tokenizer


def load_tokenizer(model_name, model):
    tokenizer = None
    path_to_model = Path(f"{shared.args.model_dir}/{model_name}/")
    if path_to_model.exists():
        if shared.args.no_use_fast:
            logger.info('正在use_fast设置为False的情况下加载词符化器。')

        tokenizer = AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=shared.args.trust_remote_code,
            use_fast=not shared.args.no_use_fast
        )

    return tokenizer


def huggingface_loader(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    params = {
        'low_cpu_mem_usage': True,
        'torch_dtype': torch.bfloat16 if shared.args.bf16 else torch.float16,
    }

    if shared.args.trust_remote_code:
        params['trust_remote_code'] = True

    if shared.args.use_flash_attention_2:
        params['use_flash_attention_2'] = True

    if shared.args.force_safetensors:
        params['force_safetensors'] = True

    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.args.trust_remote_code)

    if 'chatglm' in model_name.lower():
        LoaderClass = AutoModel
    else:
        if config.to_dict().get('is_encoder_decoder', False):
            LoaderClass = AutoModelForSeq2SeqLM
            shared.is_seq2seq = True
        else:
            LoaderClass = AutoModelForCausalLM

    # Load the model without any special settings
    if not any([shared.args.cpu, shared.args.load_in_8bit, shared.args.load_in_4bit, shared.args.auto_devices, shared.args.disk, shared.args.deepspeed, shared.args.gpu_memory is not None, shared.args.cpu_memory is not None, shared.args.compress_pos_emb > 1, shared.args.alpha_value > 1, shared.args.disable_exllama, shared.args.disable_exllamav2]):
        logger.info("TRANSFORMERS_PARAMS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(params)
        print()

        model = LoaderClass.from_pretrained(path_to_model, **params)
        if not (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit):
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                model = model.to(device)
            elif is_xpu_available():
                device = torch.device("xpu")
                model = model.to(device)
            elif is_npu_available():
                # workaround for https://gitee.com/ascend/pytorch/issues/I8KECW?from=project-issue, please remove me after fixed it.
                torch.npu.set_device(0)
                device = torch.device("npu")
                model = model.to(device)
            else:
                model = model.cuda()

    # DeepSpeed ZeRO-3
    elif shared.args.deepspeed:
        model = LoaderClass.from_pretrained(path_to_model, torch_dtype=params['torch_dtype'], trust_remote_code=params.get('trust_remote_code'))
        model = deepspeed.initialize(model=model, config_params=ds_config, model_parameters=None, optimizer=None, lr_scheduler=None)[0]
        model.module.eval()  # Inference
        logger.info(f'DeepSpeed ZeRO-3已启用：{is_deepspeed_zero3_enabled()}')

    # Load with quantization and/or offloading
    else:
        if not any((shared.args.cpu, torch.cuda.is_available(), is_xpu_available(), torch.backends.mps.is_available())):
            logger.warning('torch.cuda.is_available()和is_xpu_available()均返回False。这意味着未检测到GPU。正在回退到CPU模式。')
            shared.args.cpu = True

        if shared.args.cpu:
            params['torch_dtype'] = torch.float32
        else:
            params['device_map'] = 'auto'
            if x := get_max_memory_dict():
                params['max_memory'] = x

            if shared.args.load_in_4bit:
                # See https://github.com/huggingface/transformers/pull/23479/files
                # and https://huggingface.co/blog/4bit-transformers-bitsandbytes
                quantization_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': eval("torch.{}".format(shared.args.compute_dtype)) if shared.args.compute_dtype in ["bfloat16", "float16", "float32"] else None,
                    'bnb_4bit_quant_type': shared.args.quant_type,
                    'bnb_4bit_use_double_quant': shared.args.use_double_quant,
                    'llm_int8_enable_fp32_cpu_offload': True
                }

                params['quantization_config'] = BitsAndBytesConfig(**quantization_config_params)

            elif shared.args.load_in_8bit:
                if any((shared.args.auto_devices, shared.args.gpu_memory)):
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                else:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

                if params.get('max_memory') is not None:
                    with init_empty_weights():
                        model = LoaderClass.from_config(config, trust_remote_code=params.get('trust_remote_code'))

                    model.tie_weights()
                    params['device_map'] = infer_auto_device_map(
                        model,
                        dtype=torch.int8,
                        max_memory=params.get('max_memory'),
                        no_split_module_classes=model._no_split_modules
                    )

            if shared.args.disk:
                params['offload_folder'] = shared.args.disk_cache_dir

        if shared.args.disable_exllama or shared.args.disable_exllamav2:
            try:
                gptq_config = GPTQConfig(
                    bits=config.quantization_config.get('bits', 4),
                    disable_exllama=shared.args.disable_exllama,
                    disable_exllamav2=shared.args.disable_exllamav2,
                )

                params['quantization_config'] = gptq_config
                logger.info(f'正在以设置项disable_exllama={shared.args.disable_exllama}和disable_exllamav2={shared.args.disable_exllamav2}加载。')
            except:
                exc = traceback.format_exc()
                logger.error('禁用exllama失败。这个模型的config.json文件中是否包含必要的量化信息？')
                print(exc)

        if shared.args.compress_pos_emb > 1:
            params['rope_scaling'] = {'type': 'linear', 'factor': shared.args.compress_pos_emb}
        elif shared.args.alpha_value > 1:
            params['rope_scaling'] = {'type': 'dynamic', 'factor': RoPE.get_alpha_value(shared.args.alpha_value, shared.args.rope_freq_base)}

        logger.info("TRANSFORMERS_PARAMS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(params)
        print()
        model = LoaderClass.from_pretrained(path_to_model, **params)

    return model


def llamacpp_loader(model_name):
    from modules.llamacpp_model import LlamaCppModel

    path = Path(f'{shared.args.model_dir}/{model_name}')
    if path.is_file():
        model_file = path
    else:
        model_file = sorted(Path(f'{shared.args.model_dir}/{model_name}').glob('*.gguf'))[0]

    logger.info(f"检测到llama.cpp权重：\"{model_file}\"")
    model, tokenizer = LlamaCppModel.from_pretrained(model_file)
    return model, tokenizer


def llamacpp_HF_loader(model_name):
    from modules.llamacpp_hf import LlamacppHF

    path = Path(f'{shared.args.model_dir}/{model_name}')

    # Check if a HF tokenizer is available for the model
    if all((path / file).exists() for file in ['tokenizer_config.json']):
        logger.info(f'正在使用来自：“{path}”的词符化器')
    else:
        logger.error("无法加载模型，因为找不到Transformers格式的词符化器。")
        return None, None

    model = LlamacppHF.from_pretrained(model_name)
    return model


def AutoAWQ_loader(model_name):
    from awq import AutoAWQForCausalLM

    model_dir = Path(f'{shared.args.model_dir}/{model_name}')

    model = AutoAWQForCausalLM.from_quantized(
        quant_path=model_dir,
        max_new_tokens=shared.args.max_seq_len,
        trust_remote_code=shared.args.trust_remote_code,
        fuse_layers=not shared.args.no_inject_fused_attention,
        max_memory=get_max_memory_dict(),
        batch_size=1,
        safetensors=any(model_dir.glob('*.safetensors')),
    )

    return model


def AutoGPTQ_loader(model_name):
    import modules.AutoGPTQ_loader

    return modules.AutoGPTQ_loader.load_quantized(model_name)


def ExLlamav2_loader(model_name):
    from modules.exllamav2 import Exllamav2Model

    model, tokenizer = Exllamav2Model.from_pretrained(model_name)
    return model, tokenizer


def ExLlamav2_HF_loader(model_name):
    from modules.exllamav2_hf import Exllamav2HF

    return Exllamav2HF.from_pretrained(model_name)


def HQQ_loader(model_name):
    from hqq.core.quantize import HQQBackend, HQQLinear
    from hqq.models.hf.base import AutoHQQHFModel

    logger.info(f"正在使用后端“{shared.args.hqq_backend}”加载HQQ模型")

    model_dir = Path(f'{shared.args.model_dir}/{model_name}')
    model = AutoHQQHFModel.from_quantized(str(model_dir))
    HQQLinear.set_backend(getattr(HQQBackend, shared.args.hqq_backend))
    return model


def get_max_memory_dict():
    max_memory = {}
    max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
    if shared.args.gpu_memory:
        memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]

        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    # If --auto-devices is provided standalone, try to get a reasonable value
    # for the maximum memory of device :0
    elif shared.args.auto_devices:
        if is_xpu_available():
            total_mem = (torch.xpu.get_device_properties(0).total_memory / (1024 * 1024))
        else:
            total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        logger.warning(f"正在自动分配--gpu-memory {suggestion}给您的GPU，以尝试防止内存溢出错误。您也可以手动设置其他值。")
        max_memory[0] = f'{suggestion}GiB'
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    return max_memory if len(max_memory) > 0 else None


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        if is_xpu_available():
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()


def unload_model():
    shared.model = shared.tokenizer = None
    shared.previous_model_name = shared.model_name
    shared.model_name = 'None'
    shared.lora_names = []
    shared.model_dirty_from_training = False
    clear_torch_cache()


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)


def unload_model_if_idle():
    global last_generation_time

    logger.info(f"正在设置无操作自动卸载模型的超时时间为{shared.args.idle_timeout}分钟。")

    while True:
        shared.generation_lock.acquire()
        try:
            if time.time() - last_generation_time > shared.args.idle_timeout * 60:
                if shared.model is not None:
                    logger.info("由于长时间没有操作，正在卸载模型。")
                    unload_model()
        finally:
            shared.generation_lock.release()

        time.sleep(60)
