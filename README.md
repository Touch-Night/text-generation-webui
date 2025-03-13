# 写在前面

这是原项目[text-generation-webui](https://github.com/oobabooga/text-generation-webui)的完全汉化版。  
相较于原版，做了以下改动：
- 用户界面、命令行内容、部分文档的完整翻译  
- 使用上海交大的torch源，其他包使用清华大学源，使用mirror.ghproxy下载github上的软件包，使用hf-mirror下载模型。因此安装到使用的全过程都不需要挂梯子  
- 添加对华为昇腾NPU的支持  

以下是原项目的README翻译
# Text generation web UI

一个用于大型语言模型的Gradio Web UI。

此项目的目标是成为文本生成领域的[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

[试试看深度思考扩展](https://oobabooga.gumroad.com/l/deep_reason)

|![Image1](https://github.com/oobabooga/screenshots/raw/main/AFTER-INSTRUCT.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/AFTER-CHAT.png) |
|:---:|:---:|
|![Image1](https://github.com/oobabooga/screenshots/raw/main/AFTER-DEFAULT.png) | ![Image2](https://github.com/oobabooga/screenshots/raw/main/AFTER-PARAMETERS.png) |

## 功能

- 在一个UI/API中支持多种文本生成后端，包括 [Transformers](https://github.com/huggingface/transformers)、[llama.cpp](https://github.com/ggerganov/llama.cpp) 和 [ExLlamaV2](https://github.com/turboderp-org/exllamav2)。[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 可以通过其专属的 [Dockerfile](https://github.com/oobabooga/text-generation-webui/blob/main/docker/TensorRT-LLM/Dockerfile) 使用。Transformers 加载器兼容诸如 [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)、[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)、[HQQ](https://github.com/mobiusml/hqq) 和 [AQLM](https://github.com/Vahe1994/AQLM) 等库，但这些库需要手动安装。
- 具有聊天和补全端点的OpenAI兼容API服务器 -- 请参阅[示例](https://github.com/Touch-Night/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API#examples)。
- 自动使用模型自带的元数据中的Jinja2模板对其进行提示词格式化。
- 三种聊天模式：`instruct`（指令）、`chat-instruct`（聊天-指令）和`chat`（聊天），既可以让模型遵循指令，也可以与角色进行随意对话。`chat-instruct`模式会自动将模型的模板应用于聊天提示词，确保输出高质量而无需手动设置。
- “过往聊天”菜单，可快速切换对话或开始新对话。
- 在默认/笔记本标签页中进行自由形式的生成，不受聊天轮次的限制。可以从聊天标签页将格式化的聊天对话发送到这些标签页。
- 多种采样参数和生成选项，用于复杂的文本生成控制。
- 使用“模型”标签页，通过用户界面轻松切换不同模型，而无需重启。
- 简单的LoRA微调工具，可使用您的数据自定义模型。
- 所有内容都在一个文件夹中。所需的依赖项安装在一个独立的`installer_files`文件夹中，不会干扰系统环境。
- 支持扩展功能，提供众多内置和用户贡献的扩展。有关详细信息，请参阅 [Wiki](https://github.com/Touch-Night/text-generation-webui/wiki/07-%E2%80%90-%E6%89%A9%E5%B1%95) 和 [扩展目录](https://github.com/oobabooga/text-generation-webui-extensions)。

## 如何安装

1) 克隆或[下载](https://ghfast.top/https://github.com/Touch-Night/text-generation-webui/releases/download/v1.15/text-generation-webui-Chinese.zip)此存储库。
2) 根据您的操作系统运行`start_linux.sh`，`start_windows.bat`，`start_macos.sh`或`start_wsl.bat`脚本。
3) 问到时选择您的GPU供应商。
4) 安装结束后，访问`http://localhost:7860`。
5) 玩得开心！

要在将来重新启动Web UI，只需再次运行`start_`脚本。如果您需要重新安装依赖，只需删除该文件夹并再次启动Web UI。

此脚本接受命令行参数，例如`./start_linux.sh --help`。或者，您可以使用文本编辑器编辑`CMD_FLAGS.txt`文件并在其中添加命令行参数，例如你可以添加`--api`来启用API功能。要在将来更新此软件，请运行`update_wizard_linux.sh`，`update_wizard_windows.bat`，`update_wizard_macos.sh`或`update_wizard_wsl.bat`。

<details>
<summary>
安装细节和手动安装的信息
</summary>

### 一键安装脚本

此脚本使用Miniconda在`installer_files`文件夹中建立Conda环境。

如果您需要在`installer_files`环境中手动安装某些内容，可以使用cmd脚本启动交互式shell：`cmd_linux.sh`，`cmd_windows.bat`，`cmd_macos.sh`或`cmd_wsl.bat`。

* 无需以管理员/root用户身份运行这些脚本（`start_`，`update_wizard_`或`cmd_`）。
* 要安装扩展的依赖，您可以使用您的操作系统的`extensions_reqs`脚本。最后，此脚本将安装项目的主依赖，以确保在版本冲突的情况下它们优先。
* 有关AMD和WSL设置的其他说明，请参阅[此文档](https://github.com/Touch-Night/text-generation-webui/wiki)。
* 为了自动安装，您可以使用`GPU_CHOICE`，`USE_CUDA118`，`LAUNCH_AFTER_INSTALL`和`INSTALL_EXTENSIONS`环境变量。例如：`GPU_CHOICE=A USE_CUDA118=FALSE LAUNCH_AFTER_INSTALL=FALSE INSTALL_EXTENSIONS=TRUE ./start_linux.sh`。

### 使用Conda手动安装

如果您有使用命令行的经验，方可使用这种方式。

#### 0.安装Conda

https://docs.conda.io/en/latest/miniconda.html

在Linux或WSL上，可以使用这两个命令自动安装（ [来源](https://educe-ubc.github.io/conda.html) ）：

```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh
```

#### 1.创建一个新的Conda环境

```
conda create -n textgen python=3.11
conda activate textgen
```

#### 2.安装Pytorch

| 系统 | GPU | 命令 |
|--------|---------|---------|
| Linux/WSL | Nvidia | `pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121` |
| Linux/WSL | 仅CPU | `pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu` |
| Linux | AMD | `pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1` |
| MacOS + MPS | 任意 | `pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1` |
| Windows | Nvidia | `pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121` |
| Windows | 仅CPU | `pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1` |

最新的命令可以在这里找到：https://pytorch.org/get-started/locally/ 。

对于NVIDIA，您还需要安装CUDA运行时库：

```
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-runtime
```

如果你需要 `nvcc`  来手动编译一些库，请用下面的命令替换上述命令：

```
conda install -y -c "nvidia/label/cuda-12.1.1" cuda
```

#### 3.安装Web UI

```
git clone --recursive -b Chinese https://gitee.com/touchnight/text-generation-webui.git
cd text-generation-webui
pip install -r <根据下表确定的依赖文件>
```

要使用的依赖文件：

| GPU | CPU | 要使用的依赖文件 |
|--------|---------|---------|
| NVIDIA | 支持AVX2指令集 | `requirements.txt` |
| NVIDIA | 不支持AVX2指令集 | `requirements_noavx2.txt` |
| AMD | 支持AVX2指令集 | `requirements_amd.txt` |
| AMD | 不支持AVX2指令集 | `requirements_amd_noavx2.txt` |
| 仅CPU | 支持AVX2指令集 | `requirements_cpu_only.txt` |
| 仅CPU | 不支持AVX2指令集 | `requirements_cpu_only_noavx2.txt` |
| Apple | Intel | `requirements_apple_intel.txt` |
| Apple | Apple Silicon | `requirements_apple_silicon.txt` |

### 启动Web UI

```
conda activate textgen
cd text-generation-webui
python server.py
```

然后浏览

`http://localhost:7860/?__theme=dark`

##### Windows上的AMD GPU

1) 在上面的命令中使用 `requirements_cpu_only.txt` 或者 `requirements_cpu_only_noavx2.txt`。

2) 根据你的硬件使用适当的命令手动安装llama-cpp-python：[从PyPI安装](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration) 。
    * 使用 `LLAMA_HIPBLAS=on` 切换键。
    * 注意 [Windows remarks](https://github.com/abetlen/llama-cpp-python#windows-remarks) 。

3) 手动安装autoGPTQ：[安装方法](https://github.com/PanQiWei/AutoGPTQ#install-from-source) 。
    * 从源代码安装 - Windows没有预构建的ROCm包。

##### 较老的NVIDIA GPU

1) 对于Kepler GPU和较早的GPU，您需要安装CUDA 11.8而不是12：

```
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-runtime
```

2) bitsandbytes >= 0.39 可能无法正常工作。在这种情况下，使用 `--load-in-8bit` ，您可能必须这样降级：
    * Linux： `pip install bitsandbytes==0.38.1` 
    * Windows： `pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl` 

##### 手动安装

`requirements*.txt` 包含通过GitHub Action预编译的各种轮子。如果您想手动编译它们，或者您因为没有合适的轮子可用于您的硬件而需要这么做，则可以使用 `requirements_nowheels.txt` 然后手动安装所需的加载器。

### 另一可选方案：Docker

```
对于NVIDIA GPU:
ln -s docker/{nvidia/Dockerfile,nvidia/docker-compose.yml,.dockerignore} .
对于AMD GPU: 
ln -s docker/{amd/Dockerfile,intel/docker-compose.yml,.dockerignore} .
对于Intel GPU:
ln -s docker/{intel/Dockerfile,amd/docker-compose.yml,.dockerignore} .
对于仅CPU
ln -s docker/{cpu/Dockerfile,cpu/docker-compose.yml,.dockerignore} .
cp docker/.env.example .env
# 创建 logs/cache 目录 : 
mkdir -p logs cache
# 编辑 .env 并设置以下内容: 
#   TORCH_CUDA_ARCH_LIST （据你的GPU型号而定）
#   APP_RUNTIME_GID      你的主机用户的组ID（在终端中运行 `id -g`查看）
#   BUILD_EXTENIONS      可选地添加逗号分隔的扩展名列表以构建
# 编辑 CMD_FLAGS.txt 并在其中添加您想要执行的选项（如 --listen --cpu）
# 
docker compose up --build
```

*您需要安装Docker Compose v2.17或更高的版本。查看 [本指南](https://github.com/Touch-Night/text-generation-webui/wiki/09-%E2%80%90-Docker)获取说明。
*有关其他Docker文件，请查看[这个存储库](https://github.com/Atinoda/text-generation-webui-docker) 。

### 更新依赖

随着时间的推移，`requirements*.txt` 可能改变。要更新，请使用以下命令：

```
conda activate textgen
cd text-generation-webui
pip install -r <你曾使用过的依赖文件> --upgrade
```
</details>

<details>
<summary>
命令行参数列表
</summary>

```txt
使用方法: server.py [-h] [--multi-user] [--character CHARACTER] [--model MODEL] [--lora LORA [LORA ...]] [--model-dir MODEL_DIR] [--lora-dir LORA_DIR] [--model-menu] [--settings SETTINGS]
                 [--extensions EXTENSIONS [EXTENSIONS ...]] [--verbose] [--idle-timeout IDLE_TIMEOUT] [--loader LOADER] [--cpu] [--auto-devices] [--gpu-memory GPU_MEMORY [GPU_MEMORY ...]]
                 [--cpu-memory CPU_MEMORY] [--disk] [--disk-cache-dir DISK_CACHE_DIR] [--load-in-8bit] [--bf16] [--no-cache] [--trust-remote-code] [--force-safetensors] [--no_use_fast]
                 [--use_flash_attention_2] [--use_eager_attention] [--torch-compile] [--load-in-4bit] [--use_double_quant] [--compute_dtype COMPUTE_DTYPE] [--quant_type QUANT_TYPE] [--flash-attn]
                 [--tensorcores] [--n_ctx N_CTX] [--threads THREADS] [--threads-batch THREADS_BATCH] [--no_mul_mat_q] [--n_batch N_BATCH] [--no-mmap] [--mlock] [--n-gpu-layers N_GPU_LAYERS]
                 [--tensor_split TENSOR_SPLIT] [--numa] [--logits_all] [--no_offload_kqv] [--cache-capacity CACHE_CAPACITY] [--row_split] [--streaming-llm] [--attention-sink-size ATTENTION_SINK_SIZE]
                 [--tokenizer-dir TOKENIZER_DIR] [--gpu-split GPU_SPLIT] [--autosplit] [--max_seq_len MAX_SEQ_LEN] [--cfg-cache] [--no_flash_attn] [--no_xformers] [--no_sdpa]
                 [--num_experts_per_token NUM_EXPERTS_PER_TOKEN] [--enable_tp] [--hqq-backend HQQ_BACKEND] [--cpp-runner] [--cache_type CACHE_TYPE] [--deepspeed] [--nvme-offload-dir NVME_OFFLOAD_DIR]
                 [--local_rank LOCAL_RANK] [--alpha_value ALPHA_VALUE] [--rope_freq_base ROPE_FREQ_BASE] [--compress_pos_emb COMPRESS_POS_EMB] [--listen] [--listen-port LISTEN_PORT]
                 [--listen-host LISTEN_HOST] [--share] [--auto-launch] [--gradio-auth GRADIO_AUTH] [--gradio-auth-path GRADIO_AUTH_PATH] [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE]
                 [--subpath SUBPATH] [--old-colors] [--api] [--public-api] [--public-api-id PUBLIC_API_ID] [--api-port API_PORT] [--api-key API_KEY] [--admin-key ADMIN_KEY] [--api-enable-ipv6]
                 [--api-disable-ipv4] [--nowebui] [--multimodal-pipeline MULTIMODAL_PIPELINE] [--cache_4bit] [--cache_8bit] [--chat-buttons] [--triton] [--no_inject_fused_mlp] [--no_use_cuda_fp16]
                 [--desc_act] [--disable_exllama] [--disable_exllamav2] [--wbits WBITS] [--groupsize GROUPSIZE]

Text generation web UI

可选项：
  -h, --help                                     显示此帮助消息然后退出

基础设置：
  --multi-user                                   多用户模式。聊天历史将不保存或自动加载。警告：公开分享可能不安全。
  --character CHARACTER                          默认情况下，要在聊天模式加载的角色名称。
  --model MODEL                                  默认情况下加载的模型名称。
  --lora LORA [LORA ...]                         加载的LoRA列表。如果您想加载多个LoRA，请写下由空格分开的名称。
  --model-dir MODEL_DIR                          所有模型的目录路径。
  --lora-dir LORA_DIR                            所有LoRA的目录路径。
  --model-menu                                   UI首次启动时，在终端中显示模型菜单。
  --settings SETTINGS                            从此YAML文件加载默认接口设置。参见settings-template.yaml以获取示例。如果您创建了一个名为settings.yaml的文件，该文件将被默认加载，无需
                                                 使用--settings命令行参数。
  --extensions EXTENSIONS [EXTENSIONS ...]       加载的扩展列表。如果要加载多个扩展，请写下由空格隔开的名称。
  --verbose                                      将提示词打印到终端。
  --idle-timeout IDLE_TIMEOUT                    在这么多分钟不活动后卸载模型。当您再次尝试使用它时，模型将自动重新加载。

模型加载器：
  --loader LOADER                                手动选择模型加载器，否则，它将被自动检测。可选选项：Transformers，llama.cpp，llamacpp_HF，Exllamav2_HF，Exllamav2，
                                                 HQQ，TensorRT-LLM。

Transformers/Accelerate：
  --cpu                                          使用CPU生成文本。警告：使用CPU训练非常慢。
  --auto-devices                                 自动将模型划分到可用的GPU和CPU上。
  --gpu-memory GPU_MEMORY [GPU_MEMORY ...]       为每个GPU分配的最大GPU内存，单位为GiB。例如：单个GPU使用 --gpu-memory 10，两个GPU使用 --gpu-memory 10 5。你也可以像这样
                                                 用MiB来设置值 --gpu-memory 3500MiB。
  --cpu-memory CPU_MEMORY                        用于分配卸载权重的最大CPU内存，单位为GiB。与上面相同。
  --disk                                         如果模型对于你的GPU和CPU的总和来说太大了，将剩余的层发送到磁盘。
  --disk-cache-dir DISK_CACHE_DIR                磁盘缓存保存目录。默认为 "cache" 。
  --load-in-8bit                                 使用bitsandbytes以8位精度加载模型。
  --bf16                                         使用bfloat16精度加载模型。需要Nvidia Ampere GPU。
  --no-cache                                     生成文本时设置 `use_cache` 为 `False`。这略微减少了显存的使用，但这也导致性能损失。
  --trust-remote-code                            加载模型时设置 `trust_remote_code=True`。这对于某些模型是必需的。
  --force-safetensors                            在加载模型时设置 `use_safetensors=True`。这可以防止任意代码执行。
  --no_use_fast                                  加载词符化器时设置use_fast=false（默认情况下为true）。如果您遇到与use_fast有关的任何问题，请使用此功能。
  --use_flash_attention_2                        在加载模型时设置use_flash_attention_2=True。
  --use_eager_attention                          在加载模型时设置attn_implementation=eager。
  --torch-compile                                使用torch.compile编译模型以提高性能。

bitsandbytes 4-比特：
  --load-in-4bit                                 使用bitsandbytes以4位精度加载模型。
  --use_double_quant                             对4位精度使用use_double_quant。
  --compute_dtype COMPUTE_DTYPE                  4位精度的计算数据类型。有效选项：bfoat16, float16, float32。
  --quant_type QUANT_TYPE                        4位精度的量化类型。有效选项：nf4, fp4。

llama.cpp：
  --flash-attn                                   使用flash-attention。
  --tensorcores                                  仅限NVIDIA显卡：使用不带GGML_CUDA_FORCE_MMQ的llama-cpp-python。这在比较新型号的显卡上可能能提高性能。
  --n_ctx N_CTX                                  提示词上下文的大小。
  --threads THREADS                              要使用的线程数。
  --threads-batch THREADS_BATCH                  用于批处理/提示词处理的线程数。
  --no_mul_mat_q                                 禁用mulmat内核。
  --n_batch N_BATCH                              在调用llama_eval时批量处理的提示词词符的最大数量。
  --no-mmap                                      防止使用mmap。
  --mlock                                        强制系统将模型保留在RAM中。
  --n-gpu-layers N_GPU_LAYERS                    卸载到GPU的层数。
  --tensor_split TENSOR_SPLIT                    在多个GPU上分割模型。逗号分隔的比例列表。示例：60,40。
  --numa                                         激活Llama.cpp的NUMA任务分配。
  --logits_all                                   要使困惑度评估起效，需要设置此项。否则，请忽略它，因为它会使提示词处理变慢。
  --no_offload_kqv                               不将K、Q、V卸载到GPU。这可以节省VRAM，但会降低性能。
  --cache-capacity CACHE_CAPACITY                最大缓存容量（llama-cpp-python）。示例：2000MiB, 2GiB。如果没有提供单位，默认为字节。
  --row_split                                    将模型按行分割到多个GPU上，这可能会提高多GPU的性能。
  --streaming-llm                                激活StreamingLLM以避免在删除旧消息时重新评估整个提示词。
  --attention-sink-size ATTENTION_SINK_SIZE      StreamingLLM：下沉词符的数量。仅在修剪后的提示词与旧提示词前缀不同时使用。
  --tokenizer-dir TOKENIZER_DIR                  从此指定的文件夹加载词符化器。用于通过命令行使用llamacpp_HF加载器。

ExLlamaV2：
  --gpu-split GPU_SPLIT                          逗号分隔的列表，指定每个GPU设备用于模型层的VRAM（以GB为单位）。示 例：20,7,7。
  --autosplit                                    将模型张量自动分割到可用的GPU上。这将导致--gpu-split被忽略。
  --max_seq_len MAX_SEQ_LEN                      最大序列长度。
  --cfg-cache                                    ExLlamav2_HF：为CFG负面提示创建一个额外的缓存。使用该加载器时，必须使用CFG。
  --no_flash_attn                                强制不使用flash-attention。
  --no_xformers                                  强制不使用xformers。
  --no_sdpa                                      强制不使用Torch SDPA。
  --num_experts_per_token NUM_EXPERTS_PER_TOKEN  用于生成的专家数量。适用于MoE模型，如Mixtral。
  --enable_tp                                    启用ExLlamav2的张量并行。

HQQ:
  --hqq-backend HQQ_BACKEND                      HQQ加载器的后端。有效选项：PYTORCH, PYTORCH_COMPILE, ATEN。

TensorRT-LLM:
  --cpp-runner                                   使用ModelRunnerCpp运行器，它比默认的ModelRunner更快，但目前不支持流式传输。

缓存:
  --cache_type CACHE_TYPE                        KV 缓存类型; 可选项: llama.cpp - fp16, q8_0, q4_0; ExLlamaV2 - fp16, fp8, q8, q6, q4。

DeepSpeed：
  --deepspeed                                    通过Transformers集成启用DeepSpeed ZeRO-3进行推理。
  --nvme-offload-dir NVME_OFFLOAD_DIR            DeepSpeed：用于ZeRO-3 NVME卸载的目录。
  --local_rank LOCAL_RANK                        DeepSpeed：分布式设置的可选参数。

RoPE：
  --alpha_value ALPHA_VALUE                      NTK RoPE缩放的位置嵌入alpha因子。使用此选项或compress_pos_emb，不要同时使用两者。
  --rope_freq_base ROPE_FREQ_BASE                如果大于0，将代替alpha_value使用。这两者符合rope_freq_base = 10000 * alpha_value ^ (64 / 63)关系式。
  --compress_pos_emb COMPRESS_POS_EMB            位置嵌入的压缩因子。应设置为 (上下文长度) / (模型原始上下文长度)。等于 1/rope_freq_scale。

Gradio：
  --listen                                       使web UI能够从你的本地网络访问。
  --listen-port LISTEN_PORT                      服务器将使用的监听端口。
  --listen-host LISTEN_HOST                      服务器将使用的主机名。
  --share                                        创建一个公共URL。这对于在Google Colab或类似环境上运行web UI很有用。
  --auto-launch                                  启动时在默认浏览器中打开web UI。
  --gradio-auth GRADIO_AUTH                      设置Gradio认证密码，格式为"uername:password"。也可以提供多个凭证，格式为"u1:p1,u2:p2,u3:p3"。
  --gradio-auth-path GRADIO_AUTH_PATH            设置Gradio认证文件路径。文件应包含一个或多和上面相同格式的用户:密码对。
  --ssl-keyfile SSL_KEYFILE                      SSL证书密钥文件的路径。
  --ssl-certfile SSL_CERTFILE                    SSL证书文件的路径。
  --subpath SUBPATH                              使用反向代理时自定义gradio的子路径。
  --old-colors                                   使用2024年12月更新之前的旧版Gradio颜色。

API：
  --api                                          启用API扩展。
  --public-api                                   使用CloudFlare为API创建公共URL。
  --public-api-id PUBLIC_API_ID                  命名Cloudflare Tunnel的隧道ID。与public-api选项一起使用。
  --api-port API_PORT                            API的监听端口。
  --api-key API_KEY                              API认证密钥。
  --admin-key ADMIN_KEY                          用于加载和卸载模型等管理员任务的API认证密钥。如果未设置，将与--api-key相同。
  --api-enable-ipv6                              为API启用IPv6。
  --api-disable-ipv4                             为API禁用IPv4。
  --nowebui                                      不启动Gradio UI。想要单独启动API时很有用。

Multimodal：
  --multimodal-pipeline MULTIMODAL_PIPELINE      要使用的多模态模型管线。示例：llava-7b、llava-13b。
```

</details>

## 文档

https://github.com/Touch-Night/text-generation-webui/wiki

## 下载模型

模型应该放在`text-generation-webui/models`文件夹中。它们通常可以从[Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)和其镜像站[hf-mirror](https://hf-mirror.com/models?pipeline_tag=text-generation&sort=downloads)下载。

* GGUF模型是单个文件，应直接放入`models`文件夹。例如：

```
text-generation-webui
└── models
    └── llama-2-13b-chat.Q4_K_M.gguf
```

* 其余的模型类型（例如16-位Transformers模型和EXL2模型）由多个文件组成，必须放置在子文件夹中。例如：

```
text-generation-webui
├── models
│   ├── lmsys_vicuna-33b-v1.3
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model-00001-of-00007.bin
│   │   ├── pytorch_model-00002-of-00007.bin
│   │   ├── pytorch_model-00003-of-00007.bin
│   │   ├── pytorch_model-00004-of-00007.bin
│   │   ├── pytorch_model-00005-of-00007.bin
│   │   ├── pytorch_model-00006-of-00007.bin
│   │   ├── pytorch_model-00007-of-00007.bin
│   │   ├── pytorch_model.bin.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
```

在这两种情况下，您都可以使用 UI 中的“模型”选项卡自动从HF Mirror上下载模型。也可以通过以下命令在命令行下载

```
python download-model.py organization/model
```

运行 `python download-model.py --help` 查看所有选项。

## Google Colab笔记本

https://colab.research.google.com/github/Touch-Night/text-generation-webui/blob/Chinese/Colab-TextGen-GPU.ipynb

## 致谢

2023年8月， [安德烈·霍洛维茨（Andreessen Horowitz）](https://a16z.com/)  （A16Z）提供了一项慷慨的赠款，以鼓励和支持我对该项目的独立工作。我 **极其**  感谢他们的信任和认可。