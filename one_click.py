import argparse
import glob
import hashlib
from importlib import util as implib_util
import json
import os
import platform
import re
import signal
import site
import subprocess
import sys

# Remove the '# ' from the following lines as needed for your AMD GPU on Linux
# os.environ["ROCM_PATH"] = '/opt/rocm'
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = '10.3.0'
# os.environ["HCC_AMDGPU_TARGET"] = 'gfx1030'


# Define the required PyTorch version
TORCH_VERSION = "2.4.1"
TORCHVISION_VERSION = "0.19.1"
TORCHAUDIO_VERSION = "2.4.1"

# Environment
script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")

# Command-line flags
cmd_flags_path = os.path.join(script_dir, "CMD_FLAGS.txt")
if os.path.exists(cmd_flags_path):
    with open(cmd_flags_path, 'r') as f:
        CMD_FLAGS = ' '.join(line.strip().rstrip('\\').strip() for line in f if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'))
else:
    CMD_FLAGS = ''

flags = f"{' '.join([flag for flag in sys.argv[1:] if flag != '--update-wizard'])} {CMD_FLAGS}"


def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def is_linux():
    return sys.platform.startswith("linux")


def is_windows():
    return sys.platform.startswith("win")


def is_macos():
    return sys.platform.startswith("darwin")


def is_x86_64():
    return platform.machine() == "x86_64"


def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if implib_util.find_spec("torch") is None or implib_util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def cpu_has_avx2():
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if 'avx2' in info['flags']:
            return True
        else:
            return False
    except:
        return True


def cpu_has_amx():
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        if 'amx' in info['flags']:
            return True
        else:
            return False
    except:
        return True


def torch_version():
    site_packages_path = None
    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir and conda_env_path in sitedir:
            site_packages_path = sitedir
            break

    if site_packages_path:
        torch_version_file = open(os.path.join(site_packages_path, 'torch', 'version.py')).read().splitlines()
        torver = [line for line in torch_version_file if line.startswith('__version__')][0].split('__version__ = ')[1].strip("'")
    else:
        from torch import __version__ as torver

    return torver


def update_pytorch():
    print_big_message("正在检查PyTorch更新.")
    torver = torch_version()
    base_cmd = f"python -m pip install --upgrade torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"

    if "+cu118" in torver:
        install_cmd = f"{base_cmd} --index-url https://download.pytorch.org/whl/cu118"
    elif "+cu" in torver:
        install_cmd = f"{base_cmd} --index-url https://download.pytorch.org/whl/cu121"
    elif "+rocm" in torver:
        install_cmd = f"{base_cmd} --index-url https://download.pytorch.org/whl/rocm6.1"
    elif "+cpu" in torver:
        install_cmd = f"{base_cmd} --index-url https://download.pytorch.org/whl/cpu"
    elif "+cxx11" in torver:
        intel_extension = "intel-extension-for-pytorch==2.1.10+xpu" if is_linux() else "intel-extension-for-pytorch==2.1.10"
        install_cmd = f"{base_cmd} {intel_extension} --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    else:
        install_cmd = base_cmd

    run_cmd(install_cmd, assert_success=True, environment=True)


def is_installed():
    site_packages_path = None
    for sitedir in site.getsitepackages():
        if "site-packages" in sitedir and conda_env_path in sitedir:
            site_packages_path = sitedir
            break

    if site_packages_path:
        return os.path.isfile(os.path.join(site_packages_path, 'torch', '__init__.py'))
    else:
        return os.path.isdir(conda_env_path)


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd("conda", environment=True, capture_output=True).returncode == 0
    if not conda_exist:
        print("Conda未安装。正在退出...")
        sys.exit(1)

    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("请为此项目创建一个环境并激活它。正在退出...")
        sys.exit(1)


def get_current_commit():
    result = run_cmd("git rev-parse HEAD", capture_output=True, environment=True)
    return result.stdout.decode('utf-8').strip()


def clear_cache():
    run_cmd("conda clean -a -y", environment=True)
    run_cmd("python -m pip cache purge", environment=True)


def print_big_message(message):
    message = message.strip()
    lines = message.split('\n')
    print("\n\n*******************************************************************")
    for line in lines:
        print("*", line)

    print("*******************************************************************\n\n")


def calculate_file_hash(file_path):
    p = os.path.join(script_dir, file_path)
    if os.path.isfile(p):
        with open(p, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    else:
        return ''


def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
    # Use the conda environment
    if environment:
        if is_windows():
            conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
            cmd = f'"{conda_bat_path}" activate "{conda_env_path}" >nul && {cmd}'
        else:
            conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
            cmd = f'. "{conda_sh_path}" && conda activate "{conda_env_path}" && {cmd}'

    # Set executable to None for Windows, bash for everything else
    executable = None if is_windows() else 'bash'

    # Run shell commands
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env, executable=executable)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print(f"指令 '{cmd}' 运行失败，退出状态码为 '{str(result.returncode)}'。\n\n现在正在退出。\n请尝试再次运行启动/更新脚本。")
        sys.exit(1)

    return result


def generate_alphabetic_sequence(index):
    result = ''
    while index >= 0:
        index, remainder = divmod(index, 26)
        result = chr(ord('A') + remainder) + result
        index -= 1

    return result


def get_user_choice(question, options_dict):
    print()
    print(question)
    print()

    for key, value in options_dict.items():
        print(f"{key}) {value}")

    print()

    choice = input("输入> ").upper()
    while choice not in options_dict.keys():
        print("非法选择，请重试。")
        choice = input("输入> ").upper()

    return choice


def install_webui():
    # Ask the user for the GPU vendor
    if "GPU_CHOICE" in os.environ:
        choice = os.environ["GPU_CHOICE"].upper()
        print_big_message(f"已基于GPU_CHOICE环境变量选择\"{choice}\"。")

        # Warn about changed meanings and handle old NVIDIA choice
        if choice == "B":
            print_big_message("警告：GPU_CHOICE='B'在新版本中意味着'NVIDIA (CUDA 11.8)'。")
        elif choice == "C":
            print_big_message("警告：GPU_CHOICE='C'在新版本中意味着'AMD'。")
        elif choice == "D":
            print_big_message("警告：GPU_CHOICE='D'在新版本中意味着'Apple M Series'。")
        elif choice == "E":
            print_big_message("警告：GPU_CHOICE='E'在新版本中意味着'Intel Arc'。")
        elif choice == "A" and "USE_CUDA118" in os.environ:
            choice = "B" if os.environ.get("USE_CUDA118", "").lower() in ("yes", "y", "true", "1", "t", "on") else "A"
    else:
        choice = get_user_choice(
            "您的GPU是什么型号的?",
            {
                'A': 'NVIDIA - CUDA 12.1 (推荐)',
                'B': 'NVIDIA - CUDA 11.8 (老款GPU)',
                'C': 'AMD - 仅限Linux/macOS, 需要ROCm 6.1',
                'D': 'Apple M 系列',
                'E': 'Intel Arc (beta)',
                'F': '华为昇腾NPU',
                'N': 'CPU 模式'
            },
        )

    # Convert choices to GPU names for compatibility
    gpu_choice_to_name = {
        "A": "NVIDIA",
        "B": "NVIDIA",
        "C": "AMD",
        "D": "APPLE",
        "E": "INTEL",
        'F': "HUAWEI",
        "N": "NONE"
    }

    selected_gpu = gpu_choice_to_name[choice]
    use_cuda118 = (choice == "B")  # CUDA version is now determined by menu choice

    # Write a flag to CMD_FLAGS.txt for CPU mode
    if selected_gpu == "NONE":
        with open(cmd_flags_path, 'r+') as cmd_flags_file:
            if "--cpu" not in cmd_flags_file.read():
                print_big_message("正在添加--cpu命令行参数到CMD_FLAGS.txt。")
                cmd_flags_file.write("\n--cpu\n")

    # Handle CUDA version display
    elif any((is_windows(), is_linux())) and selected_gpu == "NVIDIA":
        if use_cuda118:
            print("CUDA: 11.8")
        else:
            print("CUDA: 12.1")

    # No PyTorch for AMD on Windows (?)
    elif is_windows() and selected_gpu == "AMD":
        print("PyTorch在Windows上的安装尚未实现。正在退出...")
        sys.exit(1)

    # Find the Pytorch installation command
    install_pytorch = f"python -m pip install torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION} "

    if selected_gpu == "NVIDIA":
        if use_cuda118 == 'Y':
            install_pytorch += "--index-url https://download.pytorch.org/whl/cu118"
        else:
            install_pytorch += "--index-url https://download.pytorch.org/whl/cu121"
    elif selected_gpu == "AMD":
        install_pytorch += "--index-url https://download.pytorch.org/whl/rocm6.1"
    elif selected_gpu in ["APPLE", "NONE"]:
        install_pytorch += "--index-url https://download.pytorch.org/whl/cpu"
    elif selected_gpu == "INTEL":
        if is_linux():
            install_pytorch = "python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
        else:
            install_pytorch = "python -m pip install torch==2.1.0a0 torchvision==0.16.0a0 torchaudio==2.1.0a0 intel-extension-for-pytorch==2.1.10 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    elif selected_gpu == "HUAWEI":
        if is_linux():
            install_pytorch = "python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && python -m pip install torch_npu==2.1.0"
        else:
            print("华为昇腾NPU仅支持Linux。正在退出...")
            sys.exit(1)

    # Install Git and then Pytorch
    print_big_message("正在安装PyTorch。")
    run_cmd(f"conda install -y -k ninja git && {install_pytorch} && python -m pip install py-cpuinfo==9.0.0", assert_success=True, environment=True)

    if selected_gpu == "INTEL":
        # Install oneAPI dependencies via conda
        print_big_message("正在安装Intel oneAPI运行时库。")
        run_cmd("conda install -y -c https://software.repos.intel.com/python/conda/ -c conda-forge dpcpp-cpp-rt=2024.0 mkl-dpcpp=2024.0")
        # Install libuv required by Intel-patched torch
        run_cmd("conda install -y libuv")

    # Install the webui requirements
    update_requirements(initial_installation=True, pull=False)


def get_extensions_names():
    return [foldername for foldername in os.listdir('extensions') if os.path.isfile(os.path.join('extensions', foldername, 'requirements.txt'))]


def install_extensions_requirements():
    print_big_message("正在安装扩展依赖。\n一些扩展可能在Windows上安装失败。\n如果看到错误消息，请不要担心，因为它们不会影响主程序。")
    extensions = get_extensions_names()
    for i, extension in enumerate(extensions):
        print(f"\n\n--- [{i + 1}/{len(extensions)}]: {extension}\n\n")
        extension_req_path = os.path.join("extensions", extension, "requirements.txt")
        run_cmd(f"python -m pip install -r {extension_req_path} --upgrade", assert_success=False, environment=True)


def update_requirements(initial_installation=False, pull=True):
    # Create .git directory if missing
    if not os.path.exists(os.path.join(script_dir, ".git")):
        run_cmd(
            "git init -b main && git remote add origin https://github.com/Touch-Night/text-generation-webui && "
            "git fetch && git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/ChineseHuggingface && "
            "git reset --hard origin/ChineseHuggingface && git branch --set-upstream-to=origin/ChineseHuggingface && git submodule update --init --recursive",
            environment=True,
            assert_success=True
        )

    torver = torch_version()
    if "+rocm" in torver:
        requirements_file = "requirements_amd" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"
    elif "+cpu" in torver or "+cxx11" in torver:
        requirements_file = "requirements_cpu_only" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"
    elif is_macos():
        requirements_file = "requirements_apple_" + ("intel" if is_x86_64() else "silicon") + ".txt"
    else:
        requirements_file = "requirements" + ("_noavx2" if not cpu_has_avx2() else "") + ".txt"

    # Load state from JSON file
    state_file = '.installer_state.json'
    current_commit = get_current_commit()
    wheels_changed = False
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            last_state = json.load(f)

        if 'wheels_changed' in last_state or last_state.get('last_installed_commit') != current_commit:
            wheels_changed = True
    else:
        wheels_changed = True

    if pull:
        # Read .whl lines before pulling
        before_pull_whl_lines = []
        if os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                before_pull_whl_lines = [line for line in f if '.whl' in line]

        print_big_message("正在使用\"git pull\"更新本地副本。")

        # Hash files before pulling
        files_to_check = [
            'start_linux.sh', 'start_macos.sh', 'start_windows.bat', 'start_wsl.bat',
            'update_wizard_linux.sh', 'update_wizard_macos.sh', 'update_wizard_windows.bat', 'update_wizard_wsl.bat',
            'one_click.py'
        ]
        before_hashes = {file: calculate_file_hash(file) for file in files_to_check}

        # Perform the git pull
        run_cmd("git pull --autostash", assert_success=True, environment=True)

        # Check hashes after pulling
        after_hashes = {file: calculate_file_hash(file) for file in files_to_check}
        if os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                after_pull_whl_lines = [line for line in f if '.whl' in line]

        wheels_changed = wheels_changed or (before_pull_whl_lines != after_pull_whl_lines)

        # Check for changes to installer files
        for file in files_to_check:
            if before_hashes[file] != after_hashes[file]:
                print_big_message(f"文件'{file}'在'git pull'期间被更新。请再次运行脚本。")

                # Save state before exiting
                current_state = {}
                if wheels_changed:
                    current_state['wheels_changed'] = True

                with open(state_file, 'w') as f:
                    json.dump(current_state, f)

                sys.exit(1)

    # Save current state
    current_state = {'last_installed_commit': current_commit}
    with open(state_file, 'w') as f:
        json.dump(current_state, f)

    if os.environ.get("INSTALL_EXTENSIONS", "").lower() in ("yes", "y", "true", "1", "t", "on"):
        install_extensions_requirements()

    # Update PyTorch
    if not initial_installation:
        update_pytorch()

    print_big_message(f"正在根据 {requirements_file} 安装webui依赖。")
    print(f"TORCH: {torver}\n")

    # Prepare the requirements file
    textgen_requirements = open(requirements_file).read().splitlines()

    if not initial_installation and not wheels_changed:
        textgen_requirements = [line for line in textgen_requirements if '.whl' not in line]

    if "+cu118" in torver:
        textgen_requirements = [
            req.replace('+cu121', '+cu118').replace('+cu122', '+cu118')
            for req in textgen_requirements
            if "autoawq" not in req.lower()
        ]

    if is_windows() and "+cu118" in torver:  # No flash-attention on Windows for CUDA 11
        textgen_requirements = [req for req in textgen_requirements if 'oobabooga/flash-attention' not in req]

    with open('temp_requirements.txt', 'w') as file:
        file.write('\n'.join(textgen_requirements))

    # Workaround for git+ packages not updating properly.
    git_requirements = [req for req in textgen_requirements if req.startswith("git+")]
    for req in git_requirements:
        url = req.replace("git+", "")
        package_name = url.split("/")[-1].split("@")[0].rstrip(".git")
        run_cmd(f"python -m pip uninstall -y {package_name}", environment=True)
        print(f"已卸载 {package_name}")

    # Install/update the project requirements
    run_cmd("python -m pip install -r temp_requirements.txt --upgrade", assert_success=True, environment=True)

    # Clean up
    os.remove('temp_requirements.txt')
    clear_cache()


def launch_webui():
    run_cmd(f"python server.py {flags}", environment=True)


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--update-wizard', action='store_true', help='启动一个带有更新选项的菜单。')
    args, _ = parser.parse_known_args()

    if args.update_wizard:
        while True:
            choice = get_user_choice(
                "您想做什么？",
                {
                    'A': '更新web UI',
                    'B': '安装/更新扩展依赖',
                    'C': '使用\"git reset --hard\"恢复本地更改',
                    'N': '没事（退出）'
                },
            )

            if choice == 'A':
                update_requirements()
            elif choice == 'B':
                choices = {'A': '所有扩展'}
                for i, name in enumerate(get_extensions_names()):
                    key = generate_alphabetic_sequence(i + 1)
                    choices[key] = name

                choice = get_user_choice("哪个扩展？", choices)

                if choice == 'A':
                    install_extensions_requirements()
                else:
                    extension_req_path = os.path.join("extensions", choices[choice], "requirements.txt")
                    run_cmd(f"python -m pip install -r {extension_req_path} --upgrade", assert_success=False, environment=True)

                update_requirements(pull=False)
            elif choice == 'C':
                run_cmd("git reset --hard", assert_success=True, environment=True)
            elif choice == 'N':
                sys.exit()
    else:
        if not is_installed():
            install_webui()
            os.chdir(script_dir)

        if os.environ.get("LAUNCH_AFTER_INSTALL", "").lower() in ("no", "n", "false", "0", "f", "off"):
            print_big_message("由于变量LAUNCH_AFTER_INSTALL不为TRUE，现在将退出。")
            sys.exit()

        # Check if a model has been downloaded yet
        if '--model-dir' in flags:
            # Splits on ' ' or '=' while maintaining spaces within quotes
            flags_list = re.split(' +(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)|=', flags)
            model_dir = [flags_list[(flags_list.index(flag) + 1)] for flag in flags_list if flag == '--model-dir'][0].strip('"\'')
        else:
            model_dir = 'models'

        if len([item for item in glob.glob(f'{model_dir}/*') if not item.endswith(('.txt', '.yaml'))]) == 0:
            print_big_message("您还没有下载任何模型。\n等Web UI启动之后，可以转到\"模型\"标签页去下载一个。")

        # Workaround for llama-cpp-python loading paths in CUDA env vars even if they do not exist
        conda_path_bin = os.path.join(conda_env_path, "bin")
        if not os.path.exists(conda_path_bin):
            os.mkdir(conda_path_bin)

        # Launch the webui
        launch_webui()
