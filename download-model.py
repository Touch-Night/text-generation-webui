'''
Downloads models from Hugging Face to models/username_modelname.

Example:
python download-model.py facebook/opt-1.3b

'''

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from time import sleep

import requests
import tqdm
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, RequestException, Timeout
from tqdm.contrib.concurrent import thread_map

base = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"


class ModelDownloader:
    def __init__(self, max_retries=5):
        self.max_retries = max_retries

    def get_session(self):
        session = requests.Session()
        if self.max_retries:
            session.mount('https://cdn-lfs.hf-mirror.com', HTTPAdapter(max_retries=self.max_retries))
            session.mount('https://hf-mirror.com', HTTPAdapter(max_retries=self.max_retries))

        if os.getenv('HF_USER') is not None and os.getenv('HF_PASS') is not None:
            session.auth = (os.getenv('HF_USER'), os.getenv('HF_PASS'))

        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            token = os.getenv("HF_TOKEN")

        if token is not None:
            session.headers = {'authorization': f'Bearer {token}'}

        return session

    def sanitize_model_and_branch_names(self, model, branch):
        if model[-1] == '/':
            model = model[:-1]

        if model.startswith(base + '/'):
            model = model[len(base) + 1:]

        model_parts = model.split(":")
        model = model_parts[0] if len(model_parts) > 0 else model
        branch = model_parts[1] if len(model_parts) > 1 else branch

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "分支名称无效。只允许使用字母、数字、点、下划线和破折号。")

        return model, branch

    def get_download_links_from_huggingface(self, model, branch, text_only=False, specific_file=None):
        session = self.get_session()
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
        classifications = []
        has_pytorch = False
        has_pt = False
        has_gguf = False
        has_safetensors = False
        is_lora = False
        while True:
            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            r = session.get(url, timeout=10)
            r.raise_for_status()
            content = r.content

            dict = json.loads(content)
            if len(dict) == 0:
                break

            for i in range(len(dict)):
                fname = dict[i]['path']
                if specific_file not in [None, ''] and fname != specific_file:
                    continue

                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                is_safetensors = re.match(r".*\.safetensors", fname)
                is_pt = re.match(r".*\.pt", fname)
                is_gguf = re.match(r'.*\.gguf', fname)
                is_tiktoken = re.match(r".*\.tiktoken", fname)
                is_tokenizer = re.match(r"(tokenizer|ice|spiece).*\.model", fname) or is_tiktoken
                is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                if any((is_pytorch, is_safetensors, is_pt, is_gguf, is_tokenizer, is_text)):
                    if 'lfs' in dict[i]:
                        sha256.append([fname, dict[i]['lfs']['oid']])

                    if is_text:
                        links.append(f"{base}/{model}/resolve/{branch}/{fname}")
                        classifications.append('text')
                        continue

                    if not text_only:
                        links.append(f"{base}/{model}/resolve/{branch}/{fname}")
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append('safetensors')
                        elif is_pytorch:
                            has_pytorch = True
                            classifications.append('pytorch')
                        elif is_pt:
                            has_pt = True
                            classifications.append('pt')
                        elif is_gguf:
                            has_gguf = True
                            classifications.append('gguf')

            cursor = base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode()) + b':50'
            cursor = base64.b64encode(cursor)
            cursor = cursor.replace(b'=', b'%3D')

        # If both pytorch and safetensors are available, download safetensors only
        if (has_pytorch or has_pt) and has_safetensors:
            for i in range(len(classifications) - 1, -1, -1):
                if classifications[i] in ['pytorch', 'pt']:
                    links.pop(i)

        # For GGUF, try to download only the Q4_K_M if no specific file is specified.
        # If not present, exclude all GGUFs, as that's likely a repository with both
        # GGUF and fp16 files.
        if has_gguf and specific_file is None:
            has_q4km = False
            for i in range(len(classifications) - 1, -1, -1):
                if 'q4_k_m' in links[i].lower():
                    has_q4km = True

            if has_q4km:
                for i in range(len(classifications) - 1, -1, -1):
                    if 'q4_k_m' not in links[i].lower():
                        links.pop(i)
            else:
                for i in range(len(classifications) - 1, -1, -1):
                    if links[i].lower().endswith('.gguf'):
                        links.pop(i)

        is_llamacpp = has_gguf and specific_file is not None
        return links, sha256, is_lora, is_llamacpp

    def get_output_folder(self, model, branch, is_lora, is_llamacpp=False):
        base_folder = 'models' if not is_lora else 'loras'

        # If the model is of type GGUF, save directly in the base_folder
        if is_llamacpp:
            return Path(base_folder)

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'

        output_folder = Path(base_folder) / output_folder
        return output_folder

    def get_single_file(self, url, output_folder, start_from_scratch=False):
        filename = Path(url.rsplit('/', 1)[1])
        output_path = output_folder / filename

        max_retries = 7
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            session = self.get_session()
            headers = {}
            mode = 'wb'

            if output_path.exists() and not start_from_scratch:
                # Resume download
                r = session.get(url, stream=True, timeout=20)
                total_size = int(r.headers.get('content-length', 0))
                if output_path.stat().st_size >= total_size:
                    return

                headers = {'Range': f'bytes={output_path.stat().st_size}-'}
                mode = 'ab'

            try:
                with session.get(url, stream=True, headers=headers, timeout=30) as r:
                    r.raise_for_status()  # If status is not 2xx, raise an error
                    total_size = int(r.headers.get('content-length', 0))
                    block_size = 1024 * 1024  # 1MB

                    tqdm_kwargs = {
                        'total': total_size,
                        'unit': 'iB',
                        'unit_scale': True,
                        'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}'
                    }

                    if 'COLAB_GPU' in os.environ:
                        tqdm_kwargs.update({
                            'position': 0,
                            'leave': True
                        })

                    with open(output_path, mode) as f:
                        with tqdm.tqdm(**tqdm_kwargs) as t:
                            count = 0
                            for data in r.iter_content(block_size):
                                f.write(data)
                                t.update(len(data))
                                if total_size != 0 and self.progress_bar is not None:
                                    count += len(data)
                                    self.progress_bar(float(count) / float(total_size), f"{filename}")

                    break  # Exit loop if successful
            except (RequestException, ConnectionError, Timeout) as e:
                print(f"下载 {filename} 时出现错误：{e}。")
                print(f"尝试次数：{attempt}/{max_retries}。", end=' ')
                if attempt < max_retries:
                    print(f"将在 {2 ** attempt} 秒后重试。")
                    sleep(2 ** attempt)
                else:
                    print("最大尝试次数后仍下载失败。")

    def start_download_threads(self, file_list, output_folder, start_from_scratch=False, threads=4):
        thread_map(lambda url: self.get_single_file(url, output_folder, start_from_scratch=start_from_scratch), file_list, max_workers=threads, disable=True)

    def download_model_files(self, model, branch, links, sha256, output_folder, progress_bar=None, start_from_scratch=False, threads=4, specific_file=None, is_llamacpp=False):
        self.progress_bar = progress_bar

        # Create the folder and writing the metadata
        output_folder.mkdir(parents=True, exist_ok=True)

        if not is_llamacpp:
            metadata = f'url: https://hf-mirror.com/{model}\n' \
                       f'branch: {branch}\n' \
                       f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'

            sha256_str = '\n'.join([f'    {item[1]} {item[0]}' for item in sha256])
            if sha256_str:
                metadata += f'sha256sum:\n{sha256_str}'

            metadata += '\n'
            (output_folder / 'huggingface-metadata.txt').write_text(metadata)

        if specific_file:
            print(f"正在下载 {specific_file} 至 {output_folder}")
        else:
            print(f"正在下载至 {output_folder}")

        self.start_download_threads(links, output_folder, start_from_scratch=start_from_scratch, threads=threads)

    def check_model_files(self, model, branch, links, sha256, output_folder):
        # Validate the checksums
        validated = True
        for i in range(len(sha256)):
            fpath = (output_folder / sha256[i][0])

            if not fpath.exists():
                print(f"以下文件丢失: {fpath}")
                validated = False
                continue

            with open(output_folder / sha256[i][0], "rb") as f:
                bytes = f.read()
                file_hash = hashlib.sha256(bytes).hexdigest()
                if file_hash != sha256[i][1]:
                    print(f'sha256校验失败: {sha256[i][0]}  {sha256[i][1]}')
                    validated = False
                else:
                    print(f'sha256已校验: {sha256[i][0]}  {sha256[i][1]}')

        if validated:
            print('[+] 已校验所有模型文件的sha256校验和！')
        else:
            print('[-] sha256校验和无效。请使用--clean标志重新运行download-model.py。')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', type=str, default=None, nargs='?')
    parser.add_argument('--branch', type=str, default='main', help='下载的Git分支的名称。')
    parser.add_argument('--threads', type=int, default=4, help='同时下载的文件数。')
    parser.add_argument('--text-only', action='store_true', help='只下载文本文件(txt/json)。')
    parser.add_argument('--specific-file', type=str, default=None, help='要下载的特定文件的名称（如果未提供，则下载所有文件）。')
    parser.add_argument('--output', type=str, default=None, help='保存模型的文件夹。')
    parser.add_argument('--clean', action='store_true', help='不恢复以前的下载。')
    parser.add_argument('--check', action='store_true', help='校验模型文件的sha256校验和。')
    parser.add_argument('--max-retries', type=int, default=5, help='在下载时出现错误时的最大重试次数。')
    args = parser.parse_args()

    branch = args.branch
    model = args.MODEL
    specific_file = args.specific_file

    if model is None:
        print("错误：请指定要下载的模型（例如'python download-model.py facebook/opt-1.3b'）。")
        sys.exit()

    downloader = ModelDownloader(max_retries=args.max_retries)
    # Clean up the model/branch names
    try:
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)
    except ValueError as err_branch:
        print(f"错误：{err_branch}")
        sys.exit()

    # Get the download links from Hugging Face
    links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=args.text_only, specific_file=specific_file)

    # Get the output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp)

    if args.check:
        # Check previously downloaded files
        downloader.check_model_files(model, branch, links, sha256, output_folder)
    else:
        # Download files
        downloader.download_model_files(model, branch, links, sha256, output_folder, specific_file=specific_file, threads=args.threads, is_llamacpp=is_llamacpp)