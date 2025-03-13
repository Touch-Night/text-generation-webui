#!/usr/bin/env bash

export LANG=zh_CN.UTF-8

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "$(pwd)" =~ " " ]]; then echo 此脚本依赖Miniconda，而它无法在包含空格的路径下静默安装。 && exit; fi

# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null

# config
CONDA_ROOT_PREFIX="$(pwd)/installer_files/conda"
INSTALL_ENV_DIR="$(pwd)/installer_files/env"

# Ascend environment variables
pythonpath=$(echo $PYTHONPATH | tr ":" "\n")
ASCEND_ENV=""
for pth in $pythonpath
do 
    if [[ $pth == *"Ascend"* ]];
    then ASCEND_ENV=$pth:$ASCEND_ENV
    fi 
done

# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="$INSTALL_ENV_DIR"
export CUDA_HOME="$CUDA_PATH"
export PYTHONPATH=$ASCEND_ENV

# activate installer env
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh" # otherwise conda complains about 'shell not initialized' (needed when running in a script)
conda activate "$INSTALL_ENV_DIR"

# update installer env
python one_click.py --update-wizard && echo -e "\n祝您愉快！"
