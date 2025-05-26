#!/bin/bash
#SBATCH --job-name=bert_train
#SBATCH --partition=rtx2080ti  # 使用RTX2080Ti分区
#SBATCH --qos=rtx2080ti
#SBATCH --gres=gpu:1  # 请求1个GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err

# 加载必要的模块
module load cuda/11.7
module load python/3.9

# 创建并激活Python环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行训练脚本
python main.py 