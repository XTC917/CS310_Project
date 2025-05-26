# BERT文本检测项目

本项目使用BERT模型来检测文本是否由AI生成。

## 本地运行指南

### 1. 环境准备
1. 安装Python 3.9
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

### 2. 数据集准备
1. 下载Ghostbuster数据集
2. 将数据集放在项目目录下的 `ghostbuster-data` 文件夹中
3. 确保数据集路径正确（在 `main.py` 中检查 `DATA_PATH` 变量）

### 3. 运行训练
```bash
python main.py
```

## 集群运行指南

### 1. 准备文件
确保你有以下文件：
- `main.py`：主训练脚本
- `data_loader.py`：数据加载脚本
- `requirements.txt`：依赖包列表
- `submit.sh`：集群作业提交脚本

### 2. 登录集群
```bash
ssh -p 10022 cse12211617@172.18.34.25
```
密码：`SKwFeThVz(8p6q*23a`

### 3. 创建项目目录
```bash
mkdir -p ~/bert_train
cd ~/bert_train
```

### 4. 上传文件
在本地Windows PowerShell中执行：
```bash
scp -P 10022 main.py data_loader.py requirements.txt submit.sh cse12211617@172.18.34.25:~/bert_train/
```

### 5. 提交作业
在集群终端中执行：
```bash
sbatch submit.sh
```

### 6. 查看作业状态
```bash
# 查看作业是否在运行
squeue -u cse12211617

# 查看训练输出
cat train_*.log

# 实时查看训练进度
tail -f train_*.log
```

### 7. 取消作业（如果需要）
```bash
# 查看作业ID
squeue -u cse12211617

# 取消作业
scancel <作业ID>
```

## 集群资源说明

### 可用分区
1. A100分区
   - 每个用户最多1个GPU
   - 每个用户最多1个作业
   - 使用 `--partition=a100` 和 `--qos=a100`

2. RTX2080Ti分区
   - 每个用户最多2个GPU
   - 每个用户最多1个作业
   - 使用 `--partition=rtx2080ti` 和 `--qos=rtx2080ti`

3. Titan分区
   - 每个用户最多2个GPU
   - 每个用户最多1个作业
   - 使用 `--partition=titan` 和 `--qos=titan`

### 修改分区
如果需要修改使用的GPU分区，编辑 `submit.sh` 文件中的以下行：
```bash
#SBATCH --partition=a100  # 改为需要的分区
#SBATCH --qos=a100       # 改为对应的qos
```

## 注意事项
1. 确保数据集路径正确
2. 作业提交后会自动安装依赖包
3. 训练过程可能需要几个小时
4. 可以随时查看日志了解训练进度
5. 如果遇到问题，可以查看错误日志：`cat train_*.err`

## 代码说明

### main.py
- 主训练脚本
- 包含模型训练和评估的完整流程
- 使用BERT模型进行文本分类
- 支持GPU训练

### data_loader.py
- 数据加载脚本
- 处理Ghostbuster数据集
- 将文本转换为BERT可用的格式

### 训练参数
- 批次大小：32
- 学习率：2e-5
- 训练轮数：4
- 最大序列长度：512
- 优化器：AdamW
- 损失函数：CrossEntropyLoss