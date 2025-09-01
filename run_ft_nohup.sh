#!/bin/bash
# 激活虚拟环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate SDWM

# 配置运行参数
CUDA_DEVICES="1,2"
NPROC_PER_NODE=2
COMMON_ENV="MsPacman"

# 获取当前日期
DATE=$(date +%Y-%m-%d)

# 创建输出目录
OUT_DIR="outputs_terminal"
mkdir -p $OUT_DIR/$DATE

# 生成带日期和时间的输出文件名
OUTFILE="$OUT_DIR/$DATE/$(date +%Y-%m-%d_%H-%M-%S)_out.file"

# 使用nohup后台运行，并重定向标准输出和标准错误
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup torchrun \
    --nproc_per_node=$NPROC_PER_NODE --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=39500 \
    src/fine_tune.py hydra/job_logging=disabled hydra/hydra_logging=disabled \
    common.env=$COMMON_ENV training.action.use_imagination=True training.action.planning_horizon=2 \
    > "$OUTFILE" 2>&1 &

# 保存主进程PID
echo $! > $OUT_DIR/$DATE/$(date +%Y-%m-%d_%H-%M-%S).pid