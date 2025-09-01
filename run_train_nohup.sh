#!/bin/bash

# 获取当前日期
DATE=$(date +%Y-%m-%d)
# 创建输出目录
mkdir -p outputs_terminal/$DATE

# 生成带日期和时间的输出文件名
OUTFILE="outputs_terminal/$DATE/$(date +%Y-%m-%d_%H-%M-%S)_out.file"

# 使用nohup后台运行，并重定向标准输出和标准错误
CUDA_VISIBLE_DEVICES=7 nohup torchrun \
    --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=39501 \
    src/train.py hydra/job_logging=disabled hydra/hydra_logging=disabled \
    > "$OUTFILE" 2>&1 &

# 保存主进程PID
echo $! > outputs_terminal/$DATE/$(date +%Y-%m-%d_%H-%M-%S).pid