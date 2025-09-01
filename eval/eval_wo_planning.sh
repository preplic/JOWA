#!/bin/bash

# checkpoints of JOWA
game=MsPacman
ckpt_path="checkpoints/""$game"
model_name="JOWA_150M"  # in [JOWA_150M, JOWA_70M, JOWA_40M]

num_rollouts=16
buffer_size=(1 2 3 4 5 6 7 8)
# 指定要使用的GPU ID列表
gpu_list=(0)

# no planning
use_planning=False
beam_width=1
horizon=0


echo model: "$ckpt_path"/"$model_name".pt
echo game: "$game"
echo buffer_size: "${buffer_size[*]}"
echo num_rollouts: "$num_rollouts" 
echo "Using GPUs: ${gpu_list[*]}"


# 为本次运行创建一个唯一的、带时间戳的日志目录
log_dir="$ckpt_path/logs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"
mkdir -p "$log_dir"

k=0
max_steps=108000

for i in "${buffer_size[@]}"; do
    # 从日志文件名中移除日期和时间，因为它们现在是目录名的一部分
    log_name="$log_dir"/"$model_name"_play_"$game"_buffer_size_"$i"_plan_"$use_planning"_bw_"$beam_width"_h_"$horizon".log
    > $log_name  # clear log

    # 从你的GPU列表中选择一个
    num_gpus_in_list=${#gpu_list[@]}
    gpu_index=$((k % num_gpus_in_list))
    device="cuda:${gpu_list[$gpu_index]}" 
    k=$((k + 1))

    python src/play.py \
    transformer=$model_name \
    critic_head=$model_name \
    initialization.path_to_checkpoint=$ckpt_path \
    initialization.jowa_model_name=$model_name \
    common.num_given_steps=$i \
    common.game_name="$game" \
    common.max_steps="$max_steps" \
    common.use_planning="$use_planning" \
    common.beam_width="$beam_width" \
    common.horizon="$horizon" \
    common.device="$device" \
    common.num_envs="$num_rollouts" \
    common.num_eval_episodes=1 \
    hydra/job_logging=disabled hydra/hydra_logging=disabled \
    >> $log_name 2>&1 &
done
echo "All evaluations started. Check logs in $log_dir for details."