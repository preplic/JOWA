# nohup python dataset/src/downsample.py > dataset/src/downsample.log 2>&1 &
import gzip
import os
import random
import time
import sys
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import gc  # 添加垃圾回收

from download import ENVS, TEST_ENVS, TRAIN_ENVS, capitalize_game_name
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# os.environ["TFDS_MAX_INTRA_OP_PARALLELISM"] = "32"
# os.environ["TFDS_NUM_PARALLEL_CALLS"] = "32"

# 新增：是否使用 TFDS
USE_TFDS = True
DATA_DIR = "/data0/share/datasets/TFDS/"
# 添加测试模式，用于快速验证流程
TEST_MODE = False
# 添加最大超时时间（秒）
MAX_TIMEOUT = 60

total_envs = ['DemonAttack']

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def compute_cumulative_num_of_episodes(envs, num_agents, start_epoch=1, end_epoch=50):
    if USE_TFDS:
        # 示例：envs=['DemonAttack'], num_agents=1, run_id=1
        env = envs[0]
        run_id = 1
        tqdm.write(f"加载数据集信息: {env}_run_{run_id}...")
        ds, info = tfds.load(
            f'rlu_atari_checkpoints_ordered/{env}_run_{run_id}:1.1.0',
            download=False, data_dir=DATA_DIR, with_info=True
        )
        checkpoints = [f'checkpoint_{i:02d}' for i in range(start_epoch - 1, end_epoch)]
        total_eps = 0
        total_steps = 0
        # cumu_episodes: env × agent × epoch
        cumu_episodes = np.zeros((len(envs), num_agents, len(checkpoints)), dtype=int)
        
        tqdm.write(f"计算累积episodes和steps...")
        for i, checkpoint in enumerate(tqdm(checkpoints, desc="处理checkpoints")):
            n_eps = info.splits[checkpoint].num_examples
            total_eps += n_eps
            
            # 统计总steps（仅采样前50个样本计算平均值）
            # tqdm.write(f"采样计算 {checkpoint} 的平均steps...")
            sample_size = min(50, n_eps)
            steps_samples = []
            
            # 采样计算平均steps
            ds_iter = iter(tfds.as_numpy(ds[checkpoint]))
            for _ in range(sample_size):
                try:
                    ex = next(ds_iter)
                    steps_samples.append(len(ex['steps']))
                except StopIteration:
                    break
                
            # 计算平均steps
            avg_steps = np.mean(steps_samples) if steps_samples else 0
            steps_in_checkpoint = int(avg_steps * n_eps)
            total_steps += steps_in_checkpoint
            
            # tqdm.write(f"  {checkpoint}: {n_eps}个episodes, 平均{avg_steps:.1f}步/episode, 估计总steps={steps_in_checkpoint}")
            
            cumu_episodes[0, 0, i] = total_eps
            # 每处理完一个checkpoint执行垃圾回收
            gc.collect()

        df = pd.DataFrame(
            [[total_eps, total_steps, total_steps / total_eps if total_eps > 0 else 0]],
            index=[env],
            columns=['Total episodes', 'Total steps', 'Steps per episode']
        )
        # 单 agent 时直接固定为 [run_id]
        env_choosed_indices = {env: [run_id]}
        return env_choosed_indices, cumu_episodes, df

    else:
        # 原始 gzip 分支不变
        df = pd.DataFrame(columns=['Total episodes', 'Total steps', 'Steps per episode'], index=envs)
        env_choosed_indices = {}
        cumu_episodes = np.zeros((len(envs), num_agents, end_epoch - start_epoch + 1), dtype=int)
        for index_env, env_ in enumerate(tqdm(envs)):
            # … 原逻辑 …
            pass
        return env_choosed_indices, cumu_episodes, df


def main(
    envs,
    env_choosed_indices,
    cumu_episodes,
    df,
    num_sample_episodes,
    save_dir,
    L,
    tau,
    traj_path,
    meta_path,
    seg_csv_path,
    env_index,
    start_epoch=1,
    end_epoch=50,
):
    if USE_TFDS:
        # 直接使用上面算好的 df、cumu_episodes、env_choosed_indices
        env = envs[env_index]
        run_id = env_choosed_indices[env][0]
        tqdm.write(f"[{env}] 开始加载数据集...")
        
        ds, info = tfds.load(
            f'rlu_atari_checkpoints_ordered/{env}_run_{run_id}:1.1.0',
            download=False, data_dir=DATA_DIR, with_info=True
        )
        checkpoints = [f'checkpoint_{i:02d}' for i in range(start_epoch - 1, end_epoch)]
        
        # 按 num_sample_episodes 抽样
        total_eps = df.loc[env, 'Total episodes']
        
        if TEST_MODE:
            # 测试模式：只处理10个样本
            sample_size = min(10, num_sample_episodes[env_index])
            sample_idxs = sorted(np.random.choice(total_eps, sample_size, replace=False))
            tqdm.write(f"[{env}] 测试模式: 只处理 {sample_size} 个样本")
        else:
            sample_idxs = sorted(np.random.choice(total_eps, num_sample_episodes[env_index], replace=False))
        
        # 打印采样计划
        tqdm.write(f"[{env}] 采样计划: 总共 {total_eps} 个episodes, 抽取 {len(sample_idxs)} 个")
        
        # 1) 先把 sample_idxs 按 checkpoint 分组
        checkpoint_to_samples = {}
        for j, global_idx in enumerate(sample_idxs):
            # 找到这个全局索引在哪个checkpoint中
            for k, checkpoint in enumerate(checkpoints):
                # 计算当前checkpoint的样本范围
                start_idx = 0 if k == 0 else cumu_episodes[env_index, 0, k-1]
                end_idx = cumu_episodes[env_index, 0, k]
                
                # 如果全局索引在这个范围内，那么它属于这个checkpoint
                if start_idx <= global_idx < end_idx:
                    # 计算局部索引
                    local_idx = global_idx - start_idx
                    checkpoint_to_samples.setdefault(checkpoint, []).append((j, local_idx))
                    break
        # print(f'>> checkpoint_to_samples: {checkpoint_to_samples}')
        # sys.exit(0)
        # check_total_samples = sum(len(samples) for samples in checkpoint_to_samples.values())
        # tqdm.write(f"[{env}] checkpoint_to_samples中采样数量: {check_total_samples}")
        
        # 打印每个checkpoint需要抽取的样本数量
        # for sp in checkpoints:
        #     samples = checkpoint_to_samples.get(sp, [])
        #     if samples:
        #         tqdm.write(f"[{env}] {sp}: 需要抽取 {len(samples)} 个样本")
        
        downsampled_steps = 0
        env_traj_data = {}
        seg_right, seg_left = [], []

        # 获取需要处理的checkpoints
        checkpoints_to_process = []
        for checkpoint in checkpoints:
            if checkpoint_to_samples.get(checkpoint, []):
                checkpoints_to_process.append(checkpoint)

        # 环境级进度条 - 最顶层
        with tqdm(total=len(checkpoints_to_process), desc=f"环境: {env}", position=0, leave=True, ncols=100) as env_pbar:
            # 2) 流式处理每个checkpoint
            for checkpoint_idx, checkpoint in enumerate(checkpoints):
                samples = checkpoint_to_samples.get(checkpoint, [])
                if not samples:
                    tqdm.write(f"[{env}] {checkpoint} ({checkpoint_idx+1}/{len(checkpoints)}) - 无需抽样，跳过")
                    continue
                    
                tqdm.write(f"[{env}] 处理 {checkpoint} ({checkpoint_idx+1}/{len(checkpoints)}) - 需抽取 {len(samples)} 个样本")
                
                # 建立局部索引到全局样本索引的映射
                local_to_global = {}
                for j, local_idx in samples:
                    local_to_global[local_idx] = j
                
                # 跟踪处理进度
                processed_count = 0
                total_episodes = info.splits[checkpoint].num_examples
                
                # tqdm.write(f"[{env}] 开始加载 {checkpoint} 数据... 这可能需要一些时间")
                # 设置加载开始时间，用于检测超时
                # load_start_time = time.time()
                
                # 创建迭代器而不是立即加载所有数据
                ds_iter = iter(tfds.as_numpy(ds[checkpoint]))

                # checkpoint级进度条 - 中间层
                with tqdm(total=total_episodes, desc=f"Checkpoint: {checkpoint}", position=1, leave=False, ncols=100) as ckpt_pbar:
                    # 处理每个episode
                    for i in range(total_episodes):
                        # 更新checkpoint进度条
                        ckpt_pbar.update(1)
                        
                        # 检查是否超时
                        # if time.time() - load_start_time > MAX_TIMEOUT and i < 5:
                        #     tqdm.write(f"警告: 加载时间超过 {MAX_TIMEOUT} 秒，可能存在问题。尝试继续...")
                        
                        # 每处理100个样本显示一次进度
                        # if i % 100 == 0:
                        #     tqdm.write(f"[{env}] 处理进度: {i}/{total_episodes}, 已找到 {processed_count}/{len(samples)} 个目标样本")
                        
                        # try:
                        #     ex = next(ds_iter)
                        # except StopIteration:
                        #     tqdm.write(f"警告: 迭代器提前结束，实际样本数少于预期 {total_episodes}")
                        #     break
                        
                        # 只处理需要的样本
                        if i in local_to_global:
                            j = local_to_global[i]  # 全局样本索引
                            
                            steps = next(ds_iter)['steps']
                            T = len(steps)
                            
                            # 移除episode级进度条，只在终端底部显示信息
                            if TEST_MODE:
                                tqdm.write(f"正在处理 Episode: 全局索引={j}, 原始索引={i} (长度: {T} 步)")
                            
                            downsampled_steps += T
                            
                            # 处理分段逻辑
                            starts = np.arange(0, T, tau)
                            ends = starts + L
                            seg_right.append(pd.DataFrame({
                                'Episode index': [j] * len(starts),
                                'Start index': starts, 
                                'End index': ends,
                                'Environment': [env] * len(starts)
                            }))
                            
                            ends2 = np.arange(T-1, -1, -tau)
                            starts2 = ends2 - L
                            seg_left.append(pd.DataFrame({
                                'Episode index': [j] * len(starts2),
                                'Start index': starts2, 
                                'End index': ends2,
                                'Environment': [env] * len(starts2)
                            }))
                            
                            # 提取数据
                            obs = np.stack([s['observation'] for s in steps])
                            act = np.array([s['action'] for s in steps])
                            rew = np.array([s['reward'] for s in steps])
                            done = np.array([s['is_terminal'] for s in steps], dtype=int)
                            
                            env_traj_data[j] = {
                                'observation': obs, 
                                'action': act,
                                'reward': rew, 
                                'terminal': done
                            }
                            
                            processed_count += 1
                            
                            # 每处理特定数量的样本执行垃圾回收
                            # if processed_count % 5 == 0:
                            #     # tqdm.write(f"  > 已处理 {processed_count}/{len(samples)} 个样本，执行内存回收")
                            #     gc.collect()
                
                tqdm.write(f"[{env}] 完成 {checkpoint}: 处理了 {processed_count}/{len(samples)} 个样本")
                # 每完成一个checkpoint执行垃圾回收
                gc.collect()
                
                # 更新环境进度条
                env_pbar.update(1)

        # 合并 seg DataFrame
        tqdm.write(f"[{env}] 合并分段数据并保存结果...")
        seg_right = pd.concat(seg_right) if seg_right else pd.DataFrame()
        seg_left = pd.concat(seg_left) if seg_left else pd.DataFrame()
        df.loc[env, 'Downsampled steps'] = downsampled_steps

        # 保存文件
        tqdm.write(f"[{env}] 保存数据到HDF5文件...")
        with h5py.File(os.path.join(traj_path, f"{env}.h5"), 'w') as f:
            for epi, traj in tqdm(env_traj_data.items(), desc=f"保存 {env} 轨迹数据", position=0, leave=True, ncols=100):
                g = f.create_group(str(epi))
                obs = traj['observation']
                if TEST_MODE:
                    tqdm.write(f"  > Episode {epi}: 原始形状: {obs.shape}, 类型 {obs.dtype}")
                if obs.ndim > 3 and obs.shape[-1] == 1:
                    obs = obs.squeeze(-1)
                if TEST_MODE:
                    tqdm.write(f"  > Episode {epi}: 调整后形状: {obs.shape}, 类型 {obs.dtype}")
                g.create_dataset('observations', data=traj['observation'])
                g.create_dataset('actions', data=traj['action'])
                g.create_dataset('rewards', data=traj['reward'])
                g.create_dataset('terminals', data=traj['terminal'])
        
        tqdm.write(f'保存 {env} 轨迹数据集完成。')
        
        df.loc[[env], :].to_csv(os.path.join(meta_path, f"{env}.csv"), index=False)
        seg_right.to_csv(f"{seg_csv_path}/{env}_right_padding.csv", index=False)
        seg_left.to_csv(f"{seg_csv_path}/{env}_left_padding.csv", index=False)
        return

    # 原 gzip 分支的 main 逻辑…
    pass


if __name__ == '__main__':
    # 保持原来的 env/agent 结构
    if TEST_MODE:
        save_dir = 'dataset/downsampled_test/'
    else:
        save_dir = 'dataset/downsampled/'
    num_agents = 1
    num_steps_per_env = 10e6
    num_processes = 1
    num_checkpoints = 1
    start_epoch, end_epoch = 1, 50

    L = 8  # segment length
    tau = 4  # checkpoint offset

    set_seed(0)

    meta_path = os.path.join(save_dir, "trajectory/meta")
    traj_path = os.path.join(save_dir, f"trajectory/data")
    seg_csv_path = os.path.join(save_dir, "segment/csv")
    os.makedirs(meta_path, exist_ok=True)
    os.makedirs(traj_path, exist_ok=True)
    os.makedirs(seg_csv_path, exist_ok=True)

    checkpoint_envs = np.array_split(total_envs, num_checkpoints)
    checkpoint_envs = [list(checkpoint) for checkpoint in checkpoint_envs]

    # 总进度条 - 处理所有环境
    with tqdm(total=len(checkpoint_envs), desc="总进度", position=0, leave=True, ncols=100) as total_pbar:
        for envs in checkpoint_envs:
            env_choosed_indices, cumu_episodes, df = compute_cumulative_num_of_episodes(
                envs, num_agents, start_epoch, end_epoch)
            
            # DEBUG 打印
            tqdm.write(">>> env_choosed_indices: " + str(env_choosed_indices))
            tqdm.write(">>> cumu_episodes.shape: " + str(cumu_episodes.shape))
            tqdm.write(">>> cumu_episodes: " + str(cumu_episodes))
            tqdm.write(">>> dataframe df:\n" + str(df))

            num_sample_episodes = np.ceil(
                num_steps_per_env / df.loc[:, 'Steps per episode'].values).astype(int)
            df['Downsampled episodes'] = num_sample_episodes
            
            for env_index in range(len(envs)):
                tqdm.write(f"处理环境 {envs[env_index]} ({env_index+1}/{len(envs)})")
                main(
                    envs, env_choosed_indices, cumu_episodes, 
                    df, num_sample_episodes, save_dir, L, tau,
                    traj_path, meta_path, seg_csv_path,
                    env_index, start_epoch, end_epoch
                )
            
            total_pbar.update(1)
    
    # check the num of transitions
    meta_files = []
    indices = []
    for file in os.listdir(meta_path):
        df = pd.read_csv(os.path.join(meta_path, file))
        meta_files.append(df)
        indices.append(file[:-4])
    concat_meta_file = pd.concat(meta_files)
    concat_meta_file.index = indices
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        tqdm.write(str(concat_meta_file))