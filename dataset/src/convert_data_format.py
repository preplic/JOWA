import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py
import glob

def process_observations(observations, verbose=False):
    """处理观察数据的形状，确保它符合 PyTorch 卷积层的格式要求 (NCHW)"""
    if verbose:
        tqdm.write(f"原始形状: {observations.shape}")
    
    # 直接去掉最后一维
    observations = observations.squeeze(-1)
    if verbose:
        tqdm.write(f"去掉最后一维后: {observations.shape}")
    
    # # 确保通道在正确位置 (NCHW格式)
    # if observations.ndim >= 4 and observations.shape[-1] == 1:
    #     # 从[B,T,H,W,C]转换为[B,T,C,H,W]或从[B,H,W,C]转换为[B,C,H,W]
    #     if observations.ndim == 5:  # [B,T,H,W,C]
    #         observations = np.transpose(observations, (0, 1, 4, 2, 3))
    #     elif observations.ndim == 4:  # [B,H,W,C]
    #         observations = np.transpose(observations, (0, 3, 1, 2))
        
    #     if verbose:
    #         tqdm.write(f"调整通道位置后: {observations.shape}")
    
    return observations

def convert_h5_file(file_path, output_path=None, verbose=False):
    """转换单个HDF5文件的数据格式"""
    if output_path is None:
        output_path = file_path
    
    tqdm.write(f"处理文件: {file_path}")
    
    # 安全读取原始数据
    try:
        with h5py.File(file_path, 'r') as f:
            data = {}
            # 列出文件中的所有组
            trajectory_ids = list(f.keys())
            tqdm.write(f"文件中的轨迹数: {len(trajectory_ids)}")
            
            # 遍历所有轨迹，添加进度条
            for traj_id in tqdm(trajectory_ids, desc="处理轨迹", leave=False):
                # 处理组内的数据
                group_data = {}
                
                # 读取组内的四个标准数据集
                for subkey in ['actions', 'observations', 'rewards', 'terminals']:
                    if subkey in f[traj_id]:
                        group_data[subkey] = f[traj_id][subkey][()]
                        if verbose:
                            tqdm.write(f"读取组 {traj_id} 中的数据集: {subkey}, 形状: {group_data[subkey].shape}, 类型: {group_data[subkey].dtype}")
                
                # 处理观察数据
                if 'observations' in group_data:
                    group_data['observations'] = process_observations(group_data['observations'], verbose)
                
                # 存储组数据
                data[traj_id] = group_data
                if verbose:
                    tqdm.write(f"处理完组: {traj_id}")

            # 将数据保存到输出文件
            with h5py.File(output_path, 'w') as out_f:
                # 添加进度条来显示保存进度
                for traj_id, traj_data in tqdm(data.items(), desc="保存轨迹", leave=False):
                    # 创建轨迹组
                    group = out_f.create_group(traj_id)
                    # 添加组内数据
                    for subkey, subvalue in traj_data.items():
                        group.create_dataset(subkey, data=subvalue)
            

            tqdm.write(f"已保存到: {output_path}")
            
            return True

    except Exception as e:
        tqdm.write(f"读取文件 {file_path} 时出错: {e}")
        return False

def convert_numpy_file(file_path, output_path=None, verbose=False):
    """转换单个NumPy文件的数据格式"""
    if output_path is None:
        output_path = file_path
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if verbose:
        tqdm.write(f"处理文件: {file_path}")
    
    # 读取原始数据
    data = np.load(file_path, allow_pickle=True)
    
    # 如果是字典格式
    if isinstance(data, np.ndarray) and data.dtype.kind == 'O':
        data_dict = data.item()
        if 'observations' in data_dict:
            data_dict['observations'] = process_observations(data_dict['observations'], verbose)
        np.save(output_path, data_dict)
    # 如果直接是观察数据
    elif isinstance(data, np.ndarray):
        processed_data = process_observations(data, verbose)
        np.save(output_path, processed_data)
    
    if verbose:
        tqdm.write(f"已保存到: {output_path}")
    
    return True

def convert_dataset(data_dir, output_dir=None, file_ext='.h5', verbose=True):
    """转换整个数据集的格式"""
    if output_dir is None:
        output_dir = data_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有文件
    files = glob.glob(os.path.join(data_dir, f"**/*{file_ext}"), recursive=True)
    
    tqdm.write(f"找到 {len(files)} 个{file_ext}文件")
    
    for file_path in tqdm(files, desc="转换数据"):
        rel_path = os.path.relpath(file_path, data_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        if file_ext.lower() == '.h5':
            convert_h5_file(file_path, output_path, verbose=verbose)
        elif file_ext.lower() == '.npy':
            convert_numpy_file(file_path, output_path, verbose=verbose)
    
    tqdm.write(f"所有文件已转换并保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='转换数据集中的观察数据格式')
    parser.add_argument('--data-dir', type=str, default='dataset/downsampled/trajectory/data/',
                        help='数据目录路径')
    parser.add_argument('--output-dir', type=str, default='dataset/downsampled/trajectory/modified/',
                        help='输出目录路径')
    parser.add_argument('--file-ext', type=str, default='.h5',
                        help='文件扩展名 (.h5 或 .npy)')
    parser.add_argument('--test-file', type=str, default='DemonAttack.h5',
                        help='测试单个文件而不是整个目录')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    if args.test_file:
        file_path = os.path.join(args.data_dir, args.test_file)
        save_path = os.path.join(args.output_dir, args.test_file)
        tqdm.write(f"测试处理单个文件: {args.test_file}")
        if args.test_file.endswith('.h5'):
            convert_h5_file(file_path, save_path, verbose=False)
        elif args.test_file.endswith('.npy'):
            convert_numpy_file(file_path, save_path, verbose=False)
    else:
        convert_dataset(args.data_dir, args.output_dir, args.file_ext, args.verbose)

if __name__ == "__main__":
    main()