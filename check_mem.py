import psutil

pid = 1576300

def get_tree_memory(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    # 收集所有进程（主进程 + 所有子进程），去重
    procs = [parent] + parent.children(recursive=True)
    unique_procs = {p.pid: p for p in procs}.values()
    print(f"All Processes:")
    for p in unique_procs:
        print(f"{p.pid}, {p.memory_info().rss / (1024 ** 3):.2f} GB, {p.name()}")
    # 计算总内存使用
    mem = sum(p.memory_info().rss for p in unique_procs)
    return mem / (1024 ** 3)  # 转为 GB

print(f"Process {pid} Used {get_tree_memory(pid):.2f} GB")