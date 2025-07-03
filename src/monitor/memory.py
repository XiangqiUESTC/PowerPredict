def get_memory_info(process):
    mem_info = process.memory_info()
    physical_mb = mem_info.rss / (1024 ** 2)  # 优先使用 rss
    return {
        "memory": physical_mb,
    }
