import psutil

def get_disk_info():
    disk_info = psutil.disk_io_counters(perdisk=False)
    return {
        "read_bytes": disk_info.read_bytes,
        "write_bytes": disk_info.write_bytes,
    }