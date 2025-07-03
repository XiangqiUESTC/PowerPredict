import psutil

def get_cpu_info():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.001),
    }