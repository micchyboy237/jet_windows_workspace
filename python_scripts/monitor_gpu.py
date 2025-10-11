# monitor_gpu.py

from typing import List, TypedDict
import subprocess
import csv
from io import StringIO
import psutil

class GPUInfo(TypedDict):
    index: str
    name: str
    utilization_gpu: str
    memory_used: str
    memory_total: str

class ProcessInfo(TypedDict):
    pid: str
    process_name: str
    used_gpu_memory: str
    full_cmdline: str
    username: str

def get_gpu_info() -> List[GPUInfo]:
    """Fetch GPU stats: index, name, utilization, memory used/total."""
    cmd = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader"
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    reader = csv.reader(StringIO(result.stdout))
    return [
        {
            "index": row[0],
            "name": row[1],
            "utilization_gpu": row[2],
            "memory_used": row[3],
            "memory_total": row[4]
        }
        for row in reader
    ]

def get_process_info() -> List[ProcessInfo]:
    """Fetch process info (compute only): PID, name, VRAM, cmdline, username."""
    cmd_compute = "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader"
    processes = []
    
    try:
        result = subprocess.run(cmd_compute, capture_output=True, text=True, shell=True)
        # Skip if command fails, output is empty, or contains an error
        if result.returncode != 0 or not result.stdout.strip() or "ERROR" in result.stdout:
            return processes
        reader = csv.reader(StringIO(result.stdout))
        for row in reader:
            # Validate PID is numeric
            if not row[0].isdigit():
                continue
            pid = row[0]
            try:
                proc = psutil.Process(int(pid))
                processes.append({
                    "pid": pid,
                    "process_name": row[1],
                    "used_gpu_memory": row[2] if len(row) > 2 else "N/A",
                    "full_cmdline": " ".join(proc.cmdline()) or "N/A",
                    "username": proc.username() or "N/A"
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                processes.append({
                    "pid": pid,
                    "process_name": row[1],
                    "used_gpu_memory": row[2] if len(row) > 2 else "N/A",
                    "full_cmdline": "N/A",
                    "username": "N/A"
                })
    except subprocess.CalledProcessError:
        pass
    return processes

def print_gpu_process_info() -> None:
    """Print GPU stats and processes with full command lines in a readable format."""
    gpu_info = get_gpu_info()
    process_info = get_process_info()
    
    for gpu in gpu_info:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  Utilization: {gpu['utilization_gpu']}")
        print(f"  Memory: {gpu['memory_used']} / {gpu['memory_total']}")
        print("  Processes using VRAM:")
        if not process_info:
            print("    None")
        for proc in process_info:
            print(f"    PID: {proc['pid']}, Name: {proc['process_name']}, "
                  f"VRAM: {proc['used_gpu_memory']}, User: {proc['username']}")
            print(f"      Command: {proc['full_cmdline']}")
        print()

if __name__ == "__main__":
    print_gpu_process_info()
