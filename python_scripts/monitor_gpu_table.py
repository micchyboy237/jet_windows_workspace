# monitor_gpu_table.py

from typing import List, TypedDict
import subprocess
import csv
from io import StringIO
import psutil
import asyncio
import argparse
import json
from pathlib import Path
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for Windows compatibility
init()

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

async def get_gpu_info_async() -> List[GPUInfo]:
    """Fetch GPU stats asynchronously: index, name, utilization, memory used/total."""
    cmd = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader"
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    reader = csv.reader(StringIO(stdout.decode()))
    return [
        {
            "index": row[0],
            "name": row[1],
            "utilization_gpu": row[2],
            "memory_used": row[3],
            "memory_total": row[4],
        }
        for row in reader
    ]

async def get_process_info_async(user: str = None, process_name: str = None) -> List[ProcessInfo]:
    """Fetch process info (compute only): PID, base name, VRAM, cmdline, username."""
    cmd_compute = "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader"
    processes = []
    
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd_compute, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        stdout_str = stdout.decode()
        # Skip if command fails, output is empty, or contains an error
        if proc.returncode != 0 or not stdout_str.strip() or "ERROR" in stdout_str:
            return processes
        reader = csv.reader(StringIO(stdout_str))
        for row in reader:
            # Validate PID is numeric
            if not row[0].isdigit():
                continue
            pid = row[0]
            # Extract base name without extension for process_name
            base_name = Path(row[1]).stem
            try:
                proc = psutil.Process(int(pid))
                cmdline = proc.cmdline()
                # Use only base name without extension for command, no args
                cmd_base = Path(cmdline[0]).stem if cmdline else "N/A"
                processes.append({
                    "pid": pid,
                    "process_name": base_name,
                    "used_gpu_memory": row[2] if len(row) > 2 else "N/A",
                    "full_cmdline": cmd_base,
                    "username": proc.username() or "N/A",
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                processes.append({
                    "pid": pid,
                    "process_name": base_name,
                    "used_gpu_memory": row[2] if len(row) > 2 else "N/A",
                    "full_cmdline": "N/A",
                    "username": "N/A",
                })
    except Exception:
        pass
    
    # Filter processes by user or process_name
    if user or process_name:
        processes = [
            p for p in processes
            if (user is None or p["username"] == user) and
               (process_name is None or p["process_name"].lower().find(process_name.lower()) != -1)
        ]
    
    # Sort by VRAM usage (descending) or PID if VRAM is N/A
    def vram_to_mib(vram: str) -> float:
        if vram == "N/A" or "[N/A]" in vram:
            return -float("inf")  # Sort N/A to the bottom
        try:
            return float(vram.split()[0])  # Extract MiB value
        except (ValueError, IndexError):
            return -float("inf")
    
    processes.sort(key=lambda p: (vram_to_mib(p["used_gpu_memory"]), -int(p["pid"])), reverse=True)
    return processes

def print_gpu_process_info(gpu_info: List[GPUInfo], process_info: List[ProcessInfo], json_output: bool = False) -> None:
    """Print GPU stats and processes in a readable, colored table format or JSON."""
    if json_output:
        print(json.dumps({"gpus": gpu_info, "processes": process_info}, indent=2))
        return
    
    for gpu in gpu_info:
        # GPU stats in cyan
        print(f"{Fore.CYAN}GPU {gpu['index']}: {gpu['name']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  Utilization: {gpu['utilization_gpu']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  Memory: {gpu['memory_used']} / {gpu['memory_total']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Processes using VRAM:{Style.RESET_ALL}")
        # Note WDDM limitation
        print(f"{Fore.RED}  Note: Per-process VRAM usage unavailable in WDDM mode (GeForce GPUs). Run with --check-tcc to verify.{Style.RESET_ALL}")
        
        if not process_info:
            print(f"{Fore.RED}    None{Style.RESET_ALL}")
        else:
            # Prepare table data
            table = [
                [
                    proc["pid"],
                    proc["process_name"],
                    proc["used_gpu_memory"],
                    proc["username"],
                    proc["full_cmdline"]
                ]
                for proc in process_info
            ]
            # Print table with headers in green
            print(tabulate(
                table,
                headers=[f"{Fore.GREEN}PID{Style.RESET_ALL}", f"{Fore.GREEN}Name{Style.RESET_ALL}",
                         f"{Fore.GREEN}VRAM{Style.RESET_ALL}", f"{Fore.GREEN}User{Style.RESET_ALL}",
                         f"{Fore.GREEN}Command{Style.RESET_ALL}"],
                tablefmt="fancy_grid",
                stralign="left",
                maxcolwidths=[None, 20, None, 20, 20]
            ))
        print()

async def main():
    """Main function to monitor GPU and processes with optional looping and filtering."""
    parser = argparse.ArgumentParser(description="Monitor GPU and processes")
    parser.add_argument("--loop-ms", type=int, default=0, help="Loop interval in milliseconds (0 for single run)")
    parser.add_argument("--user", type=str, help="Filter processes by username")
    parser.add_argument("--process-name", type=str, help="Filter processes by process name")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--check-tcc", action="store_true", help="Check TCC mode and exit")
    args = parser.parse_args()

    if args.check_tcc:
        print("Checking driver mode (run as admin to switch to TCC):")
        print("Run: nvidia-smi -q -d DRIVER_MODEL")
        print("To switch to TCC (if supported): nvidia-smi -dm 1")
        print("Note: GeForce GPUs typically only support WDDM mode.")
        return

    while True:
        gpu_info = await get_gpu_info_async()
        process_info = await get_process_info_async(user=args.user, process_name=args.process_name)
        print_gpu_process_info(gpu_info, process_info, json_output=args.json)
        if args.loop_ms <= 0:
            break
        await asyncio.sleep(args.loop_ms / 1000.0)

if __name__ == "__main__":
    asyncio.run(main())
