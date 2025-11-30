import json
import platform
import psutil
from typing import TypedDict

class DiskUsageDict(TypedDict):
    total: int
    used: int
    free: int
    percent: float

class DeviceInfoDict(TypedDict):
    system: str
    machine: str
    processor: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total: int
    ram_available: int
    disk_usage: DiskUsageDict

def get_device_info() -> DeviceInfoDict:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total": psutil.virtual_memory().total,
        "ram_available": psutil.virtual_memory().available,
        "disk_usage": psutil.disk_usage("/")._asdict(),  # type: ignore
    }

if __name__ == "__main__":
    print(json.dumps(get_device_info(), indent=2))
