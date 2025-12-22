import json
import numpy as np


def check_numpy_config() -> dict:
    """Check BLAS configuration for NumPy."""
    config = np.__config__.show(mode='dicts')
    print("\nNumpy Config:")
    print(json.dumps(config, indent=2))
    return config


def check_accelerate_usage() -> dict:
    """Check BLAS configuration for NumPy."""
    config = np.__config__.show(mode='dicts')
    blas_info = config.get('Build Dependencies', {}).get('blas', {})
    print("\nBLAS Info:")
    print(json.dumps(blas_info, indent=2))
    return blas_info


if __name__ == "__main__":
    check_numpy_config()
    check_accelerate_usage()
