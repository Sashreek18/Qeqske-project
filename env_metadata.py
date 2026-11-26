"""Shared environment metadata and statistical reporting helpers."""

import os
import platform
import subprocess
import sys

try:
    import numpy as np
except ImportError:
    np = None

try:
    import psutil
except ImportError:
    psutil = None


LEGACY_MACOS_ENVIRONMENT = {
    "platform": "macOS-26.6.2-arm64-arm-64bit-Mach-O",
    "os": "Darwin / macOS 26.6.2",
    "os_system": "Darwin",
    "os_release": "26.6.2",
    "machine": "arm64",
    "cpu_model": "Apple Silicon (arm64; exact marketing model not recorded at run time)",
    "cpu_cores_logical": 8,
    "cpu_cores_physical": 8,
    "python_version": "3.13.1",
    "python_implementation": "CPython",
    "numpy_version": "2.2.6",
    "provenance_note": "Backfilled from results/reproducibility.json recorded on the original macOS arm64 host.",
}


def get_cpu_model() -> str:
    """Return a human-readable CPU model string across Linux, macOS, and fallback hosts."""
    if sys.platform.startswith("linux") and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for prefix in ("model name", "Model", "Hardware"):
                for line in lines:
                    if line.lower().startswith(prefix.lower()) and ":" in line:
                        val = line.split(":", 1)[1].strip()
                        if val:
                            return val
        except OSError:
            pass
        try:
            out = subprocess.check_output(["lscpu"], text=True, timeout=2, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if line.lower().startswith("model name:") and ":" in line:
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out
        except Exception:
            pass

    return platform.processor() or platform.machine() or "unknown"


def get_cpu_governor() -> str | None:
    """Return the active Linux cpufreq scaling governor if readable."""
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if sys.platform.startswith("linux") and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                val = f.read().strip()
                if val:
                    return val
        except OSError:
            pass
    return None


def get_cpu_scaling_driver() -> str | None:
    """Return the active Linux cpufreq scaling driver if readable."""
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"
    if sys.platform.startswith("linux") and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                val = f.read().strip()
                if val:
                    return val
        except OSError:
            pass
    return None


def get_environment_metadata() -> dict:
    """Return standardized host/runtime metadata for results JSON files."""
    return {
        "platform": platform.platform(),
        "os": f"{platform.system()} {platform.release()}",
        "os_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": get_cpu_model(),
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": psutil.cpu_count(logical=False) if psutil is not None else None,
        "cpu_governor": get_cpu_governor(),
        "cpu_scaling_driver": get_cpu_scaling_driver(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__ if np is not None else None,
    }


def format_perm_pvalue(pvalue: float, n_permutations: int = 200) -> str:
    """Format permutation p-values using the Phipson-Smyth 1/(N+1) resolution floor."""
    floor = 1.0 / (n_permutations + 1)
    if pvalue <= floor + 1e-12:
        return f"< 1/{n_permutations + 1} ({floor:.4f})"
    return f"{pvalue:.4f}"


def perm_pvalue_metadata(pvalue: float, n_permutations: int = 200) -> dict:
    """Return numeric p-value alongside 1/(N+1) floor metadata and display string."""
    floor = 1.0 / (n_permutations + 1)
    return {
        "pvalue": float(pvalue),
        "n_permutations": int(n_permutations),
        "pvalue_floor": float(floor),
        "at_floor": bool(pvalue <= floor + 1e-12),
        "pvalue_display": format_perm_pvalue(pvalue, n_permutations),
    }
