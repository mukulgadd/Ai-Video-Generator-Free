"""Shared utilities for the vidgen pipeline.

Provides logging setup, memory monitoring, GPU memory release,
timing decorators, and file integrity checking.
"""

import gc
import functools
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_pipeline_logging(log_dir: Path, job_id: str, verbose: bool = False) -> None:
    """Configure logging with both console and file output.
    
    Args:
        log_dir: Directory to write log files.
        job_id: Job identifier for the log filename.
        verbose: If True, set console to DEBUG level.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_id}.log"
    
    # File handler (always DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    
    root_logger = logging.getLogger("vidgen")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_memory_usage_gb() -> float:
    """Get current process memory usage in GB.
    
    Uses os.getpid() and psutil if available, falls back to
    /proc/self/status on Linux or basic estimation on macOS.
    
    Returns:
        Memory usage in gigabytes.
    """
    try:
        import resource
        # macOS/Linux: get max resident set size
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS, ru_maxrss is in bytes; on Linux, in KB
        import platform
        if platform.system() == "Darwin":
            return usage.ru_maxrss / (1024 ** 3)
        else:
            return usage.ru_maxrss / (1024 ** 2)
    except (ImportError, AttributeError):
        return 0.0


def check_memory_limit(limit_gb: int) -> bool:
    """Check if current memory usage is within the limit.
    
    Args:
        limit_gb: Maximum allowed memory in GB.
        
    Returns:
        True if within limit, False if exceeded.
    """
    current = get_memory_usage_gb()
    if current > limit_gb:
        logger.warning(f"Memory usage {current:.2f}GB exceeds limit {limit_gb}GB")
        return False
    return True


def release_gpu_memory() -> None:
    """Release GPU memory between pipeline stages.
    
    Calls gc.collect() and clears MLX Metal cache if available.
    """
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
        logger.debug("MLX Metal cache cleared")
    except ImportError:
        pass


def timing_decorator(stage_name: str):
    """Decorator that logs execution time for a function.
    
    Args:
        stage_name: Human-readable name for the stage being timed.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"Starting: {stage_name}")
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"Completed: {stage_name} in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"Failed: {stage_name} after {duration:.2f}s - {e}")
                raise
        return wrapper
    return decorator


def check_file_integrity(file_path: Path, min_size_bytes: int = 0) -> bool:
    """Check that a file exists and meets minimum size requirements.
    
    Args:
        file_path: Path to the file to check.
        min_size_bytes: Minimum file size in bytes (0 = just check existence).
        
    Returns:
        True if file exists and meets size requirement.
    """
    if not file_path.exists():
        return False
    if file_path.stat().st_size < min_size_bytes:
        return False
    return True


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string.
    
    Examples:
        45.2 -> "45s"
        125.0 -> "2m 5s"
        3661.0 -> "1h 1m 1s"
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


def estimate_generation_time(
    script_duration_seconds: float,
    generation_ratio: float = 1.0,
    buffer: float = 1.5,
) -> float:
    """Estimate total generation time for a video.
    
    Args:
        script_duration_seconds: Expected video duration in seconds.
        generation_ratio: Hours per minute of video (default 1.0).
        buffer: Timeout buffer multiplier (default 1.5).
        
    Returns:
        Estimated generation time in seconds (including buffer).
    """
    minutes = script_duration_seconds / 60
    hours = minutes * generation_ratio
    seconds = hours * 3600
    return seconds * buffer
