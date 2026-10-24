"""POSIX Shared Memory buffer management for high-throughput batch ingestion."""

from internal.shm.posix_shm import (
    PodBoundaryViolationError,
    SharedMemoryRegion,
    validate_pod_boundary,
)

__all__ = [
    "PodBoundaryViolationError",
    "SharedMemoryRegion",
    "validate_pod_boundary",
]
