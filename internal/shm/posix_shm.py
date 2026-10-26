"""POSIX Shared Memory buffer management for high-throughput batch ingestion.

Enforces:
1. Strict Pod Isolation: Shared memory regions are shared strictly between co-located
   containers within the same Kubernetes Pod (8 GiB emptyDir volume at /dev/shm).
2. PodBoundaryViolationError: Raised if cross-pod or cross-node access is attempted,
   forcing remote callers to use batched gRPC over HTTP/2.
3. Deterministic Lifecycle: Safe creation, attachment, unmapping, and unlinking
   to prevent kernel memory leaks.
4. Triton KServe v2 IPC Protocol: Generates system shared memory registration payloads.
"""

from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Any

from internal.logging.secured_logger import get_secured_logger

logger = get_secured_logger("posix-shm")

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", ""})


class PodBoundaryViolationError(Exception):
    """Raised when cross-pod or cross-node POSIX shared memory access is attempted."""

    pass


def validate_pod_boundary(target_host: str, shm_base_path: str = "/dev/shm") -> bool:
    """Validate that the target host resides strictly within the local pod boundary.

    Raises PodBoundaryViolationError if target_host is a remote node, pod, or cluster IP.
    """
    clean_host = target_host.strip().lower()
    if clean_host not in LOCAL_HOSTS:
        logger.error(
            "Rejected cross-pod shared memory access attempt",
            extra={
                "status": "pod_boundary_violation",
                "target_host": clean_host,
            },
        )
        raise PodBoundaryViolationError(
            f"POSIX shared memory IPC is strictly restricted to co-located containers in the same Pod. "
            f"Remote host '{clean_host}' must connect via batched gRPC over HTTP/2."
        )
    return True


class SharedMemoryRegion:
    """Managed POSIX Shared Memory segment for zero-copy tensor passing."""

    def __init__(
        self,
        name: str,
        size_bytes: int,
        offset: int = 0,
        shm_base_path: str = "/dev/shm",
    ) -> None:
        self.name = name.strip("/")
        self.size_bytes = size_bytes
        self.offset = offset
        self.shm_base_path = shm_base_path
        self._shm: shared_memory.SharedMemory | None = None
        self._is_creator: bool = False

    @property
    def key(self) -> str:
        """POSIX shm key matching Triton registration path conventions."""
        return f"/{self.name}"

    def allocate(self) -> "SharedMemoryRegion":
        """Allocate a new shared memory segment of specified size."""
        try:
            self._shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=self.size_bytes,
            )
            self._is_creator = True
            logger.info(
                "Allocated POSIX shared memory region",
                extra={
                    "region_name": self.name,
                    "size_bytes": self.size_bytes,
                    "status": "allocated",
                },
            )
            return self
        except FileExistsError:
            # Region already exists; attach instead
            return self.attach()

    def attach(self) -> "SharedMemoryRegion":
        """Attach to an existing shared memory segment."""
        self._shm = shared_memory.SharedMemory(
            name=self.name,
            create=False,
            size=self.size_bytes,
        )
        self._is_creator = False
        return self

    def write(self, data: bytes, offset: int = 0) -> int:
        """Write binary tensor data directly into the shared memory segment."""
        if self._shm is None or self._shm.buf is None:
            raise RuntimeError(f"Shared memory region '{self.name}' is not allocated or attached.")
        end = offset + len(data)
        if end > self.size_bytes:
            raise ValueError(
                f"Data size ({len(data)} bytes) at offset {offset} exceeds region capacity ({self.size_bytes} bytes)."
            )
        buf = self._shm.buf
        buf[offset:end] = data
        return len(data)

    def read(self, size: int | None = None, offset: int = 0) -> bytes:
        """Read binary tensor data directly from the shared memory segment."""
        if self._shm is None or self._shm.buf is None:
            raise RuntimeError(f"Shared memory region '{self.name}' is not allocated or attached.")
        read_len = size if size is not None else (self.size_bytes - offset)
        end = offset + read_len
        if end > self.size_bytes:
            raise ValueError(
                f"Read range ({end} bytes) exceeds region capacity ({self.size_bytes} bytes)."
            )
        buf = self._shm.buf
        return bytes(buf[offset:end])

    def close(self) -> None:
        """Close local process file descriptor mapping."""
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception as exc:
                logger.warning("Error closing shared memory descriptor", exc_info=exc)
            finally:
                self._shm = None

    def unlink(self) -> None:
        """Unlink the shared memory segment from the operating system kernel."""
        if self._shm is not None:
            try:
                self._shm.unlink()
                logger.info(
                    "Unlinked POSIX shared memory region",
                    extra={"region_name": self.name, "status": "unlinked"},
                )
            except Exception as exc:
                logger.warning("Error unlinking shared memory region", exc_info=exc)
            finally:
                self._shm = None

    def release(self) -> None:
        """Close descriptor and unlink if this instance was the creator."""
        if self._is_creator:
            self.unlink()
        else:
            self.close()

    def to_triton_register_payload(self) -> dict[str, Any]:
        """Generate Triton System Shared Memory Registration payload."""
        return {
            "key": self.key,
            "offset": self.offset,
            "byte_size": self.size_bytes,
        }

    def to_kserve_infer_parameter(self) -> dict[str, Any]:
        """Generate KServe v2 inference input/output tensor shared memory parameter."""
        return {
            "shared_memory_region": self.name,
            "shared_memory_offset": self.offset,
            "shared_memory_byte_size": self.size_bytes,
        }

    def __enter__(self) -> "SharedMemoryRegion":
        if self._shm is None:
            self.allocate()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


@contextmanager
def temporary_shm_region(
    name: str,
    size_bytes: int,
    target_host: str = "localhost",
) -> Generator[SharedMemoryRegion, None, None]:
    """Scoped context manager guaranteeing leak-free shared memory allocation and unlinking."""
    validate_pod_boundary(target_host)
    region = SharedMemoryRegion(name=name, size_bytes=size_bytes)
    region.allocate()
    try:
        yield region
    finally:
        region.release()
