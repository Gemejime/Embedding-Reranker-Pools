"""Unit tests for POSIX shared memory IPC and pod boundary security guards."""

import unittest
import uuid

from internal.shm.posix_shm import (
    PodBoundaryViolationError,
    temporary_shm_region,
    validate_pod_boundary,
)


class TestPosixShm(unittest.TestCase):
    def test_pod_boundary_allows_local_loopback(self) -> None:
        self.assertTrue(validate_pod_boundary("localhost"))
        self.assertTrue(validate_pod_boundary("127.0.0.1"))
        self.assertTrue(validate_pod_boundary("::1"))
        self.assertTrue(validate_pod_boundary(""))

    def test_pod_boundary_rejects_cross_pod_hosts(self) -> None:
        remote_hosts = [
            "10.0.1.45",
            "192.168.1.10",
            "rag-worker-spot-7.internal",
            "k8s-node-worker-2",
        ]
        for host in remote_hosts:
            with self.assertRaises(PodBoundaryViolationError):
                validate_pod_boundary(host)

    def test_temporary_shm_region_lifecycle(self) -> None:
        region_name = f"test_shm_{uuid.uuid4().hex[:8]}"
        payload = b"test-tensor-bytes-payload-12345"
        size = 1024

        with temporary_shm_region(name=region_name, size_bytes=size) as region:
            self.assertEqual(region.name, region_name)
            self.assertEqual(region.key, f"/{region_name}")
            self.assertEqual(region.size_bytes, size)

            bytes_written = region.write(payload)
            self.assertEqual(bytes_written, len(payload))

            read_back = region.read(size=len(payload))
            self.assertEqual(read_back, payload)

            # Test Triton registration payload structure
            reg_payload = region.to_triton_register_payload()
            self.assertEqual(reg_payload["key"], f"/{region_name}")
            self.assertEqual(reg_payload["byte_size"], size)
            self.assertEqual(reg_payload["offset"], 0)

            # Test KServe infer parameter structure
            kserve_param = region.to_kserve_infer_parameter()
            self.assertEqual(kserve_param["shared_memory_region"], region_name)
            self.assertEqual(kserve_param["shared_memory_byte_size"], size)

    def test_shm_write_exceeding_capacity_raises(self) -> None:
        region_name = f"test_shm_overflow_{uuid.uuid4().hex[:8]}"
        size = 16

        with (
            temporary_shm_region(name=region_name, size_bytes=size) as region,
            self.assertRaises(ValueError),
        ):
            region.write(b"this payload is way too long for sixteen bytes")


if __name__ == "__main__":
    unittest.main()
