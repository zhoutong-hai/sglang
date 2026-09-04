import logging
import os
import socket
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)

# Runtime capability marker used by launch preflights that require frozen
# weights to survive colocated release/resume without consuming host RAM.
SUPPORTS_DISK_WEIGHT_BACKUP = True


def _disk_backup_dir() -> str:
    root = os.environ.get("SGLANG_TMS_DISK_BACKUP_ROOT")
    if not root:
        raise ValueError("SGLANG_TMS_DISK_BACKUP_ROOT is required for disk weight backup")
    if not os.path.isabs(root):
        raise ValueError("SGLANG_TMS_DISK_BACKUP_ROOT must be an absolute path")
    hostname = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in socket.gethostname()
    )
    if not hostname:
        raise ValueError("Unable to derive a safe hostname for disk weight backup")
    return os.path.join(root, hostname, str(os.getpid()))


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(
        self,
        tag: str,
        enable_cpu_backup: bool = False,
        enable_disk_backup: bool = False,
    ):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def configure_subprocess(self):
        return torch_memory_saver.configure_subprocess()

    def region(
        self,
        tag: str,
        enable_cpu_backup: bool = False,
        enable_disk_backup: bool = False,
    ):
        if enable_cpu_backup and enable_disk_backup:
            raise ValueError("CPU and disk weight backup are mutually exclusive")
        if enable_disk_backup:
            _memory_saver.set_disk_backup_dir(_disk_backup_dir())
        return _memory_saver.region(
            tag=tag,
            enable_cpu_backup=enable_cpu_backup,
            enable_disk_backup=enable_disk_backup,
        )

    def cuda_graph(self, **kwargs):
        return _memory_saver.cuda_graph(**kwargs)

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(
        self,
        tag: str,
        enable_cpu_backup: bool = False,
        enable_disk_backup: bool = False,
    ):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
