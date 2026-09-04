import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "python/sglang/srt/utils/torch_memory_saver_adapter.py"
SPEC = importlib.util.spec_from_file_location("torch_memory_saver_adapter_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def test_disk_backup_requires_an_absolute_scoped_root(monkeypatch):
    monkeypatch.delenv("SGLANG_TMS_DISK_BACKUP_ROOT", raising=False)
    with pytest.raises(ValueError, match="is required"):
        adapter._disk_backup_dir()

    monkeypatch.setenv("SGLANG_TMS_DISK_BACKUP_ROOT", "relative/path")
    with pytest.raises(ValueError, match="absolute path"):
        adapter._disk_backup_dir()


def test_disk_backup_is_forwarded_to_torch_memory_saver(monkeypatch, tmp_path):
    memory_saver = MagicMock()
    monkeypatch.setattr(adapter, "_memory_saver", memory_saver, raising=False)
    monkeypatch.setenv("SGLANG_TMS_DISK_BACKUP_ROOT", str(tmp_path))

    adapter._TorchMemorySaverAdapterReal().region("weights", enable_disk_backup=True)

    disk_dir = memory_saver.set_disk_backup_dir.call_args.args[0]
    assert disk_dir.startswith(f"{tmp_path}/")
    memory_saver.region.assert_called_once_with(
        tag="weights",
        enable_cpu_backup=False,
        enable_disk_backup=True,
    )


def test_cpu_and_disk_backup_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        adapter._TorchMemorySaverAdapterReal().region(
            "weights",
            enable_cpu_backup=True,
            enable_disk_backup=True,
        )
