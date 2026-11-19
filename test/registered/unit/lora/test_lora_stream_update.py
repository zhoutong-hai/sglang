"""Streamed LoRA updates: prefix routing, the session stash, and the
whole-adapter apply at end_weight_update."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.io_struct import EndWeightUpdateReqInput, LoRAUpdateOutput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _sha256_tensor,
    _split_lora_named_tensors,
    _WeightUpdateSession,
    _tensor_metadata,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


BASE_NAME = "model.layers.0.self_attn.q_proj.weight"
LORA_A = "model.layers.0.self_attn.q_proj.lora_A.weight"
LORA_B = "model.layers.0.self_attn.q_proj.lora_B.weight"


def _make_manager(**overrides) -> SchedulerWeightUpdaterManager:
    kwargs = dict(
        tp_worker=MagicMock(),
        draft_worker=None,
        memory_saver_adapter=MagicMock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
    )
    kwargs.update(overrides)
    with get_parallel().override(tp_group=SimpleNamespace(cpu_group=MagicMock())):
        return SchedulerWeightUpdaterManager(**kwargs)


class TestSplitLoraNamedTensors(CustomTestCase):
    def test_partitions_on_prefix(self):
        t = torch.zeros(1)
        base, lora = _split_lora_named_tensors(
            [
                (BASE_NAME, t),
                (f"miles_lora:{LORA_A}", t),
                (f"__miles_slot_3:{LORA_B}", t),
            ]
        )
        self.assertEqual([n for n, _ in base], [BASE_NAME])
        self.assertEqual(
            [n for n, _ in lora],
            [f"miles_lora:{LORA_A}", f"__miles_slot_3:{LORA_B}"],
        )

    def test_unprefixed_lora_tensor_is_a_sender_bug(self):
        with self.assertRaisesRegex(AssertionError, "without a"):
            _split_lora_named_tensors([(LORA_A, torch.zeros(1))])

    def test_pure_base_payload_passes_through(self):
        base, lora = _split_lora_named_tensors([(BASE_NAME, torch.zeros(1))])
        self.assertEqual(len(base), 1)
        self.assertEqual(lora, [])


class TestLoraStash(CustomTestCase):
    def test_structural_copy_is_owned_tp_local_and_never_hashed(self):
        mgr = _make_manager()
        mgr._session = _WeightUpdateSession(
            selector="all", sync_base=False, verify_lora_checksums=False
        )
        module = SimpleNamespace(
            tp_size=4,
            tp_rank=2,
            intermediate_size_per_partition=4,
            base_layer=SimpleNamespace(is_shared_fused_moe=False),
            slice_moe_lora_b_weights=None,
        )
        loras = mgr.tp_worker.model_runner.lora_manager
        loras.lora_modules = [{"moe": module}]
        loras.apply_streamed_adapter.return_value = SimpleNamespace(success=True)
        key = "language_model.model.layers.0.block_sparse_moe.experts.w1.lora_B.weight"
        wire = torch.arange(2 * 16 * 3).reshape(2, 16, 3).float()
        expected = wire[:, 8:12].clone()
        manifest = {"miles_lora": {key: _tensor_metadata(wire)}}
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater._sha256_tensor",
            side_effect=AssertionError("structural publication must not hash bytes"),
        ):
            mgr._stash_lora_tensors([("miles_lora:" + key, wire)], copy_tensors=True)
            owned = mgr._lora_stash["miles_lora"][key]
            self.assertEqual(
                owned.untyped_storage().nbytes(), owned.numel() * owned.element_size()
            )
            wire.zero_()
            self.assertTrue(torch.equal(owned, expected))
            self.assertTrue(mgr._apply_lora_stash(None, manifest)[0])
        loras.apply_streamed_adapter.assert_called_once()

    def test_structural_manifest_fails_closed_before_install(self):
        for defect in ("missing", "extra", "shape", "dtype", "adapter", "none"):
            with self.subTest(defect=defect):
                mgr = _make_manager()
                mgr._session = _WeightUpdateSession(
                    selector="all", verify_lora_checksums=False
                )
                mgr._stash_lora_tensors(
                    [(f"a:{LORA_A}", torch.ones(2, 3))], copy_tensors=True
                )
                manifest = {"a": {LORA_A: {"shape": [2, 3], "dtype": "torch.float32"}}}
                if defect == "missing":
                    manifest["a"].clear()
                elif defect == "extra":
                    manifest["a"][LORA_B] = manifest["a"][LORA_A]
                elif defect in ("shape", "dtype"):
                    manifest["a"][LORA_A][defect] = (
                        [3, 2] if defect == "shape" else "torch.bfloat16"
                    )
                elif defect == "adapter":
                    manifest["other"] = manifest.pop("a")
                else:
                    manifest = None
                self.assertFalse(mgr._apply_lora_stash(None, manifest)[0])
                mgr.tp_worker.model_runner.lora_manager.apply_streamed_adapter.assert_not_called()

    def test_duplicate_factor_is_rejected(self):
        mgr = _make_manager()
        bucket = [(f"a:{LORA_A}", torch.ones(2))]
        mgr._stash_lora_tensors(bucket, copy_tensors=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mgr._stash_lora_tensors(bucket, copy_tensors=True)

    def test_tp_local_stash_preserves_full_wire_manifest_and_detects_slice_corruption(
        self,
    ):
        for corrupt in (False, True):
            mgr = _make_manager()
            module = SimpleNamespace(
                tp_size=4,
                tp_rank=2,
                intermediate_size_per_partition=4,
                base_layer=SimpleNamespace(is_shared_fused_moe=False),
                slice_moe_lora_b_weights=None,
            )
            loras = mgr.tp_worker.model_runner.lora_manager
            loras.lora_modules = [{"moe": module}]
            loras.apply_streamed_adapter.return_value = SimpleNamespace(success=True)
            key = "language_model.model.layers.0.block_sparse_moe.experts.w1.lora_B.weight"
            wire = torch.arange(2 * 16 * 3).reshape(2, 16, 3).float()
            manifest = {"miles_lora": {key: _sha256_tensor(wire)}}
            mgr._stash_lora_tensors([("miles_lora:" + key, wire)], copy_tensors=True)
            retained = mgr._lora_stash["miles_lora"][key]
            self.assertTrue(torch.equal(retained, wire[:, 8:12]))
            self.assertEqual(
                retained.untyped_storage().nbytes(),
                retained.numel() * retained.element_size(),
            )
            wire.zero_()
            if corrupt:
                retained.add_(1)
            success, message = mgr._apply_lora_stash(manifest)
            self.assertEqual(success, not corrupt, message)
            if corrupt:
                loras.apply_streamed_adapter.assert_not_called()

    def test_pre_sliced_expert_factors_match_original_full_factor_loading(self):
        for tp_size in (2, 4, 16):
            for rank in range(tp_size):
                module = SimpleNamespace(
                    tp_size=tp_size,
                    intermediate_size_per_partition=4,
                    base_layer=SimpleNamespace(
                        moe_runner_config=SimpleNamespace(is_gated=True)
                    ),
                    _uses_interleaved_gate_up=False,
                )
                for name in ("_slice_moe_a", "_slice_moe_b_2d"):
                    setattr(
                        module,
                        name,
                        MethodType(getattr(FusedMoEWithLoRA, name), module),
                    )
                inter = 4 * tp_size
                a = torch.arange(3 * inter).reshape(3, inter)
                b = torch.arange(2 * inter * 3).reshape(2 * inter, 3)
                rows = slice(rank * 4, (rank + 1) * 4)
                local_b = torch.cat((b[:inter][rows], b[inter:][rows]))
                self.assertTrue(
                    torch.equal(
                        module._slice_moe_a(a, rank, "down_proj_moe"),
                        module._slice_moe_a(a[:, rows], rank, "down_proj_moe"),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        module._slice_moe_b_2d(b, rank, "gate_up_proj_moe"),
                        module._slice_moe_b_2d(local_b, rank, "gate_up_proj_moe"),
                    )
                )

    def test_stash_groups_by_adapter(self):
        mgr = _make_manager()
        a, b = torch.zeros(2), torch.ones(2)
        mgr._stash_lora_tensors(
            [(f"miles_lora:{LORA_A}", a), (f"__miles_slot_1:{LORA_A}", b)]
        )
        self.assertEqual(set(mgr._lora_stash), {"miles_lora", "__miles_slot_1"})
        self.assertIs(mgr._lora_stash["miles_lora"][LORA_A], a)
        self.assertIs(mgr._lora_stash["__miles_slot_1"][LORA_A], b)

    def test_split_at_first_colon_only(self):
        mgr = _make_manager()
        mgr._stash_lora_tensors([("name:with:colon.lora_A.weight", torch.zeros(1))])
        self.assertIn("with:colon.lora_A.weight", mgr._lora_stash["name"])

    def test_per_rank_payload_is_unwrapped(self):
        """A LoRA tensor never reaches the base loader, so the stash is the last
        place a per-rank payload can be reduced to this rank's tensor."""
        mgr = _make_manager()
        rank_tensors = [torch.zeros(2), torch.ones(2)]
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in rank_tensors]
        )
        with get_parallel().override(tp_rank=1):
            mgr._stash_lora_tensors([(f"miles_lora:{LORA_A}", payload)])
        self.assertTrue(
            torch.equal(mgr._lora_stash["miles_lora"][LORA_A], rank_tensors[1])
        )


class TestApplyLoraStash(CustomTestCase):
    def _manager_with_lora(self, apply_result=None):
        mgr = _make_manager()
        lora_manager = MagicMock()
        lora_manager.apply_streamed_adapter.return_value = (
            apply_result if apply_result is not None else LoRAUpdateOutput(success=True)
        )
        mgr.tp_worker.model_runner.lora_manager = lora_manager
        return mgr, lora_manager

    def test_empty_stash_is_noop(self):
        mgr = _make_manager()
        mgr.tp_worker.model_runner.lora_manager = None
        success, _ = mgr._apply_lora_stash(None)
        self.assertTrue(success)

    def test_stash_without_lora_manager_fails(self):
        mgr = _make_manager()
        mgr.tp_worker.model_runner.lora_manager = None
        mgr._stash_lora_tensors([(f"miles_lora:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(None)
        self.assertFalse(success)
        self.assertIn("--enable-lora", message)

    def test_applies_each_adapter_and_clears_stash(self):
        mgr, lora_manager = self._manager_with_lora()
        mgr._stash_lora_tensors(
            [(f"a:{LORA_A}", torch.zeros(1)), (f"b:{LORA_A}", torch.zeros(1))]
        )
        success, _ = mgr._apply_lora_stash(None)
        self.assertTrue(success)
        self.assertEqual(lora_manager.apply_streamed_adapter.call_count, 2)
        applied = [
            c.args[0] for c in lora_manager.apply_streamed_adapter.call_args_list
        ]
        self.assertEqual(applied, ["a", "b"])  # sorted, deterministic
        self.assertEqual(mgr._lora_stash, {})

    def test_manager_failure_propagates(self):
        mgr, _ = self._manager_with_lora(
            apply_result=LoRAUpdateOutput(success=False, error_message="unregistered")
        )
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(None)
        self.assertFalse(success)
        self.assertIn("unregistered", message)

    def test_checksum_match_passes(self):
        mgr, _ = self._manager_with_lora()
        t = torch.arange(4, dtype=torch.float32)
        mgr._stash_lora_tensors([(f"a:{LORA_A}", t)])
        success, _ = mgr._apply_lora_stash({"a": {LORA_A: _sha256_tensor(t)}})
        self.assertTrue(success)

    def test_checksum_mismatch_fails(self):
        mgr, lora_manager = self._manager_with_lora()
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(4))])
        success, message = mgr._apply_lora_stash(
            {"a": {LORA_A: _sha256_tensor(torch.ones(4))}}
        )
        self.assertFalse(success)
        self.assertIn("checksum mismatch", message)
        lora_manager.apply_streamed_adapter.assert_not_called()

    def test_checksum_name_set_mismatch_fails(self):
        mgr, _ = self._manager_with_lora()
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash({"a": {LORA_A: "x", LORA_B: "y"}})
        self.assertFalse(success)
        self.assertIn("expected manifest", message)

    def test_manifest_adapter_set_mismatch_fails(self):
        """An adapter the sender promised but streamed nothing for is a sender bug
        reported to the caller, not a scheduler kill."""
        mgr, lora_manager = self._manager_with_lora()
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(
            {"a": {LORA_A: "x"}, "b": {LORA_A: "y"}}
        )
        self.assertFalse(success)
        self.assertIn("expected manifest", message)
        self.assertIn("'b'", message)
        lora_manager.apply_streamed_adapter.assert_not_called()

    def test_forget_admits_a_different_tensor_set(self):
        """Without the forget on re-register/unload, a name that legitimately
        changes its tensor set is rejected as a partial stream forever."""
        mgr, _ = self._manager_with_lora()
        mgr._stash_lora_tensors(
            [(f"a:{LORA_A}", torch.zeros(1)), (f"a:{LORA_B}", torch.zeros(1))]
        )
        self.assertTrue(mgr._apply_lora_stash(None)[0])
        mgr.forget_lora_adapter("a")
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        self.assertTrue(mgr._apply_lora_stash(None)[0])

    def test_shrunken_tensor_set_rejected(self):
        # The adapter's tensor set is static for its lifetime; a change means a
        # partial stream, which the whole-adapter replace would silently zero.
        mgr, _ = self._manager_with_lora()
        mgr._stash_lora_tensors(
            [(f"a:{LORA_A}", torch.zeros(1)), (f"a:{LORA_B}", torch.zeros(1))]
        )
        self.assertTrue(mgr._apply_lora_stash(None)[0])
        mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.zeros(1))])
        success, message = mgr._apply_lora_stash(None)
        self.assertFalse(success)
        self.assertIn("different tensor", message)


class TestLoRAManagerStreamEntryPoints(CustomTestCase):
    def test_installed_audit_uses_resolved_live_slot_and_rejects_invalid_ownership(
        self,
    ):
        manager = self._bare_manager()
        ref = LoRARef(lora_name="miles_lora", lora_path="__stream__")
        manager.lora_refs[ref.lora_id] = ref
        values = dict(
            A_buffer={"q": [torch.arange(12).reshape(2, 2, 3).float()]},
            B_buffer={},
            embedding_A_buffer={},
            embedding_B_buffer={},
            lm_head_A_buffer={},
            lm_head_B_buffer={},
            new_embeddings_buffer={},
        )
        pool = SimpleNamespace(
            invalid_uids=set(),
            uid_to_buffer_id={ref.lora_id: 1},
            buffer_id_to_uid=[None, ref.lora_id],
            **values,
        )
        manager.memory_pool = pool
        before = manager.installed_adapter_checksums(["miles_lora"])
        self.assertEqual(len(before), 1)
        pool.A_buffer["q"][0][0].add_(5)
        self.assertEqual(manager.installed_adapter_checksums(["miles_lora"]), before)
        pool.A_buffer["q"][0][1].add_(5)
        self.assertNotEqual(manager.installed_adapter_checksums(["miles_lora"]), before)
        pool.invalid_uids.add(ref.lora_id)
        with self.assertRaisesRegex(ValueError, "invalid or nonresident"):
            manager.installed_adapter_checksums(["miles_lora"])
        pool.invalid_uids.clear()
        pool.buffer_id_to_uid[1] = "foreign"
        with self.assertRaisesRegex(ValueError, "slot ownership"):
            manager.installed_adapter_checksums(["miles_lora"])

    def _bare_manager(self):
        manager = object.__new__(LoRAManager)
        manager.lora_refs = {}
        return manager

    def test_require_registered(self):
        manager = self._bare_manager()
        ref = LoRARef(lora_name="miles_lora", lora_path="__stream__")
        manager.lora_refs[ref.lora_id] = ref
        self.assertIs(manager.require_registered("miles_lora"), ref)
        with self.assertRaisesRegex(ValueError, "unregistered"):
            manager.require_registered("missing")

    def test_register_delegates_with_empty_tensors_and_upsert(self):
        manager = self._bare_manager()
        manager.load_lora_adapter_from_tensors = MagicMock(
            return_value=LoRAUpdateOutput(success=True)
        )
        ref = LoRARef(lora_name="miles_lora", lora_path="__stream__")
        result = manager.register_lora_adapter(ref, {"r": 8})
        self.assertTrue(result.success)
        kwargs = manager.load_lora_adapter_from_tensors.call_args.kwargs
        self.assertEqual(kwargs["tensors"], {})
        self.assertEqual(kwargs["config_dict"], {"r": 8})
        self.assertTrue(kwargs["upsert"])

    def test_apply_streamed_adapter_reuses_registered_config(self):
        manager = self._bare_manager()
        ref = LoRARef(lora_name="miles_lora", lora_path="__stream__")
        manager.lora_refs[ref.lora_id] = ref
        manager.load_lora_adapter_from_tensors = MagicMock(
            return_value=LoRAUpdateOutput(success=True)
        )
        tensors = {LORA_A: torch.zeros(1)}
        result = manager.apply_streamed_adapter("miles_lora", tensors)
        self.assertTrue(result.success)
        args, kwargs = manager.load_lora_adapter_from_tensors.call_args
        self.assertIs(args[0], ref)
        self.assertIs(args[1], tensors)
        self.assertIsNone(kwargs["config_dict"])
        self.assertTrue(kwargs["upsert"])

    def test_apply_streamed_adapter_unregistered_fails(self):
        manager = self._bare_manager()
        result = manager.apply_streamed_adapter("missing", {LORA_A: torch.zeros(1)})
        self.assertFalse(result.success)
        self.assertIn("unregistered", result.error_message)


if __name__ == "__main__":
    unittest.main()


class TestWeightVersionDeferredToCommit(CustomTestCase):
    """Inside a session the version reported by a bucket must not reach the
    scheduler until end_weight_update commits; a failed commit drops it."""

    def _manager(self):
        mgr = _make_manager(scheduler=Mock())
        mgr._session = _WeightUpdateSession(selector="all")
        return mgr

    def test_structural_failure_and_abort_do_not_commit_version(self):
        for abort in (False, True):
            mgr = _make_manager(scheduler=Mock())
            mgr._session = _WeightUpdateSession(
                selector="target",
                sync_base=False,
                verify_lora_checksums=False,
                pending_version="7",
            )
            mgr._stash_lora_tensors([(f"a:{LORA_A}", torch.ones(2))], copy_tensors=True)
            with patch("torch.distributed.barrier"):
                result = mgr.end_weight_update(EndWeightUpdateReqInput(abort=abort))
            self.assertEqual(result.success, abort)
            mgr.scheduler.record_weight_version_change.assert_not_called()
            mgr.tp_worker.model_runner.lora_manager.apply_streamed_adapter.assert_not_called()
            if abort:
                self.assertEqual(mgr._lora_stash, {})
                self.assertEqual(mgr._lora_stash_metadata, {})

    def test_bucket_version_is_held_while_the_session_is_open(self):
        mgr = self._manager()
        mgr.record_weight_version_after_update("7")
        mgr.scheduler.record_weight_version_change.assert_not_called()
        self.assertEqual(mgr._session.pending_version, "7")

    def test_outside_a_session_the_version_is_recorded_at_once(self):
        mgr = _make_manager(scheduler=Mock())
        mgr.record_weight_version_after_update("7")
        mgr.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="7"
        )

    def test_commit_records_the_held_version(self):
        mgr = self._manager()
        mgr.record_weight_version_after_update("7")
        pending_version = mgr._session.pending_version
        mgr._session = None
        mgr.record_weight_version_after_update(pending_version)
        mgr.scheduler.record_weight_version_change.assert_called_once_with(
            new_version="7"
        )
