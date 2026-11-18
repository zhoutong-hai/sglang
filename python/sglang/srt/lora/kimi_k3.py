"""Kimi-K3's optional KDA Q/K/V adapter layout; the fourth gate partition stays frozen."""


class KimiK3LoRAMapping:
    @staticmethod
    def get_lora_target_module_name(name: str) -> str:
        parent, _, leaf = name.rpartition(".")
        if ".experts" in parent:
            return {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}.get(
                leaf, name
            )
        if leaf in {"q_proj", "k_proj", "v_proj"}:
            return "fused_qkvg_proj"
        return name

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        from sglang.srt.lora.utils import get_default_hidden_dim

        config = self.config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        if module_name == "fused_qkvg_proj":
            kda = config.linear_attn_config
            if not kda.get("use_full_rank_gate", False):
                raise ValueError(
                    "Kimi-K3 Q/K/V LoRA requires the fused full-rank-gate layout"
                )
            return config.hidden_size, 3 * kda["head_dim"] * kda["num_heads"]
        return get_default_hidden_dim(module_name, config, layer_idx)
