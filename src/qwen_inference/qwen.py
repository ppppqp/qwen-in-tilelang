from __future__ import annotations

from dataclasses import dataclass
import logging
import tilelang.language as T
import torch

from .basics import linear, silu, rms_norm
from .attention import grouped_attention
from .basics import RMSNorm
from .rope import RoPE
from .embedding import Embedding


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: torch.tensor,
        wk: torch.tensor,
        wv: torch.tensor,
        wo: torch.tensor,
        q_norm: torch.tensor,
        k_norm: torch.tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.rope = RoPE(self.head_dim, max_seq_len, theta)
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = float(theta)
        # empty bias for reusing
        self.empty_bias = torch.zeros(
            self.hidden_size, dtype=wq.dtype, device=wq.device
        )
        # TODO: precompile all kernels
        # TODO: BLOCK hyper parameter and autotuning

    def __call__(
        self,
        x: torch.tensor,
        is_causal: bool = True,
    ) -> torch.tensor:
        B, L, _ = x.shape
        x_flat = x.reshape(B * L, self.hidden_size)

        def project(weight: torch.Tensor) -> torch.Tensor:
            return linear(x_flat, weight, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)

        projection_q = project(self.wq).reshape(B, L, self.num_heads, self.head_dim)
        projection_k = project(self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        projection_q = rms_norm(
            projection_q.reshape(B * L * self.num_heads, self.head_dim),
            self.q_norm,
            self.rms_norm_eps,
            BLOCK_M=16,
            BLOCK_N=self.head_dim,
        ).reshape(B, L, self.num_heads, self.head_dim)
        projection_k = rms_norm(
            projection_k.reshape(B * L * self.num_kv_heads, self.head_dim),
            self.k_norm,
            self.rms_norm_eps,
            BLOCK_M=16,
            BLOCK_N=self.head_dim,
        ).reshape(B, L, self.num_kv_heads, self.head_dim)
        projection_v = project(self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)

        projection_q = self.rope(projection_q)
        projection_k = self.rope(projection_k)

        # TODO: potentially fix precision here
        # TODO: custom scale
        x = grouped_attention(
            projection_q,
            projection_k,
            projection_v,
            is_causal=is_causal,
            BLOCK_L=16,
            BLOCK_S=16,
        ).reshape(B, L, self.num_heads * self.head_dim)

        x_flat = x.reshape(B * L, self.num_heads * self.head_dim)
        output = linear(
            x_flat,
            self.wo,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=self.head_dim,
        )
        return output.reshape(B, L, self.hidden_size)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.tensor,
        w_up: torch.tensor,
        w_down: torch.tensor,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.empty_bias = torch.zeros(dim, dtype=w_down.dtype, device=w_down.device)

    # TODO: fusion
    def __call__(self, x: torch.tensor) -> torch.tensor:
        B, L, _ = x.shape
        x_flat = x.reshape(B * L, self.dim)

        def project(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return linear(input_tensor, weight, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)

        project_gate = project(x_flat, self.w_gate)

        project_gate_silu = silu(project_gate, BLOCK_M=16, BLOCK_N=self.hidden_dim)
        project_up = project(x_flat, self.w_up) * project_gate_silu
        project_down = linear(
            project_up, self.w_down, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64
        )
        return project_down.reshape(B, L, self.dim)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: torch.tensor,
        wk: torch.tensor,
        wv: torch.tensor,
        wo: torch.tensor,
        q_norm: torch.tensor,
        k_norm: torch.tensor,
        w_gate: torch.tensor,
        w_up: torch.tensor,
        w_down: torch.tensor,
        w_input_layernorm: torch.tensor,
        w_post_attention_layernorm: torch.tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):

        print("Initializing Qwen3TransformerBlock with")
        print(f"num_attention_heads: {num_attention_heads}")
        print(f"num_kv_heads: {num_kv_heads}")
        print(f"hidden_size: {hidden_size}")
        print(f"head_dim: {head_dim}")
        print(f"intermediate_size: {intermediate_size}")
        print(f"wq shape: {wq.shape}")
        print(f"wk shape: {wk.shape}")
        print(f"wv shape: {wv.shape}")
        print(f"wo shape: {wo.shape}")
        print(f"q_norm shape: {q_norm.shape}")
        print(f"k_norm shape: {k_norm.shape}")
        print(f"w_gate shape: {w_gate.shape}")
        print(f"w_up shape: {w_up.shape}")
        print(f"w_down shape: {w_down.shape}")
        print(f"w_input_layernorm shape: {w_input_layernorm.shape}")
        print(f"w_post_attention_layernorm shape: {w_post_attention_layernorm.shape}")
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
        )

    def __call__(
        self,
        x: torch.tensor,
        is_causal: bool = True,
    ) -> torch.tensor:
        r = self.self_attn(self.input_layernorm(x), is_causal=is_causal)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


@dataclass
class Qwen3ModelConfig:
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    rms_norm_eps: float
    head_dim: int
    max_position_embeddings: int = 32768
    rope_theta: int = 1000000
    tie_word_embeddings: bool = False


class Qwen3Model:
    def __init__(
        self,
        model_config: Qwen3ModelConfig,
        state_dict: dict[str, torch.Tensor],
    ):
        self.num_hidden_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        self.vocab_size = model_config.vocab_size
        precision = T.bfloat16
        self.precision = precision

        def tensor(name: str) -> torch.Tensor:
            try:
                return state_dict[name].contiguous()
            except KeyError as exc:
                raise KeyError(f"Missing model tensor: {name}") from exc

        def linear_weight(name: str) -> torch.Tensor:
            # Hugging Face stores linear weights as (out_features, in_features).
            # The local TileLang linear kernel expects (in_features, out_features).
            return tensor(name).T.contiguous()

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=tensor("model.embed_tokens.weight"),
        )
        self.layers_inner = []

        for i in range(model_config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            layer = Qwen3TransformerBlock(
                num_attention_heads=model_config.num_attention_heads,
                num_kv_heads=model_config.num_kv_heads,
                hidden_size=model_config.hidden_size,
                head_dim=model_config.head_dim,
                intermediate_size=model_config.intermediate_size,
                rms_norm_eps=model_config.rms_norm_eps,
                wq=linear_weight(f"{prefix}.self_attn.q_proj.weight"),
                wk=linear_weight(f"{prefix}.self_attn.k_proj.weight"),
                wv=linear_weight(f"{prefix}.self_attn.v_proj.weight"),
                wo=linear_weight(f"{prefix}.self_attn.o_proj.weight"),
                q_norm=tensor(f"{prefix}.self_attn.q_norm.weight"),
                k_norm=tensor(f"{prefix}.self_attn.k_norm.weight"),
                w_gate=linear_weight(f"{prefix}.mlp.gate_proj.weight"),
                w_up=linear_weight(f"{prefix}.mlp.up_proj.weight"),
                w_down=linear_weight(f"{prefix}.mlp.down_proj.weight"),
                w_input_layernorm=tensor(f"{prefix}.input_layernorm.weight"),
                w_post_attention_layernorm=tensor(
                    f"{prefix}.post_attention_layernorm.weight"
                ),
                max_seq_len=model_config.max_position_embeddings,
                theta=model_config.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            model_config.hidden_size,
            weight=tensor("model.norm.weight"),
            eps=model_config.rms_norm_eps,
        )
        if not model_config.tie_word_embeddings:
            self.w_lm_head = linear_weight("lm_head.weight")
        else:
            self.w_lm_head = None
        self.state_dict = state_dict

    def __call__(
        self,
        inputs: torch.tensor,
    ) -> torch.tensor:

        h = self.embedding(inputs)

        # FIXME: remove
        # h = torch.randn((1, 32, 128), dtype=torch.float16, device="cuda")
        print("Input after embedding shape:", h.shape)

        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, is_causal=True)
        if self.num_hidden_layers > 0:
            h = self.norm(h)
        batch_size, seq_len, _ = h.shape
        h_flat = h.reshape(batch_size * seq_len, self.hidden_size)
        if self.w_lm_head is not None:
            logits = linear(h_flat, self.w_lm_head, BLOCK_M=16, BLOCK_N=64, BLOCK_K=64)
            return logits.reshape(batch_size, seq_len, self.vocab_size)
        else:
            return self.embedding.as_linear(h)
