import math
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn

def rms_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x * scale

def _associative_scan_step(elem_i, elem_j):
    a_i, b_i = elem_i
    a_j, b_j = elem_j
    return a_j * a_i, a_j * b_i + b_j

def jax_chunkwise_vla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gammas: jnp.ndarray,
    scale: float,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """
    Hardware-accelerated Chunkwise VLA for Google Cloud TPU v5e MXU arrays.
    
    Splits sequence S into chunks of size C (e.g. 64 or 128 matching TPU systolic array tile size).
    Computes intra-chunk causal attention with O(C^2) memory and updates inter-chunk states
    using jax.lax.scan with zero intermediate memory allocation for the full sequence.
    """
    b, h, s, d = q.shape
    C = chunk_size
    pad_len = (C - (s % C)) % C
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_len), (0, 0)))

    s_p = s + pad_len
    nc = s_p // C

    q_c = jnp.reshape(q, (b, h, nc, C, d))
    k_c = jnp.reshape(k, (b, h, nc, C, d))
    v_c = jnp.reshape(v, (b, h, nc, C, d))

    steps = jnp.arange(C, dtype=jnp.float32)
    diff = steps[:, None] - steps[None, :]
    causal = diff >= 0
    intra_decay = jnp.where(
        causal[None, None, None, :, :],
        gammas[None, :, None, None, None] ** diff[None, None, None, :, :],
        0.0
    ).astype(q.dtype)

    scores = jnp.matmul(q_c, jnp.swapaxes(k_c, -2, -1)) * scale
    o_intra = jnp.matmul(scores * intra_decay, v_c)

    decay_k = (gammas[None, :, None, None, None] ** (C - 1 - steps)[None, None, None, :, None]).astype(q.dtype)
    k_decayed = k_c * decay_k
    kv_chunk = jnp.matmul(jnp.swapaxes(k_decayed, -2, -1), v_c)

    decay_chunk = (gammas[None, :, None, None] ** C).astype(q.dtype)
    decay_q = ((gammas[None, :, None, None, None] ** (steps + 1)[None, None, None, :, None]) * scale).astype(q.dtype)
    q_decayed = q_c * decay_q

    q_dec_t = jnp.transpose(q_decayed, (2, 0, 1, 3, 4))
    kv_t = jnp.transpose(kv_chunk, (2, 0, 1, 3, 4))

    init_state = jnp.zeros((b, h, d, d), dtype=q.dtype)

    def scan_fn(prev_state, xs):
        curr_q, curr_kv = xs
        o_inter_chunk = jnp.matmul(curr_q, prev_state)
        next_state = prev_state * decay_chunk + curr_kv
        return next_state, o_inter_chunk

    _, o_inter_t = jax.lax.scan(scan_fn, init_state, (q_dec_t, kv_t))
    o_inter = jnp.transpose(o_inter_t, (1, 2, 0, 3, 4))

    out = jnp.reshape(o_intra + o_inter, (b, h, s_p, d))
    if pad_len > 0:
        out = out[:, :, :s, :]
    return out


def vla_associative_scan(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gammas: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    # Full associative scan fallback for small sequences
    b, h, s, d = q.shape
    a = jnp.broadcast_to(gammas[None, :, None, None, None], (b, h, s, 1, 1))
    b_outer = jnp.einsum('bhsd,bhsm->bhsdm', k, v)
    _, s_all = jax.lax.associative_scan(_associative_scan_step, (a, b_outer), axis=2)
    out = jnp.einsum('bhsd,bhsdm->bhsm', q, s_all) * scale
    return out


class JAXVectorLinearAttention(nn.Module):
    in_dim: int
    mem_dim: int
    num_heads: int = 4
    rms_eps: float = 1e-5
    gamma_min: float = 0.90
    gamma_max: float = 0.98
    chunk_size: int = 64

    def setup(self):
        self.head_dim = self.mem_dim // self.num_heads
        self.w_q = nn.Dense(self.mem_dim, use_bias=True)
        self.w_k = nn.Dense(self.mem_dim, use_bias=True)
        self.w_v = nn.Dense(self.mem_dim, use_bias=True)
        self.w_gate = nn.Dense(self.mem_dim, use_bias=True)
        self.w_out = nn.Dense(self.mem_dim, use_bias=True)

    def get_gammas(self) -> jnp.ndarray:
        h = self.num_heads
        if h <= 1:
            return jnp.array([self.gamma_min], dtype=jnp.float32)
        return jnp.linspace(self.gamma_min, self.gamma_max, h, dtype=jnp.float32)

    def forward_parallel(self, c: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b, s, _ = c.shape
        h, d = self.num_heads, self.head_dim

        q_raw = self.w_q(c)
        k_raw = self.w_k(c)
        v_raw = self.w_v(c)

        q = rms_norm(q_raw, self.rms_eps)
        k = rms_norm(k_raw, self.rms_eps)

        q = jnp.transpose(jnp.reshape(q, (b, s, h, d)), (0, 2, 1, 3))
        k = jnp.transpose(jnp.reshape(k, (b, s, h, d)), (0, 2, 1, 3))
        v = jnp.transpose(jnp.reshape(v_raw, (b, s, h, d)), (0, 2, 1, 3))

        gammas = self.get_gammas()
        scale = 1.0 / math.sqrt(d)

        # Chunkwise VLA execution for peak TPU MXU hardware efficiency
        o_heads = jax_chunkwise_vla(q, k, v, gammas, scale, chunk_size=self.chunk_size)
        o_flat = jnp.reshape(jnp.transpose(o_heads, (0, 2, 1, 3)), (b, s, h * d))
        o_norm = rms_norm(o_flat, self.rms_eps)
        o_out = self.w_out(o_norm)

        v_norm = rms_norm(v_raw, self.rms_eps)
        v_pred = rms_norm(o_flat, self.rms_eps)
        e_vla = v_norm - v_pred
        fe_vla = jnp.mean(jnp.square(e_vla))

        return o_out, fe_vla

    def forward_step(self, c_t: jnp.ndarray, s_prev: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        b = c_t.shape[0]
        h, d = self.num_heads, self.head_dim

        q_raw = self.w_q(c_t)
        k_raw = self.w_k(c_t)
        v_raw = self.w_v(c_t)
        g_raw = jax.nn.sigmoid(self.w_gate(c_t))

        q = rms_norm(q_raw, self.rms_eps)
        k = rms_norm(k_raw, self.rms_eps)

        q_h = jnp.reshape(q, (b, h, 1, d))
        k_h = jnp.reshape(k, (b, h, d, 1))
        v_h = jnp.reshape(v_raw, (b, h, 1, d))
        g_h = jnp.reshape(g_raw, (b, h, d, 1))

        s_decayed = s_prev * g_h
        kv_outer = jnp.matmul(k_h, v_h)
        s_new = s_decayed + kv_outer

        o_h = jnp.squeeze(jnp.matmul(q_h, s_new), axis=-2)
        o_flat = jnp.reshape(o_h, (b, h * d))
        o_norm = rms_norm(o_flat, self.rms_eps)
        o_out = self.w_out(o_norm)

        v_pred = rms_norm(jnp.reshape(jnp.squeeze(jnp.matmul(q_h, s_prev), axis=-2), (b, h * d)), self.rms_eps)
        e_vla = rms_norm(v_raw, self.rms_eps) - v_pred
        fe_vla = jnp.mean(jnp.square(e_vla))

        return o_out, s_new, fe_vla
