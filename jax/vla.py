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

def vla_associative_scan(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gammas: jnp.ndarray,
    scale: float,
) -> jnp.ndarray:
    # q, k, v shape: [B, H, S, D]
    b, h, s, d = q.shape
    a = jnp.broadcast_to(gammas[None, :, None, None, None], (b, h, s, 1, 1))
    b_outer = jnp.einsum('bhsd,bhsm->bhsdm', k, v)

    # O(log S) parallel scan on TPU MXUs
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

        o_heads = vla_associative_scan(q, k, v, gammas, scale)
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
