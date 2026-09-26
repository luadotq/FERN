from typing import List, Tuple, Optional, Dict, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from safetensors import safe_open

from ..config import ModelConfig
from .vla import JAXVectorLinearAttention, rms_norm

class JAXMLP(nn.Module):
    out_dim: int
    mid_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.mid_dim, name="linear1")(x)
        h = nn.relu(h)
        return nn.Dense(self.out_dim, name="linear2")(h)


class JAXHierarchicalLayer(nn.Module):
    l: int
    d_sensory: int
    d_prev: int
    d_curr: int

    def setup(self):
        mid_pred = max(self.d_curr + self.d_prev, 32)
        mid_phi = max(self.d_sensory + 2 * self.d_curr, 32)

        self.f_pred = JAXMLP(out_dim=self.d_prev, mid_dim=mid_pred)
        self.W_up = nn.Dense(self.d_curr)
        self.W_gate = nn.Dense(self.d_curr)
        self.W_rec = nn.Dense(self.d_curr, use_bias=False)
        self.q_phi = JAXMLP(out_dim=self.d_curr, mid_dim=mid_phi)

    def __call__(self, prev_mu: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
        up = self.W_up(prev_mu)
        return rms_norm(up, eps)


class JAXFERNModel(nn.Module):
    config: ModelConfig

    def setup(self):
        cfg = self.config
        d_layers = cfg.d_layers
        self.encoder = nn.Embed(num_embeddings=cfg.vocab_size, features=d_layers[0])
        self.pad_neutral = self.param("pad_neutral", lambda rng, shape: jnp.zeros(shape), (1, d_layers[0]))

        self.layers = [
            JAXHierarchicalLayer(
                l=i,
                d_sensory=d_layers[0],
                d_prev=d_layers[i - 1],
                d_curr=d_layers[i],
                name=f"layer_{i}"
            )
            for i in range(1, len(d_layers))
        ]

        total_belief_dim = sum(d_layers[1:])
        d_mem = d_layers[-1]
        self.vla = JAXVectorLinearAttention(
            in_dim=total_belief_dim,
            mem_dim=d_mem,
            num_heads=cfg.num_heads,
            rms_eps=cfg.rms_eps,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
        )
        self.decoder = nn.Dense(cfg.vocab_size)

    def forward_parallel(self, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        eps = self.config.rms_eps
        embeddings = self.encoder(tokens)
        num_layers = len(self.layers)
        beliefs = [embeddings]
        total_fe = 0.0

        for l in range(num_layers):
            prev_mu = beliefs[l]
            up = self.layers[l].W_up(prev_mu)
            beliefs.append(rms_norm(up, eps))

        for l in range(num_layers):
            higher_mu = beliefs[l + 1]
            lower_mu = beliefs[l]
            p_hat = self.layers[l].f_pred(higher_mu)
            err = lower_mu - p_hat
            total_fe = total_fe + jnp.mean(jnp.square(err))

        concat = jnp.concatenate(beliefs[1:], axis=-1)
        m_out, fe_vla = self.vla.forward_parallel(concat)
        total_fe = total_fe + fe_vla
        avg_fe = total_fe / (num_layers + 1)

        dec_in = jnp.concatenate([concat, m_out], axis=-1)
        logits = self.decoder(dec_in)
        return logits, avg_fe


def load_safetensors_to_jax(path: str, config: ModelConfig) -> Tuple[JAXFERNModel, Dict[str, Any]]:
    model = JAXFERNModel(config=config)
    tensors = {}
    with safe_open(path, framework="np") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    params: Dict[str, Any] = {
        "encoder": {"embedding": jnp.array(tensors["encoder.weight"])},
        "pad_neutral": jnp.array(tensors["pad_neutral"]),
        "decoder": {
            "kernel": jnp.array(tensors["decoder.weight"].T),
            "bias": jnp.array(tensors["decoder.bias"]),
        },
        "vla": {
            "w_q": {"kernel": jnp.array(tensors["vla.w_q.weight"].T), "bias": jnp.array(tensors["vla.w_q.bias"])},
            "w_k": {"kernel": jnp.array(tensors["vla.w_k.weight"].T), "bias": jnp.array(tensors["vla.w_k.bias"])},
            "w_v": {"kernel": jnp.array(tensors["vla.w_v.weight"].T), "bias": jnp.array(tensors["vla.w_v.bias"])},
            "w_gate": {"kernel": jnp.array(tensors["vla.w_gate.weight"].T), "bias": jnp.array(tensors["vla.w_gate.bias"])},
            "w_out": {"kernel": jnp.array(tensors["vla.w_out.weight"].T), "bias": jnp.array(tensors["vla.w_out.bias"])},
        }
    }

    for i in range(1, len(config.d_layers)):
        l_name = f"layer_{i}"
        params[l_name] = {
            "W_up": {
                "kernel": jnp.array(tensors[f"{l_name}.W_up.weight"].T),
                "bias": jnp.array(tensors[f"{l_name}.W_up.bias"]),
            },
            "W_gate": {
                "kernel": jnp.array(tensors[f"{l_name}.W_gate.weight"].T),
                "bias": jnp.array(tensors[f"{l_name}.W_gate.bias"]),
            },
            "W_rec": {
                "kernel": jnp.array(tensors[f"{l_name}.W_rec.weight"].T),
            },
            "f_pred": {
                "linear1": {
                    "kernel": jnp.array(tensors[f"{l_name}.f_pred.linear1.weight"].T),
                    "bias": jnp.array(tensors[f"{l_name}.f_pred.linear1.bias"]),
                },
                "linear2": {
                    "kernel": jnp.array(tensors[f"{l_name}.f_pred.linear2.weight"].T),
                    "bias": jnp.array(tensors[f"{l_name}.f_pred.linear2.bias"]),
                }
            },
            "q_phi": {
                "linear1": {
                    "kernel": jnp.array(tensors[f"{l_name}.q_phi.linear1.weight"].T),
                    "bias": jnp.array(tensors[f"{l_name}.q_phi.linear1.bias"]),
                },
                "linear2": {
                    "kernel": jnp.array(tensors[f"{l_name}.q_phi.linear2.weight"].T),
                    "bias": jnp.array(tensors[f"{l_name}.q_phi.linear2.bias"]),
                }
            }
        }

    return model, {"params": params}
