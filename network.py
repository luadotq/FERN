from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import HierarchicalLayer, rms_norm
from .vla import VectorLinearAttention
from .pe import sinusoidal_pe, apply_rope

@dataclass
class NetworkState:
    mu: Optional[List[torch.Tensor]] = None
    sigma2: Optional[List[torch.Tensor]] = None
    memory: Optional[torch.Tensor] = None
    s_vla: Optional[torch.Tensor] = None
    s_vla_layers: Optional[List[torch.Tensor]] = None
    delay_step: int = 0


class FERNBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, config: ModelConfig):
        super().__init__()
        self.d_model = d_model
        self.rms_eps = config.rms_eps
        self.vla = VectorLinearAttention(
            in_dim=d_model,
            mem_dim=d_model,
            num_heads=num_heads,
            rms_eps=config.rms_eps,
            gamma_min=config.gamma_min,
            gamma_max=config.gamma_max,
            chunk_size=getattr(config, "chunk_size", 64),
        )
        mlp_dim = int(d_model * 2.5)
        self.w_up = nn.Linear(d_model, mlp_dim, bias=False)
        self.act = nn.GELU()
        self.w_down = nn.Linear(mlp_dim, d_model, bias=False)
        self._init_weights(config.init_range)

    def _init_weights(self, r: float):
        nn.init.normal_(self.w_up.weight, mean=0.0, std=r)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=r)

    def forward_parallel(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = rms_norm(x, self.rms_eps)
        vla_out, fe_vla = self.vla.forward_parallel(norm_x)
        h = x + vla_out

        norm_h = rms_norm(h, self.rms_eps)
        mlp_out = self.w_down(self.act(self.w_up(norm_h)))
        out = h + mlp_out

        p_err = norm_h - rms_norm(out, self.rms_eps)
        fe_pc = (p_err ** 2).mean()

        return out, fe_vla + fe_pc

    def forward_step(self, x_t: torch.Tensor, s_vla: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_x = rms_norm(x_t, self.rms_eps)
        vla_out, new_s_vla, fe_vla = self.vla.forward_step(norm_x, s_vla)
        h = x_t + vla_out

        norm_h = rms_norm(h, self.rms_eps)
        mlp_out = self.w_down(self.act(self.w_up(norm_h)))
        out = h + mlp_out

        p_err = norm_h - rms_norm(out, self.rms_eps)
        fe_pc = (p_err ** 2).mean()

        return out, new_s_vla, fe_vla + fe_pc


class FERNModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.is_deep = (config.num_layers is not None) or (config.d_model is not None)
        if self.is_deep:
            self.num_layers = config.num_layers or 7
            self.d_model = config.d_model or 448
            self.num_heads = config.num_heads
            self.head_dim = self.d_model // self.num_heads

            self.encoder = nn.Embedding(config.vocab_size, self.d_model)
            self.pad_neutral = nn.Parameter(torch.zeros(1, self.d_model))
            self.blocks = nn.ModuleList([
                FERNBlock(self.d_model, self.num_heads, config)
                for _ in range(self.num_layers)
            ])
            self.decoder = nn.Linear(self.d_model, config.vocab_size, bias=False)
            self.decoder.weight = self.encoder.weight
            self._reset_parameters()
        else:
            d_layers = config.d_layers
            assert len(d_layers) >= 2, "d_layers must have at least 2 layers"

            self.encoder = nn.Embedding(config.vocab_size, d_layers[0])
            self.pad_neutral = nn.Parameter(torch.zeros(1, d_layers[0]))

            self.num_layers = len(d_layers) - 1
            for i in range(1, len(d_layers)):
                setattr(self, f"layer_{i}", HierarchicalLayer(
                    l=i,
                    d_sensory=d_layers[0],
                    d_prev=d_layers[i - 1],
                    d_curr=d_layers[i]
                ))

            total_belief_dim = sum(d_layers[1:])
            d_mem = d_layers[-1]
            self.vla = VectorLinearAttention(
                in_dim=total_belief_dim,
                mem_dim=d_mem,
                num_heads=config.num_heads,
                rms_eps=config.rms_eps,
                gamma_min=config.gamma_min,
                gamma_max=config.gamma_max,
                chunk_size=getattr(config, "chunk_size", 64),
            )
            self.decoder = nn.Linear(total_belief_dim + d_mem, config.vocab_size)
            self._reset_parameters()

    @property
    def layer_list(self) -> List[HierarchicalLayer]:
        if self.is_deep:
            return []
        return [getattr(self, f"layer_{i}") for i in range(1, self.num_layers + 1)]

    def _reset_parameters(self):
        r = self.config.init_range
        nn.init.normal_(self.encoder.weight, mean=0.0, std=r)
        if not self.is_deep:
            nn.init.uniform_(self.decoder.weight, -r, r)
            nn.init.zeros_(self.decoder.bias)

    def init_state(self, batch_size: int = 1, device=None) -> NetworkState:
        device = device or next(self.parameters()).device
        if self.is_deep:
            h, d = self.num_heads, self.head_dim
            s_vla_layers = [
                torch.zeros((batch_size, h, d, d), device=device)
                for _ in range(self.num_layers)
            ]
            return NetworkState(s_vla_layers=s_vla_layers, s_vla=s_vla_layers[-1], delay_step=0)

        d_layers = self.config.d_layers
        d_mem = self.config.d_mem
        num_heads = self.config.num_heads
        head_dim = d_mem // num_heads

        mu = [torch.zeros((batch_size, d), device=device) for d in d_layers]
        sigma2 = [torch.ones((batch_size, d), device=device) for d in d_layers]
        memory = torch.zeros((batch_size, d_mem), device=device)
        s_vla = torch.zeros((batch_size, num_heads, head_dim, head_dim), device=device)
        return NetworkState(mu=mu, sigma2=sigma2, memory=memory, s_vla=s_vla, delay_step=0)

    def forward_parallel(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = self.config.rms_eps
        if self.is_deep:
            h = self.encoder(tokens)
            total_fe = torch.zeros((), device=tokens.device, dtype=h.dtype)
            use_ckpt = self.training and getattr(self.config, "gradient_checkpointing", False)
            for block in self.blocks:
                if use_ckpt:
                    import torch.utils.checkpoint as cp
                    h, fe_l = cp.checkpoint(block.forward_parallel, h, use_reentrant=False)
                else:
                    h, fe_l = block.forward_parallel(h)
                total_fe = total_fe + fe_l
            avg_fe = total_fe / len(self.blocks)
            logits = self.decoder(rms_norm(h, eps))
            return logits, avg_fe

        embeddings = self.encoder(tokens)
        layers = self.layer_list
        num_layers = len(layers)
        beliefs = [embeddings]
        total_fe = torch.zeros((), device=tokens.device, dtype=embeddings.dtype)

        for l in range(num_layers):
            prev_mu = beliefs[l]
            layer = layers[l]
            up = layer.W_up(prev_mu)
            beliefs.append(rms_norm(up, eps))

        for l in range(num_layers):
            layer = layers[l]
            higher_mu = beliefs[l + 1]
            lower_mu = beliefs[l]
            p_hat = layer.f_pred(higher_mu)
            err = lower_mu - p_hat
            total_fe = total_fe + (err ** 2).mean()

        concat = torch.cat(beliefs[1:], dim=-1)
        m_out, fe_vla = self.vla.forward_parallel(concat)
        total_fe = total_fe + fe_vla
        avg_fe = total_fe / (num_layers + 1)

        dec_in = torch.cat([concat, m_out], dim=-1)
        logits = self.decoder(dec_in)
        return logits, avg_fe

    def forward_step(self, token_t: torch.Tensor, state: NetworkState) -> Tuple[torch.Tensor, NetworkState]:
        b = token_t.shape[0]
        device = token_t.device
        eps = self.config.rms_eps

        if self.is_deep:
            is_pad = (token_t == self.config.pad_token_id)
            if is_pad.any():
                state.delay_step += 1
                pe = sinusoidal_pe(state.delay_step, self.d_model, device=device)
                e_t = self.pad_neutral + self.config.pe_scale * pe
            else:
                state.delay_step = 0
                e_t = self.encoder(token_t).view(b, self.d_model)

            h = e_t
            if state.s_vla_layers is None:
                state.s_vla_layers = [None] * self.num_layers
            new_s_layers = []
            for l, block in enumerate(self.blocks):
                h, new_s, _ = block.forward_step(h, state.s_vla_layers[l])
                new_s_layers.append(new_s)

            state.s_vla_layers = new_s_layers
            state.s_vla = new_s_layers[-1]
            logits = self.decoder(rms_norm(h, eps)).unsqueeze(1)
            return logits, state

        d_0 = self.config.d_layers[0]
        is_pad = (token_t == self.config.pad_token_id)
        if is_pad.any():
            state.delay_step += 1
            pe = sinusoidal_pe(state.delay_step, d_0, device=device)
            e_t = self.pad_neutral + self.config.pe_scale * pe
        else:
            state.delay_step = 0
            e_t = self.encoder(token_t).view(b, d_0)

        next_mu = [e_t]
        for l, layer in enumerate(self.layer_list):
            up = layer.W_up(next_mu[l])
            rec = layer.W_rec(state.mu[l + 1])
            next_mu.append(rms_norm(up + rec, eps))
        state.mu = next_mu

        roped_beliefs = []
        for l in range(1, len(self.config.d_layers)):
            mu_l = state.mu[l]
            if self.config.adaptive_k and state.delay_step > 0:
                mu_l = apply_rope(mu_l, state.delay_step)
            roped_beliefs.append(mu_l)

        concat = torch.cat(roped_beliefs, dim=-1)
        m_out, new_s_vla, _ = self.vla.forward_step(concat, state.s_vla)
        state.memory = m_out
        state.s_vla = new_s_vla

        dec_in = torch.cat([concat, m_out], dim=-1)
        logits = self.decoder(dec_in).unsqueeze(1)
        return logits, state

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        fe_weight: Optional[float] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if fe_weight is None:
            fe_weight = self.config.fe_weight

        logits, avg_fe = self.forward_parallel(tokens)

        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            total_loss = ce_loss + fe_weight * avg_fe
            return logits, total_loss, ce_loss, avg_fe
        return logits, avg_fe

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> NetworkState:
        if self.is_deep:
            return NetworkState(
                s_vla_layers=[None] * self.num_layers,
                delay_step=0
            )
        else:
            return NetworkState(
                mu=[torch.zeros(batch_size, d, device=device) for d in self.config.d_layers],
                sigma2=[torch.ones(batch_size, d, device=device) for d in self.config.d_layers],
                memory=torch.zeros(batch_size, self.config.d_mem, device=device),
                s_vla=torch.zeros(batch_size, self.config.num_heads, self.config.d_mem // self.config.num_heads, self.config.d_mem // self.config.num_heads, device=device) if self.config.legacy_ampc else None,
                delay_step=0
            )

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        self.eval()
        device = next(self.parameters()).device
        state = self.init_state(batch_size=1, device=device)
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        for t in range(prompt_tensor.shape[1]):
            tok_t = prompt_tensor[:, t:t + 1]
            logits, state = self.forward_step(tok_t, state)

        generated = []
        history = list(prompt_tokens)
        curr_logits = logits.squeeze(1)

        for _ in range(max_tokens):
            logits_step = curr_logits[0].clone()

            if repetition_penalty > 1.0:
                seen = set(history[-64:])
                for tid in seen:
                    if tid < logits_step.shape[0]:
                        if logits_step[tid] < 0:
                            logits_step[tid] *= repetition_penalty
                        else:
                            logits_step[tid] /= repetition_penalty

            if temperature > 0:
                logits_step = logits_step / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                    logits_step[logits_step < v[-1]] = -float('inf')
                probs = F.softmax(logits_step, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits_step).item()

            generated.append(next_token)
            history.append(next_token)

            if next_token == eos_id:
                break

            tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            curr_logits_out, state = self.forward_step(tok_tensor, state)
            curr_logits = curr_logits_out.squeeze(1)

        return generated
