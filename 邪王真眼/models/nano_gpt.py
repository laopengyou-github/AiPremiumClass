import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict


class SelfAttention(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg['n_embd'],
            num_heads=cfg['n_head'],
            dropout=cfg['dropout'],
            batch_first=True
        )
        
        size = cfg['max_length']
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        self.register_buffer('mask', mask)
        self.resid_dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        T = x.size(1)
        if mask is not None:
            m = mask
            if m.dim() == 3:
                m = m[0]
            attn_mask = ~m[:T, :T].bool()
        else:
            attn_mask = self.mask[:T, :T]
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return self.resid_dropout(attn_output)


class MLP(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        ne = cfg['n_embd']
        self.c_fc = nn.Linear(ne, 4 * ne, bias=True)
        self.c_proj = nn.Linear(4 * ne, ne, bias=True)
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        ne = cfg['n_embd']
        self.ln1 = nn.LayerNorm(ne, eps=1e-5)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(ne, eps=1e-5)
        self.mlp = MLP(cfg)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.wte = nn.Embedding(cfg['vocab_size'], cfg['n_embd'])
        self.wpe = nn.Embedding(cfg['max_length'], cfg['n_embd'])
        self.drop = nn.Dropout(cfg['dropout'])
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg['n_layer'])])
        self.ln_f = nn.LayerNorm(cfg['n_embd'], eps=1e-5)
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)
        
        self.lm_head.weight = self.wte.weight
        self.max_length = cfg['max_length']
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x, tag=None, mask=None) -> Dict[str, Any]:
        T = x.size(1)
        assert T <= self.max_length, "Sequence length exceeds block size"
        pos = torch.arange(T, dtype=torch.long, device=x.device).unsqueeze(0)
        token_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(token_emb + pos_emb)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        output = {'out': logits.permute(0, 2, 1)}
        if tag is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tag.view(-1))
            output['loss'] = loss
        return output
