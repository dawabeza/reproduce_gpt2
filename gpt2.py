from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size = 1024
    vocab_size = 50257
    n_layer = 12
    n_emb = 768
    n_head = 12

class CasualSelfAttention:
    def __init__(self, config):
        assert config.n_emb % config.n_head == 0, "set the number of head such that it devides the embedding"
        super().__init__()
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)

        self.n_head = config.n_head
        self.n_emb = config.n_emb

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_head, 2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        #let's go deeper to the attention part
        wei = q @ k.transpose(-2, -1)
        wei = torch.masked_fill(wei, self.bias == 0, float("-inf"))
        wei =  F.softmax(wei, -1)

        y = wei @ v
        y = y.transpose(1, 2).contigous().view(B, T, C)
        y = self.c_proj(y)

        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = (4 * config.n_emb, config.n_emb)
    
    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)
    
    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb)
        ))

        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2" : dict(n_layer=12, n_head=12, n_emb=768),
            "gpt2-medium" : dict(n_layer=24, n_head=16, n_emb=1024),
            "gpt2-large" : dict(n_layer=36, n_head=20, n_emb=1280),
            "gpt2-xl" : dict(n_layer=48, n_head=25, n_emb=1600) 
            }[model_type]
        
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config =  GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith("attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith("attn.masked_bias")]
        sd_heys_hf = [k for k in sd_keys_hf if not k.endswith("attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys) == len(sd_heys_hf), "key mismatch in the models"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in  transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
        return model
    
model = GPT.from_pretrained("gpt2")
print("fortunately it works")