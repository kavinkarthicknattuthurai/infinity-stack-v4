import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. KERNEL BINDINGS (NO FALLBACKS) ---
# If these fail, we want to know immediately. 
# We are here to use the hardware, not simulate it.
try:
    from mamba_ssm import Mamba
    from flash_attn import flash_attn_func
except ImportError as e:
    raise ImportError("❌ CRITICAL: High-Performance Kernels not found. "
                      "This code requires 'mamba-ssm' and 'flash-attn'. "
                      "Are you on the H200?") from e

print("✅ SYSTEM: H200 Kernels Locked & Loaded.")

# --- 2. CONFIGURATION ---
class ModelConfig:
    def __init__(self, vocab_size=50304, d_model=1024, n_layers=3, ctx_len=2048):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_physical_layers = n_layers 
        self.loops_per_layer = [1, 8, 1]  # The Cyclic Schedule
        self.ctx_len = ctx_len
        self.dropout = 0.0 # FlashAttn handles dropout internally if needed

# --- 3. OPTIMIZED COMPONENTS ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # We use torch.rsqrt for speed
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class KernelMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Direct binding to the CUDA Kernel
        self.mixer = Mamba(
            d_model=config.d_model,
            d_state=16,  # Standard State Expansion
            d_conv=4,    # Local Conv Width
            expand=2     # Block Expansion
        )

    def forward(self, x):
        # Mamba kernel expects [Batch, Seq, Dim]
        return self.mixer(x)

class KernelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = 16 
        self.head_dim = self.d_model // self.n_head
        
        # Fused QKV Projection (Faster than 3 separate linears)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        B, T, C = x.size()
        
        # 1. Project QKV
        qkv = self.qkv(x)
        
        # 2. Reshape for FlashAttention Layout [Batch, Seq, 3, Heads, HeadDim]
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)
        
        # 3. Unbind directly (Zero-copy view)
        q, k, v = qkv.unbind(2)
        
        # 4. THE KERNEL CALL
        # This bypasses PyTorch memory management and goes straight to SRAM
        y = flash_attn_func(
            q, k, v, 
            dropout_p=0.0, 
            causal=True, # Auto-masking
            window_size=(-1, -1) # Infinite context window
        )
        
        # 5. Reshape and Project
        y = y.reshape(B, T, C)
        return self.c_proj(y)

class StandardMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard MLP, but we will compile this graph later
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(), 
            nn.Linear(4 * config.d_model, config.d_model)
        )
    def forward(self, x):
        return self.net(x)

# --- 4. THE CYCLIC ENGINE (V4) ---

class CyclicBlock(nn.Module):
    def __init__(self, config, stage_type, num_loops):
        super().__init__()
        self.stage_type = stage_type
        self.num_loops = num_loops
        
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        
        # Bind the correct Kernel
        if stage_type == 'mamba': self.mixer = KernelMamba(config)
        elif stage_type == 'attention': self.mixer = KernelAttention(config)
        else: self.mixer = nn.Identity()
        
        # The Cyclic MLPs (Knowledge Bank)
        self.mlp_list = nn.ModuleList([StandardMLP(config) for _ in range(num_loops)])

    def forward_step(self, x, loop_idx, time_vec):
        # 1. Mixer (Kernel Ops)
        # Note: FlashAttention expects specific dtypes (BF16), handled in training loop
        if self.stage_type == 'attention':
             # Time injection
             x_mixed = x + self.mixer(self.ln1(x + time_vec))
        else:
             x_mixed = x + self.mixer(self.ln1(x))
        
        # 2. MLP (Dense Ops)
        # Direct index access is O(1)
        mlp = self.mlp_list[loop_idx]
        x_out = x_mixed + mlp(self.ln2(x_mixed))
        
        return x_out

# --- 5. THE INFINITY STACK ---

class InfinityStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.ctx_len, config.d_model)
        self.loop_emb = nn.Embedding(32, config.d_model)
        
        self.stages = nn.ModuleList()
        # Stage 1: Mamba (1 Loop)
        self.stages.append(CyclicBlock(config, 'mamba', num_loops=1))
        # Stage 2: Attention (8 Loops)
        self.stages.append(CyclicBlock(config, 'attention', num_loops=8))
        # Stage 3: Output MLP (1 Loop)
        self.stages.append(CyclicBlock(config, 'mlp', num_loops=1))
        
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.token_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)

    def forward_stage(self, x, stage_idx):
        layer = self.stages[stage_idx]
        loops = self.config.loops_per_layer[stage_idx]
        curr_x = x
        history = []
        
        for k in range(loops):
            t_vec = self.loop_emb(torch.tensor([k], device=x.device))
            out = layer.forward_step(curr_x, k, t_vec)
            history.append(curr_x)
            curr_x = out
            
        return curr_x, history