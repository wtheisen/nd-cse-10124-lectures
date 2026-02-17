import math
import torch
import torch.nn.functional as F
from collections import namedtuple
from .linear_layer import LinearLayer
from .relu import ReLU

class Transformer:
    Grad_Info = namedtuple('Grad_Info', [
        # ---- Attention sub-block cache ----
        'x',           # (B, T, C) original input (before LN1)
        'x_hat',       # (B, T, C) normalized: (X - mu) / sigma
        'x_norm',      # (B, T, C) after gamma/beta scaling
        'sigma',       # (B, T, 1) std dev (for LN1 backward)
        'q', 'k', 'v',                    # (B, T, Dh)
        'scores',                          # (B, T, T)
        'attn',                            # (B, T, T)
        'attn_v',                          # (B, T, Dh)  = attn @ v
        # ---- FFN sub-block cache ----
        'y_attn',      # (B, T, C) output after attention residual (input to FFN sub-block)
        'y_attn_hat',  # (B, T, C) normalized for LN2
        'sigma2',      # (B, T, 1) std dev (for LN2 backward)
    ])

    def __init__(self, model_dim, head_dim=None, ffn_expand=4, device='cpu'):
        """
        Single-head self-attention + FFN with Pre-LayerNorm for stability.
        model_dim: C (embedding size)
        head_dim: Dh (defaults to model_dim)
        ffn_expand: expansion factor for the FFN hidden dimension (default 4x)
        """
        self.C = model_dim
        self.Dh = head_dim if head_dim is not None else model_dim
        assert self.Dh == self.C, "For this simple single-head block, set head_dim == model_dim."

        self.device = device
        self.eps = 1e-5

        # ---- LayerNorm 1 parameters (Pre-LN for Attention) ----
        self.gamma = torch.ones(self.C, device=self.device)
        self.beta  = torch.zeros(self.C, device=self.device)

        # ---- Attention weights (Xavier init, appropriate for attention) ----
        scale_in = math.sqrt(1.0 / self.C)
        self.Wq = torch.randn(self.Dh, self.C, device=self.device) * scale_in
        self.Wk = torch.randn(self.Dh, self.C, device=self.device) * scale_in
        self.Wv = torch.randn(self.Dh, self.C, device=self.device) * scale_in
        self.bq = torch.zeros(self.Dh, device=self.device)
        self.bk = torch.zeros(self.Dh, device=self.device)
        self.bv = torch.zeros(self.Dh, device=self.device)

        self.Wo = torch.randn(self.C, self.Dh, device=self.device) * scale_in
        self.bo = torch.zeros(self.C, device=self.device)

        # ---- LayerNorm 2 parameters (Pre-LN for FFN) ----
        self.gamma2 = torch.ones(self.C, device=self.device)
        self.beta2  = torch.zeros(self.C, device=self.device)

        # ---- FFN: Linear → ReLU → Linear (uses existing classes) ----
        ffn_dim = ffn_expand * model_dim
        self.ffn_linear1 = LinearLayer(model_dim, ffn_dim, device=self.device)
        self.ffn_relu    = ReLU()
        self.ffn_linear2 = LinearLayer(ffn_dim, model_dim, device=self.device)

        self.cache = None   # holds Grad_Info from the last forward

    def forward(self, X, key_pad_mask=None, causal=False):
        """
        X: (B, T, C)
        key_pad_mask: optional (B, T) bool; True where PAD token.
        causal: if True, apply lower-triangular mask.
        Returns: Z = FFN_residual(Attn_residual(X))  (B, T, C)
        """
        B, T, C = X.shape

        # ============================================================
        # Sub-block 1: Pre-LN → Self-Attention → Residual
        # ============================================================

        # ---- Pre-LayerNorm 1 ----
        mu    = X.mean(dim=-1, keepdim=True)                                          # (B, T, 1)
        sigma = torch.sqrt(X.var(dim=-1, keepdim=True, unbiased=False) + self.eps)    # (B, T, 1)
        X_hat  = (X - mu) / sigma                                                     # (B, T, C)
        X_norm = self.gamma * X_hat + self.beta                                        # (B, T, C)

        # ---- Self-Attention (computed from normalized input) ----
        Q = X_norm @ self.Wq.T + self.bq
        K = X_norm @ self.Wk.T + self.bk
        V = X_norm @ self.Wv.T + self.bv

        scores = (Q @ K.transpose(1, 2)) / math.sqrt(self.Dh)

        # Masks (optional)
        if key_pad_mask is not None:
            mask_k = key_pad_mask.unsqueeze(1).expand(B, T, T)
            scores = scores.masked_fill(mask_k, float('-inf'))
        if causal:
            tril = torch.tril(torch.ones(T, T, device=X.device, dtype=torch.bool))
            scores = scores.masked_fill(~tril, float('-inf'))

        attn = F.softmax(scores, dim=-1)   # (B, T, T)
        attn_v = attn @ V                  # (B, T, Dh)

        y_ctx = attn_v @ self.Wo.T + self.bo   # (B, T, C)
        Y_attn = X + y_ctx                     # residual uses ORIGINAL X

        # ============================================================
        # Sub-block 2: Pre-LN → FFN (Linear → ReLU → Linear) → Residual
        # ============================================================

        # ---- Pre-LayerNorm 2 ----
        mu2    = Y_attn.mean(dim=-1, keepdim=True)                                          # (B, T, 1)
        sigma2 = torch.sqrt(Y_attn.var(dim=-1, keepdim=True, unbiased=False) + self.eps)    # (B, T, 1)
        Y_attn_hat  = (Y_attn - mu2) / sigma2                                               # (B, T, C)
        Y_attn_norm = self.gamma2 * Y_attn_hat + self.beta2                                  # (B, T, C)

        # ---- FFN ----
        ffn_out = self.ffn_linear1.forward(Y_attn_norm)   # (B, T, 4C)
        ffn_out = self.ffn_relu.forward(ffn_out)           # (B, T, 4C)
        ffn_out = self.ffn_linear2.forward(ffn_out)        # (B, T, C)

        # ---- Residual ----
        Y = Y_attn + ffn_out

        # Cache for backward
        self.cache = Transformer.Grad_Info(
            x=X, x_hat=X_hat, x_norm=X_norm, sigma=sigma,
            q=Q, k=K, v=V, scores=scores, attn=attn, attn_v=attn_v,
            y_attn=Y_attn, y_attn_hat=Y_attn_hat, sigma2=sigma2
        )
        return Y

    def backward(self, dY, key_pad_mask=None, causal=False):
        """
        dY: gradient wrt output Y, shape (B, T, C)
        Returns: dX (B, T, C)
        """
        B, T, C = dY.shape
        D = C   # feature dimension for LayerNorm
        cache = self.cache

        # ============================================================
        # Sub-block 2 backward: FFN + LN2
        # ============================================================

        # Y = Y_attn + ffn_out → residual splits gradient
        dffn_out = dY.clone()          # gradient into FFN path
        dY_attn  = dY.clone()          # gradient through residual

        # ---- FFN backward (reverse order, classes handle their own caching) ----
        dffn_out = self.ffn_linear2.backward(dffn_out)    # (B, T, 4C)
        dffn_out = self.ffn_relu.backward(dffn_out)       # (B, T, 4C)
        dffn_out = self.ffn_linear1.backward(dffn_out)    # (B, T, C)
        dY_attn_norm = dffn_out

        # ---- LayerNorm 2 backward ----
        dY_attn_hat  = dY_attn_norm * self.gamma2
        self.dgamma2 = (dY_attn_norm * cache.y_attn_hat).sum(dim=(0, 1))    # (C,)
        self.dbeta2  = dY_attn_norm.sum(dim=(0, 1))                           # (C,)

        dY_attn_ln2 = (1.0 / (D * cache.sigma2)) * (
            D * dY_attn_hat
            - dY_attn_hat.sum(dim=-1, keepdim=True)
            - cache.y_attn_hat * (dY_attn_hat * cache.y_attn_hat).sum(dim=-1, keepdim=True)
        )

        # Total gradient w.r.t. Y_attn = residual + through-LN2
        dY_attn += dY_attn_ln2

        # ============================================================
        # Sub-block 1 backward: Attention + LN1
        # ============================================================

        # Y_attn = X + y_ctx → residual splits gradient
        dy_ctx = dY_attn.clone()        # gradient into attention path
        dX     = dY_attn.clone()        # gradient through residual

        # ---- Output projection backward ----
        self.dWo = dy_ctx.reshape(-1, C).T @ cache.attn_v.reshape(-1, self.Dh)
        self.dbo = dy_ctx.sum(dim=(0, 1))
        dattn_v  = dy_ctx @ self.Wo          # (B, T, Dh)

        # ---- Attention backward ----
        dv    = cache.attn.transpose(1, 2) @ dattn_v             # (B, T, Dh)
        dattn = dattn_v @ cache.v.transpose(1, 2)                # (B, T, T)

        # Softmax backward
        tmp     = (dattn * cache.attn).sum(dim=-1, keepdim=True) # (B, T, 1)
        dscores = (dattn - tmp) * cache.attn                     # (B, T, T)

        # Respect masks in backward
        if key_pad_mask is not None:
            mask_k = key_pad_mask.unsqueeze(1).expand(B, T, T)
            dscores = dscores.masked_fill(mask_k, 0.0)
        if causal:
            tril = torch.tril(torch.ones(T, T, device=dY.device, dtype=torch.bool))
            dscores = dscores.masked_fill(~tril, 0.0)

        factor = 1.0 / math.sqrt(self.Dh)
        dqk = dscores * factor                                   # (B, T, T)

        dq = dqk @ cache.k                                       # (B, T, Dh)
        dk = dqk.transpose(1, 2) @ cache.q                       # (B, T, Dh)

        # ---- Q/K/V parameter gradients (w.r.t. X_norm) ----
        self.dWq = dq.reshape(-1, self.Dh).T @ cache.x_norm.reshape(-1, self.C)
        self.dbq = dq.sum(dim=(0, 1))

        self.dWk = dk.reshape(-1, self.Dh).T @ cache.x_norm.reshape(-1, self.C)
        self.dbk = dk.sum(dim=(0, 1))

        self.dWv = dv.reshape(-1, self.Dh).T @ cache.x_norm.reshape(-1, self.C)
        self.dbv = dv.sum(dim=(0, 1))

        # Gradient w.r.t. X_norm from Q, K, V branches
        dX_norm = dq @ self.Wq + dk @ self.Wk + dv @ self.Wv    # (B, T, C)

        # ---- LayerNorm 1 backward ----
        dX_hat      = dX_norm * self.gamma
        self.dgamma = (dX_norm * cache.x_hat).sum(dim=(0, 1))    # (C,)
        self.dbeta  = dX_norm.sum(dim=(0, 1))                     # (C,)

        dX_ln = (1.0 / (D * cache.sigma)) * (
            D * dX_hat
            - dX_hat.sum(dim=-1, keepdim=True)
            - cache.x_hat * (dX_hat * cache.x_hat).sum(dim=-1, keepdim=True)
        )

        # Total: residual + attention-through-LN1
        dX += dX_ln

        return dX

    def update(self, lr):
        # SGD update for attention weights
        self.Wq -= lr * self.dWq; self.bq -= lr * self.dbq
        self.Wk -= lr * self.dWk; self.bk -= lr * self.dbk
        self.Wv -= lr * self.dWv; self.bv -= lr * self.dbv
        self.Wo -= lr * self.dWo; self.bo -= lr * self.dbo
        # SGD update for LayerNorm 1 parameters
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta
        # SGD update for LayerNorm 2 parameters
        self.gamma2 -= lr * self.dgamma2
        self.beta2  -= lr * self.dbeta2
        # SGD update for FFN layers (LinearLayer.update handles its own W and b)
        self.ffn_linear1.update(lr)
        self.ffn_relu.update(lr)
        self.ffn_linear2.update(lr)
