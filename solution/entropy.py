import torch
import triton
import triton.language as tl


@triton.jit
def entropy_kernel(X, OUT, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = X + row.to(tl.int64) * V
    m = tl.max(tl.full([BLOCK], -3.0e38, tl.float32), 0)
    for off in range(0, V, BLOCK):
        col = off + tl.arange(0, BLOCK)
        x = tl.load(base + col, mask=col < V, other=-3.0e38).to(tl.float32)
        m = tl.maximum(m, tl.max(x, 0))
    s = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    t = tl.sum(tl.zeros([BLOCK], tl.float32), 0)
    for off in range(0, V, BLOCK):
        col = off + tl.arange(0, BLOCK)
        ok = col < V
        x = tl.load(base + col, mask=ok, other=-3.0e38).to(tl.float32)
        e = tl.where(ok, tl.exp(x - m), 0.0)
        s += tl.sum(e, 0)
        t += tl.sum(e * tl.where(ok, x, 0.0), 0)
    tl.store(OUT + row, m + tl.log(s) - t / s)


def own_entropy(logits):
    if not logits.is_cuda:
        return _orig(logits)
    shape = logits.shape
    v = shape[-1]
    x = logits.contiguous().view(-1, v)
    out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    entropy_kernel[(x.shape[0],)](x, out, v, 4096)
    return out.view(shape[:-1])


_orig = None


def patch_entropy():
    global _orig
    import trl.trainer.sft_trainer as st
    _orig = st.entropy_from_logits
    st.entropy_from_logits = own_entropy
