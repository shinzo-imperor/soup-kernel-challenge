import torch
import triton
import triton.language as tl

_NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176392, 0.7229568362236023, 1.0]
_lut = {}
STATS = {"calls": 0}


@triton.jit
def nf4_dequant_kernel(A, ABSMAX, LUT, OUT, NBYTES,
                       HALF_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NBYTES
    b = tl.load(A + offs, mask=mask, other=0).to(tl.int32)
    am = tl.load(ABSMAX + offs // HALF_BLOCK, mask=mask, other=0.0)
    hi = tl.load(LUT + (b >> 4))
    lo = tl.load(LUT + (b & 15))
    res = tl.interleave((hi * am).to(tl.float16), (lo * am).to(tl.float16))
    o = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(OUT + o, res, mask=o < 2 * NBYTES)


def _lut_for(device):
    t = _lut.get(device)
    if t is None:
        t = torch.tensor(_NF4, dtype=torch.float32, device=device)
        _lut[device] = t
    return t


def dequant_nf4(A, absmax, blocksize, out):
    nbytes = A.numel()
    grid = (triton.cdiv(nbytes, 1024),)
    nf4_dequant_kernel[grid](A, absmax, _lut_for(A.device), out,
                             nbytes, blocksize // 2, 1024)


def patch_nf4():
    from bitsandbytes.backends.cuda import ops
    orig = ops._dequantize_4bit_impl

    def impl(A, absmax, blocksize, quant_type, dtype, out):
        if (quant_type == "nf4" and dtype == torch.float16
                and A.is_cuda and A.dtype == torch.uint8
                and absmax.dtype == torch.float32
                and A.is_contiguous() and absmax.is_contiguous()
                and out.is_contiguous()
                and A.numel() * 2 == out.numel()):
            STATS["calls"] += 1
            dequant_nf4(A, absmax, blocksize, out)
        else:
            orig(A, absmax, blocksize, quant_type, dtype, out)

    ops._dequantize_4bit_impl = impl
