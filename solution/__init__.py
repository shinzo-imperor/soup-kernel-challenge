"""Own Triton kernels."""

from . import swiglu, rmsnorm, entropy, nf4


def apply():
    swiglu.patch_qwen2_mlp()
    rmsnorm.patch_rmsnorm()
    entropy.patch_entropy()
    nf4.patch_nf4()
