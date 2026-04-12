import torch
from torch.autograd import Function
# 临时注释掉有问题的导入
# from .pairwise_ext import pairwise_nlog_forward, pairwise_nlog_backward

# 尝试动态导入pairwise_ext
try:
    from .pairwise_ext import pairwise_nlog_forward, pairwise_nlog_backward
except ImportError:
    print("Warning: pairwise_ext import failed, using fallback")
    def pairwise_nlog_forward(pairwise_size, pairwise_dilation, logits):
        # 简单的fallback实现
        return torch.zeros_like(logits)

    def pairwise_nlog_backward(grad_output, logits, pairwise):
        # 简单的fallback实现
        return torch.zeros_like(logits)


class _pairwise_nlog(Function):
    @staticmethod
    def forward(ctx, logits, pairwise_size, pairwise_dilation):
        logits = logits.contiguous()
        pairwise = pairwise_nlog_forward(
            pairwise_size, pairwise_dilation, logits)
        ctx.pairwise_size = pairwise_size
        ctx.pairwise_dilation = pairwise_dilation
        ctx.save_for_backward(logits, pairwise)
        return pairwise

    @staticmethod
    def backward(ctx, g_pairwise):
        g_pairwise = g_pairwise.contiguous()
        logits, pairwise = ctx.saved_tensors
        g_logits = pairwise_nlog_backward(
            ctx.pairwise_size, ctx.pairwise_dilation, logits, pairwise, g_pairwise)
        return g_logits, None, None


pairwise_nlog = _pairwise_nlog.apply
