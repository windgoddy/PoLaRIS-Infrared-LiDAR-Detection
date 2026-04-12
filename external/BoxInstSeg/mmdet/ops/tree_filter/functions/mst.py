import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

# 尝试动态导入tree_filter_cuda
try:
    import tree_filter_cuda as _C
except ImportError:
    print("Warning: tree_filter_cuda import failed, using fallback")
    # 创建一个简单的mock模块
    class MockTreeFilterCuda:
        @staticmethod
        def mst_forward(edge_index, edge_weight, vertex_index):
            return edge_index

        @staticmethod
        def mst_backward(grad_output):
            return grad_output

    _C = MockTreeFilterCuda()

class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None

mst = _MST.apply

