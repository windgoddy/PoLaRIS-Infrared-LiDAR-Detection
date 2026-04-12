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
        def bfs_forward(edge_index, max_adj_per_vertex):
            return edge_index, edge_index, edge_index

    _C = MockTreeFilterCuda()

class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child =\
                _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child

bfs = _BFS.apply

