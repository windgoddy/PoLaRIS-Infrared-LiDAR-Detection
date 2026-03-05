#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三探针特征图可视化 (Mamba-UNet++ LiDAR)
=========================================

探针 A: conv0_0  — 浅层 CNN，看边缘/高亮斑点提取
探针 B: conv4_0  — 深层 Mamba，看长程上下文是否聚焦船目标
探针 C: output   — 最终概率热图 (sigmoid 后，阈值前)

用法:
    python vis_probes.py \
        --checkpoint model_Mamba/result/<实验名>/best_model_*.pth \
        --dataset dataset/Pohang-Canal-3k \
        --split_method 50_50_cat2 \
        --num_samples 6 \
        --output vis_probes_output.png

可选项:
    --split txt       # 用 val 集（默认），或 train
    --gpu 0
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm

# ── 项目内模块 ──────────────────────────────────────────────────────────────
from model.model_Mamba_UNetPP import mamba_unetplusplus
from model.model_DNANet import DNANet, Res_CBAM_block
from model.utils_lidar import PoLaRISTestLoader
from model.load_param_data import load_dataset, load_param


# ══════════════════════════════════════════════════════════════════════════════
# 1. 钩子收集器
# ══════════════════════════════════════════════════════════════════════════════
class ProbeCollector:
    """用 register_forward_hook 收集中间层输出"""

    def __init__(self):
        self.features = {}
        self._handles = []

    def hook(self, name):
        def _hook(module, input, output):
            # 某些层输出 tuple（MambaBlockWrapper 返回 tensor，CNN 返回 tensor）
            if isinstance(output, (list, tuple)):
                output = output[0]
            self.features[name] = output.detach().cpu()
        return _hook

    def register(self, name, module):
        h = module.register_forward_hook(self.hook(name))
        self._handles.append(h)

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.features.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 2. 特征图 → 2D 热图（对所有通道取最大值）
# ══════════════════════════════════════════════════════════════════════════════
def feat_to_heatmap(feat_tensor, target_hw=None, mode='max'):
    """
    Args:
        feat_tensor: (B, C, H, W) 或 (B, H, W, C) 或 (B, 1, H, W)
        target_hw: (H, W) 目标尺寸，None 则不缩放
        mode: 'max' | 'mean' — 跨通道聚合方式
    Returns:
        np.ndarray (H, W) 归一化到 [0, 1]
    """
    t = feat_tensor[0]  # 取 batch 第一张

    # 统一为 (C, H, W)
    if t.dim() == 3 and t.shape[-1] > t.shape[0]:
        # (H, W, C) → (C, H, W)
        t = t.permute(2, 0, 1)

    if mode == 'max':
        hmap = t.max(dim=0).values  # (H, W)
    else:
        hmap = t.mean(dim=0)        # (H, W)

    hmap = hmap.float().numpy()

    # 缩放到目标分辨率
    if target_hw is not None and (hmap.shape[0] != target_hw[0] or hmap.shape[1] != target_hw[1]):
        hmap_t = torch.tensor(hmap).unsqueeze(0).unsqueeze(0)
        hmap_t = F.interpolate(hmap_t, size=target_hw, mode='bilinear', align_corners=False)
        hmap = hmap_t[0, 0].numpy()

    # Min-max 归一化
    lo, hi = hmap.min(), hmap.max()
    if hi > lo:
        hmap = (hmap - lo) / (hi - lo)
    else:
        hmap = np.zeros_like(hmap)

    return hmap


# ══════════════════════════════════════════════════════════════════════════════
# 3. 单图可视化行（5 列）
# ══════════════════════════════════════════════════════════════════════════════
def plot_row(axes, ir_np, lidar_np, hmap_a, hmap_b, hmap_c, mask_np,
             hmap_dna=None, img_id='', threshold=0.3):
    """
    axes: [ax_ir, ax_lidar, ax_gt, ax_a, ax_b, ax_c]  (+ optional ax_dna)
    """
    H, W = ir_np.shape

    # ── IR 原图 ──
    axes[0].imshow(ir_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'IR\n{img_id}', fontsize=7)
    axes[0].axis('off')

    # ── LiDAR 深度图 ──
    lidar_has_data = lidar_np is not None and lidar_np.max() > 0
    if lidar_np is not None:
        vmax_lidar = np.percentile(lidar_np[lidar_np > 0], 95) if lidar_has_data else 1.0
        axes[1].imshow(lidar_np, cmap='plasma', vmin=0, vmax=vmax_lidar)
        if not lidar_has_data:
            axes[1].text(W / 2, H / 2, 'No LiDAR\nreturns\n(distant target)',
                         ha='center', va='center', fontsize=7,
                         color='white', bbox=dict(boxstyle='round', fc='#333333', alpha=0.7))
        axes[1].set_title('LiDAR depth\n(normalized)', fontsize=7)
    else:
        axes[1].imshow(np.zeros((H, W)), cmap='gray')
        axes[1].text(W / 2, H / 2, 'No LiDAR\nchannel',
                     ha='center', va='center', fontsize=7,
                     color='white', bbox=dict(boxstyle='round', fc='#333333', alpha=0.7))
        axes[1].set_title('LiDAR\n(N/A)', fontsize=7)
    axes[1].axis('off')

    # ── GT 标注 ──（IR 底图 + 红色掩码边框）
    axes[2].imshow(ir_np, cmap='gray', vmin=0, vmax=1)
    if mask_np is not None and mask_np.max() > 0:
        # 红色半透明遮罩
        mask_rgba = np.zeros((H, W, 4), dtype=np.float32)
        mask_rgba[..., 0] = 1.0  # R
        mask_rgba[..., 3] = (mask_np > 0.5).astype(np.float32) * 0.55
        axes[2].imshow(mask_rgba)
        # 白色轮廓
        axes[2].contour(mask_np, levels=[0.5], colors='yellow', linewidths=0.9)
    axes[2].set_title('GT Annotation\n(red=target)', fontsize=7)
    axes[2].axis('off')

    # ── 探针 A：浅层 CNN (conv0_0) ──
    im_a = axes[3].imshow(hmap_a, cmap='hot',
                           norm=PowerNorm(gamma=0.5, vmin=0, vmax=1))
    axes[3].set_title('Probe A\nconv0_0 (shallow CNN)', fontsize=7)
    axes[3].axis('off')

    # ── 探针 B：深层 Mamba (conv4_0) ──
    im_b = axes[4].imshow(hmap_b, cmap='hot',
                           norm=PowerNorm(gamma=0.5, vmin=0, vmax=1))
    axes[4].set_title('Probe B\nconv4_0 (deep Mamba)', fontsize=7)
    axes[4].axis('off')

    # ── 探针 C：最终概率热图 ──
    im_c = axes[5].imshow(hmap_c, cmap='jet', vmin=0, vmax=1)
    # 在热图上叠加阈值等高线
    axes[5].contour(hmap_c, levels=[threshold], colors='white', linewidths=0.8, linestyles='--')
    # 同时叠加 GT 轮廓（黄线），方便对比检测位置 vs 真值
    if mask_np is not None and mask_np.max() > 0:
        axes[5].contour(mask_np, levels=[0.5], colors='yellow', linewidths=0.9, linestyles='-')
    axes[5].set_title(f'Probe C (Mamba)\nOutput \u03c3 (thr={threshold:.2f}, GT=yellow)', fontsize=7)
    axes[5].axis('off')

    # 叠加 IR 轮廓帮助对位
    for ax, hmap in [(axes[3], hmap_a), (axes[4], hmap_b), (axes[5], hmap_c)]:
        ax.imshow(ir_np, cmap='gray', alpha=0.25, vmin=0, vmax=1)

    # ── DNANet 对比输出（可选第 7 列）──
    if len(axes) > 6:
        ax_dna = axes[6]
        if hmap_dna is not None:
            ax_dna.imshow(ir_np, cmap='gray', alpha=0.25, vmin=0, vmax=1)
            ax_dna.imshow(hmap_dna, cmap='jet', vmin=0, vmax=1)
            ax_dna.contour(hmap_dna, levels=[threshold], colors='white',
                           linewidths=0.8, linestyles='--')
            if mask_np is not None and mask_np.max() > 0:
                ax_dna.contour(mask_np, levels=[0.5], colors='yellow',
                               linewidths=0.9, linestyles='-')
            ax_dna.set_title(f'DNANet Output \u03c3\n(thr={threshold:.2f}, GT=yellow)', fontsize=7)
        else:
            ax_dna.text(0.5, 0.5, 'No DNANet\ncheckpoint',
                        ha='center', va='center', transform=ax_dna.transAxes,
                        fontsize=8, color='gray')
        ax_dna.axis('off')


# ══════════════════════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 加载 checkpoint ──────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # 兼容各种 checkpoint 格式（参考 test_box_iou.py 第354行）
    state_dict = ckpt.get('state_dict') or ckpt.get('model_state_dict')
    if state_dict is None:
        state_dict = ckpt

    # 从 state_dict keys/shapes 自动推断模型架构参数
    use_lidar = any('lidar_gate' in k for k in state_dict.keys())
    use_ds    = any(k.startswith('final1') or k.startswith('final2') for k in state_dict.keys())

    # 检测 lidar_gate_conv 的 kernel size（k=3 → shape[2]=3, k=7 → shape[2]=7）
    gate_kernel = 3  # 默认 Phase1
    gate_key = 'conv2_0.mamba.ss2d.lidar_gate_conv.0.weight'
    if use_lidar and gate_key in state_dict:
        gate_kernel = state_dict[gate_key].shape[2]

    # 检测是否有 lidar_gate_scale（Phase2 新增）
    has_gate_scale = any('lidar_gate_scale' in k for k in state_dict.keys())

    print(f'  use_lidar={use_lidar}, deep_supervision={use_ds}')
    if use_lidar:
        print(f'  gate_kernel={gate_kernel}, has_gate_scale={has_gate_scale}')

    # 用 monkeypatch 让 SS2D 用 checkpoint 对应的参数初始化
    # 这样避免 strict=False 导致 gate 权重乱掉
    if use_lidar:
        import model_Mamba.core.ss2d_components as _ss2d_mod
        _orig_init = _ss2d_mod.SS2D.__init__

        def _patched_init(self_inner, d_model, d_state=16, expand=2,
                          dt_rank='auto', use_lidar_gate=True):
            _orig_init(self_inner, d_model, d_state, expand, dt_rank, use_lidar_gate)
            if use_lidar_gate and hasattr(self_inner, 'lidar_gate_conv'):
                # 替换成 checkpoint 对应的 kernel size
                self_inner.lidar_gate_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(1, self_inner.d_inner,
                                    kernel_size=gate_kernel,
                                    padding=gate_kernel // 2, bias=True),
                    torch.nn.BatchNorm2d(self_inner.d_inner),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(self_inner.d_inner, self_inner.d_inner,
                                    kernel_size=1, bias=True),
                )
                # 确保 lidar_gate_scale 存在（服务器 Phase1 代码可能没有此参数）
                if not hasattr(self_inner, 'lidar_gate_scale'):
                    self_inner.lidar_gate_scale = torch.nn.Parameter(torch.ones(1))

        _ss2d_mod.SS2D.__init__ = _patched_init

    model = mamba_unetplusplus(
        in_channels=1,
        num_classes=1,
        deep_supervision=use_ds,
        use_lidar=use_lidar,
    )

    # 恢复原始 SS2D.__init__（避免影响后续代码）
    if use_lidar:
        _ss2d_mod.SS2D.__init__ = _orig_init

    # strict=False：忽略 lidar_gate_scale missing（checkpoint 无此 key 时保留默认值）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  [strict=False] missing keys: {missing}')
    if unexpected:
        print(f'  [strict=False] unexpected keys: {unexpected}')

    # Phase1 checkpoint 没有 lidar_gate_scale，等效 scale=1.0
    # 手动设回，保证推理行为与训练时一致
    if use_lidar and not has_gate_scale:
        for m in model.modules():
            if hasattr(m, 'lidar_gate_scale'):
                m.lidar_gate_scale.data.fill_(1.0)
        print('  Set lidar_gate_scale=1.0 (Phase1 checkpoint behavior)')
    model.to(device).eval()

    # ── 加载 DNANet（可选对比模型）────────────────────────────────────────────
    dnanet_model = None
    if getattr(args, 'dnanet_checkpoint', None):
        print(f'Loading DNANet checkpoint: {args.dnanet_checkpoint}')
        dna_ckpt = torch.load(args.dnanet_checkpoint, map_location='cpu')
        dna_sd = dna_ckpt.get('state_dict') or dna_ckpt.get('model_state_dict')
        if dna_sd is None:
            dna_sd = dna_ckpt
        dna_ds = any(k.startswith('final') for k in dna_sd.keys())
        # 从 checkpoint 自动检测 input_channels
        _dna_in_ch = 1
        for _k, _v in dna_sd.items():
            if 'conv0_0' in _k and 'conv1.weight' in _k:
                _dna_in_ch = _v.shape[1]  # (out_ch, in_ch, H, W)
                break
        print(f'  DNANet input_channels={_dna_in_ch} (detected from checkpoint)')
        nb_filter, num_blocks = load_param('three', 'resnet_18')
        dnanet_model = DNANet(
            num_classes=1,
            input_channels=_dna_in_ch,
            block=Res_CBAM_block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=dna_ds,
        )
        dnanet_model.load_state_dict(dna_sd, strict=False)
        dnanet_model.to(device).eval()
        print(f'  DNANet loaded, deep_supervision={dna_ds}')
    collector = ProbeCollector()
    collector.register('probe_A', model.conv0_0)   # 浅层 CNN
    collector.register('probe_B', model.conv4_0)   # 深层 Mamba (MambaBlockWrapper)

    # ── 加载数据 ─────────────────────────────────────────────────────────────
    dataset_dir = os.path.join(args.dataset_root, args.dataset)

    # 始终按 split_method 过滤 val 集；selection_file 只额外做类别筛选
    if hasattr(args, 'split_method') and args.split_method:
        _, val_ids, _ = load_dataset(args.dataset_root, args.dataset, args.split_method)
    else:
        val_ids = None  # PoLaRISTestLoader 会自动扫描

    depth_maps_dir = args.depth_maps_dir if args.depth_maps_dir else None

    testset = PoLaRISTestLoader(
        dataset_dir=dataset_dir,
        img_id=val_ids,
        base_size=256,
        crop_size=256,
        suffix='.png',
        normalize_16bit=False,
        in_channels=2 if use_lidar else 1,
        image_folder='images-8bit',
        depth_maps_dir=depth_maps_dir,
    )

    # ── 样本选取：按 selection_file 类别过滤 ──────────────────────────────────
    cat_map   = {}   # img_stem → cat_int
    include_cats = None

    sel_file = getattr(args, 'selection_file', None)
    if sel_file:
        if not os.path.isfile(sel_file):
            print(f'  [WARN] selection_file not found: {sel_file!r} — falling back to random sampling')
        else:
            inc_str = getattr(args, 'include_cats', '0,1,3')
            include_cats = set(int(c.strip()) for c in inc_str.split(','))
            with open(sel_file, 'r', encoding='utf-8') as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or '|' not in _line or _line.startswith('文件'):
                        continue
                    _parts = _line.split('|')
                    _stem = os.path.splitext(_parts[0].strip())[0]  # strip .png
                    try:
                        _cat = int(_parts[1].strip())
                    except ValueError:
                        continue
                    cat_map[_stem] = _cat
            print(f'  Selection file: {len(cat_map)} entries, '
                  f'include_cats={sorted(include_cats)}')

    # 构建候选 indices，按类别分组
    if cat_map and include_cats is not None:
        filtered: dict = {}
        for i in range(len(testset)):
            stem = os.path.splitext(testset._items[i])[0]
            cat = cat_map.get(stem)
            if cat is not None and cat in include_cats:
                filtered.setdefault(cat, []).append(i)

        if filtered:
            cats_avail = sorted(filtered.keys())
            n_per_cat  = max(1, math.ceil(args.num_samples / max(len(cats_avail), 1)))
            rng = np.random.default_rng(42)
            indices: list = []
            for _cat in cats_avail:
                _pool = filtered[_cat]
                _pick = rng.choice(_pool,
                                   size=min(n_per_cat, len(_pool)),
                                   replace=False).tolist()
                indices.extend(_pick)
            rng.shuffle(indices)
            indices = indices[:args.num_samples]
            print(f'  Per-cat pool: { {c: len(filtered[c]) for c in cats_avail} }')
            print(f'  Selected {len(indices)} samples from cats {cats_avail}')
        else:
            # include_cats 与当前 val 集无交集（如 50_50_cat2 只含 cat2 而 include_cats 排除它）
            # 回退：对 val 集全量随机采样，selection_file 仍用于类别标注
            avail_cats = set()
            for i in range(len(testset)):
                stem = os.path.splitext(testset._items[i])[0]
                c = cat_map.get(stem)
                if c is not None:
                    avail_cats.add(c)
            print(f'  [WARN] include_cats={sorted(include_cats)} has no overlap with '
                  f'val set (available cats: {sorted(avail_cats)}). '
                  f'Falling back to random sampling from full val set.')
            rng = np.random.default_rng(42)
            indices = rng.choice(len(testset),
                                 size=min(args.num_samples, len(testset)),
                                 replace=False).tolist()
    else:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(testset),
                             size=min(args.num_samples, len(testset)),
                             replace=False).tolist()

    n = len(indices)
    print(f'Visualizing {n} samples from val set ({len(testset)} total)')

    # ── 绘图布局（每行 6/7 列，N 行）──────────────────────────────────────────
    n_cols = 7 if dnanet_model is not None else 6
    fig = plt.figure(figsize=(5 * n_cols, 3.5 * n), constrained_layout=False)
    outer = gridspec.GridSpec(n, n_cols, figure=fig, hspace=0.35, wspace=0.04)

    COL_TITLES = [
        'IR Input',
        'LiDAR Depth',
        'GT Annotation',
        'Probe A\nconv0_0 (shallow CNN)',
        'Probe B\nconv4_0 (deep Mamba)',
        f'Probe C (Mamba)\nOutput σ (thr={args.threshold:.2f})',
    ]
    if dnanet_model is not None:
        COL_TITLES.append(f'DNANet Output σ\n(thr={args.threshold:.2f})')

    # ── 逐样本推理 ────────────────────────────────────────────────────────────
    for row_idx, sample_idx in enumerate(indices):
        sample = testset[sample_idx]
        img_id = testset._items[sample_idx]

        # PoLaRISTestLoader.__getitem__ 返回 dict，key='image'
        img_t = sample['image']   # (2, H, W) if use_lidar else (1, H, W)

        # 分离 IR 和 LiDAR
        if use_lidar and img_t.shape[0] == 2:
            ir_t    = img_t[0:1]    # (1, H, W)
            lidar_t = img_t[1:2]    # (1, H, W)
        else:
            ir_t    = img_t[0:1]
            lidar_t = None

        H, W = ir_t.shape[1], ir_t.shape[2]

        # batch dim
        ir_b    = ir_t.unsqueeze(0).to(device)
        lidar_b = lidar_t.unsqueeze(0).to(device) if lidar_t is not None else None

        collector.clear()
        with torch.no_grad():
            output = model(ir_b, lidar_b)
            if isinstance(output, (list, tuple)):
                output = output[0]  # deep_supervision: 取主输出
        # output: (1, 1, H, W), 已经经过 sigmoid

        # ── 提取热图 ──
        ir_np    = ir_t[0].float().numpy()          # (H, W) [0,1]
        lidar_np = lidar_t[0].float().numpy() if lidar_t is not None else None

        hmap_a = feat_to_heatmap(collector.features['probe_A'], target_hw=(H, W), mode='max')
        hmap_b = feat_to_heatmap(collector.features['probe_B'], target_hw=(H, W), mode='max')
        hmap_c = output[0, 0].cpu().float().numpy()  # (H, W) already sigmoided

        # DNANet 对比推理
        # DNANet 使用 ImageNet 归一化（见 visulization.py:33-35）
        # 输入需要：3 通道, Normalize([.485,.456,.406],[.229,.224,.225])
        hmap_dna = None
        if dnanet_model is not None:
            # 1. 重复成 3 通道（IR 灰度 → 伪 RGB）
            dna_in = ir_b.repeat(1, 3, 1, 1)   # (1,3,H,W), values in [0,1]
            # 2. ImageNet 归一化（与 visulization.py 的 transforms.Normalize 一致）
            _mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
            _std  = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
            dna_in = (dna_in - _mean) / _std
            with torch.no_grad():
                dna_out = dnanet_model(dna_in)
                if isinstance(dna_out, (list, tuple)):
                    dna_out = dna_out[-1]  # deep_supervision: take final (deepest) output
                dna_out = torch.sigmoid(dna_out)  # DNANet outputs raw logits
            hmap_dna = dna_out[0, 0].cpu().float().numpy()

        # 提取 GT mask
        mask_raw = sample.get('mask', None)
        if mask_raw is not None:
            if isinstance(mask_raw, torch.Tensor):
                mask_np = mask_raw.squeeze().float().numpy()
            else:
                mask_np = np.array(mask_raw, dtype=np.float32).squeeze()
            # 确保尺寸匹配
            if mask_np.shape != (H, W):
                mask_t = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
                mask_t = F.interpolate(mask_t, size=(H, W), mode='nearest')
                mask_np = mask_t[0, 0].numpy()
        else:
            mask_np = None

        # ── 绘制本行 ──
        axes = [fig.add_subplot(outer[row_idx, col]) for col in range(n_cols)]

        # 第一行加列标题
        if row_idx == 0:
            for col, title in enumerate(COL_TITLES):
                axes[col].set_title(title, fontsize=8, fontweight='bold', pad=4)

        # 类别标签（从 cat_map 获取）
        stem_key = os.path.splitext(img_id)[0]
        cat_label = f'cat{cat_map.get(stem_key, "?")}'  if cat_map else ''

        plot_row(axes, ir_np, lidar_np, hmap_a, hmap_b, hmap_c, mask_np,
                 hmap_dna=hmap_dna, img_id=img_id, threshold=args.threshold)

        # 左侧标签（显示样本序号 + 类别）
        axes[0].set_ylabel(f'#{sample_idx}\n{cat_label}', fontsize=7,
                           rotation=0, labelpad=36, va='center')

        print(f'  [{row_idx+1}/{n}] {img_id}  '
              f'cat={cat_map.get(os.path.splitext(img_id)[0], "?")}  '
              f'probe_A max={hmap_a.max():.3f}  '
              f'probe_B max={hmap_b.max():.3f}  '
              f'Mamba>thr={(hmap_c > args.threshold).sum()}  '
              + (f'DNA>thr={(hmap_dna > args.threshold).sum()}' if hmap_dna is not None else ''))

    # ── 保存 ─────────────────────────────────────────────────────────────────
    plt.suptitle(
        f'Three-Probe Visualization  |  ckpt: {os.path.basename(args.checkpoint)}',
        fontsize=10, y=1.001,
    )
    out_path = args.output
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f'\n✅ Saved → {out_path}')

    collector.remove_all()


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='三探针特征可视化')
    p.add_argument('--checkpoint', required=True,
                   help='模型权重路径 (.pth)')
    p.add_argument('--dataset_root', default='../dataset',
                   help='数据集根目录（默认 ../dataset）')
    p.add_argument('--dataset', default='Pohang-Canal-3k',
                   help='数据集名称（默认 Pohang-Canal-3k）')
    p.add_argument('--split_method', default='50_50_cat2',
                   help='数据划分方法（默认 50_50_cat2）')
    p.add_argument('--depth_maps_dir', default=None,
                   help='深度图目录（默认从数据集目录自动查找）')
    p.add_argument('--num_samples', type=int, default=6,
                   help='可视化样本数（默认 6）')
    p.add_argument('--threshold', type=float, default=0.3,
                   help='概率阈值（在 Probe C 上绘制白色等高线，默认 0.3）')
    p.add_argument('--selection_file', default=None,
                   help='类别选取文件路径（如 dataset/Pohang-Canal-3k/selection_summary_new.txt）')
    p.add_argument('--include_cats', default='0,1,2,3',
                   help='要展示的类别编号，逗号分隔（默认 "0,1,2,3" 即全类别；'
                        '50_50_cat2 时 val 集本身只有 cat2，无需额外限制）')
    p.add_argument('--dnanet_checkpoint', default=None,
                   help='DNANet 权重路径（可选，用于对比输出第 7 列）')
    p.add_argument('--output', default='vis_probes_output.png',
                   help='输出图片路径（默认 vis_probes_output.png）')
    p.add_argument('--gpu', type=int, default=0,
                   help='GPU 编号（默认 0）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
