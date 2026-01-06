#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MS_CAFNet 模型架构可视化 - 优化版
生成清晰易懂的架构图用于论文和演示
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 优化的颜色方案 - 更柔和、对比度更高
COLORS = {
    'input': '#E3F2FD',        # 浅蓝 - 输入
    'confidence': '#FFF3E0',   # 浅橙 - 置信度分支
    'encoder': '#BBDEFB',      # 蓝色 - 编码器
    'msblock': '#FFF9C4',      # 亮黄 - MSBlock
    'decoder': '#C8E6C9',      # 绿色 - 解码器
    'fpn': '#FFCCBC',          # 橙粉 - FPN增强
    'output': '#E1BEE7',       # 紫色 - 输出
    'gate': '#FFCDD2',         # 粉红 - 门控
    'oracle': '#F3E5F5',       # 浅紫 - Oracle Mask (训练监督)
}

def create_box(ax, x, y, width, height, text, color, fontsize=10,
               edgecolor='#424242', linewidth=2, bold=True):
    """创建圆角方框"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=0.95
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, color='#212121')

def create_arrow(ax, x1, y1, x2, y2, color='#424242', linewidth=2.5,
                style='-|>', linestyle='-', alpha=0.8):
    """创建箭头"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        mutation_scale=25,
        alpha=alpha,
        zorder=1
    )
    ax.add_patch(arrow)

def add_label(ax, x, y, text, fontsize=9, color='#1976D2',
              bgcolor='white', edge=True):
    """添加文字标签"""
    bbox_props = dict(
        boxstyle='round,pad=0.4',
        facecolor=bgcolor,
        edgecolor=color if edge else 'none',
        linewidth=1.5,
        alpha=0.95
    )
    ax.text(x, y, text, fontsize=fontsize, weight='bold',
            ha='center', va='center', color=color, bbox=bbox_props, zorder=10)

def plot_ms_cafnet_architecture():
    """绘制MS_CAFNet架构图 - 优化版"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 13)
    ax.axis('off')

    # ========== 标题 ==========
    ax.text(9, 12.2, 'MS_CAFNet: Multi-Scale Confidence-Aware Fusion Network',
            fontsize=18, weight='bold', ha='center', color='#1A237E')

    # ========== 输入层 (顶部) ==========
    y_input = 11
    create_box(ax, 3, y_input, 2, 0.8, 'Infrared\nImage',
               COLORS['input'], fontsize=11)
    create_box(ax, 6.5, y_input, 2, 0.8, 'Depth Map\n(LiDAR)',
               COLORS['input'], fontsize=11)

    # ========== 置信度分支 (右侧独立列) ==========
    x_conf = 15
    y_conf_start = 10.5

    create_box(ax, x_conf, y_conf_start, 2.5, 0.7,
               'ConfidenceNet', COLORS['confidence'], fontsize=10)
    create_arrow(ax, 6.5, y_input - 0.4, x_conf, y_conf_start + 0.35,
                color='#FF6F00', linewidth=2.5)

    y_conf_map = 9.3
    create_box(ax, x_conf, y_conf_map, 2, 0.6,
               'Confidence Map', COLORS['confidence'], fontsize=10)
    create_arrow(ax, x_conf, y_conf_start - 0.35, x_conf, y_conf_map + 0.3,
                color='#FF6F00', linewidth=2.5)

    # ========== Oracle Mask (训练监督) ==========
    x_oracle = 17.5
    y_oracle = 10.5

    create_box(ax, x_oracle, y_oracle, 2, 0.7,
               'Oracle Mask\n(GT Supervision)',
               COLORS['oracle'], fontsize=10, edgecolor='#9C27B0', linewidth=2)

    # 训练专用标签
    add_label(ax, x_oracle, y_oracle + 0.6, 'Training Only',
              fontsize=8, color='#9C27B0', bgcolor='#FCE4EC')

    # 监督连接 - 虚线箭头指向ConfidenceNet
    create_arrow(ax, x_oracle - 1, y_oracle, x_conf + 1.25, y_conf_start,
                color='#9C27B0', linewidth=2, linestyle=':', alpha=0.7)

    # 损失计算标注
    ax.text(x_conf + 2.5, y_conf_start - 0.4,
            'Loss_conf = BCE(pred, oracle)',
            fontsize=8, style='italic', color='#9C27B0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5',
                     edgecolor='#9C27B0', linewidth=1, alpha=0.8))

    # ========== 门控融合 ==========
    y_gate = 9.5
    x_gate = 4.5
    create_box(ax, x_gate, y_gate, 2.5, 1,
               'Confidence Gating\nDepth ⊗ Conf',
               COLORS['gate'], fontsize=11, edgecolor='#C62828', linewidth=2.5)

    # 创新标注1
    add_label(ax, x_gate, y_gate + 0.8, '① Gating',
              fontsize=9, color='#C62828')

    # 连接到门控
    create_arrow(ax, 6.5, y_input - 0.4, x_gate + 0.8, y_gate + 0.5,
                color='#1976D2', linewidth=2)
    create_arrow(ax, x_conf - 1, y_conf_map, x_gate + 1, y_gate,
                color='#FF6F00', linewidth=2, linestyle='--')

    # ========== 融合输入 ==========
    y_fused = 8
    create_box(ax, 4.5, y_fused, 3, 0.7,
               'Fused Input: [IR, Depth⊗Conf]',
               COLORS['gate'], fontsize=10)

    create_arrow(ax, 3, y_input - 0.4, 3.5, y_fused + 0.35,
                color='#1976D2', linewidth=2)
    create_arrow(ax, x_gate, y_gate - 0.5, x_gate, y_fused + 0.35,
                color='#C62828', linewidth=2.5)

    # ========== 编码器 (左侧列) ==========
    x_enc = 2.5
    enc_layers = [
        ('conv0_0\n16', 6.8),
        ('conv1_0\n32', 5.8),
        ('conv2_0\n64', 4.8),
        ('conv3_0\n128', 3.8),
        ('conv4_0\n256', 2.8),
    ]

    for i, (name, y) in enumerate(enc_layers):
        create_box(ax, x_enc, y, 1.8, 0.7, name, COLORS['encoder'], fontsize=10)

        if i == 0:
            create_arrow(ax, 4.5, y_fused - 0.35, x_enc, y + 0.35,
                        color='#1565C0', linewidth=2.5)
        else:
            prev_y = enc_layers[i-1][1]
            create_arrow(ax, x_enc, prev_y - 0.35, x_enc, y + 0.35,
                        color='#1565C0', linewidth=2.5)

    # ========== MSBlock ==========
    y_msblock = 1.5
    create_box(ax, x_enc, y_msblock, 2.5, 1,
               'MSBlock\n4-Branch (d=1,3,6)',
               COLORS['msblock'], fontsize=10, edgecolor='#F57F17', linewidth=2.5)

    # 创新标注2
    add_label(ax, x_enc, y_msblock - 0.7, '② Multi-Scale',
              fontsize=9, color='#F57F17')

    create_arrow(ax, x_enc, enc_layers[-1][1] - 0.35, x_enc, y_msblock + 0.5,
                color='#1565C0', linewidth=2.5)

    # ========== 解码器 (中间列) ==========
    x_dec = 7
    dec_layers = [
        ('conv3_1\n128', 3.8),
        ('conv2_2\n64', 4.8),
        ('conv1_3\n32', 5.8),
        ('conv0_4\n16', 6.8),
    ]

    for i, (name, y) in enumerate(dec_layers):
        create_box(ax, x_dec, y, 1.8, 0.7, name, COLORS['decoder'], fontsize=10)

        # 上采样箭头
        if i == 0:
            create_arrow(ax, x_enc + 1.25, y_msblock + 0.5, x_dec - 0.9, y,
                        color='#2E7D32', linewidth=2.5)
        else:
            prev_y = dec_layers[i-1][1]
            create_arrow(ax, x_dec, prev_y + 0.35, x_dec, y - 0.35,
                        color='#2E7D32', linewidth=2.5)

        # 跳跃连接
        enc_y = enc_layers[3-i][1]
        create_arrow(ax, x_enc + 0.9, enc_y, x_dec - 0.9, y,
                    color='#757575', linewidth=2, linestyle='--', alpha=0.6)

    # ========== FPN 残差增强 ==========
    y_fpn = 5.8
    x_fpn = 11
    create_box(ax, x_fpn, y_fpn, 2.5, 1.2,
               'Residual Boosting\nfeat×(1+conf)',
               COLORS['fpn'], fontsize=10, edgecolor='#D84315', linewidth=2.5)

    # 创新标注3
    add_label(ax, x_fpn, y_fpn - 0.8, '③ Boosting',
              fontsize=9, color='#D84315')

    # FPN连接
    create_arrow(ax, x_enc + 0.9, enc_layers[1][1], x_fpn - 1.25, y_fpn - 0.3,
                color='#6A1B9A', linewidth=2)
    create_arrow(ax, x_conf, y_conf_map - 0.3, x_fpn + 1.25, y_fpn + 0.4,
                color='#FF6F00', linewidth=2, linestyle='--')
    create_arrow(ax, x_fpn, y_fpn - 0.6, x_dec - 0.9, dec_layers[2][1],
                color='#D84315', linewidth=2.5)

    # ========== 输出层 ==========
    x_out = 11
    y_out_seg = 7.5

    create_box(ax, x_out, y_out_seg, 2, 0.7, 'Final Conv 1×1',
               COLORS['output'], fontsize=10)
    create_arrow(ax, x_dec + 0.9, dec_layers[3][1], x_out - 1, y_out_seg,
                color='#2E7D32', linewidth=2.5)

    y_out_final = 8.5
    create_box(ax, x_out, y_out_final, 2.2, 0.8,
               'Segmentation\nOutput',
               COLORS['output'], fontsize=11, edgecolor='#6A1B9A', linewidth=2.5)
    create_arrow(ax, x_out, y_out_seg + 0.35, x_out, y_out_final - 0.4,
                color='#6A1B9A', linewidth=3)

    # 置信度输出
    y_conf_final = 8.5
    create_box(ax, x_conf, y_conf_final, 2, 0.8,
               'Confidence\nOutput',
               COLORS['confidence'], fontsize=11, edgecolor='#FF6F00', linewidth=2.5)
    create_arrow(ax, x_conf, y_conf_map - 0.3, x_conf, y_conf_final - 0.4,
                color='#FF6F00', linewidth=2.5, linestyle='--')

    # ========== 图例 ==========
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['input'], edgecolor='#424242',
                      linewidth=2, label='Input'),
        mpatches.Patch(facecolor=COLORS['confidence'], edgecolor='#424242',
                      linewidth=2, label='Confidence Branch'),
        mpatches.Patch(facecolor=COLORS['gate'], edgecolor='#C62828',
                      linewidth=2, label='Gating'),
        mpatches.Patch(facecolor=COLORS['encoder'], edgecolor='#424242',
                      linewidth=2, label='Encoder'),
        mpatches.Patch(facecolor=COLORS['msblock'], edgecolor='#F57F17',
                      linewidth=2, label='MSBlock'),
        mpatches.Patch(facecolor=COLORS['decoder'], edgecolor='#424242',
                      linewidth=2, label='Decoder'),
        mpatches.Patch(facecolor=COLORS['fpn'], edgecolor='#D84315',
                      linewidth=2, label='FPN Boosting'),
        mpatches.Patch(facecolor=COLORS['output'], edgecolor='#6A1B9A',
                      linewidth=2, label='Output'),
        mpatches.Patch(facecolor=COLORS['oracle'], edgecolor='#9C27B0',
                      linewidth=2, label='Oracle Mask (Training)'),
    ]

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=10,
              frameon=True, fancybox=True, shadow=True)

    # ========== 流程标注 ==========
    # 数据流标注
    ax.text(1, 11.5, 'Input', fontsize=9, style='italic', color='#1565C0')
    ax.text(1, 4, 'Encoding', fontsize=9, style='italic', color='#1565C0')
    ax.text(6, 4, 'Decoding', fontsize=9, style='italic', color='#2E7D32')
    ax.text(14, 11.5, 'Confidence', fontsize=9, style='italic', color='#FF6F00')

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # 生成优化后的架构图
    fig = plot_ms_cafnet_architecture()

    # 保存高清图片
    output_path = 'MS_CAFNet_Architecture.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ 架构图已保存: {output_path}")

    # 保存PDF版本
    pdf_path = 'MS_CAFNet_Architecture.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ PDF版本已保存: {pdf_path}")

    plt.close(fig)
    print("\n📊 优化要点:")
    print("  • 垂直对齐布局，减少视觉混乱")
    print("  • 清晰的颜色分区")
    print("  • 简化标签文字")
    print("  • 减少箭头交叉")
    print("  • 突出三大创新点")
    print("  • 添加Oracle Mask训练监督流程")
