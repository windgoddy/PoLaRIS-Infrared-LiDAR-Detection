"""
Mamba Stage 深度诊断 - 定位 NaN 来源
====================================

针对性诊断 Mamba stage 内部哪一层产生 NaN

Usage:
    python model_Mamba/debug_mamba_stage.py
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_Mamba.core.polaris_mamba import polaris_mamba_tiny
from model_Mamba.core.ss2d_components import SS2D


def check_tensor_stats(name, tensor, prefix=""):
    """打印张量统计信息"""
    if tensor is None:
        print(f"{prefix}  ⚠️  {name}: None")
        return True
    
    arr = tensor.detach().cpu().numpy()
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    min_val = tensor.min().item() if not has_nan else float('nan')
    max_val = tensor.max().item() if not has_nan else float('nan')
    mean_val = tensor.mean().item() if not has_nan else float('nan')
    
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(f"{prefix}  ❌ {name}: NaN={nan_count}, Inf={inf_count}, shape={tuple(tensor.shape)}")
        return True
    else:
        print(f"{prefix}  ✅ {name}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, shape={tuple(tensor.shape)}")
        return False


def diagnose_ss2d_block(block, x, name="SS2D"):
    """深度诊断单个 SS2D block"""
    print(f"\n{'='*60}")
    print(f"诊断 {name}")
    print(f"{'='*60}")
    
    check_tensor_stats(f"{name} input", x)
    
    # Check parameters first
    print(f"\n  参数检查:")
    for param_name, param in block.named_parameters():
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        if has_nan or has_inf:
            print(f"    ❌ {param_name}: 参数包含 NaN/Inf")
        else:
            norm_val = param.norm().item()
            if norm_val > 1000 or norm_val < 1e-6:
                print(f"    ⚠️  {param_name}: norm={norm_val:.4e} (异常)")
            else:
                print(f"    ✅ {param_name}: norm={norm_val:.4f}")
    
    # Manual step-by-step forward
    print(f"\n  逐步 forward:")
    try:
        with torch.no_grad():
            # Step 1: Norm
            if hasattr(block, 'norm'):
                x_norm = block.norm(x)
                check_tensor_stats("after norm", x_norm, prefix="    ")
            else:
                x_norm = x
            
            # Step 2: in_proj
            if hasattr(block, 'in_proj'):
                x_proj = block.in_proj(x_norm)
                check_tensor_stats("after in_proj", x_proj, prefix="    ")
            else:
                x_proj = x_norm
            
            # Step 3: conv2d (if exists)
            if hasattr(block, 'conv2d'):
                x_conv = block.conv2d(x_proj)
                check_tensor_stats("after conv2d", x_conv, prefix="    ")
            else:
                x_conv = x_proj
            
            # Step 4: act
            if hasattr(block, 'act'):
                x_act = block.act(x_conv)
                check_tensor_stats("after activation", x_act, prefix="    ")
            else:
                x_act = x_conv
            
            # Step 5: full forward to see where NaN appears
            output = block(x)
            check_tensor_stats(f"{name} final output", output)
            
    except Exception as e:
        print(f"    ❌ Forward 失败: {e}")
        import traceback
        traceback.print_exc()


def diagnose_vmamba_block(block, x, name="VSSBlock"):
    """诊断 VSSBlock (包含 SS2D + MLP)"""
    print(f"\n{'='*60}")
    print(f"诊断 {name}")
    print(f"{'='*60}")
    
    check_tensor_stats(f"{name} input", x)
    
    # Step 1: Normalization
    if hasattr(block, 'ln_1'):
        with torch.no_grad():
            x_norm1 = block.ln_1(x)
        check_tensor_stats("ln_1 output", x_norm1, prefix="  ")
    else:
        x_norm1 = x
    
    # Step 2: SS2D block
    if hasattr(block, 'self_attention'):
        print(f"\n  检查 self_attention (SS2D):")
        diagnose_ss2d_block(block.self_attention, x_norm1, name="SS2D")
    
    # Step 3: Full forward
    try:
        with torch.no_grad():
            output = block(x)
        check_tensor_stats(f"{name} output", output)
    except Exception as e:
        print(f"  ❌ {name} forward 失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("="*80)
    print("Mamba Stage 深度诊断工具")
    print("="*80)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # Create model
    print(f"\n创建模型...")
    model = polaris_mamba_tiny(use_lidar=True)
    model = model.to(device)
    model.eval()
    
    # Create test input
    print(f"\n创建测试输入...")
    B, C, H, W = 1, 1, 480, 640
    
    ir_img = torch.randn(B, C, H, W).to(device) * 0.5 + 0.5
    lidar_img = torch.zeros(B, C, H, W).to(device)
    
    check_tensor_stats("ir_img", ir_img)
    check_tensor_stats("lidar_img", lidar_img)
    
    # Step 1: Patch embedding
    print(f"\n{'='*80}")
    print("Step 1: Patch Embedding")
    print(f"{'='*80}")
    
    with torch.no_grad():
        x = model.patch_embed(ir_img)
    check_tensor_stats("patch_embed output", x)
    
    # Step 2: Diagnose first stage
    print(f"\n{'='*80}")
    print("Step 2: 第一个 Mamba Stage (stage_0)")
    print(f"{'='*80}")
    
    if hasattr(model, 'stages') and len(model.stages) > 0:
        stage_0 = model.stages[0]
        
        print(f"\nStage 0 类型: {type(stage_0)}")
        
        # Check if stage is a Sequential of VSSBlocks
        if isinstance(stage_0, nn.Sequential):
            print(f"Stage 0 包含 {len(stage_0)} 个 blocks")
            
            x_current = x
            for i, block in enumerate(stage_0):
                print(f"\n{'─'*60}")
                print(f"Block {i}")
                print(f"{'─'*60}")
                
                diagnose_vmamba_block(block, x_current, name=f"Block_{i}")
                
                # Forward through this block
                try:
                    with torch.no_grad():
                        x_current = block(x_current)
                    
                    has_issue = check_tensor_stats(f"Block {i} output", x_current)
                    
                    if has_issue:
                        print(f"\n❌ NaN 首次出现在 Block {i}！")
                        print(f"   定位到问题 block，停止诊断")
                        break
                        
                except Exception as e:
                    print(f"\n❌ Block {i} forward 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        else:
            # Single block or other structure - diagnose it directly
            print(f"Stage 0 不是 Sequential，直接诊断")
            diagnose_vmamba_block(stage_0, x, name="Stage_0")
    
    print(f"\n{'='*80}")
    print("诊断完成")
    print(f"{'='*80}")
    
    print(f"\n建议检查项:")
    print(f"  1. LayerNorm 的 eps 参数（可能需要增大到 1e-5）")
    print(f"  2. SS2D 中的 selective_scan 实现")
    print(f"  3. 参数初始化（是否有未初始化的参数）")
    print(f"  4. DWConv 的 padding 和 groups 设置")


if __name__ == '__main__':
    main()
