import torch
import torch.nn as nn
import torch.nn.functional as F

# 从现有的 model_DNANet 导入基础模块
# 注意：这里我们统一使用 Res_CBAM_block，因为它是 DNANet 的核心模块
from model.model_DNANet import Res_CBAM_block

class MSBlock(nn.Module):
    """
    MSBlock (Multi-Scale Context Block)
    参考 YOLO-MST 的 MSFA 思想，利用多路不同空洞率的卷积提取多尺度特征。
    """
    def __init__(self, in_channels, out_channels):
        super(MSBlock, self).__init__()
        
        # Branch 1: 1x1 Conv (保留微小细节)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 Conv, dilation 1 (局部上下文)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 3x3 Conv, dilation 3 (中距上下文, 模拟 7x7)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 Conv, dilation 6 (长距上下文, 模拟 13x13)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Aggregation: Concat -> 1x1 Conv -> BN -> ReLU
        self.aggregation = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        # 拼接多尺度特征
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.aggregation(out)
        return out

class ConfidenceNet(nn.Module):
    """
    轻量级置信度预测网络 (Confidence Branch)
    输入: 稀疏深度图 (1通道)
    输出: 置信度图 (1通道, 0~1)
    """
    def __init__(self, in_channels=1):
        super(ConfidenceNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid() # 关键：输出概率 0-1
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        conf_map = self.layer3(x)
        return conf_map

class MS_CAFNet(nn.Module):
    """
    Multi-Scale Confidence-Aware Fusion Network (Phase 3 Core Model)
    集成了:
    1. 置信度门控 (Confidence Gating)
    2. MSBlock 多尺度感知 (Multi-Scale Context)
    3. 密集嵌套骨干 (Dense Nested Backbone)
    4. LiDAR 门控特征金字塔 (LiDAR-Guided Gated FPN) - 新增
    """
    def __init__(self, num_classes=1, input_channels=2):
        super(MS_CAFNet, self).__init__()

        # 滤波器通道数配置 (与 DNANet 保持一致)
        nb_filter = [16, 32, 64, 128, 256]

        # --- 1. 置信度分支 ---
        self.conf_net = ConfidenceNet(in_channels=1)

        # --- 2. 编码器 (Encoder) ---
        self.pool = nn.MaxPool2d(2, 2)

        # 使用 Res_CBAM_block 构建骨干网络
        # 注意：Res_CBAM_block(in, out)
        self.conv0_0 = Res_CBAM_block(2, nb_filter[0])
        self.conv1_0 = Res_CBAM_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Res_CBAM_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Res_CBAM_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Res_CBAM_block(nb_filter[3], nb_filter[4])

        # --- 3. MSBlock 模块 (核心改进：适配大目标) ---
        # 放置在最深层 (Encoder 和 Decoder 之间)
        self.ms_context = MSBlock(nb_filter[4], nb_filter[4])

        # --- 4. 解码器 (Decoder) ---
        # 保持密集跳跃连接
        self.conv3_1 = Res_CBAM_block(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = Res_CBAM_block(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = Res_CBAM_block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = Res_CBAM_block(nb_filter[0] + nb_filter[1], nb_filter[0])

        # --- 5. LiDAR 增强 FPN 模块 (Residual Boosting) ---
        # 用于浅层特征的残差增强，保留远距离目标的红外信息
        self.shallow_reducer = nn.Sequential(
            nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # input shape: [Batch, 2, H, W]
        # 假设通道 0 是红外(IR), 通道 1 是深度(Depth)
        x_ir = input[:, 0:1, :, :]
        x_depth = input[:, 1:2, :, :]

        # === Step A: 置信度生成 ===
        # 预测 LiDAR 哪里可信 (受到 Oracle Mask 监督)
        pred_conf = self.conf_net(x_depth)

        # === Step B: 门控融合 (Input Gating) ===
        # 核心逻辑：Depth 只在 Conf 高的地方生效
        # 融合结果 = Concat(IR, Depth * Conf)
        x_gated_depth = x_depth * pred_conf
        x_fused = torch.cat([x_ir, x_gated_depth], dim=1)

        # === Step C: 编码 (Encoding) ===
        x0_0 = self.conv0_0(x_fused)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # === Step D: 多尺度增强 (MSBlock) ===
        # 这里让网络"看"得更广，同时保留细节
        x4_0_enhanced = self.ms_context(x4_0)

        # === Step E: 解码 (Decoding) ===
        # 注意：这里上采样使用的是 MSBlock 增强后的特征
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_enhanced)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))

        # === Step F: LiDAR 残差增强 FPN (Residual Boosting) ===
        # 关键改进：不抑制远距离目标，而是根据 LiDAR 置信度进行残差增强

        # F1. 提取浅层高分辨率特征（选择 x1_0，包含丰富的空间细节）
        feat_shallow = x1_0  # shape: [B, nb_filter[1]=32, H/2, W/2]

        # F2. 处理置信度图
        # pred_conf 原始尺寸: [B, 1, H, W]
        # 下采样到与 feat_shallow 相同尺寸: [B, 1, H/2, W/2]
        att_map = F.interpolate(pred_conf, size=feat_shallow.shape[2:],
                                mode='bilinear', align_corners=True)
        # att_map 已经通过 Sigmoid（在 ConfidenceNet 中），范围 [0, 1]

        # F3. 残差增强（核心创新）
        # 公式：feat_enhanced = feat_shallow × (1 + att_map)
        # - 当 att_map ≈ 0（无 LiDAR）: feat_enhanced ≈ feat_shallow × 1 (保留原始特征)
        # - 当 att_map ≈ 1（LiDAR 确认）: feat_enhanced ≈ feat_shallow × 2 (特征增强 2 倍)
        feat_enhanced = feat_shallow * (1.0 + att_map)

        # F4. 通过 1x1 卷积进一步提炼增强后的特征
        feat_enhanced_processed = self.shallow_reducer(feat_enhanced)

        # F5. 继续解码（在 x1_3 处融合增强特征）
        x1_3_base = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        # 残差连接：深层语义特征 + 浅层增强特征
        x1_3_fused = x1_3_base + feat_enhanced_processed

        # F6. 完成最后一层解码
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3_fused)], 1))

        # F7. 最终分割输出
        output = self.final(x0_4)

        # 返回两个结果：[检测图, 置信度图]
        return output, pred_conf

    def up(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


# ============================================================================
# [NEW] 双流几何引导升级版 (Dual-Geometry Enhanced Version)
# 创建日期: 2026-01-06
# 设计理念: 双保险门锁 - LiDAR (电子锁) + Visual (机械锁)
# ============================================================================

class VisualStructureExtractor(nn.Module):
    """
    轻量级视觉结构提取器 (Visual Structure Branch)
    灵感来源: SG-LLIE 的结构先验提取思想

    作用: 从红外图像中提取边缘/纹理结构，作为 LiDAR 失效时的几何先验
    输入: 单通道红外图像 (1通道)
    输出: 视觉结构图 (1通道, 0~1) - 作为注意力图

    设计理念: "机械锁" - 不依赖 LiDAR，只从视觉本身提取结构信息
    """
    def __init__(self, in_channels=1):
        super(VisualStructureExtractor, self).__init__()

        # 方案1: 使用轻量级卷积模拟边缘提取 (类似 Sobel)
        # 单层 3x3 卷积 + BN + ReLU，保持极轻量
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # 1x1 卷积降维到单通道
            nn.Conv2d(8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # 确保输出 [0, 1]，作为结构先验强度
        )

    def forward(self, x_ir):
        """
        Args:
            x_ir: 红外图像 [B, 1, H, W]
        Returns:
            structure_map: 视觉结构图 [B, 1, H, W], 范围 [0, 1]
        """
        structure_map = self.edge_extractor(x_ir)
        return structure_map


class MS_CAFNet_DualGeo(nn.Module):
    """
    MS_CAFNet with Adaptive Dual-Geometry Guidance (升级版)

    核心升级:
    1. 置信度门控 (Confidence Gating) - 保留
    2. MSBlock 多尺度感知 (Multi-Scale Context) - 保留
    3. 密集嵌套骨干 (Dense Nested Backbone) - 保留
    4. [NEW] 自适应双流几何引导 (Adaptive Dual-Geometry Guidance)
       - LiDAR 几何分支 (电子锁): 从深度图提取几何信息
       - 视觉结构分支 (机械锁): 从红外提取边缘/纹理结构
       - 自适应融合门控 (智能联动): 可学习的权重平衡

    公式升级:
    旧版: feat_enhanced = feat × (1 + lidar_conf)
    新版: feat_enhanced = feat × (1 + α·lidar_conf + β·visual_struct)

    参数量: 相比原始 MS_CAFNet 仅增加 82 个参数 (<0.003%)
    """
    def __init__(self, num_classes=1, input_channels=2):
        super(MS_CAFNet_DualGeo, self).__init__()

        # 保存输入通道数（1=仅IR，2=IR+Depth）
        self.input_channels = input_channels

        # 滤波器通道数配置 (与 DNANet 保持一致)
        nb_filter = [16, 32, 64, 128, 256]

        # --- 1. 置信度分支 (LiDAR 几何引导) ---
        self.conf_net = ConfidenceNet(in_channels=1)

        # --- [NEW] 视觉结构分支 (Visual 几何引导) ---
        self.visual_extractor = VisualStructureExtractor(in_channels=1)

        # --- 2. 编码器 (Encoder) ---
        self.pool = nn.MaxPool2d(2, 2)

        # 使用 Res_CBAM_block 构建骨干网络
        # 编码器第一层的输入通道数根据 input_channels 设置
        self.conv0_0 = Res_CBAM_block(input_channels, nb_filter[0])
        self.conv1_0 = Res_CBAM_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = Res_CBAM_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = Res_CBAM_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = Res_CBAM_block(nb_filter[3], nb_filter[4])

        # --- 3. MSBlock 模块 (核心改进：适配大目标) ---
        self.ms_context = MSBlock(nb_filter[4], nb_filter[4])

        # --- 4. 解码器 (Decoder) ---
        self.conv3_1 = Res_CBAM_block(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = Res_CBAM_block(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = Res_CBAM_block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = Res_CBAM_block(nb_filter[0] + nb_filter[1], nb_filter[0])

        # --- 5. [NEW] 自适应双流融合 FPN 模块 ---
        self.shallow_reducer = nn.Sequential(
            nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(nb_filter[1]),
            nn.ReLU(inplace=True)
        )

        # [NEW] 自适应门控参数 (Adaptive Gating Parameters)
        # gate_lidar (alpha): LiDAR 置信度分支的权重
        # gate_visual (beta): 视觉结构分支的权重
        # 初始化为 0.5，让网络学习如何平衡两个分支
        self.gate_lidar = nn.Parameter(torch.tensor(0.5))   # alpha: 电子锁权重
        self.gate_visual = nn.Parameter(torch.tensor(0.5))  # beta: 机械锁权重

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # input shape: [Batch, 1 or 2, H, W]
        # - 1 channel: 仅红外 (IR only)
        # - 2 channels: 红外 + 深度 (IR + Depth)

        if self.input_channels == 1:
            # 单通道模式：仅使用红外图像
            x_ir = input  # [B, 1, H, W]
            # 创建全零深度图（用于置信度网络）
            x_depth = torch.zeros_like(x_ir)
        else:
            # 双通道模式：红外 + 深度
            x_ir = input[:, 0:1, :, :]     # 通道 0: 红外
            x_depth = input[:, 1:2, :, :]  # 通道 1: 深度

        # === Step A: 置信度生成 ===
        pred_conf = self.conf_net(x_depth)

        # === Step B: 门控融合 (Input Gating) ===
        if self.input_channels == 1:
            # 单通道模式：直接使用红外（不融合深度）
            x_fused = x_ir
        else:
            # 双通道模式：融合红外和门控深度
            x_gated_depth = x_depth * pred_conf
            x_fused = torch.cat([x_ir, x_gated_depth], dim=1)

        # === Step C: 编码 (Encoding) ===
        x0_0 = self.conv0_0(x_fused)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # === Step D: 多尺度增强 (MSBlock) ===
        x4_0_enhanced = self.ms_context(x4_0)

        # === Step E: 解码 (Decoding) ===
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_enhanced)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))

        # === Step F: [NEW] 自适应双流几何融合 FPN ===
        # 升级：从单流 LiDAR 增强 → 双流自适应几何引导
        # 核心理念：LiDAR (电子锁) + Visual (机械锁) = 双保险

        # F1. 提取浅层高分辨率特征
        feat_shallow = x1_0  # shape: [B, nb_filter[1]=32, H/2, W/2]

        # F2. 第一流：LiDAR 几何引导 (Geometry from LiDAR)
        lidar_guidance = F.interpolate(pred_conf, size=feat_shallow.shape[2:],
                                       mode='bilinear', align_corners=True)

        # F2b. [NEW] 第二流：视觉结构引导 (Geometry from Visual)
        visual_structure = self.visual_extractor(x_ir)  # [B, 1, H, W]
        visual_guidance = F.interpolate(visual_structure, size=feat_shallow.shape[2:],
                                        mode='bilinear', align_corners=True)

        # F3. [NEW] 自适应双流残差增强
        # 新公式: feat_enhanced = feat_shallow × (1 + alpha*lidar + beta*visual)
        #
        # 工作模式：
        # - 富裕模式 (LiDAR 强): alpha↑, beta 适中 → 主导 LiDAR，视觉辅助修边
        # - 低保模式 (LiDAR 弱): alpha↓, beta↑ → 依赖视觉结构先验
        feat_enhanced = feat_shallow * (
            1.0 +
            self.gate_lidar * lidar_guidance +    # 电子锁分支
            self.gate_visual * visual_guidance     # 机械锁分支
        )

        # F4. 通过 1x1 卷积进一步提炼增强后的特征
        feat_enhanced_processed = self.shallow_reducer(feat_enhanced)

        # F5. 继续解码（在 x1_3 处融合增强特征）
        x1_3_base = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x1_3_fused = x1_3_base + feat_enhanced_processed

        # F6. 完成最后一层解码
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3_fused)], 1))

        # F7. 最终分割输出
        output = self.final(x0_4)

        # 返回两个结果：[检测图, 置信度图]
        return output, pred_conf

    def up(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
