#!/bin/bash

# 数据集准备脚本 - 一键运行
# 功能：将22,183张红外图像和LiDAR数据整理为训练数据集

echo "================================================================"
echo "完整数据集准备 - Pohang-Canal-all"
echo "================================================================"
echo ""
echo "此脚本将："
echo "  1. 从原始数据目录读取所有红外图像和LiDAR点云"
echo "  2. 进行时间戳同步（处理帧率不一致问题）"
echo "  3. 过滤和投影点云到图像平面"
echo "  4. 生成深度图"
echo "  5. 创建训练/测试数据集划分（8:2）"
echo ""
echo "原始数据位置："
echo "  - 红外: /home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/infrared/images"
echo "  - LiDAR: /home/b311/data2/25-zhangxizhe/Pohang Canal Dataset And PoLaRIS/Pohang Canal Dataset/00/lidar_front/points"
echo ""
echo "目标位置："
echo "  - /home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/dataset/Pohang-Canal-all"
echo ""
echo "预计处理时间: 30-60分钟（取决于硬盘速度）"
echo "================================================================"
echo ""

# 确认是否继续
read -p "是否继续？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消"
    exit 1
fi

# 检查 Python 环境
if ! command -v python &> /dev/null
then
    echo "❌ 错误: 未找到 Python"
    exit 1
fi

# 检查必要的 Python 包
python -c "import numpy, pandas, cv2, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 错误: 缺少必要的 Python 包"
    echo "请先安装: pip install numpy pandas opencv-python tqdm"
    exit 1
fi

# 运行数据准备脚本
python scripts/prepare_all_data.py

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "✅ 数据集准备完成！"
    echo "================================================================"
    echo ""
    echo "下一步："
    echo "  1. 检查数据集: ls dataset/Pohang-Canal-all/"
    echo "  2. 查看训练集: head dataset/Pohang-Canal-all/split_data/train.txt"
    echo "  3. 开始训练: ./scripts/run_Phase3_improved_v3.sh"
    echo ""
else
    echo ""
    echo "================================================================"
    echo "❌ 数据集准备失败，请检查错误信息"
    echo "================================================================"
    echo ""
    exit 1
fi
