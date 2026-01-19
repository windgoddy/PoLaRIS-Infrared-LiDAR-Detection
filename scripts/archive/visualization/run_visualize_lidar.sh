#!/bin/bash
# ./scripts/run_visualize_lidar.sh
# 点云投影到红外图像可视化

echo "========================================================================"
echo "🎨 点云投影到红外图像可视化"
echo "========================================================================"
echo ""
echo "此脚本将："
echo "  1. 读取 Pohang-Canal-all 数据集"
echo "  2. 将 LiDAR 点云投影到红外图像上"
echo "  3. 使用深度信息为点着色（近=紫色，远=黄色）"
echo "  4. 保存可视化结果"
echo ""
echo "选项："
echo "  1) 可视化测试集（3550 张）"
echo "  2) 可视化训练集（14196 张）"
echo "  3) 可视化全部（17746 张）"
echo "  4) 可视化测试集前 100 张（快速预览）"
echo ""
echo "========================================================================"
echo ""

read -p "请选择 (1-4): " -n 1 -r choice
echo ""

case $choice in
    1)
        echo ">>> 可视化测试集（3550 张）"
        python scripts/visualize_lidar_projection.py --split test
        ;;
    2)
        echo ">>> 可视化训练集（14196 张）"
        python scripts/visualize_lidar_projection.py --split train
        ;;
    3)
        echo ">>> 可视化全部（17746 张）"
        python scripts/visualize_lidar_projection.py --split all
        ;;
    4)
        echo ">>> 可视化测试集前 100 张（快速预览）"
        python scripts/visualize_lidar_projection.py --split test --num_samples 100
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ 可视化完成！"
    echo "========================================================================"
    echo ""
    echo "📁 输出位置:"
    echo "   dataset/Pohang-Canal-all/vis_lidar_projection/"
    echo ""
    echo "💡 提示:"
    echo "   - 点的颜色表示深度（紫色=近，黄色=远）"
    echo "   - 可以使用图片查看器浏览结果"
    echo ""
    echo "========================================================================"
else
    echo ""
    echo "❌ 可视化失败"
fi
