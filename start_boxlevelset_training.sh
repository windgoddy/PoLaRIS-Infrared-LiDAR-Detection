#!/bin/bash
# BoxLevelset训练启动脚本 - MMCV兼容性问题解决版本
# 最终解决方案

set -e

echo "🚀 启动BoxLevelset训练 - MMCV兼容性修复版"
echo "=" * 60

# 检查并安装必要的依赖
echo "📋 检查环境依赖..."

# 检查MMCV是否可用
python -c "
import sys
try:
    import mmcv
    print(f'✅ MMCV版本: {mmcv.__version__}')
except ImportError:
    print('❌ MMCV未安装，正在安装...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mmcv-lite==2.1.0'], check=True)
    import mmcv
    print(f'✅ MMCV安装完成: {mmcv.__version__}')

# 检查Config
try:
    from mmcv import Config
    print('✅ mmcv.Config可用')
except ImportError:
    try:
        from mmengine import Config
        print('✅ 使用mmengine.Config作为替代')
    except ImportError:
        print('❌ Config不可用')
        sys.exit(1)
"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection/external/BoxInstSeg:$PYTHONPATH"

# 设置MMCV fallback模式
export MMCV_WITH_OPS=0
export FORCE_MLU=0
export CUDA_LAUNCH_BLOCKING=1

echo "🔧 环境配置:"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  MMCV Ops: 禁用 (使用CPU fallback)"
echo "  工作目录: work_dirs/box_levelset_nuaa_r50_fpn_3x"

# 创建工作目录
mkdir -p work_dirs/box_levelset_nuaa_r50_fpn_3x

# 检查数据集
echo "📂 检查数据集..."
if [ ! -f "dataset/NUAA-SIRST/boxinstseg_coco/annotations/instances_train2017.json" ]; then
    echo "⚠️  数据集未准备，请先运行:"
    echo "python baselines/boxinstseg/prepare_irstd_coco.py"
    exit 1
fi

# 启动训练
echo "🎯 启动训练..."
python -c "
import os
import sys
import warnings

# 禁用警告
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['MMCV_WITH_OPS'] = '0'
os.environ['FORCE_MLU'] = '0'

# 添加路径
sys.path.insert(0, 'external/BoxInstSeg')

try:
    # 导入配置
    try:
        from mmcv import Config
    except ImportError:
        from mmengine import Config

    from mmdet.apis import train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    # 加载配置
    cfg = Config.fromfile('baselines/boxinstseg/configs/box_levelset_nuaa_r50_fpn_3x.py')
    cfg.work_dir = 'work_dirs/box_levelset_nuaa_r50_fpn_3x'

    print('📋 配置加载成功')
    print(f'📁 工作目录: {cfg.work_dir}')

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]
    print(f'🗂️  训练数据集大小: {len(datasets[0])}')

    # 构建模型
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    print('🏗️  模型构建成功')

    # 开始训练
    print('🎯 开始训练...')
    train_detector(model, datasets, cfg, distributed=False, validate=True)

    print('✅ 训练完成！')

except Exception as e:
    print(f'❌ 训练失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "🎉 BoxLevelset训练启动完成！"