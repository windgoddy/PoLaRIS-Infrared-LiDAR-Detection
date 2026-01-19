#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复脚本文件的行尾符问题（CRLF -> LF）

Windows 和 Unix/Linux 系统使用不同的行尾符：
- Windows: CRLF (\r\n)
- Unix/Linux: LF (\n)

这会导致 bash 脚本在 Linux 上报错：/bin/bash^M: 解释器错误
"""

import os
import sys

def fix_line_endings(filepath):
    """将文件的行尾符从 CRLF 转换为 LF"""
    try:
        # 读取文件（二进制模式）
        with open(filepath, 'rb') as f:
            content = f.read()

        # 检查是否包含 CRLF
        if b'\r\n' in content or b'\r' in content:
            # 转换：CRLF -> LF, CR -> LF
            content = content.replace(b'\r\n', b'\n')
            content = content.replace(b'\r', b'\n')

            # 写回文件
            with open(filepath, 'wb') as f:
                f.write(content)

            return True  # 已修复
        else:
            return False  # 无需修复

    except Exception as e:
        print(f"❌ 错误: 无法处理 {filepath}: {e}")
        return None

def main():
    print("=" * 80)
    print("🔧 修复脚本文件的行尾符问题")
    print("=" * 80)

    # 需要修复的文件列表
    files_to_fix = [
        'scripts/run_filter_and_prepare.sh',
        'scripts/filter_dataset.py',
        'scripts/prepare_labels_for_all.py',
        'scripts/run_prepare_all_data.sh',
        'scripts/prepare_all_data.py',
        'scripts/run_Phase3_improved_v3.sh',
        'scripts/run_Phase3_DualGeo.sh',
        'scripts/run_baseline1.sh',
    ]

    fixed_count = 0
    skip_count = 0
    error_count = 0

    for filepath in files_to_fix:
        if not os.path.exists(filepath):
            print(f"⏭️  跳过: {filepath} (文件不存在)")
            skip_count += 1
            continue

        result = fix_line_endings(filepath)

        if result is True:
            print(f"✅ 已修复: {filepath}")
            fixed_count += 1
        elif result is False:
            print(f"✓  无需修复: {filepath}")
        else:
            error_count += 1

    print("\n" + "=" * 80)
    print("📊 总结:")
    print(f"  ✅ 已修复: {fixed_count} 个文件")
    print(f"  ✓  无需修复: {len(files_to_fix) - fixed_count - skip_count - error_count} 个文件")
    print(f"  ⏭️  跳过: {skip_count} 个文件")
    print(f"  ❌ 错误: {error_count} 个文件")
    print("=" * 80)

    if fixed_count > 0:
        print("\n💡 提示: 文件已修复，现在可以运行 bash 脚本了")
        print("   例如: ./scripts/run_filter_and_prepare.sh")

    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
