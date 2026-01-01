#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyQt5 Windows DLL 修復工具
解決 "DLL load failed while importing QtWidgets" 錯誤
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """運行命令"""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print('='*60)
    
    print(f"\n執行: {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def test_pyqt5():
    """測試 PyQt5 是否可用"""
    try:
        from PyQt5.QtWidgets import QApplication
        print("\n✓ PyQt5 可以使用")
        return True
    except ImportError as e:
        print(f"\n✗ PyQt5 導入失敗: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  PyQt5 Windows DLL 修復工具")
    print("="*60)
    
    # 步驟 1: 卸載舊版本
    print("\n[步驟 1/5] 卸載舊的 PyQt5...")
    run_command(
        "pip uninstall PyQt5 PyQt5-sip PyQt5-Qt5 -y",
        "卸載 PyQt5"
    )
    
    # 步驟 2: 清理緩存
    print("\n[步驟 2/5] 清理 pip 緩存...")
    run_command("pip cache purge", "清理 pip 緩存")
    
    # 步驟 3: 升級 pip
    print("\n[步驟 3/5] 升級 pip...")
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "升級 pip"
    )
    
    # 步驟 4: 安裝新版本
    print("\n[步驟 4/5] 安裝 PyQt5 相容版本...")
    success = run_command(
        "pip install PyQt5==5.15.9 --no-cache-dir",
        "安裝 PyQt5 5.15.9"
    )
    
    # 步驟 5: 測試
    print("\n[步驟 5/5] 測試 PyQt5...")
    if not test_pyqt5():
        print("\n嘗試另一個版本: PyQt5==5.15.10")
        run_command(
            "pip install PyQt5==5.15.10 --no-cache-dir --force-reinstall",
            "安裝 PyQt5 5.15.10"
        )
        test_pyqt5()
    
    # 最終結果
    print("\n" + "="*60)
    print("  修復流程完成")
    print("="*60)
    
    if test_pyqt5():
        print("\n✓ PyQt5 已成功安裝！")
        print("\n接下來執行:")
        print("  python model_ensemble_gui.py\n")
    else:
        print("\n✗ PyQt5 安裝仍然失敗")
        print("\n請嘗試以下方案之一:")
        print("  1. 完全重置虛擬環境")
        print("     deactivate")
        print("     rmdir /s /q .venv")
        print("     python -m venv .venv")
        print("     .venv\\Scripts\\activate")
        print("     pip install PyQt5==5.15.9 pandas numpy matplotlib huggingface-hub")
        print("\n  2. 安裝 Visual C++ Runtime:")
        print("     https://support.microsoft.com/en-us/help/2977003/\n")

if __name__ == "__main__":
    main()
