"""
GPU 診斷腳本 - 掃描並修復 GPU 配置問題
"""

import sys
import subprocess

def check_nvidia_gpu():
    """掃描 NVIDIA GPU"""
    print("="*60)
    print("1. 掃描 NVIDIA GPU")
    print("="*60)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("錯誤: nvidia-smi 不可用")
            print("原因: NVIDIA 驅動不存在或未正確安裝")
            return False
    except FileNotFoundError:
        print("錯誤: 找不到 nvidia-smi")
        print("原因: NVIDIA CUDA Toolkit 未安裝")
        return False

def check_cuda():
    """掃描 CUDA"""
    print("\n" + "="*60)
    print("2. 掃描 CUDA")
    print("="*60)
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("錯誤: nvcc 不可用")
            return False
    except FileNotFoundError:
        print("錯誤: 找不到 nvcc")
        print("原因: CUDA Toolkit 未正確安裝或 PATH 未配置")
        return False

def check_cudnn():
    """掃描 cuDNN"""
    print("\n" + "="*60)
    print("3. 掃描 cuDNN")
    print("="*60)
    try:
        import ctypes
        # 試圖加載 cuDNN
        libcudnn_path = None
        try:
            # Windows
            libcudnn_path = ctypes.CDLL('cudnn64_8.dll')
            print("[OK] 找到 cuDNN (Windows)")
            return True
        except:
            try:
                # Linux
                libcudnn_path = ctypes.CDLL('libcudnn.so.8')
                print("[OK] 找到 cuDNN (Linux)")
                return True
            except:
                print("[NO] 找不到 cuDNN")
                print("原因: cuDNN 未正確安裝")
                return False
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return False

def check_tensorflow():
    """掃描 TensorFlow GPU 支援"""
    print("\n" + "="*60)
    print("4. 掃描 TensorFlow GPU 支援")
    print("="*60)
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow 版本: {tf.__version__}")
        
        # 掃描 GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] 掃描到 {len(gpus)} 個 GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            return True
        else:
            print("[NO] 未掃描到 GPU")
            return False
    except ImportError:
        print("[NO] TensorFlow 未安裝")
        return False
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return False

def check_tensorflow_build():
    """掃描 TensorFlow 編譯細節"""
    print("\n" + "="*60)
    print("5. 掃描 TensorFlow 編譯配置")
    print("="*60)
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow 編譯信息:")
        print(f"   版本: {tf.__version__}")
        print(f"   CUDA 支援: {tf.test.is_built_with_cuda()}")
        
        # 判斷是否支援 CUDA
        if tf.test.is_built_with_cuda():
            print("[OK] 已編譯 CUDA 支援")
        else:
            print("[NO] 未編譯 CUDA 支援 (這是主要問題!)")
            return False
        
        return True
    except ImportError:
        print("[NO] TensorFlow 未安裝")
        return False
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return False

def check_environment():
    """掃描環境變數"""
    print("\n" + "="*60)
    print("6. 掃描環境變數")
    print("="*60)
    import os
    
    env_vars = [
        'CUDA_HOME',
        'CUDA_PATH',
        'CUDNN_PATH',
        'PATH',
        'LD_LIBRARY_PATH',  # Linux
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '未設置')
        if var == 'PATH' or var == 'LD_LIBRARY_PATH':
            # 只顯示部分
            if isinstance(value, str) and value != '未設置':
                paths = value.split(';' if sys.platform == 'win32' else ':')
                cuda_related = [p for p in paths if 'cuda' in p.lower()]
                if cuda_related:
                    print(f"[OK] {var}:")
                    for p in cuda_related:
                        print(f"    {p}")
                else:
                    print(f"[NO] {var} 中沒有 CUDA 路徑")
            else:
                print(f"[NO] {var}: 未設置")
        else:
            if value != '未設置':
                print(f"[OK] {var}: {value}")
            else:
                print(f"[NO] {var}: 未設置")

def show_recommendations():
    """顯示建議"""
    print("\n" + "="*60)
    print("修復建議")
    print("="*60)
    print("""
如果 GPU 仍止未掃描到，請按以下步驟操作:

步驟 1: 確認 NVIDIA 驅動器
  - Windows: 下載最新 NVIDIA 驅動 (https://www.nvidia.com/Download/driverDetails.aspx)
  - Linux: sudo apt-get install nvidia-driver-535
  - 驗證: 執行 nvidia-smi

步驟 2: 安裝 CUDA Toolkit 12.4+
  - 下載: https://developer.nvidia.com/cuda-downloads
  - 選擇你的作業系統和 GPU
  - 例子 (Windows): CUDA Toolkit 12.4
  - 例子 (Linux): CUDA Toolkit 12.4 for Ubuntu 22.04
  - 驗證: 執行 nvcc --version

步驟 3: 安裝 cuDNN 8.0+
  - 下載: https://developer.nvidia.com/cudnn (需要 NVIDIA 帳戶)
  - 解壓縮並複製到 CUDA 安裝路徑
  - Windows 例子:
    解壓縮 cudnn-windows-x86_64-8.9.7.29.zip
    複製到 C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4
  - Linux 例子:
    tar -xzf cudnn-linux-x86_64-8.9.7.29.tar.xz
    cp -r cudnn-linux-x86_64-8.9.7.29/include/* /usr/local/cuda-12.4/include/
    cp -r cudnn-linux-x86_64-8.9.7.29/lib/* /usr/local/cuda-12.4/lib64/

步驟 4: 重新安裝 TensorFlow (最重要!)
  - 卸載舊版本:
    pip uninstall tensorflow tensorflow-intel tensorflow-macos -y
  - 安裝新版本:
    pip install tensorflow[and-cuda]==2.14.0
  - 驗證: python check_gpu.py

步驟 5: 設置環境變數 (如果需要)

  Windows:
    1. 按 Win+X，選擇「系統設置」
    2. 進入「系統」→「進階系統設置」
    3. 點擊「環境變數」
    4. 新增或編輯用戶變數:
       CUDA_HOME = C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4
       CUDNN_PATH = C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4
    5. 編輯 PATH，新增:
       C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin
    6. 重新啟動 Python 環境

  Linux:
    編輯 ~/.bashrc 或 ~/.zshrc，新增:
    export CUDA_HOME=/usr/local/cuda-12.4
    export CUDNN_HOME=/usr/local/cuda-12.4
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    然後執行:
    source ~/.bashrc

步驟 6: 快速驗證
  python
  >>> import tensorflow as tf
  >>> tf.config.list_physical_devices('GPU')
  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

如果還是不行，最常見原因是 TensorFlow 未編譯 CUDA 支援，
請確認執行: pip install tensorflow[and-cuda]==2.14.0
    """)

def main():
    print("\n[GPU 診斷工具 v1.0]\n")
    
    results = {
        'nvidia_gpu': check_nvidia_gpu(),
        'cuda': check_cuda(),
        'cudnn': check_cudnn(),
        'tensorflow': check_tensorflow(),
        'tensorflow_build': check_tensorflow_build(),
    }
    
    check_environment()
    
    # 總結
    print("\n" + "="*60)
    print("診斷結果")
    print("="*60)
    
    all_good = all(results.values())
    
    if all_good:
        print("[OK] 完美! 你的 GPU 配置應該勞作無誤")
        print("[OK] 你可以開始使用 GPU 訓練 LSTM 了")
    else:
        print("[NO] 查出以下問題:")
        if not results['nvidia_gpu']:
            print("  - 找不到 NVIDIA GPU 或驅動")
        if not results['cuda']:
            print("  - CUDA Toolkit 未正確安裝")
        if not results['cudnn']:
            print("  - cuDNN 未正確安裝")
        if not results['tensorflow']:
            print("  - TensorFlow 未掃描到 GPU")
        if not results['tensorflow_build']:
            print("  - TensorFlow 未編譯 CUDA 支援 (主要問題!)")
        
        show_recommendations()

if __name__ == '__main__':
    main()
