"""
GPU 診斷脚本 - 棄探並修複 GPU 配置問題
"""

import sys
import subprocess

def check_nvidia_gpu():
    """棄探 NVIDIA GPU"""
    print("="*60)
    print("1. 棄探 NVIDIA GPU")
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
    """棄探 CUDA"""
    print("\n" + "="*60)
    print("2. 棄探 CUDA")
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
    """棄探 cuDNN"""
    print("\n" + "="*60)
    print("3. 棄探 cuDNN")
    print("="*60)
    try:
        import ctypes
        # 試圖加載 cuDNN
        libcudnn_path = None
        try:
            # Windows
            libcudnn_path = ctypes.CDLL('cudnn64_8.dll')
            print("絠誊: 找到 cuDNN (Windows)")
            return True
        except:
            try:
                # Linux
                libcudnn_path = ctypes.CDLL('libcudnn.so.8')
                print("絠誊: 找到 cuDNN (Linux)")
                return True
            except:
                print("錯誤: 找不到 cuDNN")
                print("原因: cuDNN 未正確安裝")
                return False
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return False

def check_tensorflow():
    """棄探 TensorFlow GPU 支援"""
    print("\n" + "="*60)
    print("4. 棄探 TensorFlow GPU 支援")
    print("="*60)
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 版本: {tf.__version__}")
        
        # 棄探 GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ 棄探到 {len(gpus)} 個 GPU:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            return True
        else:
            print("❌ 未棄探到 GPU")
            return False
    except ImportError:
        print("錯誤: TensorFlow 未安裝")
        return False
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return False

def check_tensorflow_build():
    """棄探 TensorFlow 編譯細節"""
    print("\n" + "="*60)
    print("5. 棄探 TensorFlow 編譯配置")
    print("="*60)
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 編譯信息:")
        print(f"   版本: {tf.__version__}")
        print(f"   CUDA 支援: {tf.test.is_built_with_cuda()}")
        
        # 判斷是否支援 CUDA
        if tf.test.is_built_with_cuda():
            print("✅ 已編譯 CUDA 支援")
        else:
            print("❌ 未編譯 CUDA 支援 (這是主要問題!)")
            return False
        
        return True
    except ImportError:
        print("錯誤: TensorFlow 未安裝")
        return False
    except Exception as e:
        print(f"錯誤: {str(e)}")
        return False

def check_environment():
    """棄探環境變量"""
    print("\n" + "="*60)
    print("6. 棄探環境變量")
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
            # 只顯示部份
            if isinstance(value, str):
                paths = value.split(';' if sys.platform == 'win32' else ':')
                cuda_related = [p for p in paths if 'cuda' in p.lower()]
                if cuda_related:
                    print(f✅ {var}:")
                    for p in cuda_related:
                        print(f"    {p}")
                else:
                    print(f"❌ {var} 中沒有 CUDA 路徑")
            else:
                print(f"❌ {var}: 未設置")
        else:
            if value != '未設置':
                print(f"✅ {var}: {value}")
            else:
                print(f"❌ {var}: 未設置")

def show_recommendations():
    """顯示建議"""
    print("\n" + "="*60)
    print("修複建議")
    print("="*60)
    print("""
如果 GPU 仍止未棄探到，請按以下步驟操作:

1. 確認 NVIDIA 驅動器:
   - Windows: 下載最新 NVIDIA 驅動 (https://www.nvidia.com/Download/driverDetails.aspx)
   - Linux: sudo apt-get install nvidia-driver-535 (例子)

2. 安裝 CUDA Toolkit 12.0+:
   - 下載: https://developer.nvidia.com/cuda-downloads
   - 選擇你的作業系統和 GPU
   - 例子 (Windows): CUDA Toolkit 12.4
   - 例子 (Linux): CUDA Toolkit 12.4 for Ubuntu 22.04

3. 安裝 cuDNN 8.0+:
   - 下載: https://developer.nvidia.com/cudnn
   - 需要 NVIDIA 賬戶 (免費)
   - 解壓並複制檔案到 CUDA 安裝路徑

4. 重新安裝 TensorFlow-GPU:
   - pip uninstall tensorflow tensorflow-intel tensorflow-macos -y
   - pip install tensorflow[and-cuda]==2.14.0

5. 驗證安裝:
   - python check_gpu.py (例子)
   - 或從 Python 中律驗:
     import tensorflow as tf
     print(tf.config.list_physical_devices('GPU'))

6. 如果仍程探测不到：
   - 棄探 NVIDIA 驅動: nvidia-smi
   - 棄探 CUDA: nvcc --version
   - 棄探環境: echo %CUDA_HOME% (Windows) 或 echo $CUDA_HOME (Linux)
    """)

def main():
    print("\n⚡️  GPU 診斷工具 v1.0\n")
    
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
        print("✅ 恰好! 你的 GPU 配置应該劳作无亙")
        print("你可以開始使用 GPU 訓練 LSTM 了")
    else:
        print("❌ 查出以下問題:")
        if not results['nvidia_gpu']:
            print("  - 找不到 NVIDIA GPU 或驅動")
        if not results['cuda']:
            print("  - CUDA Toolkit 未正確安裝")
        if not results['cudnn']:
            print("  - cuDNN 未正確安裝")
        if not results['tensorflow']:
            print("  - TensorFlow 未棄探到 GPU")
        if not results['tensorflow_build']:
            print("  - TensorFlow 未編譯 CUDA 支援 (這是最常見的問題)")
        
        show_recommendations()

if __name__ == '__main__':
    main()
