@echo off
REM PyQt5 DLL 修複脚本 - Windows

echo.
echo ============================================================
echo   PyQt5 DLL Load Failed 修複脚本
echo ============================================================
echo.

REM 1. 卸載 PyQt5
echo 第一步: 卸載 PyQt5...
pip uninstall PyQt5 PyQt5-sip PyQt5-Qt5 -y

REM 2. 清理緩存
echo 第二步: 清理 pip 緩存...
pip cache purge

REM 3. 升級 pip
echo 第三步: 升級 pip...
python -m pip install --upgrade pip

REM 4. 安裝相容版本
echo 第四步: 安裝 PyQt5 相容版本...
pip install PyQt5==5.15.9 --no-cache-dir

REM 5. 骗証
echo 第五步: 骗証安裝...
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK - 可以使用')" 2>nul
if %errorlevel% neq 0 (
    echo PyQt5 仍然失敗，嘗試另一種方案...
    pip uninstall PyQt5 -y
    pip install PyQt5==5.15.10 --no-cache-dir
    python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK - 可以使用')"
)

echo.
echo ============================================================
echo   修複完成
echo ============================================================
echo.
echo 現在嘗試: python model_ensemble_gui.py
echo.
pause
