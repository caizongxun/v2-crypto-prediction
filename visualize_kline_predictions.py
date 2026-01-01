"""
K 棒預測可視化工具

功能:
  - 將模型預測疊加在 K 棒圖上
  - 顯示上升/下降機率曲線
  - 互動式 Plotly 圖表 (可縮放、平移、懸停)
  - 支持本地 CSV 和 Hugging Face 數據

使用:
  python visualize_kline_predictions.py --pred 1.csv --start 2024-01-01 --end 2024-12-31
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_predictions(csv_path: str) -> pd.DataFrame:
    """
    加載預測結果 CSV
    
    Args:
        csv_path: CSV 文件路徑
    
    Returns:
        pd.DataFrame: 包含 'prediction' 和 'probability' 的數據框
    """
    df = pd.read_csv(csv_path)
    
    required_cols = ['prediction', 'probability']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV 必須包含列: {required_cols}, 實際: {df.columns.tolist()}")
    
    print(f"加載預測數據: {csv_path}")
    print(f"  樣本數: {len(df)}")
    print(f"  預測分佈:")
    print(f"    上升 (1): {(df['prediction'] == 1).sum()} ({(df['prediction'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"    下降 (0): {(df['prediction'] == 0).sum()} ({(df['prediction'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  概率統計:")
    print(f"    最小: {df['probability'].min():.4f}")
    print(f"    最大: {df['probability'].max():.4f}")
    print(f"    平均: {df['probability'].mean():.4f}")
    print()
    
    return df


def load_kline_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    加載 K 線數據 (從本地或 HF)
    
    Args:
        start_date: 開始日期 (YYYY-MM-DD)
        end_date: 結束日期 (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: OHLCV 數據
    """
    try:
        from data import load_btc_data
        from config import HF_TOKEN
        
        print("正在加載 BTC K 線數據...")
        df_kline = load_btc_data(
            hf_token=HF_TOKEN,
            start_date=start_date,
            end_date=end_date
        )
        
        if df_kline is None:
            raise Exception("無法加載 K 線數據")
        
        print(f"K 線數據加載成功: {len(df_kline)} 筆記錄")
        print()
        
        return df_kline
    
    except Exception as e:
        print(f"錯誤: {e}")
        print()
        return None


def align_predictions_with_klines(df_kline: pd.DataFrame, df_pred: pd.DataFrame) -> tuple:
    """
    對齊預測和 K 線數據
    
    Args:
        df_kline: K 線數據 (帶時間索引)
        df_pred: 預測數據
    
    Returns:
        tuple: (對齊後的 K 線, 對齊後的預測)
    """
    # 使用索引進行對齊 (假設同步)
    n_rows = min(len(df_kline), len(df_pred))
    
    print(f"對齊數據:")
    print(f"  K 線記錄: {len(df_kline)}")
    print(f"  預測記錄: {len(df_pred)}")
    print(f"  使用記錄: {n_rows}")
    print()
    
    df_kline_aligned = df_kline.iloc[:n_rows].copy()
    df_pred_aligned = df_pred.iloc[:n_rows].copy()
    
    # 重置索引以便於合併
    df_kline_aligned = df_kline_aligned.reset_index()
    df_pred_aligned = df_pred_aligned.reset_index(drop=True)
    
    # 合併
    df_merged = pd.concat([df_kline_aligned, df_pred_aligned], axis=1)
    
    return df_merged


def create_kline_chart(
    df: pd.DataFrame,
    title: str = "BTC 預測 K 線圖",
    confidence_threshold: float = 0.6,
    show_signals: bool = True
) -> go.Figure:
    """
    創建互動式 K 線圖表，疊加預測信號
    
    Args:
        df: 合併後的數據框 (包含 OHLCV 和預測)
        title: 圖表標題
        confidence_threshold: 信號信心閾值 (>= 此值才標記)
        show_signals: 是否顯示信號標記
    
    Returns:
        go.Figure: Plotly 圖表
    """
    # 創建子圖 (2 行 1 列)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("K 線圖", "預測概率曲線")
    )
    
    # K 線數據
    timestamps = df['timestamp'] if 'timestamp' in df.columns else range(len(df))
    
    # 添加 K 線
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="K 線",
            increasing_line_color="green",
            decreasing_line_color="red"
        ),
        row=1, col=1
    )
    
    # 添加預測概率曲線
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=df['probability'],
            name="上升機率",
            line=dict(color="blue", width=1),
            fill="tozeroy",
            fillcolor="rgba(0, 0, 255, 0.2)"
        ),
        row=2, col=1
    )
    
    # 添加 0.5 參考線 (中性)
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="50% 中性",
        annotation_position="right",
        row=2, col=1
    )
    
    # 添加信號標記
    if show_signals:
        # 強烈看多信號 (prediction=1, prob >= threshold)
        bullish_mask = (df['prediction'] == 1) & (df['probability'] >= confidence_threshold)
        bullish_idx = df[bullish_mask].index.tolist()
        
        if bullish_idx:
            fig.add_trace(
                go.Scatter(
                    x=timestamps[bullish_idx],
                    y=df.loc[bullish_idx, 'low'] * 0.99,  # 在 K 線下方
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="lime",
                        symbol="triangle-up",
                        line=dict(color="darkgreen", width=2)
                    ),
                    name=f"強烈看多 (prob >= {confidence_threshold})",
                    hovertemplate="<b>看多信號</b><br>概率: %{customdata:.2%}<extra></extra>",
                    customdata=df.loc[bullish_idx, 'probability']
                ),
                row=1, col=1
            )
        
        # 強烈看空信號 (prediction=0, prob >= threshold)
        bearish_mask = (df['prediction'] == 0) & ((1 - df['probability']) >= confidence_threshold)
        bearish_idx = df[bearish_mask].index.tolist()
        
        if bearish_idx:
            fig.add_trace(
                go.Scatter(
                    x=timestamps[bearish_idx],
                    y=df.loc[bearish_idx, 'high'] * 1.01,  # 在 K 線上方
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red",
                        symbol="triangle-down",
                        line=dict(color="darkred", width=2)
                    ),
                    name=f"強烈看空 (prob >= {confidence_threshold})",
                    hovertemplate="<b>看空信號</b><br>概率: %{customdata:.2%}<extra></extra>",
                    customdata=1 - df.loc[bearish_idx, 'probability']
                ),
                row=1, col=1
            )
    
    # 更新軸標籤和標題
    fig.update_xaxes(title_text="時間", row=2, col=1)
    fig.update_yaxes(title_text="價格 (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="概率", row=2, col=1)
    
    fig.update_layout(
        title_text=title,
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    return fig


def create_performance_stats(df: pd.DataFrame) -> dict:
    """
    計算性能統計
    
    Args:
        df: 合併後的數據框
    
    Returns:
        dict: 統計信息
    """
    # 計算實際漲跌 (close vs open)
    df_stats = df.copy()
    df_stats['actual_direction'] = (df_stats['close'] >= df_stats['open']).astype(int)
    
    # 準確度
    accuracy = (df_stats['prediction'] == df_stats['actual_direction']).sum() / len(df_stats)
    
    # 按預測分類
    bullish_pred = df_stats[df_stats['prediction'] == 1]
    bearish_pred = df_stats[df_stats['prediction'] == 0]
    
    bullish_acc = (bullish_pred['prediction'] == bullish_pred['actual_direction']).sum() / len(bullish_pred) if len(bullish_pred) > 0 else 0
    bearish_acc = (bearish_pred['prediction'] == bearish_pred['actual_direction']).sum() / len(bearish_pred) if len(bearish_pred) > 0 else 0
    
    # 信心度分析
    high_conf_pred = df_stats[(df_stats['probability'] >= 0.6) | (df_stats['probability'] <= 0.4)]
    high_conf_acc = (high_conf_pred['prediction'] == high_conf_pred['actual_direction']).sum() / len(high_conf_pred) if len(high_conf_pred) > 0 else 0
    
    stats = {
        'total_samples': len(df_stats),
        'overall_accuracy': accuracy,
        'bullish_predictions': len(bullish_pred),
        'bullish_accuracy': bullish_acc,
        'bearish_predictions': len(bearish_pred),
        'bearish_accuracy': bearish_acc,
        'high_confidence_samples': len(high_conf_pred),
        'high_confidence_accuracy': high_conf_acc,
        'avg_probability': df_stats['probability'].mean(),
        'std_probability': df_stats['probability'].std()
    }
    
    return stats


def print_performance_stats(stats: dict):
    """
    打印性能統計
    
    Args:
        stats: 統計字典
    """
    print("\n" + "="*60)
    print("模型性能統計")
    print("="*60)
    print(f"\n總樣本數: {stats['total_samples']}")
    print(f"\n整體準確度: {stats['overall_accuracy']:.2%}")
    print(f"\n預測分佈:")
    print(f"  看多: {stats['bullish_predictions']} 個 ({stats['bullish_predictions']/stats['total_samples']*100:.1f}%) - 準確度: {stats['bullish_accuracy']:.2%}")
    print(f"  看空: {stats['bearish_predictions']} 個 ({stats['bearish_predictions']/stats['total_samples']*100:.1f}%) - 準確度: {stats['bearish_accuracy']:.2%}")
    print(f"\n高信心度預測 (prob >= 0.6 或 <= 0.4):")
    print(f"  樣本數: {stats['high_confidence_samples']} ({stats['high_confidence_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  準確度: {stats['high_confidence_accuracy']:.2%}")
    print(f"\n概率統計:")
    print(f"  平均: {stats['avg_probability']:.4f}")
    print(f"  標準差: {stats['std_probability']:.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="K 棒預測可視化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python visualize_kline_predictions.py --pred 1.csv
  
  # 指定時間範圍
  python visualize_kline_predictions.py --pred 1.csv --start 2024-06-01 --end 2024-12-31
  
  # 調整信號信心閾值
  python visualize_kline_predictions.py --pred 1.csv --threshold 0.7
  
  # 保存為 HTML
  python visualize_kline_predictions.py --pred 1.csv --output chart.html
        """
    )
    
    parser.add_argument(
        "--pred", "-p",
        required=True,
        help="預測結果 CSV 路徑"
    )
    parser.add_argument(
        "--start", "-s",
        default=None,
        help="開始日期 (YYYY-MM-DD, 默認: 自動)"
    )
    parser.add_argument(
        "--end", "-e",
        default=None,
        help="結束日期 (YYYY-MM-DD, 默認: 自動)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.6,
        help="信號信心閾值 (默認: 0.6)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="輸出 HTML 文件路徑 (默認: 在瀏覽器中打開)"
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="不顯示信號標記"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("K 棒預測可視化")
    print("="*60 + "\n")
    
    # 加載預測
    try:
        df_pred = load_predictions(args.pred)
    except Exception as e:
        print(f"錯誤: 無法加載預測 CSV - {e}")
        return
    
    # 加載 K 線數據
    df_kline = load_kline_data(start_date=args.start, end_date=args.end)
    
    if df_kline is None:
        print("警告: 無法加載 K 線數據，使用生成的模擬數據")
        # 生成模擬 K 線數據
        n = len(df_pred)
        base_price = 40000
        prices = base_price + np.cumsum(np.random.randn(n) * 100)
        
        df_kline = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='15min'),
            'open': prices + np.random.randn(n) * 20,
            'high': prices + np.random.randn(n) * 50 + 100,
            'low': prices + np.random.randn(n) * 50 - 100,
            'close': prices + np.random.randn(n) * 20,
            'volume': np.random.uniform(1000, 10000, n)
        })
        df_kline.set_index('timestamp', inplace=True)
    
    # 對齊數據
    df_merged = align_predictions_with_klines(df_kline, df_pred)
    
    # 創建圖表
    print(f"正在創建圖表...")
    fig = create_kline_chart(
        df_merged,
        confidence_threshold=args.threshold,
        show_signals=not args.no_signals
    )
    
    # 計算性能統計
    stats = create_performance_stats(df_merged)
    print_performance_stats(stats)
    
    # 輸出
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"圖表已保存: {output_path}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
