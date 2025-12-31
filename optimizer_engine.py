#!/usr/bin/env python3
"""
指標適化引擎

通過不斷回測优化指標參数，使准確率达到峰值

機制:
1. 適化趨勢指標 (Trend)
2. 適化方向指標 (Direction)
3. 適化波動性指標 (Volatility)
4. 导出最优组合信號
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import os
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


@dataclass
class OptimizationResult:
    """优化结果"""
    accuracy: float
    parameters: Dict
    best_threshold_buy: float
    best_threshold_sell: float
    win_rate: float
    profit_factor: float
    max_trades: int
    

class IndicatorOptimizer:
    """
    指標適化引擎
    
    目标: 找到使准確率最高的指標參数组合
    """
    
    def __init__(self, df: pd.DataFrame, max_trials: int = 200):
        """
        Args:
            df: OHLCV 数据
            max_trials: 最大试验次数
        """
        self.df = df
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.max_trials = max_trials
        self.best_accuracy = 0
        self.optimization_history = []
    
    def calculate_trend_fast(self, fast_ema: int, slow_ema: int, atr_period: int) -> np.ndarray:
        """
        快速計算趨勢指標
        """
        close_series = pd.Series(self.close)
        
        fast_ema_vals = close_series.ewm(span=fast_ema, adjust=False).mean().values
        slow_ema_vals = close_series.ewm(span=slow_ema, adjust=False).mean().values
        
        # ATR
        high_low = self.high - self.low
        high_close = np.abs(self.high - np.roll(self.close, 1))
        low_close = np.abs(self.low - np.roll(self.close, 1))
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        tr_series = pd.Series(tr)
        atr = tr_series.ewm(span=atr_period, adjust=False).mean().values
        
        # 趨勢指標
        ema_ratio = (fast_ema_vals - slow_ema_vals) / (slow_ema_vals + 1e-10) * 100
        atr_ratio = (atr / self.close) * 100
        
        trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-np.maximum(atr_ratio, 0) / 0.5))
        trend_value = (trend_score + 1) / 2
        
        return np.clip(trend_value, 0, 1)
    
    def calculate_direction_fast(self, rsi_period: int, roc_period: int) -> np.ndarray:
        """
        快速計算方向指標
        """
        close_series = pd.Series(self.close)
        
        # RSI
        delta = close_series.diff().values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.ewm(span=rsi_period, adjust=False).mean().values
        avg_loss = loss_series.ewm(span=rsi_period, adjust=False).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # ROC
        roc = (self.close - np.roll(self.close, roc_period)) / (np.roll(self.close, roc_period) + 1e-10) * 100
        
        rsi_signal = (rsi - 50) / 50
        roc_signal = np.tanh(roc / 5)
        direction_score = (rsi_signal + roc_signal) / 2
        direction_value = (direction_score + 1) / 2
        
        return np.clip(direction_value, 0, 1)
    
    def calculate_volatility_fast(self, sma_period: int, bb_std: float) -> np.ndarray:
        """
        快速計算波動性指標
        """
        close_series = pd.Series(self.close)
        
        sma = close_series.rolling(window=sma_period).mean().values
        std = close_series.rolling(window=sma_period).std().values
        
        volatility_ratio = (std / (sma + 1e-10)) * 100
        volatility_score = np.sqrt(volatility_ratio / 2)
        
        return np.clip(volatility_score, 0, 1)
    
    def generate_signals(self, trend: np.ndarray, direction: np.ndarray, 
                        threshold_buy: float, threshold_sell: float) -> np.ndarray:
        """
        生成交易信號
        
        Returns:
            np.ndarray: 1 = 買入, -1 = 賣出, 0 = 空仕
        """
        signals = np.zeros(len(trend))
        
        for i in range(len(trend)):
            combined = (trend[i] + direction[i]) / 2
            
            if combined > threshold_buy:
                signals[i] = 1
            elif combined < threshold_sell:
                signals[i] = -1
        
        return signals
    
    def backtest(self, signals: np.ndarray) -> Tuple[float, float, float, int]:
        """
        回測上下趨性
        
        Returns:
            Tuple: (accuracy, win_rate, profit_factor, max_consecutive_wins)
        """
        correct = 0
        total = 0
        
        wins = 0
        losses = 0
        
        for i in range(len(signals) - 1):
            if signals[i] != 0:
                # 下一根 K 線的实际涨跌
                actual_return = self.close[i+1] - self.close[i]
                
                # 预测是否正确
                if (signals[i] == 1 and actual_return > 0) or (signals[i] == -1 and actual_return < 0):
                    correct += 1
                    wins += 1
                else:
                    losses += 1
                
                total += 1
        
        # 計算准确率
        accuracy = correct / total if total > 0 else 0
        
        # 計算胜率
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # 計算豪利因子
        profit_factor = 1.0  # 简化計算
        
        return accuracy, win_rate, profit_factor, total
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna 优化目标函数
        """
        try:
            # 趨勢參数
            fast_ema = trial.suggest_int('fast_ema', 10, 30)
            slow_ema = trial.suggest_int('slow_ema', 40, 150)
            atr_period = trial.suggest_int('atr_period', 5, 20)
            
            # 方向參数
            rsi_period = trial.suggest_int('rsi_period', 5, 20)
            roc_period = trial.suggest_int('roc_period', 5, 15)
            
            # 波動性參数
            sma_period = trial.suggest_int('sma_period', 20, 50)
            bb_std = trial.suggest_float('bb_std', 1.5, 3.5, step=0.1)
            
            # 信號閾值
            threshold_buy = trial.suggest_float('threshold_buy', 0.55, 0.75)
            threshold_sell = trial.suggest_float('threshold_sell', 0.25, 0.45)
            
            if fast_ema >= slow_ema:
                return 0
            
            # 計算指標
            trend = self.calculate_trend_fast(fast_ema, slow_ema, atr_period)
            direction = self.calculate_direction_fast(rsi_period, roc_period)
            volatility = self.calculate_volatility_fast(sma_period, bb_std)
            
            # 遣成信號
            signals = self.generate_signals(trend, direction, threshold_buy, threshold_sell)
            
            # 回測
            accuracy, win_rate, profit_factor, total_trades = self.backtest(signals)
            
            # 記錄优化历史
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.optimization_history.append({
                    'iteration': len(self.optimization_history),
                    'accuracy': accuracy,
                    'win_rate': win_rate,
                    'parameters': trial.params,
                    'timestamp': datetime.now().isoformat()
                })
            
            return accuracy
            
        except Exception as e:
            return 0
    
    def optimize(self) -> OptimizationResult:
        """
        執行优化
        """
        print("\n" + "="*80)
        print("指標適化引擎 - 优化指標參数")
        print("="*80)
        
        print(f"\n正在執行 {self.max_trials} 次试验...\n")
        
        # 创建 Optuna 研究
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='maximize'
        )
        
        # 优化
        study.optimize(
            self.objective,
            n_trials=self.max_trials,
            show_progress_bar=True
        )
        
        # 获取最优结果
        best_trial = study.best_trial
        best_params = best_trial.params
        best_accuracy = best_trial.value
        
        print("\n" + "="*80)
        print("优化完成")
        print("="*80)
        
        print(f"\n最高准確率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"\n最优參数:")
        for key, value in sorted(best_params.items()):
            if isinstance(value, float):
                print(f"  {key:25s} = {value:.4f}")
            else:
                print(f"  {key:25s} = {value}")
        
        # 进行最终回測
        print(f"\n正在执行最终回測...")
        
        trend = self.calculate_trend_fast(
            best_params['fast_ema'],
            best_params['slow_ema'],
            best_params['atr_period']
        )
        direction = self.calculate_direction_fast(
            best_params['rsi_period'],
            best_params['roc_period']
        )
        signals = self.generate_signals(
            trend, direction,
            best_params['threshold_buy'],
            best_params['threshold_sell']
        )
        
        accuracy, win_rate, profit_factor, max_trades = self.backtest(signals)
        
        return OptimizationResult(
            accuracy=accuracy,
            parameters=best_params,
            best_threshold_buy=best_params['threshold_buy'],
            best_threshold_sell=best_params['threshold_sell'],
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_trades=max_trades
        )


def main():
    """
    主程序
    """
    print("\n" + "#"*80)
    print("# 指標適化引擎 - 不断优化直到准確率达到峰值")
    print("#"*80)
    
    # 加載数据
    print("\n[一] 加載数据...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        
        # 過濾日期
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        print(f"✓ 数据加載成功")
        print(f"  時間範围: {df.index[0]} ~ {df.index[-1]}")
        print(f"  K 线数量: {len(df)}")
    except Exception as e:
        print(f"✗ 数据加載失败: {e}")
        return
    
    # 適化指標
    print("\n[二] 优化指標...")
    optimizer = IndicatorOptimizer(df, max_trials=100)
    result = optimizer.optimize()
    
    # 打印结果
    print(f"\n准确率: {result.accuracy:.4f}")
    print(f"胜率: {result.win_rate:.4f}")
    print(f"交易数: {result.max_trades}")
    
    # 保存结果
    print("\n[三] 保存结果...")
    os.makedirs("results", exist_ok=True)
    
    output_file = "results/optimization_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_parameters': result.parameters,
            'best_accuracy': result.accuracy,
            'win_rate': result.win_rate,
            'max_trades': result.max_trades,
            'optimization_history': optimizer.optimization_history
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ 结果已保存: {output_file}")
    
    print("\n" + "#"*80 + "\n")


if __name__ == "__main__":
    main()
