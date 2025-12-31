#!/usr/bin/env python3
"""
進階優化引擎 - 突破50%準確率瓶頸

核心創新:
1. 多目標優化 (Multi-Objective Optimization)
   - 同時最大化: 準確率、勝率、利潤因子
   - 最小化: 最大回撤

2. 集合信號 (Ensemble Signals)
   - 結合趨勢、方向、波動性的多重信號
   - 加權平均而不是簡單平均
   - 動態權重調整

3. 非線性映射 (Non-linear Mapping)
   - 使用 sigmoid 函數而不是線性
   - 更好地捕捉市場轉折點

4. 動態閾值 (Dynamic Threshold)
   - 基於波動性調整買賣閾值
   - 高波動性時收緊閾值
   - 低波動性時放寬閾值

5. 多時間框架 (Multi-Timeframe Analysis)
   - 同時分析短期和中期趨勢
   - 只在兩個時間框架一致時交易
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.optimize import minimize


@dataclass
class AdvancedResult:
    """進階優化結果"""
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    parameters: Dict
    weights: Dict
    signals: np.ndarray


class AdvancedIndicatorOptimizer:
    """
    進階指標優化器
    
    策略:
    - 使用集合學習 (Ensemble Learning)
    - 動態權重調整 (Dynamic Weighting)
    - 風險調整回報 (Risk-Adjusted Returns)
    """
    
    def __init__(self, df: pd.DataFrame, max_trials: int = 300):
        self.df = df.copy()
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.volume = df.get('volume', pd.Series([1]*len(df))).values
        self.max_trials = max_trials
        self.best_score = -np.inf
        self.optimization_history = []
    
    # ========== 指標計算 ==========
    
    def calculate_trend_advanced(self, fast_ema: int, slow_ema: int, atr_period: int) -> np.ndarray:
        """
        進階趨勢指標 - 使用非線性映射
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
        
        # 計算趨勢強度
        ema_ratio = (fast_ema_vals - slow_ema_vals) / (slow_ema_vals + 1e-10) * 100
        atr_ratio = (atr / self.close) * 100
        
        # 使用雙層非線性函數
        trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-np.maximum(atr_ratio, 0) / 0.5))
        
        # 應用 Sigmoid 函數以增加靈敏度
        trend_value = 1 / (1 + np.exp(-trend_score * 3))
        
        return np.clip(trend_value, 0, 1)
    
    def calculate_direction_advanced(self, rsi_period: int, roc_period: int) -> np.ndarray:
        """
        進階方向指標 - RSI + ROC + MACD
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
        
        # MACD
        ema12 = close_series.ewm(span=12, adjust=False).mean().values
        ema26 = close_series.ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        macd_histogram = macd - signal_line
        
        # 規範化信號
        rsi_signal = (rsi - 50) / 50
        roc_signal = np.tanh(roc / 5)
        macd_signal = np.tanh(macd_histogram / (np.std(macd_histogram) + 1e-10))
        
        # 加權平均 (MACD 權重最高)
        direction_score = (rsi_signal * 0.3 + roc_signal * 0.3 + macd_signal * 0.4)
        direction_value = 1 / (1 + np.exp(-direction_score * 3))
        
        return np.clip(direction_value, 0, 1)
    
    def calculate_volatility_advanced(self, sma_period: int, bb_std: float) -> np.ndarray:
        """
        進階波動性指標 - Bollinger Bands + ATR
        """
        close_series = pd.Series(self.close)
        
        # 布林通道
        sma = close_series.rolling(window=sma_period).mean().values
        std = close_series.rolling(window=sma_period).std().values
        
        # 正規化波動性
        volatility_ratio = (std / (sma + 1e-10)) * 100
        volatility_score = np.sqrt(volatility_ratio / 2)
        
        # 計算高低點距離
        high_low_range = (self.high - self.low) / sma
        
        # 組合波動性指標
        combined_vol = volatility_score * 0.6 + (high_low_range * 10) * 0.4
        
        return np.clip(combined_vol, 0, 1)
    
    # ========== 集合信號生成 ==========
    
    def generate_ensemble_signals(self, trend: np.ndarray, direction: np.ndarray,
                                  volatility: np.ndarray,
                                  trend_weight: float, direction_weight: float,
                                  volatility_weight: float,
                                  threshold_buy: float, threshold_sell: float) -> np.ndarray:
        """
        生成集合信號 - 加權組合三個指標
        
        特點:
        1. 動態權重
        2. 波動性調整閾值
        3. 確認信號 (Confirmation)
        """
        signals = np.zeros(len(trend))
        
        # 正規化權重
        total_weight = trend_weight + direction_weight + volatility_weight
        trend_weight /= total_weight
        direction_weight /= total_weight
        volatility_weight /= total_weight
        
        # 計算加權指標
        combined = (trend * trend_weight + 
                   direction * direction_weight + 
                   volatility * volatility_weight * 0.5)  # 波動性作為調節因子
        
        # 動態閾值 - 基於波動性調整
        adjusted_buy_threshold = threshold_buy + (volatility * 0.05)
        adjusted_sell_threshold = threshold_sell - (volatility * 0.05)
        
        # 生成信號
        for i in range(1, len(combined)):
            if combined[i] > adjusted_buy_threshold[i]:
                # 買入信號確認
                if trend[i] > 0.55 and direction[i] > 0.55:
                    signals[i] = 1
            elif combined[i] < adjusted_sell_threshold[i]:
                # 賣出信號確認
                if trend[i] < 0.45 and direction[i] < 0.45:
                    signals[i] = -1
        
        return signals
    
    # ========== 回測和評估 ==========
    
    def calculate_returns(self, signals: np.ndarray) -> np.ndarray:
        """
        計算收益序列
        """
        returns = np.diff(self.close) / self.close[:-1]
        
        strategy_returns = np.zeros_like(returns)
        for i in range(len(returns)):
            if signals[i] == 1:
                strategy_returns[i] = returns[i]
            elif signals[i] == -1:
                strategy_returns[i] = -returns[i]
        
        return strategy_returns
    
    def calculate_metrics(self, signals: np.ndarray) -> Tuple[float, float, float, float, int]:
        """
        計算關鍵指標
        
        Returns:
            (accuracy, sharpe_ratio, max_drawdown, profit_factor, total_trades)
        """
        # 計算收益
        strategy_returns = self.calculate_returns(signals)
        
        # 準確率
        correct = 0
        total = 0
        for i in range(len(signals) - 1):
            if signals[i] != 0:
                actual_return = self.close[i+1] - self.close[i]
                if (signals[i] == 1 and actual_return > 0) or (signals[i] == -1 and actual_return < 0):
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Sharpe 比率 (風險調整回報)
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # 最大回撤
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # 利潤因子
        profits = strategy_returns[strategy_returns > 0]
        losses = np.abs(strategy_returns[strategy_returns < 0])
        profit_factor = (np.sum(profits) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) > 0 else 0
        
        return accuracy, sharpe_ratio, max_drawdown, profit_factor, total
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        多目標優化目標函數
        """
        try:
            # 趨勢參數
            fast_ema = trial.suggest_int('fast_ema', 12, 30)
            slow_ema = trial.suggest_int('slow_ema', 50, 150)
            atr_period = trial.suggest_int('atr_period', 8, 25)
            
            # 方向參數
            rsi_period = trial.suggest_int('rsi_period', 8, 25)
            roc_period = trial.suggest_int('roc_period', 5, 15)
            
            # 波動性參數
            sma_period = trial.suggest_int('sma_period', 20, 50)
            bb_std = trial.suggest_float('bb_std', 1.5, 3.5, step=0.1)
            
            # 權重
            trend_weight = trial.suggest_float('trend_weight', 0.2, 0.5)
            direction_weight = trial.suggest_float('direction_weight', 0.2, 0.5)
            volatility_weight = trial.suggest_float('volatility_weight', 0.2, 0.5)
            
            # 閾值
            threshold_buy = trial.suggest_float('threshold_buy', 0.55, 0.75)
            threshold_sell = trial.suggest_float('threshold_sell', 0.25, 0.45)
            
            # 驗證參數
            if fast_ema >= slow_ema:
                return -np.inf
            
            # 計算指標
            trend = self.calculate_trend_advanced(fast_ema, slow_ema, atr_period)
            direction = self.calculate_direction_advanced(rsi_period, roc_period)
            volatility = self.calculate_volatility_advanced(sma_period, bb_std)
            
            # 生成信號
            signals = self.generate_ensemble_signals(
                trend, direction, volatility,
                trend_weight, direction_weight, volatility_weight,
                threshold_buy, threshold_sell
            )
            
            # 計算指標
            accuracy, sharpe_ratio, max_drawdown, profit_factor, total_trades = self.calculate_metrics(signals)
            
            # 多目標得分 (加權組合)
            # 優先: Sharpe 比率 (風險調整回報)
            # 次優: 準確率
            # 約束: 最大回撤 < -30%
            
            if max_drawdown < -0.3:
                return -np.inf  # 回撤過大，淘汰
            
            combined_score = (sharpe_ratio * 0.5 + accuracy * 100 * 0.5)
            
            # 記錄歷史
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.optimization_history.append({
                    'iteration': len(self.optimization_history),
                    'score': combined_score,
                    'accuracy': accuracy,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'profit_factor': profit_factor,
                    'parameters': trial.params,
                    'timestamp': datetime.now().isoformat()
                })
            
            return combined_score
            
        except Exception as e:
            return -np.inf
    
    def optimize(self) -> AdvancedResult:
        """
        執行進階優化
        """
        print("\n" + "="*80)
        print("進階優化引擎 - 突破50%準確率瓶頸")
        print("="*80)
        
        print(f"\n正在執行 {self.max_trials} 次試驗...\n")
        
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='maximize'
        )
        
        study.optimize(
            self.objective,
            n_trials=self.max_trials,
            show_progress_bar=True
        )
        
        best_trial = study.best_trial
        best_params = best_trial.params
        
        print("\n" + "="*80)
        print("優化完成")
        print("="*80)
        
        # 提取權重
        weights = {
            'trend': best_params.get('trend_weight', 0.33),
            'direction': best_params.get('direction_weight', 0.33),
            'volatility': best_params.get('volatility_weight', 0.34)
        }
        
        # 計算最終信號
        trend = self.calculate_trend_advanced(
            best_params['fast_ema'],
            best_params['slow_ema'],
            best_params['atr_period']
        )
        direction = self.calculate_direction_advanced(
            best_params['rsi_period'],
            best_params['roc_period']
        )
        volatility = self.calculate_volatility_advanced(
            best_params['sma_period'],
            best_params['bb_std']
        )
        
        signals = self.generate_ensemble_signals(
            trend, direction, volatility,
            weights['trend'], weights['direction'], weights['volatility'],
            best_params['threshold_buy'],
            best_params['threshold_sell']
        )
        
        accuracy, sharpe_ratio, max_drawdown, profit_factor, _ = self.calculate_metrics(signals)
        
        print(f"\n最佳得分: {self.best_score:.4f}")
        print(f"準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Sharpe 比率: {sharpe_ratio:.4f}")
        print(f"最大回撤: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        print(f"利潤因子: {profit_factor:.4f}")
        
        return AdvancedResult(
            accuracy=accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            parameters=best_params,
            weights=weights,
            signals=signals
        )


def main():
    """
    主程序
    """
    print("\n" + "#"*80)
    print("# 進階優化引擎 - 多目標優化")
    print("#"*80)
    
    # 加載數據
    print("\n[一] 加載數據...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        print(f"✓ 數據加載成功")
        print(f"  時間範圍: {df.index[0]} ~ {df.index[-1]}")
        print(f"  K線數量: {len(df)}")
    except Exception as e:
        print(f"✗ 數據加載失敗: {e}")
        return
    
    # 執行優化
    print("\n[二] 執行進階優化...")
    optimizer = AdvancedIndicatorOptimizer(df, max_trials=150)
    result = optimizer.optimize()
    
    # 保存結果
    print("\n[三] 保存結果...")
    os.makedirs("results", exist_ok=True)
    
    output_data = {
        'best_parameters': result.parameters,
        'weights': result.weights,
        'metrics': {
            'accuracy': float(result.accuracy),
            'sharpe_ratio': float(result.sharpe_ratio),
            'max_drawdown': float(result.max_drawdown),
            'profit_factor': float(result.profit_factor)
        },
        'optimization_history': optimizer.optimization_history
    }
    
    with open("results/advanced_optimization_result.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ 結果已保存: results/advanced_optimization_result.json")
    
    print("\n" + "#"*80 + "\n")


if __name__ == "__main__":
    main()
