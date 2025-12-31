#!/usr/bin/env python3
"""
回測驗證器 - 驗證遺傳算法是否過擬合

目標:
1. 分割數據魚誓 (In-Sample) 和 驗證 (Out-of-Sample)
2. 用 In-Sample 數據テスト優化的配方
3. 用 Out-of-Sample 數據驗證真實表現
4. 比较鍩玩上下文 (漲跌)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime
import sys


class BacktestValidator:
    """
    回測驗證器
    """
    
    def __init__(self, df: pd.DataFrame, best_gene: Dict):
        """
        Args:
            df: 全体 OHLCV 数据
            best_gene: 优化器找到的最优基因
        """
        self.df = df.copy()
        self.best_gene = best_gene
        self.results = {}
    
    def split_data(self, split_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分割数据为 In-Sample 和 Out-of-Sample
        
        Args:
            split_ratio: In-Sample 批次 (0.7 = 70% In-Sample, 30% Out-of-Sample)
        
        Returns:
            (in_sample_df, out_of_sample_df)
        """
        split_point = int(len(self.df) * split_ratio)
        
        in_sample = self.df.iloc[:split_point]
        out_of_sample = self.df.iloc[split_point:]
        
        return in_sample, out_of_sample
    
    def calculate_indicators(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        計算三個指標
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 趨勢指標
        close_series = pd.Series(close)
        fast_ema = close_series.ewm(span=int(self.best_gene['fast_ema']), adjust=False).mean().values
        slow_ema = close_series.ewm(span=int(self.best_gene['slow_ema']), adjust=False).mean().values
        
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        tr_series = pd.Series(tr)
        atr = tr_series.ewm(span=int(self.best_gene['atr_period']), adjust=False).mean().values
        
        ema_ratio = (fast_ema - slow_ema) / (slow_ema + 1e-10) * 100
        atr_ratio = (atr / close) * 100
        
        trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-np.maximum(atr_ratio, 0) / 0.5))
        trend_value = 1 / (1 + np.exp(-trend_score * self.best_gene['sigmoid_strength']))
        
        # 方向指標
        delta = close_series.diff().values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.ewm(span=int(self.best_gene['rsi_period']), adjust=False).mean().values
        avg_loss = loss_series.ewm(span=int(self.best_gene['roc_period']), adjust=False).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        roc = (close - np.roll(close, int(self.best_gene['roc_period']))) / (np.roll(close, int(self.best_gene['roc_period'])) + 1e-10) * 100
        
        ema12 = close_series.ewm(span=12, adjust=False).mean().values
        ema26 = close_series.ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        macd_histogram = macd - signal_line
        
        rsi_signal = (rsi - 50) / 50
        roc_signal = np.tanh(roc / 5)
        macd_signal = np.tanh(macd_histogram / (np.std(macd_histogram) + 1e-10))
        
        direction_score = (rsi_signal * 0.3 + roc_signal * 0.3 + macd_signal * 0.4)
        direction_value = 1 / (1 + np.exp(-direction_score * self.best_gene['sigmoid_strength']))
        
        # 波動性指標
        sma = close_series.rolling(window=int(self.best_gene['sma_period'])).mean().values
        std = close_series.rolling(window=int(self.best_gene['sma_period'])).std().values
        
        volatility_ratio = (std / (sma + 1e-10)) * 100
        volatility_score = np.sqrt(volatility_ratio / 2)
        
        high_low_range = (high - low) / sma
        combined_vol = volatility_score * 0.6 + (high_low_range * 10) * 0.4
        volatility_value = np.clip(combined_vol, 0, 1)
        
        return np.clip(trend_value, 0, 1), np.clip(direction_value, 0, 1), volatility_value
    
    def generate_signals(self, trend: np.ndarray, direction: np.ndarray, volatility: np.ndarray) -> np.ndarray:
        """
        生成交易信號
        """
        signals = np.zeros(len(trend))
        
        total_weight = self.best_gene['trend_weight'] + self.best_gene['direction_weight'] + self.best_gene['volatility_weight']
        trend_w = self.best_gene['trend_weight'] / total_weight
        direction_w = self.best_gene['direction_weight'] / total_weight
        volatility_w = self.best_gene['volatility_weight'] / total_weight
        
        combined = (trend * trend_w + direction * direction_w + volatility * volatility_w * 0.5)
        
        adjusted_buy_threshold = self.best_gene['threshold_buy'] + (volatility * 0.05)
        adjusted_sell_threshold = self.best_gene['threshold_sell'] - (volatility * 0.05)
        
        for i in range(1, len(combined)):
            if combined[i] > adjusted_buy_threshold[i]:
                if (trend[i] > self.best_gene['confirmation_strength'] and 
                    direction[i] > self.best_gene['confirmation_strength']):
                    signals[i] = 1
            elif combined[i] < adjusted_sell_threshold[i]:
                if (trend[i] < (1 - self.best_gene['confirmation_strength']) and 
                    direction[i] < (1 - self.best_gene['confirmation_strength'])):
                    signals[i] = -1
        
        return signals
    
    def calculate_metrics(self, signals: np.ndarray, close_prices: np.ndarray) -> Dict:
        """
        計算評估指標
        """
        # 準確率
        correct = 0
        total = 0
        for i in range(len(signals) - 1):
            if signals[i] != 0:
                actual_return = close_prices[i+1] - close_prices[i]
                if (signals[i] == 1 and actual_return > 0) or (signals[i] == -1 and actual_return < 0):
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # 回報
        strategy_returns = np.zeros(len(close_prices) - 1)
        for i in range(len(signals) - 1):
            if signals[i] == 1:
                strategy_returns[i] = (close_prices[i+1] - close_prices[i]) / close_prices[i]
            elif signals[i] == -1:
                strategy_returns[i] = -(close_prices[i+1] - close_prices[i]) / close_prices[i]
        
        cumulative_returns = np.cumprod(1 + strategy_returns)
        total_return = (cumulative_returns[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0
        
        # Sharpe 比率
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # 最大回撤
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # 买卖橫打
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        
        return {
            'accuracy': accuracy,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_signals': total,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def validate(self) -> Dict:
        """
        執行完整回測驗證
        """
        print("\n" + "="*80)
        print("回測驗證 - 播鼠橫打策略")
        print("="*80)
        
        # 分割数据
        print("\n[一] 分割數据...")
        in_sample, out_of_sample = self.split_data(split_ratio=0.7)
        
        print(f"✓ In-Sample: {len(in_sample):6d} K线 ({len(in_sample)/len(self.df)*100:.1f}%)")
        print(f"  时间: {in_sample.index[0]} ~ {in_sample.index[-1]}")
        print(f"\n✓ Out-of-Sample: {len(out_of_sample):6d} K线 ({len(out_of_sample)/len(self.df)*100:.1f}%)")
        print(f"  时间: {out_of_sample.index[0]} ~ {out_of_sample.index[-1]}")
        
        # In-Sample 回測
        print("\n[二] In-Sample 回測...")
        trend_is, direction_is, volatility_is = self.calculate_indicators(in_sample)
        signals_is = self.generate_signals(trend_is, direction_is, volatility_is)
        metrics_is = self.calculate_metrics(signals_is, in_sample['close'].values)
        
        print(f"\n  准确率: {metrics_is['accuracy']*100:6.2f}%")
        print(f"  总回报: {metrics_is['total_return_pct']:7.2f}%")
        print(f"  Sharpe 比率: {metrics_is['sharpe_ratio']:7.4f}")
        print(f"  最大回撤: {metrics_is['max_drawdown']*100:7.2f}%")
        print(f"  交易信號: {metrics_is['total_signals']:6d} (买: {metrics_is['buy_signals']}, 卖: {metrics_is['sell_signals']})")
        
        self.results['in_sample'] = metrics_is
        
        # Out-of-Sample 回測
        print("\n[三] Out-of-Sample 回測 (此后文根据)...")
        trend_oos, direction_oos, volatility_oos = self.calculate_indicators(out_of_sample)
        signals_oos = self.generate_signals(trend_oos, direction_oos, volatility_oos)
        metrics_oos = self.calculate_metrics(signals_oos, out_of_sample['close'].values)
        
        print(f"\n  准确率: {metrics_oos['accuracy']*100:6.2f}%")
        print(f"  总回报: {metrics_oos['total_return_pct']:7.2f}%")
        print(f"  Sharpe 比率: {metrics_oos['sharpe_ratio']:7.4f}")
        print(f"  最大回撤: {metrics_oos['max_drawdown']*100:7.2f}%")
        print(f"  交易信號: {metrics_oos['total_signals']:6d} (买: {metrics_oos['buy_signals']}, 卖: {metrics_oos['sell_signals']})")
        
        self.results['out_of_sample'] = metrics_oos
        
        # 比较
        print("\n[四] 比较 In-Sample vs Out-of-Sample")
        print("="*80)
        
        accuracy_diff = (metrics_is['accuracy'] - metrics_oos['accuracy']) * 100
        sharpe_diff = metrics_is['sharpe_ratio'] - metrics_oos['sharpe_ratio']
        
        print(f"\n准确率变化: {accuracy_diff:+7.2f}% ({metrics_is['accuracy']*100:6.2f}% -> {metrics_oos['accuracy']*100:6.2f}%)")
        print(f"Sharpe 比率变化: {sharpe_diff:+7.4f} ({metrics_is['sharpe_ratio']:7.4f} -> {metrics_oos['sharpe_ratio']:7.4f})")
        print(f"\u56de报变化: {metrics_is['total_return_pct']:+7.2f}% -> {metrics_oos['total_return_pct']:+7.2f}%")
        
        # 过负合检测
        print("\n[五] 过负合检测")
        print("="*80)
        
        if accuracy_diff > 10:
            print("\n⚠ï 警告: 检测到明显的过负合迹象!")
            print(f"  In-Sample 准确率比 Out-of-Sample 高出 {accuracy_diff:.2f}%")
            print(f"  建议: 扩大上一丶撨數据或使用更保守的參数")
        else:
            print(f"\n✓ 模型泛化性一般")
            print(f"  In-Sample 和 Out-of-Sample 准确率接近")
        
        # 保存结果
        print("\n[六] 保存结果...")
        output_data = {
            'best_formula_combination': self.best_gene,
            'in_sample_metrics': self.results['in_sample'],
            'out_of_sample_metrics': self.results['out_of_sample'],
            'comparison': {
                'accuracy_difference_pct': accuracy_diff,
                'sharpe_ratio_difference': sharpe_diff,
                'overfitting_detected': accuracy_diff > 10
            },
            'data_split': {
                'in_sample_count': len(in_sample),
                'out_of_sample_count': len(out_of_sample),
                'total_count': len(self.df),
                'in_sample_ratio': 0.7
            },
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs("results", exist_ok=True)
        with open("results/backtest_validation.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ 结果已保存: results/backtest_validation.json")
        
        return output_data


def main():
    print("\n" + "#"*80)
    print("# 回測驗證器 - 驗證遺傳算法是否過擬合")
    print("#"*80)
    
    # 加載数据
    print("\n[一] 加載数据和优化结果...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # 加載优化结果
        with open("results/genetic_algorithm_result.json", 'r', encoding='utf-8') as f:
            ga_result = json.load(f)
        
        best_gene = ga_result['best_formula_combination']
        
        print(f"✓ 数据加載成功")
        print(f"  K线数量: {len(df)}")
        print(f"\n✓ 优化结果加載成功")
        print(f"  In-Sample 准确率: {best_gene['accuracy']*100:.2f}%")
        
    except Exception as e:
        print(f"✗ 加載失败: {e}")
        return
    
    # 执行驗證
    print("\n[二] 执行回測驗證...")
    validator = BacktestValidator(df, best_gene)
    results = validator.validate()
    
    print("\n" + "#"*80 + "\n")


if __name__ == "__main__":
    main()
