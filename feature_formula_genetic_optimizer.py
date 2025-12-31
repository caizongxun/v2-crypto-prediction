#!/usr/bin/env python3
"""
特徵公式遺傳優化器

目標: 演化出3套獨立的公式指標
  1. 波動性公式: 衡量價格波動大小 (0-1)
  2. 趨勢公式: 衡量趨勢強度 (0-1)
  3. 方向公式: 衡量方向確定性 (0-1)

這3套公式將作為特徵輸入到後續的機器學習模型中
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json
import os
from datetime import datetime
import random
from copy import deepcopy


@dataclass
class FeatureFormula:
    """特徵公式基因"""
    # 波動性參數
    volatility_atr_period: int
    volatility_bb_period: int
    volatility_bb_std: float
    volatility_roc_period: int
    
    # 趨勢參數
    trend_fast_ema: int
    trend_slow_ema: int
    trend_macd_signal: int
    trend_adx_period: int
    
    # 方向參數
    direction_rsi_period: int
    direction_stoch_k: int
    direction_stoch_d: int
    direction_roc_period: int
    
    # 公式權重組合
    volatility_weights: Dict[str, float]  # {atr, bb, roc}
    trend_weights: Dict[str, float]  # {ema, macd, adx}
    direction_weights: Dict[str, float]  # {rsi, stoch, roc}
    
    # 性能指標
    volatility_correlation: float = 0.0
    trend_correlation: float = 0.0
    direction_correlation: float = 0.0
    overall_fitness: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'volatility_params': {
                'atr_period': self.volatility_atr_period,
                'bb_period': self.volatility_bb_period,
                'bb_std': self.volatility_bb_std,
                'roc_period': self.volatility_roc_period,
                'weights': self.volatility_weights
            },
            'trend_params': {
                'fast_ema': self.trend_fast_ema,
                'slow_ema': self.trend_slow_ema,
                'macd_signal': self.trend_macd_signal,
                'adx_period': self.trend_adx_period,
                'weights': self.trend_weights
            },
            'direction_params': {
                'rsi_period': self.direction_rsi_period,
                'stoch_k': self.direction_stoch_k,
                'stoch_d': self.direction_stoch_d,
                'roc_period': self.direction_roc_period,
                'weights': self.direction_weights
            },
            'performance': {
                'volatility_correlation': self.volatility_correlation,
                'trend_correlation': self.trend_correlation,
                'direction_correlation': self.direction_correlation,
                'overall_fitness': self.overall_fitness
            }
        }


class FeatureFormulaOptimizer:
    """
    特徵公式優化器
    目標: 找到最優的3套特徵公式
    """
    
    def __init__(self, df: pd.DataFrame, population_size: int = 30, generations: int = 50):
        self.df = df.copy()
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.population_size = population_size
        self.generations = generations
        self.population: List[FeatureFormula] = []
        self.evolution_history = []
    
    def calculate_volatility_formula(self, gene: FeatureFormula) -> np.ndarray:
        """
        計算波動性公式
        輸出: [0, 1] 的波動度分數
        
        組成:
        1. ATR (Average True Range): 絕對波動
        2. Bollinger Bands 寬度: 相對波動
        3. ROC 變化率: 速度變化
        """
        close_series = pd.Series(self.close)
        
        # 1. ATR 計算
        high_low = self.high - self.low
        high_close = np.abs(self.high - np.roll(self.close, 1))
        low_close = np.abs(self.low - np.roll(self.close, 1))
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        tr_series = pd.Series(tr)
        atr = tr_series.ewm(span=gene.volatility_atr_period, adjust=False).mean().values
        atr_normalized = (atr / (self.close + 1e-10)) * 100
        atr_signal = np.clip(atr_normalized / np.percentile(atr_normalized[100:], 75), 0, 1)
        
        # 2. Bollinger Bands 寬度
        sma = close_series.rolling(window=gene.volatility_bb_period).mean().values
        std = close_series.rolling(window=gene.volatility_bb_period).std().values
        bb_width = (2 * gene.volatility_bb_std * std) / (sma + 1e-10)
        bb_signal = np.clip(bb_width / np.percentile(bb_width[100:], 75), 0, 1)
        
        # 3. ROC 變化率
        roc = (self.close - np.roll(self.close, gene.volatility_roc_period)) / \
              (np.roll(self.close, gene.volatility_roc_period) + 1e-10) * 100
        roc_abs = np.abs(roc)
        roc_signal = np.clip(roc_abs / np.percentile(roc_abs[100:], 75), 0, 1)
        
        # 權重組合
        weights = gene.volatility_weights
        total_weight = weights['atr'] + weights['bb'] + weights['roc']
        volatility_score = (
            atr_signal * weights['atr'] +
            bb_signal * weights['bb'] +
            roc_signal * weights['roc']
        ) / total_weight
        
        return np.clip(volatility_score, 0, 1)
    
    def calculate_trend_formula(self, gene: FeatureFormula) -> np.ndarray:
        """
        計算趨勢公式
        輸出: [0, 1] 的趨勢強度分數
        
        組成:
        1. EMA 差異: 快慢線差
        2. MACD: 動量指標
        3. ADX: 趨勢方向指數
        """
        close_series = pd.Series(self.close)
        
        # 1. EMA 差異
        fast_ema = close_series.ewm(span=gene.trend_fast_ema, adjust=False).mean().values
        slow_ema = close_series.ewm(span=gene.trend_slow_ema, adjust=False).mean().values
        ema_ratio = (fast_ema - slow_ema) / (slow_ema + 1e-10) * 100
        ema_signal = np.tanh(ema_ratio / 5)
        ema_signal = (ema_signal + 1) / 2  # 轉換到 [0, 1]
        
        # 2. MACD
        ema12 = close_series.ewm(span=12, adjust=False).mean().values
        ema26 = close_series.ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=gene.trend_macd_signal, adjust=False).mean().values
        macd_histogram = macd - signal_line
        macd_signal = np.tanh(macd_histogram / (np.std(macd_histogram) + 1e-10) / 5)
        macd_signal = (macd_signal + 1) / 2
        
        # 3. ADX (簡化版)
        delta = close_series.diff().values
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        plus_di = pd.Series(up).rolling(window=gene.trend_adx_period).sum().values / \
                  (pd.Series(np.abs(delta)).rolling(window=gene.trend_adx_period).sum().values + 1e-10) * 100
        minus_di = pd.Series(down).rolling(window=gene.trend_adx_period).sum().values / \
                   (pd.Series(np.abs(delta)).rolling(window=gene.trend_adx_period).sum().values + 1e-10) * 100
        
        di_sum = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_signal = np.clip(di_sum, 0, 1)
        
        # 權重組合
        weights = gene.trend_weights
        total_weight = weights['ema'] + weights['macd'] + weights['adx']
        trend_score = (
            ema_signal * weights['ema'] +
            macd_signal * weights['macd'] +
            adx_signal * weights['adx']
        ) / total_weight
        
        return np.clip(trend_score, 0, 1)
    
    def calculate_direction_formula(self, gene: FeatureFormula) -> np.ndarray:
        """
        計算方向公式
        輸出: [0, 1] 的方向確定性分數 (0=看跌, 0.5=中性, 1=看漲)
        
        組成:
        1. RSI: 超買超賣
        2. Stochastic: 動量強度
        3. ROC: 方向速度
        """
        close_series = pd.Series(self.close)
        
        # 1. RSI
        delta = close_series.diff().values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.ewm(span=gene.direction_rsi_period, adjust=False).mean().values
        avg_loss = loss_series.ewm(span=gene.direction_rsi_period, adjust=False).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = rsi / 100  # 轉換到 [0, 1]
        
        # 2. Stochastic
        low_min = close_series.rolling(window=gene.direction_stoch_k).min().values
        high_max = close_series.rolling(window=gene.direction_stoch_k).max().values
        
        stoch_k = (self.close - low_min) / (high_max - low_min + 1e-10) * 100
        stoch_k_series = pd.Series(stoch_k)
        stoch_d = stoch_k_series.rolling(window=gene.direction_stoch_d).mean().values
        stoch_signal = stoch_d / 100
        
        # 3. ROC
        roc = (self.close - np.roll(self.close, gene.direction_roc_period)) / \
              (np.roll(self.close, gene.direction_roc_period) + 1e-10) * 100
        roc_signal = np.tanh(roc / 10)
        roc_signal = (roc_signal + 1) / 2  # 轉換到 [0, 1]
        
        # 權重組合
        weights = gene.direction_weights
        total_weight = weights['rsi'] + weights['stoch'] + weights['roc']
        direction_score = (
            rsi_signal * weights['rsi'] +
            stoch_signal * weights['stoch'] +
            roc_signal * weights['roc']
        ) / total_weight
        
        return np.clip(direction_score, 0, 1)
    
    def evaluate_formula_quality(self, gene: FeatureFormula) -> Tuple[float, float, float]:
        """
        評估公式品質
        
        衡量指標: 與價格波動/趨勢/方向的相關性
        
        Returns:
            (volatility_corr, trend_corr, direction_corr)
        """
        # 計算3個公式
        volatility_signal = self.calculate_volatility_formula(gene)
        trend_signal = self.calculate_trend_formula(gene)
        direction_signal = self.calculate_direction_formula(gene)
        
        # 1. 波動性與實際波動的相關性
        actual_volatility = pd.Series(self.close).pct_change().rolling(20).std().values
        volatility_corr = np.corrcoef(
            volatility_signal[50:-50],
            actual_volatility[50:-50]
        )[0, 1]
        volatility_corr = 0 if np.isnan(volatility_corr) else volatility_corr
        
        # 2. 趨勢與價格方向的相關性
        price_change = np.diff(self.close)
        trend_direction = (price_change > 0).astype(float)
        trend_corr = np.corrcoef(
            trend_signal[:-1][50:-50],
            trend_direction[50:-50]
        )[0, 1]
        trend_corr = 0 if np.isnan(trend_corr) else trend_corr
        
        # 3. 方向信號與實際價格變化的相關性
        direction_signal_shifted = direction_signal[:-1]
        price_direction = (price_change > 0).astype(float)
        direction_corr = np.corrcoef(
            direction_signal_shifted[50:-50],
            price_direction[50:-50]
        )[0, 1]
        direction_corr = 0 if np.isnan(direction_corr) else direction_corr
        
        return volatility_corr, trend_corr, direction_corr
    
    def create_random_gene(self) -> FeatureFormula:
        """
        隨機創建基因
        """
        return FeatureFormula(
            # 波動性參數
            volatility_atr_period=random.randint(8, 20),
            volatility_bb_period=random.randint(15, 30),
            volatility_bb_std=random.uniform(1.5, 2.5),
            volatility_roc_period=random.randint(5, 15),
            
            # 趨勢參數
            trend_fast_ema=random.randint(12, 26),
            trend_slow_ema=random.randint(50, 100),
            trend_macd_signal=random.randint(7, 12),
            trend_adx_period=random.randint(10, 25),
            
            # 方向參數
            direction_rsi_period=random.randint(8, 20),
            direction_stoch_k=random.randint(10, 20),
            direction_stoch_d=random.randint(3, 8),
            direction_roc_period=random.randint(5, 15),
            
            # 權重
            volatility_weights={
                'atr': random.uniform(0.2, 0.5),
                'bb': random.uniform(0.2, 0.5),
                'roc': random.uniform(0.2, 0.5)
            },
            trend_weights={
                'ema': random.uniform(0.2, 0.5),
                'macd': random.uniform(0.2, 0.5),
                'adx': random.uniform(0.2, 0.5)
            },
            direction_weights={
                'rsi': random.uniform(0.2, 0.5),
                'stoch': random.uniform(0.2, 0.5),
                'roc': random.uniform(0.2, 0.5)
            }
        )
    
    def crossover(self, gene1: FeatureFormula, gene2: FeatureFormula) -> FeatureFormula:
        """
        交叉
        """
        child = FeatureFormula(
            volatility_atr_period=random.choice([gene1.volatility_atr_period, gene2.volatility_atr_period]),
            volatility_bb_period=random.choice([gene1.volatility_bb_period, gene2.volatility_bb_period]),
            volatility_bb_std=(gene1.volatility_bb_std + gene2.volatility_bb_std) / 2 + random.uniform(-0.1, 0.1),
            volatility_roc_period=random.choice([gene1.volatility_roc_period, gene2.volatility_roc_period]),
            
            trend_fast_ema=random.choice([gene1.trend_fast_ema, gene2.trend_fast_ema]),
            trend_slow_ema=random.choice([gene1.trend_slow_ema, gene2.trend_slow_ema]),
            trend_macd_signal=random.choice([gene1.trend_macd_signal, gene2.trend_macd_signal]),
            trend_adx_period=random.choice([gene1.trend_adx_period, gene2.trend_adx_period]),
            
            direction_rsi_period=random.choice([gene1.direction_rsi_period, gene2.direction_rsi_period]),
            direction_stoch_k=random.choice([gene1.direction_stoch_k, gene2.direction_stoch_k]),
            direction_stoch_d=random.choice([gene1.direction_stoch_d, gene2.direction_stoch_d]),
            direction_roc_period=random.choice([gene1.direction_roc_period, gene2.direction_roc_period]),
            
            volatility_weights={k: (gene1.volatility_weights[k] + gene2.volatility_weights[k]) / 2 for k in gene1.volatility_weights},
            trend_weights={k: (gene1.trend_weights[k] + gene2.trend_weights[k]) / 2 for k in gene1.trend_weights},
            direction_weights={k: (gene1.direction_weights[k] + gene2.direction_weights[k]) / 2 for k in gene1.direction_weights}
        )
        
        # 確保有效性
        child.volatility_bb_std = max(1.5, min(2.5, child.volatility_bb_std))
        
        return child
    
    def mutate(self, gene: FeatureFormula, mutation_rate: float = 0.15) -> FeatureFormula:
        """
        變異
        """
        mutant = deepcopy(gene)
        
        if random.random() < mutation_rate:
            mutant.volatility_atr_period = random.randint(8, 20)
        if random.random() < mutation_rate:
            mutant.trend_slow_ema = random.randint(50, 100)
        if random.random() < mutation_rate:
            mutant.direction_rsi_period = random.randint(8, 20)
        if random.random() < mutation_rate:
            mutant.volatility_weights['atr'] = random.uniform(0.2, 0.5)
        if random.random() < mutation_rate:
            mutant.trend_weights['macd'] = random.uniform(0.2, 0.5)
        if random.random() < mutation_rate:
            mutant.direction_weights['rsi'] = random.uniform(0.2, 0.5)
        
        return mutant
    
    def evolve(self):
        """
        進化過程
        """
        print("\n" + "="*80)
        print("特徵公式遺傳優化器")
        print("目標: 演化3套獨立的特徵公式")
        print("  1. 波動性公式 (Volatility)")
        print("  2. 趨勢公式 (Trend)")
        print("  3. 方向公式 (Direction)")
        print("="*80)
        
        # 初始化
        print(f"\n[1] 初始化人口 ({self.population_size} 個個體)...")
        self.population = [self.create_random_gene() for _ in range(self.population_size)]
        
        print(f"\n[2] 進化過程 ({self.generations} 代)...\n")
        
        for generation in range(self.generations):
            # 評估
            for gene in self.population:
                vol_corr, trend_corr, dir_corr = self.evaluate_formula_quality(gene)
                gene.volatility_correlation = vol_corr
                gene.trend_correlation = trend_corr
                gene.direction_correlation = dir_corr
                # 綜合適應度
                gene.overall_fitness = (vol_corr + trend_corr + dir_corr) / 3
            
            # 排序
            self.population.sort(key=lambda x: x.overall_fitness, reverse=True)
            
            best = self.population[0]
            avg_fitness = np.mean([g.overall_fitness for g in self.population])
            
            # 記錄
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': best.overall_fitness,
                'best_volatility_corr': best.volatility_correlation,
                'best_trend_corr': best.trend_correlation,
                'best_direction_corr': best.direction_correlation,
                'avg_fitness': avg_fitness,
                'best_parameters': best.to_dict()
            })
            
            # 顯示進度
            if (generation + 1) % max(1, self.generations // 10) == 0:
                print(f"第 {generation+1:3d} 代: 適應度={best.overall_fitness:.4f} "
                      f"(波動性={best.volatility_correlation:+.4f}, "
                      f"趨勢={best.trend_correlation:+.4f}, "
                      f"方向={best.direction_correlation:+.4f})")
            
            # 生成新一代
            new_population = []
            elite_size = max(1, self.population_size // 5)
            new_population.extend(self.population[:elite_size])
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
        
        # 最終排序
        for gene in self.population:
            vol_corr, trend_corr, dir_corr = self.evaluate_formula_quality(gene)
            gene.volatility_correlation = vol_corr
            gene.trend_correlation = trend_corr
            gene.direction_correlation = dir_corr
            gene.overall_fitness = (vol_corr + trend_corr + dir_corr) / 3
        
        self.population.sort(key=lambda x: x.overall_fitness, reverse=True)
        best = self.population[0]
        
        print("\n" + "="*80)
        print("進化完成!")
        print("="*80)
        
        print(f"\n最優特徵公式組合:")
        print(f"\n  波動性公式相關性: {best.volatility_correlation:+.4f}")
        print(f"  趨勢公式相關性: {best.trend_correlation:+.4f}")
        print(f"  方向公式相關性: {best.direction_correlation:+.4f}")
        print(f"  綜合適應度: {best.overall_fitness:.4f}")
        
        print(f"\n  波動性公式參數:")
        print(f"    ATR 週期: {best.volatility_atr_period}")
        print(f"    BB 週期: {best.volatility_bb_period}")
        print(f"    BB 標準差: {best.volatility_bb_std:.3f}")
        print(f"    權重 (ATR:BB:ROC) = {best.volatility_weights['atr']:.3f}:{best.volatility_weights['bb']:.3f}:{best.volatility_weights['roc']:.3f}")
        
        print(f"\n  趨勢公式參數:")
        print(f"    快速EMA週期: {best.trend_fast_ema}")
        print(f"    緩慢EMA週期: {best.trend_slow_ema}")
        print(f"    權重 (EMA:MACD:ADX) = {best.trend_weights['ema']:.3f}:{best.trend_weights['macd']:.3f}:{best.trend_weights['adx']:.3f}")
        
        print(f"\n  方向公式參數:")
        print(f"    RSI 週期: {best.direction_rsi_period}")
        print(f"    Stochastic K/D: {best.direction_stoch_k}/{best.direction_stoch_d}")
        print(f"    權重 (RSI:Stoch:ROC) = {best.direction_weights['rsi']:.3f}:{best.direction_weights['stoch']:.3f}:{best.direction_weights['roc']:.3f}")
        
        return best
    
    def _tournament_selection(self, tournament_size: int = 3):
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.overall_fitness)
    
    def save_results(self, best_gene: FeatureFormula):
        """
        保存結果
        """
        print(f"\n[3] 保存結果...")
        os.makedirs('results', exist_ok=True)
        
        output = {
            'best_feature_formulas': best_gene.to_dict(),
            'evolution_history': self.evolution_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('results/feature_formula_optimization.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ 結果已保存: results/feature_formula_optimization.json")


def main():
    print("\n" + "#"*80)
    print("# 特徵公式遺傳優化器")
    print("# 目標: 演化3套獨立的特徵提取公式")
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
    
    # 優化
    print(f"\n[二] 執行特徵公式優化...")
    optimizer = FeatureFormulaOptimizer(df, population_size=30, generations=50)
    best_gene = optimizer.evolve()
    
    # 保存
    optimizer.save_results(best_gene)
    
    print("\n" + "#"*80 + "\n")


if __name__ == "__main__":
    main()
