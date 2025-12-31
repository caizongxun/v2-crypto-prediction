#!/usr/bin/env python3
"""
遺傳算法公式優化器

目標: 自動發現最優的公式組合

機制:
1. 隨機初始化公式組合人口 (Population)
2. 評估每個公式組合的適應度 (Fitness)
3. 選擇最優的公式組合進行交叉 (Crossover)
4. 進行隨機變異 (Mutation)
5. 重複進化過程直到收斂

可優化的參數:
- EMA 週期: fast_ema, slow_ema
- ATR 週期: atr_period
- RSI 週期: rsi_period
- ROC 週期: roc_period
- SMA 週期: sma_period
- Bollinger Bands 標準差倍數: bb_std
- 指標權重: trend_weight, direction_weight, volatility_weight
- 閾值: threshold_buy, threshold_sell
- 非線性映射強度: sigmoid_strength (新增)
- 信號確認強度: confirmation_strength (新增)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import os
from datetime import datetime
import random
from copy import deepcopy


@dataclass
class FormulaGene:
    """
    公式基因 - 代表一個完整的公式組合
    """
    # 趨勢指標參數
    fast_ema: int
    slow_ema: int
    atr_period: int
    
    # 方向指標參數
    rsi_period: int
    roc_period: int
    
    # 波動性指標參數
    sma_period: int
    bb_std: float
    
    # 權重
    trend_weight: float
    direction_weight: float
    volatility_weight: float
    
    # 閾值
    threshold_buy: float
    threshold_sell: float
    
    # 非線性映射強度
    sigmoid_strength: float = 3.0
    
    # 信號確認強度
    confirmation_strength: float = 0.55
    
    # 績效指標
    fitness: float = 0.0
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    
    def to_dict(self) -> Dict:
        """
        轉換為字典
        """
        return {
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'atr_period': self.atr_period,
            'rsi_period': self.rsi_period,
            'roc_period': self.roc_period,
            'sma_period': self.sma_period,
            'bb_std': self.bb_std,
            'trend_weight': self.trend_weight,
            'direction_weight': self.direction_weight,
            'volatility_weight': self.volatility_weight,
            'threshold_buy': self.threshold_buy,
            'threshold_sell': self.threshold_sell,
            'sigmoid_strength': self.sigmoid_strength,
            'confirmation_strength': self.confirmation_strength,
            'fitness': self.fitness,
            'accuracy': self.accuracy,
            'sharpe_ratio': self.sharpe_ratio
        }


class GeneticFormulaOptimizer:
    """
    遺傳算法公式優化器
    """
    
    def __init__(self, df: pd.DataFrame, population_size: int = 50, generations: int = 100):
        self.df = df.copy()
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.population_size = population_size
        self.generations = generations
        self.population: List[FormulaGene] = []
        self.best_genes: List[FormulaGene] = []
        self.evolution_history = []
    
    # ========== 公式計算 ==========
    
    def calculate_trend(self, gene: FormulaGene) -> np.ndarray:
        """
        計算趨勢指標
        
        公式演化:
        基礎: trend_value = (tanh(ema_ratio/2) * (1-exp(-atr_ratio/0.5)) + 1) / 2
        進階: 使用 sigmoid_strength 調整敏感度
        """
        close_series = pd.Series(self.close)
        
        fast_ema = close_series.ewm(span=gene.fast_ema, adjust=False).mean().values
        slow_ema = close_series.ewm(span=gene.slow_ema, adjust=False).mean().values
        
        # ATR
        high_low = self.high - self.low
        high_close = np.abs(self.high - np.roll(self.close, 1))
        low_close = np.abs(self.low - np.roll(self.close, 1))
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        tr_series = pd.Series(tr)
        atr = tr_series.ewm(span=gene.atr_period, adjust=False).mean().values
        
        # 趨勢分數
        ema_ratio = (fast_ema - slow_ema) / (slow_ema + 1e-10) * 100
        atr_ratio = (atr / self.close) * 100
        
        trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-np.maximum(atr_ratio, 0) / 0.5))
        
        # 使用進化的 sigmoid_strength
        trend_value = 1 / (1 + np.exp(-trend_score * gene.sigmoid_strength))
        
        return np.clip(trend_value, 0, 1)
    
    def calculate_direction(self, gene: FormulaGene) -> np.ndarray:
        """
        計算方向指標 (含 MACD)
        """
        close_series = pd.Series(self.close)
        
        # RSI
        delta = close_series.diff().values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.ewm(span=gene.rsi_period, adjust=False).mean().values
        avg_loss = loss_series.ewm(span=gene.rsi_period, adjust=False).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # ROC
        roc = (self.close - np.roll(self.close, gene.roc_period)) / (np.roll(self.close, gene.roc_period) + 1e-10) * 100
        
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
        
        # 加權組合
        direction_score = (rsi_signal * 0.3 + roc_signal * 0.3 + macd_signal * 0.4)
        direction_value = 1 / (1 + np.exp(-direction_score * gene.sigmoid_strength))
        
        return np.clip(direction_value, 0, 1)
    
    def calculate_volatility(self, gene: FormulaGene) -> np.ndarray:
        """
        計算波動性指標
        """
        close_series = pd.Series(self.close)
        
        sma = close_series.rolling(window=gene.sma_period).mean().values
        std = close_series.rolling(window=gene.sma_period).std().values
        
        volatility_ratio = (std / (sma + 1e-10)) * 100
        volatility_score = np.sqrt(volatility_ratio / 2)
        
        high_low_range = (self.high - self.low) / sma
        combined_vol = volatility_score * 0.6 + (high_low_range * 10) * 0.4
        
        return np.clip(combined_vol, 0, 1)
    
    # ========== 信號生成 ==========
    
    def generate_signals(self, gene: FormulaGene) -> np.ndarray:
        """
        生成交易信號
        
        公式演化:
        1. 基礎: 加權組合三指標
        2. 進階: 使用 confirmation_strength 調整確認強度
        3. 動態閾值: 基於波動性調整
        """
        trend = self.calculate_trend(gene)
        direction = self.calculate_direction(gene)
        volatility = self.calculate_volatility(gene)
        
        signals = np.zeros(len(trend))
        
        # 正規化權重
        total_weight = gene.trend_weight + gene.direction_weight + gene.volatility_weight
        trend_w = gene.trend_weight / total_weight
        direction_w = gene.direction_weight / total_weight
        volatility_w = gene.volatility_weight / total_weight
        
        # 計算加權指標
        combined = (trend * trend_w + direction * direction_w + volatility * volatility_w * 0.5)
        
        # 動態閾值
        adjusted_buy_threshold = gene.threshold_buy + (volatility * 0.05)
        adjusted_sell_threshold = gene.threshold_sell - (volatility * 0.05)
        
        # 生成信號
        for i in range(1, len(combined)):
            if combined[i] > adjusted_buy_threshold[i]:
                # 使用 confirmation_strength 調整確認強度
                if (trend[i] > gene.confirmation_strength and 
                    direction[i] > gene.confirmation_strength):
                    signals[i] = 1
            elif combined[i] < adjusted_sell_threshold[i]:
                if (trend[i] < (1 - gene.confirmation_strength) and 
                    direction[i] < (1 - gene.confirmation_strength)):
                    signals[i] = -1
        
        return signals
    
    # ========== 適應度評估 ==========
    
    def evaluate_fitness(self, gene: FormulaGene) -> Tuple[float, float, float]:
        """
        評估基因的適應度
        
        Returns:
            (fitness, accuracy, sharpe_ratio)
        """
        try:
            signals = self.generate_signals(gene)
            
            # 計算準確率
            correct = 0
            total = 0
            for i in range(len(signals) - 1):
                if signals[i] != 0:
                    actual_return = self.close[i+1] - self.close[i]
                    if (signals[i] == 1 and actual_return > 0) or (signals[i] == -1 and actual_return < 0):
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            # 計算 Sharpe 比率
            strategy_returns = np.zeros(len(self.close) - 1)
            for i in range(len(signals) - 1):
                if signals[i] == 1:
                    strategy_returns[i] = (self.close[i+1] - self.close[i]) / self.close[i]
                elif signals[i] == -1:
                    strategy_returns[i] = -(self.close[i+1] - self.close[i]) / self.close[i]
            
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # 多目標適應度
            fitness = sharpe_ratio * 0.5 + accuracy * 100 * 0.5
            
            return fitness, accuracy, sharpe_ratio
            
        except Exception as e:
            return 0.0, 0.0, 0.0
    
    # ========== 遺傳操作 ==========
    
    def create_random_gene(self) -> FormulaGene:
        """
        隨機創建基因
        """
        return FormulaGene(
            fast_ema=random.randint(12, 30),
            slow_ema=random.randint(50, 150),
            atr_period=random.randint(8, 25),
            rsi_period=random.randint(8, 25),
            roc_period=random.randint(5, 15),
            sma_period=random.randint(20, 50),
            bb_std=random.uniform(1.5, 3.5),
            trend_weight=random.uniform(0.2, 0.5),
            direction_weight=random.uniform(0.2, 0.5),
            volatility_weight=random.uniform(0.2, 0.5),
            threshold_buy=random.uniform(0.55, 0.75),
            threshold_sell=random.uniform(0.25, 0.45),
            sigmoid_strength=random.uniform(2.0, 4.0),
            confirmation_strength=random.uniform(0.50, 0.65)
        )
    
    def crossover(self, gene1: FormulaGene, gene2: FormulaGene) -> FormulaGene:
        """
        交叉 (Crossover) - 結合兩個基因
        """
        child = FormulaGene(
            fast_ema=random.choice([gene1.fast_ema, gene2.fast_ema]),
            slow_ema=random.choice([gene1.slow_ema, gene2.slow_ema]),
            atr_period=random.choice([gene1.atr_period, gene2.atr_period]),
            rsi_period=random.choice([gene1.rsi_period, gene2.rsi_period]),
            roc_period=random.choice([gene1.roc_period, gene2.roc_period]),
            sma_period=random.choice([gene1.sma_period, gene2.sma_period]),
            bb_std=(gene1.bb_std + gene2.bb_std) / 2 + random.uniform(-0.1, 0.1),
            trend_weight=(gene1.trend_weight + gene2.trend_weight) / 2 + random.uniform(-0.05, 0.05),
            direction_weight=(gene1.direction_weight + gene2.direction_weight) / 2 + random.uniform(-0.05, 0.05),
            volatility_weight=(gene1.volatility_weight + gene2.volatility_weight) / 2 + random.uniform(-0.05, 0.05),
            threshold_buy=(gene1.threshold_buy + gene2.threshold_buy) / 2 + random.uniform(-0.02, 0.02),
            threshold_sell=(gene1.threshold_sell + gene2.threshold_sell) / 2 + random.uniform(-0.02, 0.02),
            sigmoid_strength=(gene1.sigmoid_strength + gene2.sigmoid_strength) / 2 + random.uniform(-0.2, 0.2),
            confirmation_strength=(gene1.confirmation_strength + gene2.confirmation_strength) / 2 + random.uniform(-0.02, 0.02)
        )
        
        # 確保有效性
        child.fast_ema = max(12, min(30, child.fast_ema))
        child.slow_ema = max(50, min(150, child.slow_ema))
        child.bb_std = max(1.5, min(3.5, child.bb_std))
        child.sigmoid_strength = max(2.0, min(4.0, child.sigmoid_strength))
        child.confirmation_strength = max(0.5, min(0.65, child.confirmation_strength))
        
        return child
    
    def mutate(self, gene: FormulaGene, mutation_rate: float = 0.1) -> FormulaGene:
        """
        變異 (Mutation) - 隨機修改基因
        """
        mutant = deepcopy(gene)
        
        if random.random() < mutation_rate:
            mutant.fast_ema = random.randint(12, 30)
        if random.random() < mutation_rate:
            mutant.slow_ema = random.randint(50, 150)
        if random.random() < mutation_rate:
            mutant.atr_period = random.randint(8, 25)
        if random.random() < mutation_rate:
            mutant.rsi_period = random.randint(8, 25)
        if random.random() < mutation_rate:
            mutant.roc_period = random.randint(5, 15)
        if random.random() < mutation_rate:
            mutant.sma_period = random.randint(20, 50)
        if random.random() < mutation_rate:
            mutant.bb_std = random.uniform(1.5, 3.5)
        if random.random() < mutation_rate:
            mutant.trend_weight = random.uniform(0.2, 0.5)
        if random.random() < mutation_rate:
            mutant.direction_weight = random.uniform(0.2, 0.5)
        if random.random() < mutation_rate:
            mutant.volatility_weight = random.uniform(0.2, 0.5)
        if random.random() < mutation_rate:
            mutant.threshold_buy = random.uniform(0.55, 0.75)
        if random.random() < mutation_rate:
            mutant.threshold_sell = random.uniform(0.25, 0.45)
        if random.random() < mutation_rate:
            mutant.sigmoid_strength = random.uniform(2.0, 4.0)
        if random.random() < mutation_rate:
            mutant.confirmation_strength = random.uniform(0.50, 0.65)
        
        return mutant
    
    # ========== 進化主循環 ==========
    
    def evolve(self):
        """
        執行進化過程
        """
        print("\n" + "="*80)
        print("遺傳算法公式優化器 - 發現最優公式組合")
        print("="*80)
        
        # 初始化人口
        print(f"\n[1] 初始化人口 ({self.population_size} 個個體)...")
        self.population = [self.create_random_gene() for _ in range(self.population_size)]
        
        # 進化迴圈
        print(f"\n[2] 進化過程 ({self.generations} 代)...\n")
        
        for generation in range(self.generations):
            # 評估適應度
            for gene in self.population:
                fitness, accuracy, sharpe = self.evaluate_fitness(gene)
                gene.fitness = fitness
                gene.accuracy = accuracy
                gene.sharpe_ratio = sharpe
            
            # 按適應度排序
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 記錄最佳基因
            best_gene = self.population[0]
            self.best_genes.append(deepcopy(best_gene))
            
            # 記錄進化歷史
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': best_gene.fitness,
                'best_accuracy': best_gene.accuracy,
                'best_sharpe': best_gene.sharpe_ratio,
                'avg_fitness': np.mean([g.fitness for g in self.population]),
                'best_parameters': best_gene.to_dict()
            })
            
            # 顯示進度
            if (generation + 1) % max(1, self.generations // 10) == 0:
                print(f"第 {generation+1:3d} 代: 最佳適應度={best_gene.fitness:8.4f}, "
                      f"準確率={best_gene.accuracy*100:6.2f}%, "
                      f"Sharpe={best_gene.sharpe_ratio:8.4f}")
            
            # 選擇和變異
            new_population = []
            
            # 精英保留 (Top 20%)
            elite_size = max(1, self.population_size // 5)
            new_population.extend(self.population[:elite_size])
            
            # 生成新個體
            while len(new_population) < self.population_size:
                # 選擇父母 (輪盤選擇)
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # 交叉
                child = self.crossover(parent1, parent2)
                
                # 變異
                child = self.mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
        
        # 最終排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_gene = self.population[0]
        
        print("\n" + "="*80)
        print("進化完成")
        print("="*80)
        
        print(f"\n最優公式組合:")
        print(f"  快速 EMA 週期: {best_gene.fast_ema}")
        print(f"  緩慢 EMA 週期: {best_gene.slow_ema}")
        print(f"  ATR 週期: {best_gene.atr_period}")
        print(f"  RSI 週期: {best_gene.rsi_period}")
        print(f"  ROC 週期: {best_gene.roc_period}")
        print(f"  SMA 週期: {best_gene.sma_period}")
        print(f"  Bollinger Bands 標準差倍數: {best_gene.bb_std:.2f}")
        print(f"\n  趨勢權重: {best_gene.trend_weight:.4f}")
        print(f"  方向權重: {best_gene.direction_weight:.4f}")
        print(f"  波動性權重: {best_gene.volatility_weight:.4f}")
        print(f"\n  買入閾值: {best_gene.threshold_buy:.4f}")
        print(f"  賣出閾值: {best_gene.threshold_sell:.4f}")
        print(f"  Sigmoid 強度: {best_gene.sigmoid_strength:.2f}")
        print(f"  確認強度: {best_gene.confirmation_strength:.4f}")
        
        print(f"\n績效指標:")
        print(f"  最佳適應度: {best_gene.fitness:.4f}")
        print(f"  準確率: {best_gene.accuracy*100:.2f}%")
        print(f"  Sharpe 比率: {best_gene.sharpe_ratio:.4f}")
        
        return best_gene
    
    def _tournament_selection(self, tournament_size: int = 3) -> FormulaGene:
        """
        競賽選擇 (Tournament Selection)
        """
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)


def main():
    print("\n" + "#"*80)
    print("# 遺傳算法公式優化器")
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
    
    # 執行遺傳算法
    print("\n[二] 執行遺傳算法優化...")
    optimizer = GeneticFormulaOptimizer(df, population_size=50, generations=50)
    best_gene = optimizer.evolve()
    
    # 保存結果
    print("\n[三] 保存結果...")
    os.makedirs("results", exist_ok=True)
    
    output_data = {
        'best_formula_combination': best_gene.to_dict(),
        'evolution_history': optimizer.evolution_history,
        'population_size': optimizer.population_size,
        'generations': optimizer.generations,
        'timestamp': datetime.now().isoformat()
    }
    
    with open("results/genetic_algorithm_result.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ 結果已保存: results/genetic_algorithm_result.json")
    
    print("\n" + "#"*80 + "\n")


if __name__ == "__main__":
    main()
