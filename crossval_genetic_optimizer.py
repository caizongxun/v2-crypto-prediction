#!/usr/bin/env python3
"""
K-Fold 交叉驗證遺傳算法優化器

目標: 防止過擬合

策略:
1. 將數據分成 5 個 Fold (摺疊)
2. 每次用 4 個 Fold 訓練, 1 個 Fold 驗證
3. 最終準確率 = 5 個 Fold 驗證結果的平均值
4. 這樣得到的參數更穩健, 更能泛化到未來數據
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime
import random
from copy import deepcopy


@dataclass
class RobustFormulaGene:
    """強健公式基因 - 通過交叉驗證驗證"""
    fast_ema: int
    slow_ema: int
    atr_period: int
    rsi_period: int
    roc_period: int
    sma_period: int
    bb_std: float
    trend_weight: float
    direction_weight: float
    volatility_weight: float
    threshold_buy: float
    threshold_sell: float
    sigmoid_strength: float = 3.0
    confirmation_strength: float = 0.55
    
    # 交叉驗證指標
    cv_accuracy: float = 0.0  # 平均準確率
    cv_accuracy_std: float = 0.0  # 標準差 (越小越穩定)
    cv_fold_accuracies: List[float] = None
    fitness: float = 0.0
    
    def to_dict(self) -> Dict:
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
            'cv_accuracy': self.cv_accuracy,
            'cv_accuracy_std': self.cv_accuracy_std,
            'cv_fold_accuracies': self.cv_fold_accuracies if self.cv_fold_accuracies else [],
            'fitness': self.fitness
        }


class CrossValidationGeneticOptimizer:
    """
    K-Fold 交叉驗證遺傳算法優化器
    """
    
    def __init__(self, df: pd.DataFrame, n_splits: int = 5, population_size: int = 40, generations: int = 50):
        self.df = df.copy()
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.n_splits = n_splits
        self.population_size = population_size
        self.generations = generations
        self.population: List[RobustFormulaGene] = []
        self.folds = self._create_time_series_folds()
        self.evolution_history = []
    
    def _create_time_series_folds(self) -> List[Tuple[int, int]]:
        """
        創建時間序列 Fold (而不是隨機 Fold)
        
        重要: 金融數據必須按時間分割!
        """
        fold_size = len(self.df) // self.n_splits
        folds = []
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else len(self.df)
            folds.append((start_idx, end_idx))
        
        return folds
    
    def _get_train_test_folds(self, test_fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        獲取訓練和測試 Fold
        
        test_fold_idx: 哪個 Fold 用作測試集
        """
        test_start, test_end = self.folds[test_fold_idx]
        
        # 訓練集 = 所有其他 Fold
        train_indices = []
        for i, (start, end) in enumerate(self.folds):
            if i != test_fold_idx:
                train_indices.extend(range(start, end))
        
        test_indices = list(range(test_start, test_end))
        
        train_df = self.df.iloc[train_indices]
        test_df = self.df.iloc[test_indices]
        
        return train_df, test_df
    
    def calculate_indicators(self, df: pd.DataFrame, gene: RobustFormulaGene) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        計算三個指標
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # 趨勢指標
        close_series = pd.Series(close)
        fast_ema = close_series.ewm(span=gene.fast_ema, adjust=False).mean().values
        slow_ema = close_series.ewm(span=gene.slow_ema, adjust=False).mean().values
        
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        tr_series = pd.Series(tr)
        atr = tr_series.ewm(span=gene.atr_period, adjust=False).mean().values
        
        ema_ratio = (fast_ema - slow_ema) / (slow_ema + 1e-10) * 100
        atr_ratio = (atr / close) * 100
        
        trend_score = np.tanh(ema_ratio / 2) * (1 - np.exp(-np.maximum(atr_ratio, 0) / 0.5))
        trend_value = 1 / (1 + np.exp(-trend_score * gene.sigmoid_strength))
        
        # 方向指標
        delta = close_series.diff().values
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.ewm(span=gene.rsi_period, adjust=False).mean().values
        avg_loss = loss_series.ewm(span=gene.rsi_period, adjust=False).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        roc = (close - np.roll(close, gene.roc_period)) / (np.roll(close, gene.roc_period) + 1e-10) * 100
        
        ema12 = close_series.ewm(span=12, adjust=False).mean().values
        ema26 = close_series.ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        macd_histogram = macd - signal_line
        
        rsi_signal = (rsi - 50) / 50
        roc_signal = np.tanh(roc / 5)
        macd_signal = np.tanh(macd_histogram / (np.std(macd_histogram) + 1e-10))
        
        direction_score = (rsi_signal * 0.3 + roc_signal * 0.3 + macd_signal * 0.4)
        direction_value = 1 / (1 + np.exp(-direction_score * gene.sigmoid_strength))
        
        # 波動性指標
        sma = close_series.rolling(window=gene.sma_period).mean().values
        std = close_series.rolling(window=gene.sma_period).std().values
        
        volatility_ratio = (std / (sma + 1e-10)) * 100
        volatility_score = np.sqrt(volatility_ratio / 2)
        
        high_low_range = (high - low) / sma
        combined_vol = volatility_score * 0.6 + (high_low_range * 10) * 0.4
        volatility_value = np.clip(combined_vol, 0, 1)
        
        return np.clip(trend_value, 0, 1), np.clip(direction_value, 0, 1), volatility_value
    
    def generate_signals(self, trend: np.ndarray, direction: np.ndarray, volatility: np.ndarray, gene: RobustFormulaGene) -> np.ndarray:
        """
        生成交易信號
        """
        signals = np.zeros(len(trend))
        
        total_weight = gene.trend_weight + gene.direction_weight + gene.volatility_weight
        trend_w = gene.trend_weight / total_weight
        direction_w = gene.direction_weight / total_weight
        volatility_w = gene.volatility_weight / total_weight
        
        combined = (trend * trend_w + direction * direction_w + volatility * volatility_w * 0.5)
        
        adjusted_buy_threshold = gene.threshold_buy + (volatility * 0.05)
        adjusted_sell_threshold = gene.threshold_sell - (volatility * 0.05)
        
        for i in range(1, len(combined)):
            if combined[i] > adjusted_buy_threshold[i]:
                if (trend[i] > gene.confirmation_strength and direction[i] > gene.confirmation_strength):
                    signals[i] = 1
            elif combined[i] < adjusted_sell_threshold[i]:
                if (trend[i] < (1 - gene.confirmation_strength) and direction[i] < (1 - gene.confirmation_strength)):
                    signals[i] = -1
        
        return signals
    
    def calculate_accuracy(self, signals: np.ndarray, close_prices: np.ndarray) -> float:
        """
        計算準確率
        """
        correct = 0
        total = 0
        for i in range(len(signals) - 1):
            if signals[i] != 0:
                actual_return = close_prices[i+1] - close_prices[i]
                if (signals[i] == 1 and actual_return > 0) or (signals[i] == -1 and actual_return < 0):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_with_cv(self, gene: RobustFormulaGene) -> Tuple[float, float, List[float]]:
        """
        使用 K-Fold 交叉驗證評估基因
        
        Returns:
            (平均準確率, 標準差, 每個 Fold 的準確率)
        """
        fold_accuracies = []
        
        for fold_idx in range(self.n_splits):
            train_df, test_df = self._get_train_test_folds(fold_idx)
            
            # 在訓練集上計算指標
            trend_train, direction_train, volatility_train = self.calculate_indicators(train_df, gene)
            
            # 在測試集上計算指標
            trend_test, direction_test, volatility_test = self.calculate_indicators(test_df, gene)
            
            # 在測試集上生成信號並評估
            signals_test = self.generate_signals(trend_test, direction_test, volatility_test, gene)
            accuracy = self.calculate_accuracy(signals_test, test_df['close'].values)
            
            fold_accuracies.append(accuracy)
        
        # 計算平均值和標準差
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        
        return mean_accuracy, std_accuracy, fold_accuracies
    
    def create_random_gene(self) -> RobustFormulaGene:
        """
        隨機創建基因
        """
        return RobustFormulaGene(
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
            confirmation_strength=random.uniform(0.50, 0.65),
            cv_fold_accuracies=[]
        )
    
    def crossover(self, gene1: RobustFormulaGene, gene2: RobustFormulaGene) -> RobustFormulaGene:
        """
        交叉
        """
        child = RobustFormulaGene(
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
            confirmation_strength=(gene1.confirmation_strength + gene2.confirmation_strength) / 2 + random.uniform(-0.02, 0.02),
            cv_fold_accuracies=[]
        )
        
        # 確保有效性
        child.fast_ema = max(12, min(30, child.fast_ema))
        child.slow_ema = max(50, min(150, child.slow_ema))
        child.bb_std = max(1.5, min(3.5, child.bb_std))
        child.sigmoid_strength = max(2.0, min(4.0, child.sigmoid_strength))
        child.confirmation_strength = max(0.5, min(0.65, child.confirmation_strength))
        
        return child
    
    def mutate(self, gene: RobustFormulaGene, mutation_rate: float = 0.1) -> RobustFormulaGene:
        """
        變異
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
            mutant.threshold_buy = random.uniform(0.55, 0.75)
        if random.random() < mutation_rate:
            mutant.threshold_sell = random.uniform(0.25, 0.45)
        
        return mutant
    
    def evolve(self):
        """
        進化過程
        """
        print("\n" + "="*80)
        print("K-Fold 交叉驗證遺傳算法優化器 - 防止過擬合")
        print(f"使用 {self.n_splits}-Fold 交叉驗證")
        print("="*80)
        
        # 初始化人口
        print(f"\n[1] 初始化人口 ({self.population_size} 個個體)...")
        self.population = [self.create_random_gene() for _ in range(self.population_size)]
        
        # 進化迴圈
        print(f"\n[2] 進化過程 ({self.generations} 代)...\n")
        
        for generation in range(self.generations):
            # 評估適應度
            for gene in self.population:
                cv_accuracy, cv_std, fold_accs = self.evaluate_with_cv(gene)
                gene.cv_accuracy = cv_accuracy
                gene.cv_accuracy_std = cv_std
                gene.cv_fold_accuracies = fold_accs
                # 適應度 = 準確率 - 0.5 * 標準差 (獎勵穩定性)
                gene.fitness = cv_accuracy - 0.5 * cv_std
            
            # 排序
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            best_gene = self.population[0]
            
            # 記錄歷史
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': best_gene.fitness,
                'best_cv_accuracy': best_gene.cv_accuracy,
                'best_cv_std': best_gene.cv_accuracy_std,
                'avg_fitness': np.mean([g.fitness for g in self.population]),
                'best_parameters': best_gene.to_dict()
            })
            
            # 顯示進度
            if (generation + 1) % max(1, self.generations // 10) == 0:
                print(f"第 {generation+1:3d} 代: 準確率={best_gene.cv_accuracy*100:6.2f}% ± {best_gene.cv_accuracy_std*100:5.2f}%, "
                      f"適應度={best_gene.fitness:8.4f}")
            
            # 選擇和進化
            new_population = []
            
            # 精英保留 (Top 20%)
            elite_size = max(1, self.population_size // 5)
            new_population.extend(self.population[:elite_size])
            
            # 生成新個體
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                child = self.crossover(parent1, parent2)
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
        print(f"  K-Fold CV 準確率: {best_gene.cv_accuracy*100:.2f}% ± {best_gene.cv_accuracy_std*100:.2f}%")
        print(f"  (Fold 結果: {', '.join([f'{acc*100:.1f}%' for acc in best_gene.cv_fold_accuracies])})")
        print(f"  適應度: {best_gene.fitness:.4f}")
        print(f"\n  快速 EMA 週期: {best_gene.fast_ema}")
        print(f"  緩慢 EMA 週期: {best_gene.slow_ema}")
        print(f"  ATR 週期: {best_gene.atr_period}")
        
        return best_gene
    
    def _tournament_selection(self, tournament_size: int = 3) -> RobustFormulaGene:
        """
        競賽選擇
        """
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)


def main():
    print("\n" + "#"*80)
    print("# K-Fold 交叉驗證遺傳算法優化器")
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
    print("\n[二] 執行 K-Fold 交叉驗證優化...")
    optimizer = CrossValidationGeneticOptimizer(df, n_splits=5, population_size=40, generations=50)
    best_gene = optimizer.evolve()
    
    # 保存結果
    print("\n[三] 保存結果...")
    os.makedirs("results", exist_ok=True)
    
    output_data = {
        'best_formula_combination': best_gene.to_dict(),
        'cross_validation_info': {
            'n_splits': optimizer.n_splits,
            'cv_accuracy': best_gene.cv_accuracy,
            'cv_accuracy_std': best_gene.cv_accuracy_std,
            'fold_accuracies': best_gene.cv_fold_accuracies
        },
        'evolution_history': optimizer.evolution_history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open("results/crossval_optimization_result.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ 結果已保存: results/crossval_optimization_result.json")
    
    print("\n" + "#"*80 + "\n")


if __name__ == "__main__":
    main()
