#!/usr/bin/env python3
"""
高級特徵生成器

概念:
- 供給 30 个基础指标 (积木)
- 遺傳算法自動組合不同的突變
- 对每一个突變进行整个数据集的對比
- 根据相关性不斷技擦我们的 3 个公式

最优化實際效果:
- 波動性公式: 由特定的積木組合成
- 趨勢公式: 由特定的積木組合成
- 方向公式: 由特定的積木組合成
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import json
import os
from datetime import datetime
import random
from copy import deepcopy


@dataclass
class Indicator:
    """
    基礎指標 (積木)
    """
    name: str
    values: np.ndarray
    normalized: np.ndarray  # [0, 1] 正規化值


class BasicIndicatorBuilder:
    """
    構建 30 个基礎指標
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.volume = df['volume'].values if 'volume' in df.columns else np.ones_like(self.close)
        
        self.indicators: Dict[str, Indicator] = {}
        self.build_all_indicators()
    
    def _normalize(self, values: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """正規化值到 [0, 1]"""
        if method == 'minmax':
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
            if vmax == vmin:
                return np.ones_like(values) * 0.5
            return (values - vmin) / (vmax - vmin + 1e-10)
        elif method == 'zscore':
            mean = np.nanmean(values)
            std = np.nanstd(values)
            normalized = (values - mean) / (std + 1e-10)
            return (normalized + 3) / 6  # 轉換到 [-0.5, 0.5] 到 [0, 1]
        else:
            return np.clip(values, 0, 1)
    
    def build_all_indicators(self):
        """構建所有 30 个指標"""
        close_series = pd.Series(self.close)
        
        # 動量提供器
        # ===== 住簡指標 (SMA) =====
        for period in [5, 10, 20, 50]:
            sma = close_series.rolling(window=period).mean().values
            self.indicators[f'SMA_{period}'] = Indicator(
                name=f'SMA_{period}',
                values=sma,
                normalized=self._normalize(np.abs(self.close - sma))
            )
        
        # ===== 指數住簡平均 (EMA) =====
        for period in [5, 12, 26, 50]:
            ema = close_series.ewm(span=period, adjust=False).mean().values
            self.indicators[f'EMA_{period}'] = Indicator(
                name=f'EMA_{period}',
                values=ema,
                normalized=self._normalize(np.abs(self.close - ema))
            )
        
        # ===== 相對強度指數 (RSI) =====
        for period in [7, 14, 21]:
            delta = close_series.diff().values
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
            avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            self.indicators[f'RSI_{period}'] = Indicator(
                name=f'RSI_{period}',
                values=rsi,
                normalized=self._normalize(rsi)
            )
        
        # ===== 一日真實橋段 (ATR) =====
        for period in [10, 14, 21]:
            tr = np.maximum(
                np.maximum(self.high - self.low, np.abs(self.high - np.roll(self.close, 1))),
                np.abs(self.low - np.roll(self.close, 1))
            )
            atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
            self.indicators[f'ATR_{period}'] = Indicator(
                name=f'ATR_{period}',
                values=atr,
                normalized=self._normalize(atr)
            )
        
        # ===== 計數指數易也 (不整整站) =====
        # ROC (不同週期)
        for period in [5, 10, 20]:
            roc = (self.close - np.roll(self.close, period)) / \
                  (np.roll(self.close, period) + 1e-10) * 100
            self.indicators[f'ROC_{period}'] = Indicator(
                name=f'ROC_{period}',
                values=roc,
                normalized=self._normalize(np.abs(roc))
            )
        
        # ===== 波動率指數 =====
        for period in [10, 20, 30]:
            returns = close_series.pct_change().rolling(window=period).std().values
            self.indicators[f'VOLATILITY_{period}'] = Indicator(
                name=f'VOLATILITY_{period}',
                values=returns,
                normalized=self._normalize(returns)
            )
        
        # ===== 典幫 =====
        self.indicators['PRICE'] = Indicator(
            name='PRICE',
            values=self.close,
            normalized=self._normalize(self.close)
        )
        
        self.indicators['PRICE_CHANGE'] = Indicator(
            name='PRICE_CHANGE',
            values=np.abs(np.diff(self.close, prepend=self.close[0])),
            normalized=self._normalize(np.abs(np.diff(self.close, prepend=self.close[0])))
        )
    
    def get_indicators_list(self) -> List[str]:
        """獲取所有指標名稱"""
        return list(self.indicators.keys())
    
    def get_indicator_values(self, name: str) -> np.ndarray:
        """獲取指標正規化值"""
        if name in self.indicators:
            return self.indicators[name].normalized
        raise ValueError(f"Indicator {name} not found")


class FormulaGene:
    """
    一個公式基因
    根據指標名稱与操作符組成
    """
    
    def __init__(self, components: List[str], weights: List[float], operations: List[str]):
        """
        Args:
            components: 指標名稱 e.g. ['SMA_20', 'RSI_14', 'ATR_10']
            weights: 權重 e.g. [0.4, 0.3, 0.3]
            operations: 操作符 e.g. ['*', '+', '-']
        """
        self.components = components
        self.weights = np.array(weights) / np.sum(weights)  # 正規化
        self.operations = operations
        self.fitness = 0.0
        self.correlation = 0.0
    
    def calculate(self, indicator_builder: BasicIndicatorBuilder) -> np.ndarray:
        """計算公式值"""
        if not self.components:
            return np.ones(len(indicator_builder.close)) * 0.5
        
        # 獲取所有指標值
        values = []
        for comp in self.components:
            try:
                val = indicator_builder.get_indicator_values(comp)
                values.append(val)
            except ValueError:
                # 備用指標
                values.append(np.ones(len(indicator_builder.close)) * 0.5)
        
        values = np.array(values)
        result = np.zeros(len(indicator_builder.close))
        
        # 根據操作符進行逐步計算
        for i, (val, weight, op) in enumerate(zip(values, self.weights, self.operations)):
            if i == 0:
                result = val * weight
            else:
                if op == '+':
                    result = result + val * weight
                elif op == '-':
                    result = result - val * weight
                elif op == '*':
                    result = result * (val * weight + 0.5)
                elif op == '/':
                    result = result / (val * weight + 0.1)
                elif op == 'max':
                    result = np.maximum(result, val * weight)
                elif op == 'min':
                    result = np.minimum(result, val * weight)
        
        return np.clip(result, 0, 1)
    
    def to_dict(self) -> Dict:
        return {
            'components': self.components,
            'weights': self.weights.tolist(),
            'operations': self.operations,
            'fitness': self.fitness,
            'correlation': self.correlation
        }
    
    def __repr__(self) -> str:
        formula_str = str(self.components[0]) if self.components else 'EMPTY'
        for comp, weight, op in zip(self.components[1:], self.weights[1:], self.operations[1:]):
            formula_str += f" {op} {weight:.2f}*{comp}"
        return formula_str


class AdvancedFeatureOptimizer:
    """
    高級特徵優化器
    使用遺傳算法自動結合積木
    """
    
    def __init__(self, df: pd.DataFrame, population_size: int = 50, generations: int = 100):
        self.df = df
        self.indicator_builder = BasicIndicatorBuilder(df)
        self.indicators_list = self.indicator_builder.get_indicators_list()
        self.population_size = population_size
        self.generations = generations
        self.population: List[FormulaGene] = []
        self.evolution_history = []
        
        indicators_count = len(self.indicators_list)
        print(f"\n[基礎指標] 完全構建了 {indicators_count} 个積木")
        print(f"積木列表: {', '.join(self.indicators_list[:10])}...")
    
    def create_random_gene(self, num_components: int = None) -> FormulaGene:
        """隨機創建一個基因"""
        if num_components is None:
            num_components = random.randint(2, 5)  # 2-5个指標
        
        components = random.sample(self.indicators_list, min(num_components, len(self.indicators_list)))
        weights = [random.uniform(0.1, 1.0) for _ in components]
        operations = [random.choice(['+', '-', '*', '/', 'max', 'min']) for _ in range(len(components)-1)]
        
        return FormulaGene(components, weights, operations)
    
    def crossover(self, gene1: FormulaGene, gene2: FormulaGene) -> FormulaGene:
        """交叉
        """
        # 會合不同的指標
        all_components = list(set(gene1.components + gene2.components))
        if not all_components:
            return self.create_random_gene()
        
        # 需要的組件數
        num_components = min(len(all_components), random.randint(2, 5))
        child_components = random.sample(all_components, num_components)
        
        # 權重為兩個父代的權重混合
        parent1_weights = {c: w for c, w in zip(gene1.components, gene1.weights)}
        parent2_weights = {c: w for c, w in zip(gene2.components, gene2.weights)}
        
        child_weights = []
        for comp in child_components:
            if comp in parent1_weights and comp in parent2_weights:
                w = (parent1_weights[comp] + parent2_weights[comp]) / 2
            elif comp in parent1_weights:
                w = parent1_weights[comp]
            elif comp in parent2_weights:
                w = parent2_weights[comp]
            else:
                w = random.uniform(0.1, 1.0)
            child_weights.append(w)
        
        child_operations = [random.choice(['+', '-', '*', '/', 'max', 'min']) 
                           for _ in range(len(child_components)-1)]
        
        return FormulaGene(child_components, child_weights, child_operations)
    
    def mutate(self, gene: FormulaGene, mutation_rate: float = 0.2) -> FormulaGene:
        """變率
        """
        mutant = deepcopy(gene)
        
        # 改變權重
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutant.weights)-1)
            mutant.weights[idx] = random.uniform(0.1, 1.0)
            mutant.weights = mutant.weights / np.sum(mutant.weights)
        
        # 改變操作符
        if random.random() < mutation_rate and len(mutant.operations) > 0:
            idx = random.randint(0, len(mutant.operations)-1)
            mutant.operations[idx] = random.choice(['+', '-', '*', '/', 'max', 'min'])
        
        # 改變指標
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutant.components)-1)
            # 列表中去掉該指標
            remaining = [ind for ind in self.indicators_list if ind not in mutant.components]
            if remaining:
                mutant.components[idx] = random.choice(remaining)
        
        return mutant
    
    def evaluate_formula_for_target(self, gene: FormulaGene, target_values: np.ndarray) -> float:
        """評估公式是否採合目標值
        
        Args:
            gene: 公式基因
            target_values: 目標值 (例如市場波動性)
        
        Returns:
            相關性 [-1, 1]
        """
        try:
            formula_values = gene.calculate(self.indicator_builder)
            # 計算相關性
            valid_idx = ~np.isnan(formula_values) & ~np.isnan(target_values)
            if np.sum(valid_idx) < 10:
                return 0.0
            
            correlation = np.corrcoef(
                formula_values[valid_idx],
                target_values[valid_idx]
            )[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def evolve_for_target(self, target_values: np.ndarray, target_name: str, num_generations: int = None):
        """
        針對特定目標進行進化
        
        Args:
            target_values: 目標值
            target_name: 目標名稱 (e.g., 'volatility', 'trend', 'direction')
            num_generations: 進化代數
        """
        if num_generations is None:
            num_generations = self.generations
        
        print(f"\n[齊法] 量化目標: {target_name}")
        print(f"[進化] 開始進化... ({num_generations} 代)\n")
        
        # 初始化
        self.population = [self.create_random_gene() for _ in range(self.population_size)]
        
        best_gene = None
        best_fitness = -np.inf
        
        for generation in range(num_generations):
            # 評估
            for gene in self.population:
                gene.correlation = self.evaluate_formula_for_target(gene, target_values)
                gene.fitness = abs(gene.correlation)  # 使用絕對值
            
            # 排序
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            current_best = self.population[0]
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_gene = deepcopy(current_best)
            
            # 記錄
            avg_fitness = np.mean([g.fitness for g in self.population])
            if (generation + 1) % max(1, num_generations // 10) == 0:
                gen_str = f"第 {generation+1:3d} 代: 最佳相關性={current_best.correlation:+.4f} "
                print(gen_str + f"平均={avg_fitness:.4f} | {current_best}")
            
            # 生成下一代
            new_population = []
            elite_size = max(1, self.population_size // 5)
            new_population.extend(self.population[:elite_size])
            
            while len(new_population) < self.population_size:
                parent1 = self.population[random.randint(0, min(10, len(self.population)-1))]
                parent2 = self.population[random.randint(0, min(10, len(self.population)-1))]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
        
        result_str = f"\n[結果] {target_name} 公式優化完成"
        print(result_str)
        print(f"最佳相關性: {best_gene.correlation:+.4f}")
        print(f"公式: {best_gene}")
        
        return best_gene
    
    def save_results(self, results: Dict):
        """保存結果"""
        os.makedirs('results', exist_ok=True)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'formulas': {
                'volatility': results['volatility'].to_dict() if 'volatility' in results else None,
                'trend': results['trend'].to_dict() if 'trend' in results else None,
                'direction': results['direction'].to_dict() if 'direction' in results else None
            },
            'indicators_used': self.indicators_list
        }
        
        with open('results/advanced_formula_optimization.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
        save_str = "[保存] 結果已保存: results/advanced_formula_optimization.json"
        print(f"\n{save_str}")


def main():
    print("\n" + "#" * 80)
    print("# 高級特徵生成器 (Advanced Feature Builder)")
    print("# 基於 30 个積木的自動優化")
    print("#" * 80)
    
    # 加載數據
    print("\n[一] 加載數據...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print(f"✓ 數據加載成功: {len(df)} 根 K 線")
    except Exception as e:
        print(f"✗ 數據加載失敗: {e}")
        return
    
    # 創建優化器
    print("\n[二] 創建優化器...")
    optimizer = AdvancedFeatureOptimizer(df, population_size=50, generations=100)
    
    # 計算目標值
    print("\n[三] 計算目標值...")
    
    # 1. 波動性目標: 市場實際波動性
    close_series = pd.Series(df['close'].values)
    volatility_target = close_series.pct_change().rolling(window=20).std().values
    print("✓ 波動性目標計算完成")
    
    # 2. 趨勢目標: 上漲下跌的次數
    price_change = np.diff(df['close'].values)
    trend_target = (price_change > 0).astype(float)
    print("✓ 趨勢目標計算完成")
    
    # 3. 方向目標: 下一根 K 線是否上漲
    direction_target = trend_target  # 可以上移一個週期
    print("✓ 方向目標計算完成")
    
    # 優化 3 个公式
    print("\n" + "=" * 80)
    print("開始上師自動優化公式...")
    print("=" * 80)
    
    results = {}
    
    # 優化波動性公式
    print("\n" + "#" * 80)
    print("# 優化 1: 波動性公式")
    print("#" * 80)
    volatility_gene = optimizer.evolve_for_target(volatility_target, 'volatility', num_generations=50)
    results['volatility'] = volatility_gene
    
    # 優化趨勢公式
    print("\n" + "#" * 80)
    print("# 優化 2: 趨勢公式")
    print("#" * 80)
    trend_gene = optimizer.evolve_for_target(trend_target, 'trend', num_generations=50)
    results['trend'] = trend_gene
    
    # 優化方向公式
    print("\n" + "#" * 80)
    print("# 優化 3: 方向公式")
    print("#" * 80)
    direction_gene = optimizer.evolve_for_target(direction_target, 'direction', num_generations=50)
    results['direction'] = direction_gene
    
    # 保存結果
    print("\n" + "=" * 80)
    optimizer.save_results(results)
    print("=" * 80 + "\n")
    
    # 打印最終整合
    print("\n[最終結果]")
    vol_str = f"波動性公式: {volatility_gene}"
    trend_str = f"趨勢公式: {trend_gene}"
    dir_str = f"方向公式: {direction_gene}"
    print(f"\n{vol_str}")
    print(f"{trend_str}")
    print(f"{dir_str}")


if __name__ == "__main__":
    main()
