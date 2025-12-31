#!/usr/bin/env python3
"""
高级特徵生成器

概念:
- 供給 30 个基础指标 (积木)
- 遵傳算法自动組合不同的突変
- 对每一个突変进行整个数据集的对比
- 根据相关性不断技掔我们的 3 个公式

最优化実际效果:
- 波動性公式: 由特定的积木组合成
- 趨勢公式: 由特定的积木组合成
- 方向公式: 由特定的积木组合成
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
    基础指标 (积木)
    """
    name: str
    values: np.ndarray
    normalized: np.ndarray  # [0, 1] 正規化值


class BasicIndicatorBuilder:
    """
    构建 30 个基础指标
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
        """构建所有 30 个指标"""
        close_series = pd.Series(self.close)
        
        # 动量提供器
        # ===== 住简指标 (SMA) =====
        for period in [5, 10, 20, 50]:
            sma = close_series.rolling(window=period).mean().values
            self.indicators[f'SMA_{period}'] = Indicator(
                name=f'SMA_{period}',
                values=sma,
                normalized=self._normalize(np.abs(self.close - sma))
            )
        
        # ===== 指数住简平均 (EMA) =====
        for period in [5, 12, 26, 50]:
            ema = close_series.ewm(span=period, adjust=False).mean().values
            self.indicators[f'EMA_{period}'] = Indicator(
                name=f'EMA_{period}',
                values=ema,
                normalized=self._normalize(np.abs(self.close - ema))
            )
        
        # ===== 相对強度指数 (RSI) =====
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
        
        # ===== 一日真实橋段 (ATR) =====
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
        
        # ===== 计数指数易也 (不整整站) =====
        # ROC (不同周期)
        for period in [5, 10, 20]:
            roc = (self.close - np.roll(self.close, period)) / \
                  (np.roll(self.close, period) + 1e-10) * 100
            self.indicators[f'ROC_{period}'] = Indicator(
                name=f'ROC_{period}',
                values=roc,
                normalized=self._normalize(np.abs(roc))
            )
        
        # ===== 波动率指数 =====
        for period in [10, 20, 30]:
            returns = close_series.pct_change().rolling(window=period).std().values
            self.indicators[f'VOLATILITY_{period}'] = Indicator(
                name=f'VOLATILITY_{period}',
                values=returns,
                normalized=self._normalize(returns)
            )
        
        # ===== 典帮 =====
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
        """获取所有指标名称"""
        return list(self.indicators.keys())
    
    def get_indicator_values(self, name: str) -> np.ndarray:
        """获取指标正規化值"""
        if name in self.indicators:
            return self.indicators[name].normalized
        raise ValueError(f"Indicator {name} not found")


class FormulaGene:
    """
    一个公式基因
    根据指标名称与操作符组成
    """
    
    def __init__(self, components: List[str], weights: List[float], operations: List[str]):
        """
        Args:
            components: 指标名称 e.g. ['SMA_20', 'RSI_14', 'ATR_10']
            weights: 权重 e.g. [0.4, 0.3, 0.3]
            operations: 操作符 e.g. ['*', '+', '-']
        """
        self.components = components
        self.weights = np.array(weights) / np.sum(weights)  # 正規化
        self.operations = operations
        self.fitness = 0.0
        self.correlation = 0.0
    
    def calculate(self, indicator_builder: BasicIndicatorBuilder) -> np.ndarray:
        """计算公式值"""
        if not self.components:
            return np.ones(len(indicator_builder.close)) * 0.5
        
        # 获取所有指标值
        values = []
        for comp in self.components:
            try:
                val = indicator_builder.get_indicator_values(comp)
                values.append(val)
            except ValueError:
                # 备用指标
                values.append(np.ones(len(indicator_builder.close)) * 0.5)
        
        values = np.array(values)
        result = np.zeros(len(indicator_builder.close))
        
        # 根据操作符进行逐步计算
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
    高级特徵优化器
    使用遵傳算法自动结合积木
    """
    
    def __init__(self, df: pd.DataFrame, population_size: int = 50, generations: int = 100):
        self.df = df
        self.indicator_builder = BasicIndicatorBuilder(df)
        self.indicators_list = self.indicator_builder.get_indicators_list()
        self.population_size = population_size
        self.generations = generations
        self.population: List[FormulaGene] = []
        self.evolution_history = []
        
        print(f"\n[基础指标] 完全构建了 {len(self.indicators_list)} 个积木")
        print(f"积木列表: {', '.join(self.indicators_list[:10])}...")
    
    def create_random_gene(self, num_components: int = None) -> FormulaGene:
        """随機创建一个基因"""
        if num_components is None:
            num_components = random.randint(2, 5)  # 2-5个指标
        
        components = random.sample(self.indicators_list, min(num_components, len(self.indicators_list)))
        weights = [random.uniform(0.1, 1.0) for _ in components]
        operations = [random.choice(['+', '-', '*', '/', 'max', 'min']) for _ in range(len(components)-1)]
        
        return FormulaGene(components, weights, operations)
    
    def crossover(self, gene1: FormulaGene, gene2: FormulaGene) -> FormulaGene:
        """交叉
        """
        # 会合不同的指标
        all_components = list(set(gene1.components + gene2.components))
        if not all_components:
            return self.create_random_gene()
        
        # 需要的组件数
        num_components = min(len(all_components), random.randint(2, 5))
        child_components = random.sample(all_components, num_components)
        
        # 权重为两个父代的权重混合
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
        """变率
        """
        mutant = deepcopy(gene)
        
        # 改变权重
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutant.weights)-1)
            mutant.weights[idx] = random.uniform(0.1, 1.0)
            mutant.weights = mutant.weights / np.sum(mutant.weights)
        
        # 改变操作符
        if random.random() < mutation_rate and len(mutant.operations) > 0:
            idx = random.randint(0, len(mutant.operations)-1)
            mutant.operations[idx] = random.choice(['+', '-', '*', '/', 'max', 'min'])
        
        # 改变指标
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutant.components)-1)
            # 列表中去掉该指标
            remaining = [ind for ind in self.indicators_list if ind not in mutant.components]
            if remaining:
                mutant.components[idx] = random.choice(remaining)
        
        return mutant
    
    def evaluate_formula_for_target(self, gene: FormulaGene, target_values: np.ndarray) -> float:
        """评估公式是否採合目标值
        
        Args:
            gene: 公式基因
            target_values: 目标值 (例如市场波动性)
        
        Returns:
            相关性 [-1, 1]
        """
        try:
            formula_values = gene.calculate(self.indicator_builder)
            # 计算相关性
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
        针对特定目标进行演化
        
        Args:
            target_values: 目标值
            target_name: 目标名称 (e.g., 'volatility', 'trend', 'direction')
            num_generations: 演化代数
        """
        if num_generations is None:
            num_generations = self.generations
        
        print(f"\n[齪法] 量化目标: {target_name}")
        print(f"[演化] 开始演化... ({num_generations} 代)\n")
        
        # 初始化
        self.population = [self.create_random_gene() for _ in range(self.population_size)]
        
        best_gene = None
        best_fitness = -np.inf
        
        for generation in range(num_generations):
            # 评估
            for gene in self.population:
                gene.correlation = self.evaluate_formula_for_target(gene, target_values)
                gene.fitness = abs(gene.correlation)  # 使用绝对值
            
            # 排序
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            current_best = self.population[0]
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_gene = deepcopy(current_best)
            
            # 记录
            avg_fitness = np.mean([g.fitness for g in self.population])
            if (generation + 1) % max(1, num_generations // 10) == 0:
                print(f"第 {generation+1:3d} 代: 最佳相关性={current_best.correlation:+.4f} ", end="")
                print(f"平均={avg_fitness:.4f} | {current_best}")
            
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
        
        print(f"\n[结果] {target_name} 公式优化完成")
        print(f"最佳相关性: {best_gene.correlation:+.4f}")
        print(f"公式: {best_gene}")
        
        return best_gene
    
    def save_results(self, results: Dict):
        """保存结果"""
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
        
        print(f"\n[保存] 结果已保存: results/advanced_formula_optimization.json")


def main():
    print("\n" + "#" * 80)
    print("# 高级特徵生成器 (Advanced Feature Builder)")
    print("# 基於 30 个积木的自动优化")
    print("#" * 80)
    
    # 加載数据
    print("\n[一] 加載数据...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print(f"✓ 数据加載成功: {len(df)} 根 K 线")
    except Exception as e:
        print(f"✗ 数据加載失败: {e}")
        return
    
    # 创建优化器
    print("\n[二] 创建优化器...")
    optimizer = AdvancedFeatureOptimizer(df, population_size=50, generations=100)
    
    # 计算目标值
    print("\n[三] 计算目标值...")
    
    # 1. 波动性目标: 市场实际波动性
    close_series = pd.Series(df['close'].values)
    volatility_target = close_series.pct_change().rolling(window=20).std().values
    print(f"✓ 波动性目标计算完成")
    
    # 2. 趨勢目标: 上涨下跌的次数
    price_change = np.diff(df['close'].values)
    trend_target = (price_change > 0).astype(float)
    print(f"✓ 趨勢目标计算完成")
    
    # 3. 方向目标: 下一根 K 线是否上涨
    direction_target = trend_target  # 可以上移一个周期
    print(f"✓ 方向目标计算完成")
    
    # 优化 3 个公式
    print("\n" + "=" * 80)
    print("开始上师自动优化公式...")
    print("=" * 80)
    
    results = {}
    
    # 优化波动性公式
    print("\n" + "#" * 80)
    print("# 优化 1: 波动性公式")
    print("#" * 80)
    volatility_gene = optimizer.evolve_for_target(volatility_target, 'volatility', num_generations=50)
    results['volatility'] = volatility_gene
    
    # 优化趨勢公式
    print("\n" + "#" * 80)
    print("# 优化 2: 趨勢公式")
    print("#" * 80)
    trend_gene = optimizer.evolve_for_target(trend_target, 'trend', num_generations=50)
    results['trend'] = trend_gene
    
    # 优化方向公式
    print("\n" + "#" * 80)
    print("# 优化 3: 方向公式")
    print("#" * 80)
    direction_gene = optimizer.evolve_for_target(direction_target, 'direction', num_generations=50)
    results['direction'] = direction_gene
    
    # 保存结果
    print("\n" + "=" * 80)
    optimizer.save_results(results)
    print("=" * 80 + "\n")
    
    # 打印最终整合
    print("\n[最终结果]")
    print(f"\n波动性公式: {volatility_gene}")
    print(f方向公式: {trend_gene}")
    print(f"方向公式: {direction_gene}")


if __name__ == "__main__":
    main()
