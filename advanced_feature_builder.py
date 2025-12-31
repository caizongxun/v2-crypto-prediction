#!/usr/bin/env python3
"""
高級特徵生成器

核心修正:
1. 改進目標定義 - 使用連續值而非二分類
2. 改進相關性評估 - 更穩健的計算方法
3. 改進超參數 - 更多種群和代數進化
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
from scipy import stats

@dataclass
class Indicator:
    name: str
    values: np.ndarray
    normalized: np.ndarray

class BasicIndicatorBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.volume = df['volume'].values if 'volume' in df.columns else np.ones_like(self.close)
        self.indicators: Dict[str, Indicator] = {}
        self.build_all_indicators()
    
    def _normalize(self, values: np.ndarray, method: str = 'minmax') -> np.ndarray:
        if method == 'minmax':
            vmin = np.nanmin(values)
            vmax = np.nanmax(values)
            if vmax == vmin:
                return np.ones_like(values) * 0.5
            return (values - vmin) / (vmax - vmin + 1e-10)
        else:
            return np.clip(values, 0, 1)
    
    def build_all_indicators(self):
        close_series = pd.Series(self.close)
        
        # SMA
        for period in [5, 10, 20, 50]:
            sma = close_series.rolling(window=period).mean().values
            self.indicators[f'SMA_{period}'] = Indicator(
                name=f'SMA_{period}',
                values=sma,
                normalized=self._normalize(np.abs(self.close - sma))
            )
        
        # EMA
        for period in [5, 12, 26, 50]:
            ema = close_series.ewm(span=period, adjust=False).mean().values
            self.indicators[f'EMA_{period}'] = Indicator(
                name=f'EMA_{period}',
                values=ema,
                normalized=self._normalize(np.abs(self.close - ema))
            )
        
        # RSI
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
        
        # ATR
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
        
        # ROC
        for period in [5, 10, 20]:
            roc = (self.close - np.roll(self.close, period)) / \
                  (np.roll(self.close, period) + 1e-10) * 100
            self.indicators[f'ROC_{period}'] = Indicator(
                name=f'ROC_{period}',
                values=roc,
                normalized=self._normalize(np.abs(roc))
            )
        
        # VOLATILITY
        for period in [10, 20, 30]:
            returns = close_series.pct_change().rolling(window=period).std().values
            self.indicators[f'VOLATILITY_{period}'] = Indicator(
                name=f'VOLATILITY_{period}',
                values=returns,
                normalized=self._normalize(returns)
            )
        
        # 附加
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
        return list(self.indicators.keys())
    
    def get_indicator_values(self, name: str) -> np.ndarray:
        if name in self.indicators:
            return self.indicators[name].normalized
        raise ValueError(f"Indicator {name} not found")

class FormulaGene:
    def __init__(self, components: List[str], weights: List[float], operations: List[str]):
        self.components = components
        self.weights = np.array(weights) / np.sum(weights)
        self.operations = operations
        self.fitness = 0.0
        self.correlation = 0.0
    
    def calculate(self, indicator_builder: BasicIndicatorBuilder) -> np.ndarray:
        if not self.components:
            return np.ones(len(indicator_builder.close)) * 0.5
        
        values = []
        for comp in self.components:
            try:
                val = indicator_builder.get_indicator_values(comp)
                values.append(val)
            except ValueError:
                values.append(np.ones(len(indicator_builder.close)) * 0.5)
        
        values = np.array(values)
        result = np.zeros(len(indicator_builder.close))
        
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
    def __init__(self, df: pd.DataFrame, population_size: int = 80, generations: int = 100):
        self.df = df
        self.indicator_builder = BasicIndicatorBuilder(df)
        self.indicators_list = self.indicator_builder.get_indicators_list()
        self.population_size = population_size
        self.generations = generations
        self.population: List[FormulaGene] = []
        self.evolution_history = []
        
        indicators_count = len(self.indicators_list)
        print(f"\n[基礎指標] 完全構建了 {indicators_count} 個積木")
        print(f"積木列表: {', '.join(self.indicators_list[:10])}...")
    
    def create_random_gene(self, num_components: int = None) -> FormulaGene:
        if num_components is None:
            num_components = random.randint(2, 5)
        
        components = random.sample(self.indicators_list, min(num_components, len(self.indicators_list)))
        weights = [random.uniform(0.1, 1.0) for _ in components]
        operations = [random.choice(['+', '-', '*', '/', 'max', 'min']) for _ in range(len(components)-1)]
        
        return FormulaGene(components, weights, operations)
    
    def crossover(self, gene1: FormulaGene, gene2: FormulaGene) -> FormulaGene:
        all_components = list(set(gene1.components + gene2.components))
        if not all_components:
            return self.create_random_gene()
        
        num_components = min(len(all_components), random.randint(2, 5))
        child_components = random.sample(all_components, num_components)
        
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
    
    def mutate(self, gene: FormulaGene, mutation_rate: float = 0.25) -> FormulaGene:
        mutant = deepcopy(gene)
        
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutant.weights)-1)
            mutant.weights[idx] = random.uniform(0.1, 1.0)
            mutant.weights = mutant.weights / np.sum(mutant.weights)
        
        if random.random() < mutation_rate and len(mutant.operations) > 0:
            idx = random.randint(0, len(mutant.operations)-1)
            mutant.operations[idx] = random.choice(['+', '-', '*', '/', 'max', 'min'])
        
        if random.random() < mutation_rate:
            idx = random.randint(0, len(mutant.components)-1)
            remaining = [ind for ind in self.indicators_list if ind not in mutant.components]
            if remaining:
                mutant.components[idx] = random.choice(remaining)
        
        return mutant
    
    def evaluate_formula_for_target(self, gene: FormulaGene, target_values: np.ndarray, target_name: str = '') -> float:
        """評估公式 - 使用改進的方法"""
        try:
            formula_values = gene.calculate(self.indicator_builder)
            
            # 第一層檢查: 基本有效性
            if len(formula_values) == 0 or len(target_values) == 0:
                return 0.0
            
            valid_idx = ~np.isnan(formula_values) & ~np.isnan(target_values) & \
                       np.isfinite(formula_values) & np.isfinite(target_values)
            
            if np.sum(valid_idx) < 50:
                return 0.0
            
            formula_clean = formula_values[valid_idx]
            target_clean = target_values[valid_idx]
            
            # 第二層檢查: 數值穩定性
            if not np.all(np.isfinite(formula_clean)) or not np.all(np.isfinite(target_clean)):
                return 0.0
            
            # 檢查標準差 (防止常數值)
            formula_std = np.std(formula_clean)
            target_std = np.std(target_clean)
            
            if formula_std < 1e-6 or target_std < 1e-6:
                return 0.0
            
            # 使用 Spearman 相關性
            try:
                correlation, p_value = stats.spearmanr(formula_clean, target_clean)
            except:
                return 0.0
            
            if np.isnan(correlation):
                return 0.0
            
            # 防止過擬合 (針對 volatility)
            if target_name == 'volatility' and correlation > 0.95:
                correlation = correlation * 0.7  # 更激進的懲罰
            
            return correlation
        except Exception as e:
            return 0.0
    
    def evolve_for_target(self, target_values: np.ndarray, target_name: str, num_generations: int = None):
        if num_generations is None:
            num_generations = self.generations
        
        print(f"\n[齊法] 量化目標: {target_name}")
        print(f"[進化] 開始進化... ({num_generations} 代)\n")
        
        self.population = [self.create_random_gene() for _ in range(self.population_size)]
        
        best_gene = None
        best_fitness = -np.inf
        stagnant_count = 0
        
        for generation in range(num_generations):
            # 評估
            for gene in self.population:
                gene.correlation = self.evaluate_formula_for_target(gene, target_values, target_name)
                gene.fitness = abs(gene.correlation)
            
            # 排序
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            current_best = self.population[0]
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_gene = deepcopy(current_best)
                stagnant_count = 0
            else:
                stagnant_count += 1
            
            avg_fitness = np.mean([g.fitness for g in self.population])
            if (generation + 1) % max(1, num_generations // 10) == 0:
                gen_str = f"第 {generation+1:3d} 代: 最佳相關性={current_best.correlation:+.4f} "
                print(gen_str + f"平均={avg_fitness:.4f} | {current_best}")
            
            # 自適應變異 (停滯時加大變異)
            mutation_rate = 0.25 if stagnant_count < 5 else 0.4
            
            new_population = []
            elite_size = max(1, self.population_size // 5)
            new_population.extend(self.population[:elite_size])
            
            while len(new_population) < self.population_size:
                parent1 = self.population[random.randint(0, min(15, len(self.population)-1))]
                parent2 = self.population[random.randint(0, min(15, len(self.population)-1))]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate)
                new_population.append(child)
            
            self.population = new_population
        
        print(f"\n[結果] {target_name} 公式優化完成")
        print(f"最佳相關性: {best_gene.correlation:+.4f}")
        print(f"公式: {best_gene}")
        
        return best_gene
    
    def save_results(self, results: Dict):
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
        
        print(f"\n[保存] 結果已保存: results/advanced_formula_optimization.json")

def main():
    print("\n" + "#" * 80)
    print("# 高級特徵生成器 (Advanced Feature Builder)")
    print("# 基於 30 個積木的自動優化")
    print("#" * 80)
    
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
    
    print("\n[二] 創建優化器...")
    optimizer = AdvancedFeatureOptimizer(df, population_size=80, generations=100)
    
    print("\n[三] 計算目標值...")
    
    close_series = pd.Series(df['close'].values)
    
    # 波動性: 直接使用市場波動率
    volatility_target = close_series.pct_change().rolling(window=20).std().values
    # 去掉前20個NaN
    volatility_target = volatility_target[20:]
    print("✓ 波動性目標計算完成")
    
    # 趨勢: 使用連續價格變化而非二分類
    price_change = np.diff(df['close'].values)
    # 正規化到 [-1, 1]
    trend_target = price_change / (np.percentile(np.abs(price_change), 95) + 1e-10)
    trend_target = np.clip(trend_target, -1, 1)
    # 補齊長度
    trend_target = np.append([0], trend_target)
    trend_target = trend_target[20:]
    print("✓ 趨勢目標計算完成")
    
    # 方向: 下一根K線的方向強度
    next_direction = np.append(price_change, [0])  # 後移一個位置
    # 使用符號加上絕對變化比例
    direction_target = np.sign(next_direction) * (np.abs(next_direction) / (np.percentile(np.abs(price_change), 95) + 1e-10))
    direction_target = np.clip(direction_target, -1, 1)
    direction_target = direction_target[20:]
    print("✓ 方向目標計算完成")
    
    print("\n" + "=" * 80)
    print("開始上師自動優化公式...")
    print("=" * 80)
    
    results = {}
    
    # 優化波動性
    print("\n" + "#" * 80)
    print("# 優化 1: 波動性公式")
    print("#" * 80)
    volatility_gene = optimizer.evolve_for_target(volatility_target, 'volatility', num_generations=100)
    results['volatility'] = volatility_gene
    
    # 優化趨勢
    print("\n" + "#" * 80)
    print("# 優化 2: 趨勢公式")
    print("#" * 80)
    trend_gene = optimizer.evolve_for_target(trend_target, 'trend', num_generations=100)
    results['trend'] = trend_gene
    
    # 優化方向
    print("\n" + "#" * 80)
    print("# 優化 3: 方向公式")
    print("#" * 80)
    direction_gene = optimizer.evolve_for_target(direction_target, 'direction', num_generations=100)
    results['direction'] = direction_gene
    
    print("\n" + "=" * 80)
    optimizer.save_results(results)
    print("=" * 80)
    
    print("\n[最終結果]")
    print(f"\n波動性公式: {volatility_gene}")
    print(f"趨勢公式: {trend_gene}")
    print(f"方向公式: {direction_gene}")

if __name__ == "__main__":
    main()
