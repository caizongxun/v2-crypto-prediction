"""
公式生成器 - AI 逆向推理

使用遟傳演算法進行优化，一自動發現最優公式
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple
import random
from .trend_strength import TrendStrengthFormula
from .volatility_index import VolatilityIndexFormula
from .direction_confirmation import DirectionConfirmationFormula


class FormulaGenerator:
    """
    使用遟傳演算法的公式生成器
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        indicators_dict: Dict,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        """
        初始化生成器
        
        Args:
            df: K線數據
            indicators_dict: 指標字典 {'指標名': 值序列, ...}
            population_size: 人口數
            generations: 演化代數
            mutation_rate: 突變率
            crossover_rate: 交叉率
        """
        self.df = df
        self.indicators = indicators_dict
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.trend_formula = None
        self.volatility_formula = None
        self.direction_formula = None
    
    def generate(
        self,
        formula_types: List[str] = None
    ) -> Dict:
        """
        生成三個黃金公式
        
        Args:
            formula_types: 公式類形
        
        Returns:
            Dict: 三個公式
        """
        if formula_types is None:
            formula_types = ['trend_strength', 'volatility_index', 'direction_confirmation']
        
        results = {}
        
        # 第 1 個公式: 趨勢強度
        print("\n" + "="*70)
        print("正在發現趨勢強度公式...")
        print("="*70)
        self.trend_formula = TrendStrengthFormula()
        results['trend_strength'] = self._evolve_formula(
            self.trend_formula,
            formula_type='trend_strength'
        )
        
        # 第 2 個公式: 波動率
        print("\n" + "="*70)
        print("正在發現波動率公式...")
        print("="*70)
        self.volatility_formula = VolatilityIndexFormula()
        results['volatility_index'] = self._evolve_formula(
            self.volatility_formula,
            formula_type='volatility_index'
        )
        
        # 第 3 個公式: 方向確認
        print("\n" + "="*70)
        print("正在發現方向確認公式...")
        print("="*70)
        self.direction_formula = DirectionConfirmationFormula()
        results['direction_confirmation'] = self._evolve_formula(
            self.direction_formula,
            formula_type='direction_confirmation'
        )
        
        return results
    
    def _evolve_formula(
        self,
        formula_obj,
        formula_type: str
    ) -> Dict:
        """
        使用遟傳演算法优化公式
        
        Args:
            formula_obj: 公式對象
            formula_type: 公式類形
        
        Returns:
            Dict: 优化了的公式配置
        """
        # 初始化原始人口
        population = [
            formula_obj.generate_random_coefficients()
            for _ in range(self.population_size)
        ]
        
        best_fitness = -float('inf')
        best_individual = None
        
        for gen in range(self.generations):
            # 計算每個個体的適应度
            fitness_scores = [
                self._evaluate_fitness(individual, formula_type)
                for individual in population
            ]
            
            # 找到最好的個体
            best_idx = np.argmax(fitness_scores)
            current_best = population[best_idx]
            current_best_fitness = fitness_scores[best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best.copy()
            
            print(f"  代数 {gen+1}/{self.generations} - 最佳適应度: {best_fitness:.4f}")
            
            # 選拧、交叉、突變
            new_population = [best_individual]  # 精英保留
            
            while len(new_population) < self.population_size:
                # 適应度選拧
                parent1_idx = self._select_parent(fitness_scores)
                parent2_idx = self._select_parent(fitness_scores)
                
                parent1 = population[parent1_idx].copy()
                parent2 = population[parent2_idx].copy()
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # 突變
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population[:self.population_size]
        
        print(f"\n  最終最佳適应度: {best_fitness:.4f}")
        print(f"  最优需数算: {best_individual}")
        
        return {
            'coefficients': best_individual,
            'fitness': best_fitness,
            'formula_type': formula_type
        }
    
    def _evaluate_fitness(
        self,
        coefficients: Dict,
        formula_type: str
    ) -> float:
        """
        計算適应度 (根據公式輸出的準確性)
        
        Args:
            coefficients: 公式係數
            formula_type: 公式類形
        
        Returns:
            float: 適应度值
        """
        try:
            if formula_type == 'trend_strength':
                result = self.trend_formula.calculate(self.indicators, coefficients)
            elif formula_type == 'volatility_index':
                result = self.volatility_formula.calculate(self.indicators, coefficients)
            elif formula_type == 'direction_confirmation':
                result = self.direction_formula.calculate(self.indicators, coefficients)
            else:
                return -float('inf')
            
            # 適应度 = 值位於 0-1的比例 + 低方差
            valid_count = np.sum((result >= 0) & (result <= 1))
            valid_ratio = valid_count / len(result)
            
            std_dev = np.std(result)
            
            # 適应度 = 有效比例 + 方差 (低方差比高方差更好)
            fitness = valid_ratio + (1 - std_dev)
            
            return fitness
        
        except Exception as e:
            return -float('inf')
    
    def _select_parent(self, fitness_scores: List[float]) -> int:
        """
        使用矩形選拧法選拧父本
        """
        fitness_array = np.array(fitness_scores)
        
        # 碰到負值，穩定化
        if np.min(fitness_array) < 0:
            fitness_array = fitness_array - np.min(fitness_array) + 1e-6
        
        # 檈率与適应度成正比
        probabilities = fitness_array / np.sum(fitness_array)
        
        return np.random.choice(len(fitness_scores), p=probabilities)
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        两点交叉
        """
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """
        高斯突變
        """
        mutant = individual.copy()
        key = random.choice(list(mutant.keys()))
        mutant[key] += np.random.normal(0, 0.1)
        mutant[key] = np.clip(mutant[key], -1, 1)  # 限制每个系数
        return mutant
