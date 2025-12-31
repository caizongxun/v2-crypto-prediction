"""
參數優化引擎

支援不同的優化方法：
- Grid Search (羗格搜尋)
- Random Search (隨機搜尋)
- Bayesian Optimization (Optuna)

目標函數：
- Sharpe Ratio
- 勝率
- 收益率
- 最大回撤
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass
import itertools
import json
from datetime import datetime

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass
class OptimizationResult:
    """優化結果"""
    best_params: Dict
    best_score: float
    all_trials: List[Tuple[Dict, float]]
    search_method: str
    total_trials: int
    duration_seconds: float


class ParameterOptimizer:
    """
    參數優化器
    """
    
    def __init__(self, objective_func: Callable):
        """
        初始化優化器
        
        Args:
            objective_func: 目標函数 (params -> score)
        """
        self.objective_func = objective_func
        self.all_trials = []
    
    def grid_search(
        self,
        param_grid: Dict[str, List],
        verbose: bool = True
    ) -> OptimizationResult:
        """
        羗格搜尋
        
        Args:
            param_grid: 參數筒 {參數名: [值1, 值2, ...]}
            verbose: 是否打印進度
        
        Returns:
            OptimizationResult: 優化結果
        """
        import time
        start_time = time.time()
        
        # 生成所有組合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(itertools.product(*param_values))
        total = len(combinations)
        
        best_score = -np.inf
        best_params = None
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            try:
                score = self.objective_func(params)
            except Exception as e:
                if verbose:
                    print(f"  試驗 {i+1}/{total} 失敗: {str(e)[:50]}")
                score = -np.inf
            
            self.all_trials.append((params, score))
            
            if score > best_score:
                best_score = score
                best_params = params
            
            if verbose and (i + 1) % max(1, total // 10) == 0:
                print(f"  進度: {i+1}/{total}, 最佳得分: {best_score:.4f}")
        
        duration = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=self.all_trials,
            search_method="Grid Search",
            total_trials=total,
            duration_seconds=duration
        )
    
    def random_search(
        self,
        param_space: Dict[str, Tuple],
        n_trials: int = 100,
        random_state: int = 42,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        隨機搜尋
        
        Args:
            param_space: 參數空間 {參數名: (min, max)} 或 (list of choices)
            n_trials: 試驗次數
            random_state: 隈种子
            verbose: 是否打印進度
        
        Returns:
            OptimizationResult: 優化結果
        """
        import time
        start_time = time.time()
        
        np.random.seed(random_state)
        
        best_score = -np.inf
        best_params = None
        
        for trial in range(n_trials):
            params = {}
            
            for param_name, param_range in param_space.items():
                if isinstance(param_range[0], (int, float)):
                    # 數值篆度
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
                    # 如果是整數伪伈对齐
                    if isinstance(param_range[0], int):
                        params[param_name] = int(params[param_name])
                else:
                    # 選擇列表
                    params[param_name] = np.random.choice(param_range)
            
            try:
                score = self.objective_func(params)
            except Exception as e:
                if verbose:
                    print(f"  試驗 {trial+1}/{n_trials} 失敗: {str(e)[:50]}")
                score = -np.inf
            
            self.all_trials.append((params, score))
            
            if score > best_score:
                best_score = score
                best_params = params
            
            if verbose and (trial + 1) % max(1, n_trials // 10) == 0:
                print(f"  進度: {trial+1}/{n_trials}, 最佳得分: {best_score:.4f}")
        
        duration = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=self.all_trials,
            search_method="Random Search",
            total_trials=n_trials,
            duration_seconds=duration
        )
    
    def bayesian_search(
        self,
        param_space: Dict[str, Tuple],
        n_trials: int = 100,
        random_state: int = 42,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        伯賽亞最优化 (Optuna)
        
        Args:
            param_space: 參數空間
            n_trials: 試驗次数
            random_state: 隈种子
            verbose: 是否打印進度
        
        Returns:
            OptimizationResult: 優化結果
        """
        if not HAS_OPTUNA:
            print("Optuna 未安裝, 使用 Random Search 替代")
            return self.random_search(param_space, n_trials, random_state, verbose)
        
        import time
        start_time = time.time()
        
        def optuna_objective(trial):
            params = {}
            
            for param_name, param_range in param_space.items():
                if isinstance(param_range[0], (int, float)):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            try:
                score = self.objective_func(params)
            except Exception as e:
                if verbose:
                    print(f"  試驗失敗: {str(e)[:50]}")
                score = -np.inf
            
            self.all_trials.append((params.copy(), score))
            return score
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=verbose)
        
        best_params = study.best_params
        best_score = study.best_value
        
        duration = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=self.all_trials,
            search_method="Bayesian Optimization (Optuna)",
            total_trials=n_trials,
            duration_seconds=duration
        )
    
    @staticmethod
    def print_result(result: OptimizationResult):
        """列印優化結果"""
        print("\n" + "=" * 70)
        print("參數優化結果")
        print("=" * 70)
        
        print(f"\n搜尋方法: {result.search_method}")
        print(f"試驗次數: {result.total_trials}")
        print(f"恭費時間: {result.duration_seconds:.2f} 秒")
        
        print(f"\n最佳得分: {result.best_score:.4f}")
        print(f"\n最佳參數:")
        for key, value in result.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 70 + "\n")
    
    @staticmethod
    def export_results(
        result: OptimizationResult,
        filepath: str
    ):
        """匯出結果為 JSON"""
        export_data = {
            "search_method": result.search_method,
            "total_trials": result.total_trials,
            "duration_seconds": result.duration_seconds,
            "best_score": float(result.best_score),
            "best_params": result.best_params,
            "timestamp": datetime.now().isoformat(),
            "all_trials": [
                {"params": params, "score": float(score)}
                for params, score in result.all_trials
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n結果已保存到: {filepath}")
