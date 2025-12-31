#!/usr/bin/env python3
"""
公式分析器

功能:
1. 分析演化后的公式結構
2. 可視化積木的貢獻
3. 計算公式的有效性
"""

import pandas as pd
import numpy as np
import json
import os
from advanced_feature_builder import (
    BasicIndicatorBuilder,
    AdvancedFeatureOptimizer,
    FormulaGene
)


class FormulaAnalyzer:
    """
    公式分析器
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.indicator_builder = BasicIndicatorBuilder(df)
    
    def analyze_formula(self, gene: FormulaGene) -> dict:
        """
        渐進註解公式成分
        """
        analysis = {
            'formula_str': str(gene),
            'components': gene.components,
            'weights': gene.weights.tolist(),
            'operations': gene.operations,
            'fitness': gene.fitness,
            'correlation': gene.correlation,
            'component_analysis': []
        }
        
        # 分析每个積木的貢獻
        for comp, weight in zip(gene.components, gene.weights):
            try:
                comp_values = self.indicator_builder.get_indicator_values(comp)
                contribution = weight * comp_values
                
                analysis['component_analysis'].append({
                    'name': comp,
                    'weight': weight,
                    'mean_value': float(np.nanmean(comp_values)),
                    'std_value': float(np.nanstd(comp_values)),
                    'contribution_mean': float(np.nanmean(contribution))
                })
            except:
                pass
        
        return analysis
    
    def print_formula_report(self, gene: FormulaGene, name: str):
        """
        打印公式報告
        """
        analysis = self.analyze_formula(gene)
        
        print(f"\n" + "=" * 80)
        name_upper = name.upper()
        print(f"公式: {name_upper}")
        print("=" * 80)
        
        print(f"\n公式表達式:")
        print(f"  {gene}")
        
        print(f"\n查證指標:")
        print(f"  相關性: {gene.correlation:+.4f}")
        print(f"  適合度: {gene.fitness:.4f}")
        
        num_components = len(gene.components)
        print(f"\n組件詳情 ({num_components} 個積木):")
        print(f"  {'\u7b2c':>3} | {'\u540d称':<20} | {'\u6b0a\u91cd':>8} | {'\u5e73\u5747\u503c':>8} | {'\u8ca2\u737b':>8}")
        print(f"  {'-'*3}+{'-'*22}+{'-'*10}+{'-'*10}+{'-'*10}")
        
        for i, comp_info in enumerate(analysis['component_analysis'], 1):
            comp_name = comp_info['name']
            comp_weight = comp_info['weight']
            comp_mean = comp_info['mean_value']
            comp_contrib = comp_info['contribution_mean']
            print(f"  {i:3d} | {comp_name:<20} | "
                  f"{comp_weight:8.4f} | "
                  f"{comp_mean:8.4f} | "
                  f"{comp_contrib:8.4f}")
        
        print(f"\n" + "=" * 80)
    
    def export_formula_code(self, genes: dict, output_file: str = 'results/evolved_formulas.py'):
        """
        射出公式為 Python 代碼
        """
        code = """
# 自動演化的 3 套公式
# 根據一日画市場數據自動優化

import numpy as np
import pandas as pd

"""
        
        # 生成每个公式的代碼
        for target, gene in genes.items():
            code += f"""def {target}_formula(df, indicator_builder):
    \"\"\"
    {target.upper()} 公式
    相關性: {gene.correlation:+.4f}
    使用 {len(gene.components)} 個積木
    \"\"\"
    result = np.zeros(len(df))
    
"""
            
            operations_list = gene.operations + ['']
            for i, (comp, weight, op) in enumerate(zip(gene.components, gene.weights, operations_list)):
                code += f"    # {i+1}. {comp} (\u6b0a\u91cd: {weight:.4f})\n"
                code += f"    try:\n"
                code += f"        val_{i} = indicator_builder.get_indicator_values('{comp}')\n"
                code += f"    except:\n"
                code += f"        val_{i} = np.ones(len(df)) * 0.5\n"
                
                if i == 0:
                    code += f"    result = val_{i} * {weight:.4f}\n\n"
                else:
                    prev_op = gene.operations[i-1]
                    if prev_op == '+':
                        code += f"    result = result + val_{i} * {weight:.4f}\n\n"
                    elif prev_op == '-':
                        code += f"    result = result - val_{i} * {weight:.4f}\n\n"
                    elif prev_op == '*':
                        code += f"    result = result * (val_{i} * {weight:.4f} + 0.5)\n\n"
                    elif prev_op == '/':
                        code += f"    result = result / (val_{i} * {weight:.4f} + 0.1)\n\n"
                    elif prev_op == 'max':
                        code += f"    result = np.maximum(result, val_{i} * {weight:.4f})\n\n"
                    elif prev_op == 'min':
                        code += f"    result = np.minimum(result, val_{i} * {weight:.4f})\n\n"
            
            code += f"    return np.clip(result, 0, 1)\n\n\n"
        
        os.makedirs('results', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        export_msg = f"✓ 公式代碼已射出: {output_file}"
        print(f"\n{export_msg}")
    
    def export_formula_description(self, genes: dict, output_file: str = 'results/formula_description.md'):
        """
        射出公式描述 (Markdown 格式)
        """
        md = "# 自動演化的 3 套公式\n\n"
        
        for target, gene in genes.items():
            target_upper = target.upper()
            md += f"## {target_upper} 公式\n\n"
            md += f"**相關性**: {gene.correlation:+.4f}\n\n"
            md += f"**使用積木**: {len(gene.components)} 個\n\n"
            
            md += f"### 公式積木\n\n"
            md += f"```\n{gene}\n```\n\n"
            
            md += f"### 組件詳情\n\n"
            md += f"| 蟶標 | 名称 | 權重 | 積木 |\n"
            md += f"|:----:|:----:|:----:|:----:|\n"
            
            operations_list = gene.operations + ['']
            for i, (comp, weight, op) in enumerate(zip(gene.components, gene.weights, operations_list)):
                md += f"| {i+1} | `{comp}` | {weight:.4f} | {op} |\n"
            
            md += f"\n---\n\n"
        
        os.makedirs('results', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        export_desc_msg = f"✓ 公式描述已射出: {output_file}"
        print(f"{export_desc_msg}")


def main():
    print("\n" + "#" * 80)
    print("# 公式分析器 (Formula Analyzer)")
    print("#" * 80)
    
    # 加載优化結果
    print("\n[一] 加載优化結果...")
    try:
        with open('results/advanced_formula_optimization.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        print("✓ 結果加載成功")
    except Exception as e:
        print(f"✗ 加載失敗: {e}")
        print("提示: 需要先運行 advanced_feature_builder.py")
        return
    
    # 加載數據
    print("\n[二] 加載數據...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print(f"✓ 數據加載成功")
    except Exception as e:
        print(f"✗ 加載失敗: {e}")
        return
    
    # 創建分析器
    analyzer = FormulaAnalyzer(df)
    
    # 重建公式對象
    print("\n[三] 重建公式對象...")
    genes = {}
    
    for target in ['volatility', 'trend', 'direction']:
        if results['formulas'].get(target):
            formula_dict = results['formulas'][target]
            gene = FormulaGene(
                components=formula_dict['components'],
                weights=formula_dict['weights'],
                operations=formula_dict['operations']
            )
            gene.fitness = formula_dict['fitness']
            gene.correlation = formula_dict['correlation']
            genes[target] = gene
    
    # 分析並打印
    print("\n[四] 分析公式...\n")
    for target, gene in genes.items():
        analyzer.print_formula_report(gene, target)
    
    # 射出
    print("\n[五] 射出公式...")
    analyzer.export_formula_code(genes)
    analyzer.export_formula_description(genes)
    
    print("\n" + "#" * 80)
    print("分析完成!")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
