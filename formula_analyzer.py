#!/usr/bin/env python3
"""
公式分析器

功能:
1. 分析演化后的公式结构
2. 可视化积木的贡献
3. 计算公式的有效性
"""

import pandas as pd
import numpy as np
import json
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
    
    def analyze_formula(self, gene: FormulaGene) -> Dict:
        """
        渐进注析公式成分
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
        
        # 分析每个突変的贡献
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
        打印公式报告
        """
        analysis = self.analyze_formula(gene)
        
        print(f"\n" + "=" * 80)
        print(f公式: {name.upper()}")
        print("=" * 80)
        
        print(f"\n公式表达式:")
        print(f"  {gene}")
        
        print(f"\n查证指標:")
        print(f"  相关性: {gene.correlation:+.4f}")
        print(f"  適合度: {gene.fitness:.4f}")
        
        print(f"\n组件详情 ({len(gene.components)} 个积木):")
        print(f"  {'\u7b2c':>3} | {'\u540d称':<20} | {'\u6b0a重':>8} | {'\u5e73\u5747\u503c':>8} | {'贡\u732e':>8}")
        print(f"  {'-'*3}+{'-'*22}+{'-'*10}+{'-'*10}+{'-'*10}")
        
        for i, comp_info in enumerate(analysis['component_analysis'], 1):
            print(f"  {i:3d} | {comp_info['name']:<20} | "
                  f"{comp_info['weight']:8.4f} | "
                  f"{comp_info['mean_value']:8.4f} | "
                  f"{comp_info['contribution_mean']:8.4f}")
        
        print(f"\n" + "=" * 80)
    
    def export_formula_code(self, genes: Dict[str, FormulaGene], output_file: str = 'results/evolved_formulas.py'):
        """
        射出公式为 Python 代码
        """
        code = """
# 自动演化的 3 套公式
# 根据一日画市场数据自动優化

import numpy as np
import pandas as pd

"""
        
        # 生成每个公式的代码
        for target, gene in genes.items():
            code += f"""
def {target}_formula(df, indicator_builder):
    \"\"\"
    {target.upper()} 公式
    相关性: {gene.correlation:+.4f}
    使用 {len(gene.components)} 个积木
    \"\"\"
    result = np.zeros(len(df))
    
"""
            
            for i, (comp, weight, op) in enumerate(zip(gene.components, gene.weights, gene.operations + [''])):
                code += f"    # {i+1}. {comp} (\u6b0a\u91cd: {weight:.4f})\n"
                code += f"    try:\n"
                code += f"        val_{i} = indicator_builder.get_indicator_values('{comp}')\n"
                code += f"    except:\n"
                code += f"        val_{i} = np.ones(len(df)) * 0.5\n"
                
                if i == 0:
                    code += f"    result = val_{i} * {weight:.4f}\n\n"
                else:
                    op = gene.operations[i-1]
                    if op == '+':
                        code += f"    result = result + val_{i} * {weight:.4f}\n\n"
                    elif op == '-':
                        code += f"    result = result - val_{i} * {weight:.4f}\n\n"
                    elif op == '*':
                        code += f"    result = result * (val_{i} * {weight:.4f} + 0.5)\n\n"
                    elif op == '/':
                        code += f"    result = result / (val_{i} * {weight:.4f} + 0.1)\n\n"
                    elif op == 'max':
                        code += f"    result = np.maximum(result, val_{i} * {weight:.4f})\n\n"
                    elif op == 'min':
                        code += f"    result = np.minimum(result, val_{i} * {weight:.4f})\n\n"
            
            code += f"    return np.clip(result, 0, 1)\n\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"\n✓ 公式代码已射出: {output_file}")
    
    def export_formula_description(self, genes: Dict[str, FormulaGene], output_file: str = 'results/formula_description.md'):
        """
        射出公式描述 (Markdown 格式)
        """
        md = "# 自动演化的 3 套公式\n\n"
        
        for target, gene in genes.items():
            md += f"## {target.upper()} 公式\n\n"
            md += f"**相关性**: {gene.correlation:+.4f}\n\n"
            md += f"**使用突変**: {len(gene.components)} 个\n\n"
            
            md += f"### 公式突変\n\n"
            md += f"```\n{gene}\n```\n\n"
            
            md += f"### 组件详情\n\n"
            md += f"| 蟶标 | 名称 | 權重 | 突变 |\n"
            md += f"|:----:|:----:|:----:|:----:|\n"
            
            for i, (comp, weight, op) in enumerate(zip(gene.components, gene.weights, gene.operations + [''])):
                md += f"| {i+1} | `{comp}` | {weight:.4f} | {op} |\n"
            
            md += f"\n---\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"\u2713 公式描述已射出: {output_file}")


def main():
    print("\n" + "#" * 80)
    print("# 公式分析器 (Formula Analyzer)")
    print("#" * 80)
    
    # 加載优化结果
    print("\n[一] 加載优化结果...")
    try:
        with open('results/advanced_formula_optimization.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        print("✓ 结果加載成功")
    except Exception as e:
        print(f"✗ 加載失败: {e}")
        print("提示: 需要先运行 advanced_feature_builder.py")
        return
    
    # 加載数据
    print("\n[二] 加載数据...")
    try:
        df = pd.read_parquet("./data/btc_15m.parquet")
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31 23:59:59')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print(f"✓ 数据加載成功")
    except Exception as e:
        print(f"✗ 加載失败: {e}")
        return
    
    # 创建分析器
    analyzer = FormulaAnalyzer(df)
    
    # 重建公式对象
    print("\n[三] 重建公式对象...")
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
