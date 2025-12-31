"""
掃描整個專案結構並輸出詳細信息
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any


class ProjectScanner:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.structure = {}
        self.file_info = []
        self.import_info = {}
    
    def scan(self) -> Dict[str, Any]:
        """掃描整個專案"""
        print(f"\n掃描專案: {self.root_path.absolute()}")
        print("="*80)
        
        # 掃描目錄結構
        self._scan_directory(self.root_path)
        
        # 掃描 Python 文件的導入
        self._scan_python_imports()
        
        # 生成報告
        self._generate_report()
        
        return {
            'structure': self.structure,
            'files': self.file_info,
            'imports': self.import_info
        }
    
    def _scan_directory(self, path: Path, prefix: str = ""):
        """遞迴掃描目錄"""
        try:
            items = sorted(path.iterdir())
        except PermissionError:
            return
        
        # 排序：目錄優先
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # 跳過的目錄
        skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache', '.idea'}
        
        dirs = [d for d in dirs if d.name not in skip_dirs]
        
        for item in dirs + files:
            if item.name.startswith('.'):
                continue
            
            is_last = item == (dirs + files)[-1]
            current_prefix = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")
            
            if item.is_dir():
                print(f"{prefix}{current_prefix}📁 {item.name}/")
                self._scan_directory(item, next_prefix)
            else:
                size = item.stat().st_size
                size_str = self._format_size(size)
                print(f"{prefix}{current_prefix}📄 {item.name} ({size_str})")
                
                self.file_info.append({
                    'name': item.name,
                    'path': str(item.relative_to(self.root_path)),
                    'size': size,
                    'type': item.suffix
                })
    
    def _scan_python_imports(self):
        """掃描 Python 文件的導入信息"""
        print("\n" + "="*80)
        print("Python 文件分析")
        print("="*80)
        
        python_files = list(self.root_path.rglob('*.py'))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                imports = self._extract_imports(content)
                classes = self._extract_classes(content)
                functions = self._extract_functions(content)
                
                rel_path = py_file.relative_to(self.root_path)
                line_count = len(content.split('\n'))
                self.import_info[str(rel_path)] = {
                    'imports': imports,
                    'classes': classes,
                    'functions': functions,
                    'lines': line_count
                }
                
                if imports or classes or functions:
                    print(f"\n📄 {rel_path}")
                    print(f"   行數: {line_count}")
                    if imports:
                        imports_str = ', '.join(imports[:5])
                        if len(imports) > 5:
                            imports_str += '...'
                        print(f"   導入: {imports_str}")
                    if classes:
                        print(f"   類: {', '.join(classes)}")
                    if functions:
                        functions_str = ', '.join(functions[:3])
                        if len(functions) > 3:
                            functions_str += '...'
                        print(f"   函數: {functions_str}")
            except Exception as e:
                print(f"   ⚠️ 掃描失敗: {e}")
    
    def _extract_imports(self, content: str) -> List[str]:
        """提取導入語句"""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line[:60])  # 限制長度
        return imports
    
    def _extract_classes(self, content: str) -> List[str]:
        """提取類定義"""
        classes = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('class '):
                class_name = line.split('class ')[1].split('(')[0].split(':')[0]
                classes.append(class_name)
        return classes
    
    def _extract_functions(self, content: str) -> List[str]:
        """提取函數定義"""
        functions = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('def ') and not line.startswith('def __'):
                func_name = line.split('def ')[1].split('(')[0]
                functions.append(func_name)
        return functions
    
    def _generate_report(self):
        """生成詳細報告"""
        print("\n" + "="*80)
        print("文件統計")
        print("="*80)
        
        # 按類型統計
        type_stats = {}
        for file_info in self.file_info:
            file_type = file_info['type'] or 'no_extension'
            if file_type not in type_stats:
                type_stats[file_type] = {'count': 0, 'total_size': 0}
            type_stats[file_type]['count'] += 1
            type_stats[file_type]['total_size'] += file_info['size']
        
        for file_type, stats in sorted(type_stats.items()):
            total_size = self._format_size(stats['total_size'])
            print(f"{file_type:15} {stats['count']:5} 個文件  {total_size:>10}")
        
        print(f"\n總計: {len(self.file_info)} 個文件")
        
        # Python 文件統計
        python_count = sum(1 for f in self.file_info if f['type'] == '.py')
        print(f"Python 文件: {python_count} 個")
    
    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    def save_json(self, filename: str = 'project_structure.json'):
        """保存為 JSON"""
        output = {
            'root': str(self.root_path.absolute()),
            'structure': self.structure,
            'files': self.file_info,
            'imports': self.import_info,
            'summary': {
                'total_files': len(self.file_info),
                'python_files': sum(1 for f in self.file_info if f['type'] == '.py')
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 項目結構已保存至: {filename}")


def main():
    scanner = ProjectScanner(".")
    scanner.scan()
    scanner.save_json('project_structure.json')
    
    # 額外信息
    print("\n" + "="*80)
    print("🔍 關鍵發現")
    print("="*80)
    print("\n請查看 project_structure.json 了解完整的導入和類結構")
    print("\n常見的導入問題:")
    print("  1. 如果看到 'data/__init__.py'，說明 data 是一個包")
    print("  2. 檢查 data/__init__.py 中導出了什麼類")
    print("  3. 如果需要導入 DataHandler，應該找到它定義的位置")


if __name__ == '__main__':
    main()
