import os
import csv
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any, Union

def extract_elements(value: str) -> List[Tuple[str, float]]:
    """文字列から要素（例：r1, ut など）とその数値を抽出する
    
    例えば:
    - "100r1" -> [("r", 100.0)]
    - "50ut" -> [("ut", 50.0)]
    - "25b30c" -> [("b", 25.0), ("c", 30.0)]
    - "c,c1,c2" -> [("c", 1.0), ("c", 1.0), ("c", 1.0)]  # c1とc2はcとしてカウント
    
    Args:
        value: 分析する文字列
        
    Returns:
        要素名と数値のタプルのリスト
    """
    # まず、カンマ区切りの場合を処理
    if ',' in value:
        results = []
        elements = value.split(',')
        for element in elements:
            element = element.strip()
            if not element:
                continue
                
            # 要素をそのまま追加（r1, r2などを個別に扱う）
            results.append((element, 1.0))  # 要素をそのまま追加
        return results
    
    # カンマがない場合、数値+要素の形式を処理
    results = []
    
    # パターン1: 数値+文字（例：100r1）
    pattern1 = r'(\d+)([a-zA-Z]+\d*)'
    matches1 = re.findall(pattern1, value)
    
    # パターン2: 文字+数値（例：r1 100）
    pattern2 = r'([a-zA-Z]+\d*)\s+(\d+)'
    matches2 = re.findall(pattern2, value)
    
    # パターン1の結果を処理
    for num_str, element in matches1:
        try:
            num = float(num_str)
            # 要素をそのまま追加（r1, r2などを個別に扱う）
            results.append((element, num))  # 要素をそのまま追加
        except ValueError:
            pass
    
    # パターン2の結果を処理
    for element, num_str in matches2:
        try:
            num = float(num_str)
            # 要素をそのまま追加（r1, r2などを個別に扱う）
            results.append((element, num))  # 要素をそのまま追加
        except ValueError:
            pass
    
    # もし数値が含まれる要素が見つからなかった場合、
    # 値自体が要素名である可能性があるので、そのまま返す
    if not results and re.match(r'^[a-zA-Z]+\d*$', value):
        # 要素をそのまま追加
        results.append((value, 1.0))  # 存在するだけなので値は1とする
    
    return results

def analyze_csv_files(directory_path: str) -> Dict[str, Dict[str, Any]]:
    """指定ディレクトリ内のすべてのCSVファイルを分析し、要素の平均値を計算する
    
    Args:
        directory_path: CSVファイルが格納されているディレクトリのパス
        
    Returns:
        各要素の統計情報を含む辞書
    """
    # 結果を格納するデータ構造
    elements_data = defaultdict(lambda: {"total": 0.0, "files": set(), "values": []})
    
    # 処理したファイルの数
    processed_files = 0
    
    # ディレクトリ内のCSVファイルを処理
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    
                    # ヘッダー行をスキップ
                    headers = next(csv_reader, None)
                    
                    # ファイル内の各行を処理
                    for row in csv_reader:
                        for cell in row:
                            # 空のセルはスキップ
                            if not cell or cell.strip() == '':
                                continue
                            
                            # セルから要素と数値を抽出
                            extracted = extract_elements(cell)
                            
                            for element, value in extracted:
                                elements_data[element]["total"] += value
                                elements_data[element]["files"].add(filename)
                                elements_data[element]["values"].append(value)
                
                processed_files += 1
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # 平均値と統計情報を計算
    results = {}
    for element, data in elements_data.items():
        file_count = len(data["files"])
        
        # この要素がいくつのファイルに出現したか
        results[element] = {
            "total": data["total"],
            "file_count": file_count,
            "files": list(data["files"]),
            "count": len(data["values"]),
            "average_per_file": data["total"] / file_count if file_count > 0 else 0,
            "average_per_occurrence": data["total"] / len(data["values"]) if data["values"] else 0
        }
    
    # 処理結果の要約
    summary = {
        "total_files_processed": processed_files,
        "total_elements_found": len(results),
        "elements": results
    }
    
    return summary

def main():
    """メイン実行関数"""
    # 現在のディレクトリまたは指定のディレクトリからCSVファイルを分析
    directory = input("CSVファイルが格納されているディレクトリパスを入力してください（Enterで現在のディレクトリ）: ")
    
    if not directory:
        directory = '.'  # 現在のディレクトリ
    
    if not os.path.exists(directory):
        print(f"エラー: ディレクトリ '{directory}' が見つかりません。")
        return
    
    print(f"ディレクトリ '{directory}' のCSVファイルを分析中...")
    results = analyze_csv_files(directory)
    
    print("\n分析結果:")
    print(f"処理したファイル数: {results['total_files_processed']}")
    print(f"見つかった要素の総数: {results['total_elements_found']}")
    
    print("\n各要素の平均値:")
    for element, stats in sorted(results['elements'].items()):
        print(f"{element}:")
        print(f"  - 合計値: {stats['total']}")
        print(f"  - 出現したファイル数: {stats['file_count']}")
        print(f"  - ファイルあたりの平均値: {stats['average_per_file']:.2f}")
        print(f"  - 出現あたりの平均値: {stats['average_per_occurrence']:.2f}")
        print(f"  - 出現回数: {stats['count']}")
        print(f"  - 出現したファイル: {', '.join(stats['files'][:10])}{'...' if len(stats['files']) > 10 else ''}")
        print()

if __name__ == "__main__":
    main()