import os
import glob
import pandas as pd
from collections import defaultdict, deque

# CSV上の部屋コード → config.py上のキー へのマッピング例
CSV_TO_CONFIG_MAP = {
    "l": "L",    # リビング
    "d": "D",    # ダイニング (例：'d'をダイニングと解釈)
    "k": "K",    # キッチン
    "r": "r",    # 部屋
    "t": "t",    # トイレ
    "b": "B",    # 風呂
    "c": "c",    # クローゼット
    "s": "s",    # 階段
    "e": "e",    # 玄関
    "h": "H",    # ホール (小文字をHに変換)
    "co": "co",  # 廊下
    "ut": "ut",  # 脱衣所
    # "ldk": → 必要に応じて L,D,K に分割する等
}

def analyze_1f_csv(data_1f_dir="data/1F"):
    csv_files = glob.glob(os.path.join(data_1f_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_1f_dir}")
        return

    code_blocks = defaultdict(list)

    for filepath in csv_files:
        print(f"\nAnalyzing: {filepath}")
        df = pd.read_csv(filepath, header=None).fillna("")
        grid = df.values.tolist()
        rows = len(grid)
        cols = len(grid[0]) if rows else 0

        # 小文字→大文字変換＆空マス整形
        normalized_grid = []
        for r in range(rows):
            row_list = []
            for c in range(cols):
                val = str(grid[r][c]).strip()
                if val == "" or val == ".":
                    row_list.append(".")
                else:
                    lower_val = val.lower()
                    if lower_val in CSV_TO_CONFIG_MAP:
                        row_list.append(CSV_TO_CONFIG_MAP[lower_val])
                    else:
                        row_list.append(".")
            normalized_grid.append(row_list)

        visited = [[False]*cols for _ in range(rows)]

        def neighbors(rr, cc):
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = rr+dr, cc+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

        # BFSで同じコードの塊(ブロック)を探索
        for rr in range(rows):
            for cc in range(cols):
                if normalized_grid[rr][cc] != "." and not visited[rr][cc]:
                    code = normalized_grid[rr][cc]
                    queue = deque([(rr, cc)])
                    visited[rr][cc] = True

                    min_r, max_r = rr, rr
                    min_c, max_c = cc, cc

                    while queue:
                        r0, c0 = queue.popleft()
                        if r0 < min_r: min_r = r0
                        if r0 > max_r: max_r = r0
                        if c0 < min_c: min_c = c0
                        if c0 > max_c: max_c = c0

                        for nr, nc in neighbors(r0, c0):
                            if not visited[nr][nc] and normalized_grid[nr][nc] == code:
                                visited[nr][nc] = True
                                queue.append((nr, nc))

                    block_h = (max_r - min_r) + 1
                    block_w = (max_c - min_c) + 1
                    code_blocks[code].append((block_h, block_w))

    print("\n=== Analysis Result (Min-Max bounding box per code) ===")
    for code, hw_list in code_blocks.items():
        heights = [h for (h,_) in hw_list]
        widths  = [w for (_,w) in hw_list]
        min_h, max_h = min(heights), max(heights)
        min_w, max_w = min(widths), max(widths)
        min_side = min(min_h, min_w)
        max_side = max(max_h, max_w)
        print(f"Code: {code}")
        print(f"  #Blocks: {len(hw_list)}")
        print(f"  bounding-box heights = {heights}")
        print(f"  bounding-box widths  = {widths}")
        print(f"  Proposed size range: ({min_side}, {max_side})\n")

if __name__ == "__main__":
    analyze_1f_csv()