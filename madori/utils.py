import numpy as np

def is_space_available(madori, row, col, height, width):
    """指定された範囲がすべて空いているかチェック"""
    return np.all(madori[row:row+height, col:col+width] == ".")


def check_adjacency(madori, row, col, height, width, adjacent_to):
    # ... (既存のコード) ...
    """指定された範囲の周囲に、隣接すべき部屋があるかチェック"""
    if not adjacent_to:
        return True

    rows, cols = madori.shape
    for r in range(row - 1, row + height + 1):
        for c in range(col - 1, col + width + 1):
            if (r < 0 or r >= rows or c < 0 or c >= cols or
                (row <= r < row + height and col <= c < col + width)):
                continue

            if madori[r, c] in adjacent_to:
                return True

    return False


def find_unconnected_rooms(madori, rooms):
    # ... (修正済みのコード, is_connected関数を適切に利用) ...
    """
        廊下でつなぐべき部屋のペアを見つける関数
        """
    rows, cols = madori.shape
    room_positions = {}  # 各部屋の位置を記録 {room_code: [(row, col), ...]}
    unconnected_rooms = []

    # 各部屋の位置を記録
    for r in range(rows):
        for c in range(cols):
            room_code = madori[r, c]
            if room_code != "." and room_code != "co":  # 空きマスと廊下は除く
                if room_code not in room_positions:
                    room_positions[room_code] = []
                room_positions[room_code].append((r, c))
    print(f"room_positions: {room_positions}") # デバッグ出力

    # 部屋のペアを総当たりでチェック
    room_codes = list(room_positions.keys())
    for i in range(len(room_codes)):
        for j in range(i + 1, len(room_codes)):
            room1_code = room_codes[i]
            room2_code = room_codes[j]
            print(f"Checking connection between {room1_code} and {room2_code}") # デバッグ出力
            # 部屋1と部屋2が接続されているかチェック
            connected = False
            for pos1 in room_positions[room1_code]:
                for pos2 in room_positions[room2_code]:
                    print(f"  Checking positions: {pos1} and {pos2}") # デバッグ出力
                    if is_connected(madori, pos1, pos2, room2_code):#第3引数にroom2_codeを追加
                        connected = True
                        print(f"  {room1_code} and {room2_code} are connected") # デバッグ出力
                        break  # 1つでも接続されていればOK
                if connected:
                    break

            if not connected:
                # 接続されていない場合は、部屋の代表点(ここでは、各部屋の座標リストの最初の点)をペアに追加
                unconnected_rooms.append(
                    (room_positions[room1_code][0], room_positions[room2_code][0])
                )  # 接続されていないペア
                print(f"  {room1_code} and {room2_code} are NOT connected") # デバッグ出力
    return unconnected_rooms


def is_connected(madori, pos1, pos2, room2_code):
    """
    2つの部屋が接続されているか(廊下、または他の部屋経由で)を判定する関数
    """
    # ... (修正済みのコード) ...
    rows, cols = madori.shape
    visited = set()
    queue = [pos1]
    #print(f"    Starting is_connected check from {pos1} to {pos2}") # 追加

    while queue:
        current_pos = queue.pop(0)
        #print(f"    Current position: {current_pos}, visited: {visited}, queue: {queue}") # 追加
        if current_pos == pos2:
            #print(f"    Reached {pos2}, returning True") # 追加
            return True

        visited.add(current_pos)

        row, col = current_pos
        # 上下左右を探索
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and
                (new_row, new_col) not in visited
                and madori[new_row, new_col] != "."):  # 修正: 空きマス以外なら移動可能
                #print(f"      Adding neighbor: ({new_row}, {new_col}) to queue") # 追加
                queue.append((new_row, new_col))
    #print(f"    Could not reach {pos2}, returning False") # 追加
    return False


def find_path(madori, start, goal):
    # ... (既存のコード) ...
    """A*アルゴリズムで2点間の最短経路を求める"""
    rows, cols = madori.shape

    def heuristic(a, b):
        """マンハッタン距離によるヒューリスティック関数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = {start}
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    while open_set:
        current = min(open_set, key=lambda x: fscore.get(x, float('inf')))

        if current == goal:
            return reconstruct_path(came_from, current)

        open_set.remove(current)

        for neighbor in get_neighbors(madori, current):
            tentative_gscore = gscore.get(current, float('inf')) + 1

            if tentative_gscore < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    return None  # 経路が見つからない場合はNoneを返す

def get_neighbors(madori, pos):
    # ... (既存のコード) ...
    """指定された位置の隣接マス(移動可能なマス)のリストを返す"""
    rows, cols = madori.shape
    row, col = pos
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 上下左右
        new_row, new_col = row + dr, col + dc
        if (0 <= new_row < rows and 0 <= new_col < cols and
                madori[new_row, new_col] == "."):  # 移動先が範囲内かつ空きマス
            neighbors.append((new_row, new_col))
    return neighbors

def reconstruct_path(came_from, current):
    # ... (既存のコード) ...
    """発見した経路を再構築する"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # リストを逆順にして返す