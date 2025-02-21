import numpy as np
import random

# 部屋の定義 (辞書形式)
rooms = {
    "L": {"name": "リビング", "size": (3, 5), "adjacent_to": ["D", "K"]},
    "D": {"name": "ダイニング", "size": (2, 4), "adjacent_to": ["L", "K"]},
    "K": {"name": "キッチン", "size": (2, 3), "adjacent_to": ["L", "D"]},
    "B": {"name": "風呂", "size": (1, 2), "adjacent_to": ["ut"]},
    "ut": {"name": "脱衣所", "size": (1, 1), "adjacent_to": ["B"]},
    "t": {"name": "トイレ", "size": (1, 1), "adjacent_to": []},
    "H": {"name": "ホール", "size": (1, 3), "adjacent_to": []},
    "e": {"name": "玄関", "size": (1, 2), "adjacent_to": ["H"]},
    "c": {"name": "クローゼット", "size": (1, 2), "adjacent_to": []},
    "co": {"name": "廊下", "size": (1, 1), "adjacent_to": []},
    "r": {"name": "部屋", "size": (2, 4), "adjacent_to": []},
    "s": {"name": "階段", "size": (1, 2), "adjacent_to": ["H"]},
}

def initialize_madori(rows, cols):
    """指定されたサイズの空の間取りを作成"""
    return np.full((rows, cols), ".")

def generate_madori_rule_based(rows, cols, rooms):
    """
    ルールベースで間取りを生成する関数(全種類の部屋配置)
    """
    madori = initialize_madori(rows, cols)

    # 必須ルールに基づく主要な部屋（例：LDK）の配置
    placed = False
    while not placed:
        start_row = random.randint(0, rows - 1)
        start_col = random.randint(0, cols - 1)
        placed = try_place_ldk(madori, start_row, start_col, rooms)

    # 他の部屋を配置 (優先順位順)
    priority_rooms = ["e", "H", "s", "B", "ut", "t", "r", "c"]
    for room_code in priority_rooms:
        place_room(madori, room_code, rooms)

    # 廊下を配置
    place_corridor(madori, rooms)

    return madori

def try_place_ldk(madori, row, col, rooms):
    """
    指定された位置にLDKを配置試みる関数
    """
    rows, cols = madori.shape
    ldk_order = ["L", "D", "K"]
    directions = [(0, 1), (1, 0)]
    random.shuffle(directions)

    for direction in directions:
        current_row, current_col = row, col
        placement = []
        valid_placement = True

        for room_code in ldk_order:
            room = rooms[room_code]
            room_height = random.randint(room["size"][0], room["size"][1])
            room_width = random.randint(room["size"][0], room["size"][1])

            if (current_row + room_height > rows or
                current_col + room_width > cols or
                not is_space_available(madori, current_row, current_col, room_height, room_width)):
                valid_placement = False
                break

            placement.append((room_code, current_row, current_col, room_height, room_width))
            current_row += direction[0] * room_height
            current_col += direction[1] * room_width

        if valid_placement:
            for room_code, r, c, h, w in placement:
                madori[r:r+h, c:c+w] = room_code
            return True

    return False

def place_room(madori, room_code, rooms):
    """指定された種類の部屋を1つ配置する関数"""
    room = rooms[room_code]
    rows, cols = madori.shape

    for _ in range(100):
        start_row = random.randint(0, rows - 1)
        start_col = random.randint(0, cols - 1)
        room_height = random.randint(room["size"][0], room["size"][1])
        room_width = random.randint(room["size"][0], room["size"][1])

        if is_space_available(madori, start_row, start_col, room_height, room_width):
            if check_adjacency(madori, start_row, start_col, room_height, room_width, room["adjacent_to"]):
                madori[start_row:start_row + room_height, start_col:start_col + room_width] = room_code
                return

def is_space_available(madori, row, col, height, width):
    """指定された範囲がすべて空いているかチェック"""
    return np.all(madori[row:row+height, col:col+width] == ".")

def check_adjacency(madori, row, col, height, width, adjacent_to):
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

def place_corridor(madori, rooms):
    """
    廊下を配置する関数
    """
    rows, cols = madori.shape
    # 廊下でつなぐべき部屋のペアを見つける
    unconnected_rooms = find_unconnected_rooms(madori, rooms)
    #print(f"unconnected_rooms: {unconnected_rooms}")

    for (room1_row, room1_col), (room2_row, room2_col) in unconnected_rooms:
        # 部屋1と部屋2の間の経路を探索 (A*アルゴリズムなど)
        path = find_path(madori, (room1_row,room1_col), (room2_row,room2_col))
        #print(f"path{path}")

        if path:
            # 経路に沿って廊下を配置
            for row, col in path:
                if madori[row, col] == ".":  # 他の部屋と重ならないように
                    madori[row, col] = "co"

def find_unconnected_rooms(madori, rooms):
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
    # BFS(幅優先探索)で接続性をチェック
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
                #print(f"      Adding neighbor: ({new_row, {new_col}) to queue") # 追加
                queue.append((new_row, new_col))
    #print(f"    Could not reach {pos2}, returning False") # 追加
    return False

def find_path(madori, start, goal):
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
    """発見した経路を再構築する"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # リストを逆順にして返す