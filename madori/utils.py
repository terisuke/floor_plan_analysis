import numpy as np

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


def find_unconnected_rooms(madori, rooms):
    """
    廊下でつなぐべき部屋のペアを見つける関数
    """
    rows, cols = madori.shape
    room_positions = {}
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

    room_codes = list(room_positions.keys())
    for i in range(len(room_codes)):
        for j in range(i + 1, len(room_codes)):
            room1_code = room_codes[i]
            room2_code = room_codes[j]
            print(f"Checking connection between {room1_code} and {room2_code}")
            connected = False
            for pos1 in room_positions[room1_code]:
                for pos2 in room_positions[room2_code]:
                    print(f"  Checking positions: {pos1} and {pos2}")
                    if is_connected(madori, pos1, pos2):
                        connected = True
                        print(f"  {room1_code} and {room2_code} are connected")
                        break
                if connected:
                    break

            if not connected:
                unconnected_rooms.append(
                    (room_positions[room1_code][0], room_positions[room2_code][0])
                )
                print(f"  {room1_code} and {room2_code} are NOT connected")

    return unconnected_rooms


def is_connected(madori, pos1, pos2):
    """
    2つの部屋(=pos1, pos2)が、空きマス('.')または廊下('co')を経由して
    接続されているかを判定する関数。

    ここでは pos1/pos2 は“部屋マス”そのものなので、
    まず部屋の周囲にある '.' or 'co' マスをスタート/ゴールとしてBFSを試みる。
    """
    rows, cols = madori.shape

    # 部屋の周囲にある空きマス/廊下マスを取得するヘルパー
    def get_adjacent_walkables(r, c):
        result = []
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if madori[nr, nc] in ('.', 'co'):
                    result.append((nr, nc))
        return result

    start_positions = get_adjacent_walkables(pos1[0], pos1[1])
    goal_positions  = set(get_adjacent_walkables(pos2[0], pos2[1]))

    # もし周囲に空きマス/廊下がなければ接続不可
    if not start_positions or not goal_positions:
        return False

    visited = set(start_positions)
    queue = list(start_positions)

    while queue:
        cur = queue.pop(0)
        # goal_positions のいずれかに到達すれば接続あり
        if cur in goal_positions:
            return True

        r, c = cur
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in visited and madori[nr, nc] in ('.','co'):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return False


def find_path(madori, start, goal):
    """
    A*アルゴリズムで2点間の最短経路を求める。
    空きマス('.')を移動可能とみなし、start～goalを結ぶ経路を探索する。
    """
    rows, cols = madori.shape

    def heuristic(a, b):
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

    return None


def get_neighbors(madori, pos):
    """
    指定された位置の隣接マス(移動可能なマス)のリストを返す
    ※ここでは空きマス('.')のみ移動可能。'co'も通れた方が良いなら調整可
    """
    rows, cols = madori.shape
    row, col = pos
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = row+dr, col+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            # 現在は '.' のみ通行可能にしている。廊下(co)も通すなら条件を ('.','co') に。
            if madori[nr, nc] == ".":
                neighbors.append((nr, nc))
    return neighbors


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]