import numpy as np
from . import config, utils  # 同じパッケージ内のモジュールをインポート


def initialize_madori(rows, cols):
    """指定されたサイズの空の間取りを作成"""
    return np.full((rows, cols), ".")


def generate_madori_rule_based(rows, cols):
    """
    ルールベースで間取りを生成する関数(全種類の部屋配置)
    """
    madori = initialize_madori(rows, cols)
    rooms = config.ROOMS

    # 必須ルールに基づく主要な部屋（例：LDK）の配置
    placed = False
    while not placed:
        start_row = np.random.randint(0, rows - 1)
        start_col = np.random.randint(0, cols - 1)
        placed = try_place_ldk(madori, start_row, start_col, rooms)

    # 他の部屋を配置 (優先順位順)
    priority_rooms = ["e", "H", "s", "B", "ut", "t", "r", "c"]
    for room_code in priority_rooms:
        place_room(madori, room_code, rooms)

    # 廊下を配置
    place_corridor(madori, rooms)

    return madori


def try_place_ldk(madori, row, col, rooms):
    # ... (既存のコード, 必要に応じてutilsの関数を利用) ...
    rows, cols = madori.shape
    ldk_order = ["L", "D", "K"]
    directions = [(0, 1), (1, 0)]
    np.random.shuffle(directions)

    for direction in directions:
        current_row, current_col = row, col
        placement = []
        valid_placement = True

        for room_code in ldk_order:
            room = rooms[room_code]
            room_height = np.random.randint(room["size"][0], room["size"][1])
            room_width = np.random.randint(room["size"][0], room["size"][1])

            if (current_row + room_height > rows or
                current_col + room_width > cols or
                not utils.is_space_available(madori, current_row, current_col, room_height, room_width)):
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
    # ... (既存のコード, 必要に応じてutilsの関数を利用) ...
    room = rooms[room_code]
    rows, cols = madori.shape

    for _ in range(100):
        start_row = np.random.randint(0, rows - 1)
        start_col = np.random.randint(0, cols - 1)

        # room["size"]の範囲が不正でないか確認
        if room["size"][0] >= room["size"][1]:
            print(f"Invalid room size range for {room_code}: {room['size']}")
            return

        room_height = np.random.randint(room["size"][0], room["size"][1])
        room_width = np.random.randint(room["size"][0], room["size"][1])

        if utils.is_space_available(madori, start_row, start_col, room_height, room_width):
            if utils.check_adjacency(madori, start_row, start_col, room_height, room_width, room["adjacent_to"]):
                madori[start_row:start_row + room_height, start_col:start_col + room_width] = room_code
                return

def place_corridor(madori, rooms):
  # ... (既存のコード, 必要に応じてutilsの関数を利用) ...
  # 接続判定ロジックの変更 (動的な評価)。
  # A* アルゴリズムの改良 (コスト関数)。
  # 廊下の自動生成ロジックの追加。
    rows, cols = madori.shape
    # 廊下でつなぐべき部屋のペアを見つける
    unconnected_rooms = utils.find_unconnected_rooms(madori, rooms)
    #print(f"unconnected_rooms: {unconnected_rooms}")

    for (room1_row, room1_col), (room2_row, room2_col) in unconnected_rooms:
        # 部屋1と部屋2の間の経路を探索 (A*アルゴリズムなど)
        path = utils.find_path(madori, (room1_row,room1_col), (room2_row,room2_col))
        #print(f"path{path}")

        if path:
            # 経路に沿って廊下を配置
            for row, col in path:
                if madori[row, col] == ".":  # 他の部屋と重ならないように
                    madori[row, col] = "co"