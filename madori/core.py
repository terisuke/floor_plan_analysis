import numpy as np
from . import config, utils

def initialize_madori(rows, cols):
    """指定されたサイズの空の間取りを作成"""
    return np.full((rows, cols), ".")


### 新関数: 2つの部屋をセットで配置 (例: B & ut, e & Hなど)
def place_room_set(madori, room_codeA, room_codeB, rooms, max_tries=100):
    """
    room_codeAを配置 → その周囲(隣接可能な)空きマスに room_codeB を配置し、
    2つの部屋を必ず隣接させる。
    """
    rows, cols = madori.shape
    roomA = rooms[room_codeA]
    roomB = rooms[room_codeB]

    import random

    for _ in range(max_tries):
        # 先に codeA を適当配置
        if not place_room_single(madori, room_codeA, rooms, tries=1):  
            # 配置失敗 → 次のトライ
            continue

        # codeA の配置箇所を探索
        # (最後に配置された場所)を取りたいが、シンプルにはマップを走査
        placed_positions = []
        for r in range(rows):
            for c in range(cols):
                if madori[r, c] == room_codeA:
                    placed_positions.append((r, c))

        if not placed_positions:
            # Aが置けていない
            continue

        # 代表点(配置領域の左上)を取得する簡易的な方法:
        # ここでは placed_positions の最小r,cを代表とする
        min_r = min(pos[0] for pos in placed_positions)
        min_c = min(pos[1] for pos in placed_positions)

        # codeA のサイズを推定
        max_r = max(pos[0] for pos in placed_positions)
        max_c = max(pos[1] for pos in placed_positions)
        heightA = (max_r - min_r) + 1
        widthA  = (max_c - min_c) + 1

        # 次に roomB を codeA の周囲に配置
        # 周囲 = (min_r-1 ~ max_r+1, min_c-1 ~ max_c+1)
        # -> ここでは何度かランダム試行
        placedB = False
        for _2 in range(max_tries):
            b_height = np.random.randint(roomB["size"][0], roomB["size"][1]) \
                       if roomB["size"][0] < roomB["size"][1] else roomB["size"][0]
            b_width  = np.random.randint(roomB["size"][0], roomB["size"][1]) \
                       if roomB["size"][0] < roomB["size"][1] else roomB["size"][0]

            # 試しに、周囲の近い行列あたりでランダム配置
            rr = np.random.randint(max(min_r - b_height, 0), min(max_r + 2, rows - b_height + 1))
            cc = np.random.randint(max(min_c - b_width, 0),  min(max_c + 2, cols - b_width + 1))

            # 配置可能か?
            if utils.is_space_available(madori, rr, cc, b_height, b_width):
                # 隣接チェック(隣接先は room_codeA)
                if utils.check_adjacency(madori, rr, cc, b_height, b_width, [room_codeA]):
                    # 配置
                    madori[rr:rr+b_height, cc:cc+b_width] = room_codeB
                    placedB = True
                    break
        
        if placedB:
            return True
        else:
            # B が置けなかった → Aを取り消して再トライ
            madori[madori == room_codeA] = "."

    return False


def place_room_single(madori, room_code, rooms, tries=100):
    """
    単独部屋を配置する(既存 place_room の簡易ラップ)。
    tries 回試して成功すれば True, 失敗すれば False
    """
    for _ in range(tries):
        if place_room(madori, room_code, rooms):
            return True
    return False


### 新関数: 部屋を「中央付近」に配置(階段用)
def place_room_central(madori, room_code, rooms, tries=100):
    """
    指定した room_code を、マップ中央付近に配置しやすいように座標探索。
    """
    rcenter = madori.shape[0] // 2
    ccenter = madori.shape[1] // 2

    room = rooms[room_code]
    min_size, max_size = room["size"]
    import numpy as np

    # BFS的に「中心から外へ」広がる座標を試行する例(簡易)
    coords = []
    maxdist = max(rcenter, madori.shape[0]-rcenter, ccenter, madori.shape[1]-ccenter)
    for dist in range(maxdist+1):
        for dr in range(-dist, dist+1):
            for dc in range(-dist, dist+1):
                rr = rcenter + dr
                cc = ccenter + dc
                if 0 <= rr < madori.shape[0] and 0 <= cc < madori.shape[1]:
                    coords.append((rr, cc))

    # coords には中心→外側 順で並んでいる
    for _ in range(tries):
        for (r, c) in coords:
            if min_size == max_size:
                h = w = min_size
            else:
                h = np.random.randint(min_size, max_size)
                w = np.random.randint(min_size, max_size)
            if (r + h <= madori.shape[0]) and (c + w <= madori.shape[1]):
                if utils.is_space_available(madori, r, c, h, w):
                    if utils.check_adjacency(madori, r, c, h, w, room["adjacent_to"]):
                        madori[r:r+h, c:c+w] = room_code
                        return True
    return False


### (オプション) 外周から内側へ部屋を配置する例
def place_room_outside_in(madori, room_code, rooms, max_tries=100):
    """
    壁際(外周)から順に内側へ探して、最初に見つかった空きに配置
    """
    rows, cols = madori.shape
    room = rooms[room_code]
    min_size, max_size = room["size"]
    import numpy as np

    def get_layer_coords(layer):
        # layer-th outer ring
        top, left = layer, layer
        bottom = rows - 1 - layer
        right  = cols - 1 - layer
        coords = []
        if top>bottom or left>right:
            return coords
        # 上辺
        for c in range(left,right+1):
            coords.append((top,c))
        # 右辺
        for r in range(top+1,bottom):
            coords.append((r,right))
        # 下辺
        if bottom>top:
            for c in range(right,left-1,-1):
                coords.append((bottom,c))
        # 左辺
        if right>left:
            for r in range(bottom-1,top,-1):
                coords.append((r,left))
        return coords

    for _ in range(max_tries):
        # loop layers
        layer = 0
        placed = False
        while True:
            perimeter = get_layer_coords(layer)
            if not perimeter:
                break

            for (r,c) in perimeter:
                if min_size == max_size:
                    h = w = min_size
                else:
                    h = np.random.randint(min_size, max_size)
                    w = np.random.randint(min_size, max_size)
                if (r+h <= rows and c+w <= cols):
                    if utils.is_space_available(madori, r, c, h, w):
                        if utils.check_adjacency(madori, r, c, h, w, room["adjacent_to"]):
                            madori[r:r+h, c:c+w] = room_code
                            placed = True
                            break
            if placed:
                return True
            layer += 1
    return False


def place_room(madori, room_code, rooms):
    """
    従来の単体配置: 100回トライしてランダムに置く。
    (呼び出し側からplace_room_singleしても良い)
    """
    room = rooms[room_code]
    rows, cols = madori.shape
    min_size, max_size = room["size"]
    import numpy as np

    for _ in range(100):
        start_row = np.random.randint(0, rows - 1)
        start_col = np.random.randint(0, cols - 1)
        # サイズ
        if min_size > max_size:
            print(f"Invalid room size range for {room_code}: {room['size']}")
            return False
        if min_size == max_size:
            h = w = min_size
        else:
            h = np.random.randint(min_size, max_size)
            w = np.random.randint(min_size, max_size)

        if start_row + h <= rows and start_col + w <= cols:
            if utils.is_space_available(madori, start_row, start_col, h, w):
                if utils.check_adjacency(madori, start_row, start_col, h, w, room["adjacent_to"]):
                    madori[start_row:start_row + h, start_col:start_col + w] = room_code
                    return True
    return False


def try_place_ldk(madori, row, col, rooms):
    """
    既存のLDK配置(まとめて並べる)。
    """
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
            min_size, max_size = room["size"]

            if min_size > max_size:
                valid_placement = False
                break
            elif min_size == max_size:
                rh = rw = min_size
            else:
                rh = np.random.randint(min_size, max_size)
                rw = np.random.randint(min_size, max_size)

            if (current_row+rh>rows or current_col+rw>cols or
                not utils.is_space_available(madori, current_row, current_col, rh, rw)):
                valid_placement = False
                break

            placement.append((room_code, current_row, current_col, rh, rw))
            current_row += direction[0]*rh
            current_col += direction[1]*rw

        if valid_placement:
            for rc, rr, cc, hh, ww in placement:
                madori[rr:rr+hh, cc:cc+ww] = rc
            return True
    return False


def generate_madori_rule_based(rows, cols):
    """
    新しい優先度＆セット配置ロジック:
      1. (B, ut) セット配置
      2. t (トイレ) - B,ut があれば置く
      3. (e,H) セット配置
      4. s (階段) 中央配置
      5. LDK
      6. 残り(r,c) (簡易)
      7. place_corridor
    """
    madori = initialize_madori(rows, cols)
    rooms = config.ROOMS

    # 1. 風呂B & 脱衣所ut セット
    placed_BUt = place_room_set(madori, "B", "ut", rooms)

    # 2. トイレ(t) - もしB,utが置けたら優先して単独配置
    if placed_BUt:
        place_room_single(madori, "t", rooms, tries=30)

    # 3. 玄関(e)&ホール(H) セット
    place_room_set(madori, "e", "H", rooms)

    # 4. 階段(s) 中央
    place_room_central(madori, "s", rooms)

    # 5. LDK
    #   - ランダムな位置(row,col)を試す
    #     ここでは簡易的に 100回ぐらい試行
    tries = 100
    placed_LDK = False
    while tries>0 and not placed_LDK:
        row = np.random.randint(0, rows-1)
        col = np.random.randint(0, cols-1)
        placed_LDK = try_place_ldk(madori, row, col, rooms)
        tries -= 1

    # 6. 残りの部屋(r)とクローゼット(c)
    #    例: "r" と "c" を外周から置く
    place_room_outside_in(madori, "r", rooms, max_tries=50)
    place_room_outside_in(madori, "c", rooms, max_tries=50)

    # 7. 廊下配置
    place_corridor(madori, rooms)

    return madori


def place_corridor(madori, rooms):
    """
    部屋同士の未接続部分を廊下(co)で繋ぐ。
    A* を用いて空きマス('.')を移動可能とみなし、経路上を 'co' にする。
    """
    rows, cols = madori.shape
    unconnected_rooms = utils.find_unconnected_rooms(madori, rooms)

    for (room1_row, room1_col), (room2_row, room2_col) in unconnected_rooms:
        path = utils.find_path(madori, (room1_row, room1_col), (room2_row, room2_col))
        if path:
            for r, c in path:
                if madori[r, c] == ".":
                    madori[r, c] = "co"