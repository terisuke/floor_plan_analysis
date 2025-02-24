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

    for _ in range(max_tries):
        # 先に codeA を適当配置
        if not place_room_single(madori, room_codeA, rooms, tries=1):
            continue

        # codeA の配置された座標を調べる
        placed_positions = []
        for r in range(rows):
            for c in range(cols):
                if madori[r,c] == room_codeA:
                    placed_positions.append((r,c))

        if not placed_positions:
            continue

        min_r = min(pos[0] for pos in placed_positions)
        min_c = min(pos[1] for pos in placed_positions)
        max_r = max(pos[0] for pos in placed_positions)
        max_c = max(pos[1] for pos in placed_positions)
        heightA = (max_r - min_r) + 1
        widthA  = (max_c - min_c) + 1

        import numpy as np

        placedB = False
        for _2 in range(max_tries):
            # roomB のサイズ
            if roomB["size"][0] < roomB["size"][1]:
                b_height = np.random.randint(roomB["size"][0], roomB["size"][1])
                b_width  = np.random.randint(roomB["size"][0], roomB["size"][1])
            else:
                b_height = roomB["size"][0]
                b_width  = roomB["size"][0]

            rr = np.random.randint(max(min_r - b_height, 0), 
                                   min(max_r + 2, rows - b_height + 1))
            cc = np.random.randint(max(min_c - b_width, 0),  
                                   min(max_c + 2, cols - b_width + 1))

            if utils.is_space_available(madori, rr, cc, b_height, b_width):
                if utils.check_adjacency(madori, rr, cc, b_height, b_width, [room_codeA]):
                    madori[rr:rr+b_height, cc:cc+b_width] = room_codeB
                    placedB = True
                    break

        if placedB:
            return True
        else:
            # B配置失敗 → Aをクリアして再トライ
            madori[madori == room_codeA] = "."

    return False


def place_room_single(madori, room_code, rooms, tries=100):
    """単独部屋を配置する(既存 place_room の簡易ラップ)。"""
    for _ in range(tries):
        if place_room(madori, room_code, rooms):
            return True
    return False


def place_room_central(madori, room_code, rooms, tries=100):
    """指定した room_code を、マップ中央付近に配置しやすいように座標探索。"""
    rcenter = madori.shape[0] // 2
    ccenter = madori.shape[1] // 2
    room = rooms[room_code]
    min_size, max_size = room["size"]
    import numpy as np

    coords = []
    maxdist = max(rcenter, madori.shape[0]-rcenter, ccenter, madori.shape[1]-ccenter)
    for dist in range(maxdist+1):
        for dr in range(-dist, dist+1):
            for dc in range(-dist, dist+1):
                rr, cc = rcenter+dr, ccenter+dc
                if 0<=rr<madori.shape[0] and 0<=cc<madori.shape[1]:
                    coords.append((rr, cc))

    for _ in range(tries):
        for (r,c) in coords:
            if min_size == max_size:
                h = w = min_size
            else:
                h = np.random.randint(min_size, max_size)
                w = np.random.randint(min_size, max_size)

            if (r+h<=madori.shape[0]) and (c+w<=madori.shape[1]):
                if utils.is_space_available(madori, r,c, h,w):
                    if utils.check_adjacency(madori, r,c, h,w, room["adjacent_to"]):
                        madori[r:r+h, c:c+w] = room_code
                        return True
    return False


def place_room_outside_in(madori, room_code, rooms, max_tries=100):
    """壁際(外周)から順に内側へ探して、最初に見つかった空きに配置"""
    rows, cols = madori.shape
    room = rooms[room_code]
    min_size, max_size = room["size"]
    import numpy as np

    def get_layer_coords(layer):
        top, left = layer, layer
        bottom = rows-1-layer
        right  = cols-1-layer
        coords=[]
        if top>bottom or left>right:
            return coords
        # 上辺
        for c in range(left, right+1):
            coords.append((top,c))
        # 右辺
        for r in range(top+1, bottom):
            coords.append((r,right))
        # 下辺
        if bottom>top:
            for c in range(right, left-1, -1):
                coords.append((bottom,c))
        # 左辺
        if right>left:
            for r in range(bottom-1, top, -1):
                coords.append((r,left))
        return coords

    for _ in range(max_tries):
        layer=0
        placed=False
        while True:
            perimeter = get_layer_coords(layer)
            if not perimeter:
                break

            for (r,c) in perimeter:
                if min_size>max_size:
                    print(f"Invalid room size range for {room_code}: {room['size']}")
                    return False
                if min_size==max_size:
                    h=w=min_size
                else:
                    h=np.random.randint(min_size, max_size)
                    w=np.random.randint(min_size, max_size)
                if (r+h<=rows) and (c+w<=cols):
                    if utils.is_space_available(madori, r,c, h,w):
                        if utils.check_adjacency(madori, r,c,h,w, room["adjacent_to"]):
                            madori[r:r+h, c:c+w] = room_code
                            placed=True
                            break
            if placed:
                return True
            layer+=1
    return False


def place_room(madori, room_code, rooms):
    """従来の単体配置: 100回トライしてランダムに置く。"""
    room = rooms[room_code]
    rows, cols = madori.shape
    min_size, max_size = room["size"]
    import numpy as np

    for _ in range(100):
        start_row = np.random.randint(0, rows - 1)
        start_col = np.random.randint(0, cols - 1)

        if min_size>max_size:
            print(f"Invalid room size range for {room_code}: {room['size']}")
            return False

        if min_size==max_size:
            h=w=min_size
        else:
            h=np.random.randint(min_size, max_size)
            w=np.random.randint(min_size, max_size)

        if start_row+h<=rows and start_col+w<=cols:
            if utils.is_space_available(madori, start_row, start_col, h, w):
                if utils.check_adjacency(madori, start_row, start_col, h, w, room["adjacent_to"]):
                    madori[start_row:start_row+h, start_col:start_col+w] = room_code
                    return True
    return False


def try_place_ldk(madori, row, col, rooms):
    """既存のLDK配置(まとめて並べる)。"""
    rows, cols = madori.shape
    ldk_order=["L","D","K"]
    directions=[(0,1),(1,0)]
    np.random.shuffle(directions)

    for direction in directions:
        current_row, current_col = row, col
        placement=[]
        valid_placement=True

        for room_code in ldk_order:
            room=rooms[room_code]
            min_size, max_size=room["size"]

            if min_size>max_size:
                valid_placement=False
                break
            elif min_size==max_size:
                rh=rw=min_size
            else:
                rh=np.random.randint(min_size, max_size)
                rw=np.random.randint(min_size, max_size)

            if (current_row+rh>rows or current_col+rw>cols or
                not utils.is_space_available(madori, current_row, current_col, rh, rw)):
                valid_placement=False
                break

            placement.append((room_code, current_row, current_col, rh, rw))
            current_row+=direction[0]*rh
            current_col+=direction[1]*rw

        if valid_placement:
            for rc,rr,cc,hh,ww in placement:
                madori[rr:rr+hh, cc:cc+ww] = rc
            return True
    return False


def place_corridor(madori, rooms):
    """部屋同士の未接続部分を廊下(co)で繋ぐ。"""
    unconnected_rooms = utils.find_unconnected_rooms(madori, rooms)
    for (r1,c1),(r2,c2) in unconnected_rooms:
        path=utils.find_path(madori,(r1,c1),(r2,c2))
        if path:
            for (r,c) in path:
                if madori[r,c]==".":
                    madori[r,c]="co"


#####################
# 新：必須部屋配置関数
#####################
def place_mandatory_rooms(madori, rooms):
    """
    必須部屋(B,ut,e,H,s,L,D,K)を配置する。
    順番:
      (B, ut)セット → t(オプション)
      (e, H)セット
      s(中央)
      LDK
    成功したかどうかを True/False で返す。
    """
    # 1. 風呂B & 脱衣所ut セット
    ok_but = place_room_set(madori, "B", "ut", rooms)
    # 2. t (もしB,ut成功なら)
    if ok_but:
        place_room_single(madori, "t", rooms, tries=30)

    # 3. 玄関(e) & ホール(H)
    ok_eh = place_room_set(madori, "e","H", rooms)

    # 4. 階段(s) 中央
    ok_s = place_room_central(madori, "s", rooms)

    # 5. LDK
    tries=100
    ok_ldk=False
    while tries>0 and not ok_ldk:
        row = np.random.randint(0, madori.shape[0]-1)
        col = np.random.randint(0, madori.shape[1]-1)
        if try_place_ldk(madori, row, col, rooms):
            ok_ldk=True
        tries-=1

    # 全部必須のうち, (B,ut), (e,H), s, L,D,K が配置成功したか判定
    # ※ Kの配置成功は LDK成功がベース
    #  - ここでは厳密に "B も ut もいる?" "L も D も K もいる?" などを確認

    # 簡易チェック: madori内に B, ut, e, H, s, L, D, K が１つ以上あるか
    needed_codes=["B","ut","e","H","s","L","D","K"]
    for code in needed_codes:
        if code not in madori:
            return False
    return True


def generate_madori_rule_based(rows, cols):
    """
    必須部屋をすべて置くまでループ
    (最大 attempt=10 回などでリトライ)
    1. 必須部屋 (B,ut,e,H,s,LDK) -> 全部揃わないと失敗
    2. 残り(r,c)配置 (外周)
    3. 廊下敷設
    """
    rooms=config.ROOMS
    attempt=0
    max_attempts=100

    while True:
        attempt+=1
        print(f"--- Attempt {attempt} ---")

        # 1) マップ初期化
        madori=initialize_madori(rows, cols)

        # 2) 必須部屋配置
        ok=place_mandatory_rooms(madori, rooms)

        if not ok:
            # 失敗 → やり直し
            if attempt>=max_attempts:
                print("Failed to place all mandatory rooms after max_attempts.")
                # そのまま返す or 空マップなどを返す
                return madori
            else:
                continue  # 次のattemptへ

        # 3) 残りの部屋
        place_room_outside_in(madori, "r", rooms, max_tries=50)
        place_room_outside_in(madori, "c", rooms, max_tries=50)

        # 4) 廊下
        place_corridor(madori, rooms)

        # 全部成功したので終了
        return madori