# import numpy as np
# from . import config, utils

# #########################################################
# # 基本ユーティリティ
# #########################################################

# def initialize_madori(rows, cols):
#     """空の間取りを作成"""
#     return np.full((rows, cols), ".")

# def is_on_outer_wall(rows, cols, r, c, h, w):
#     """部屋が外壁に面しているか(上下or左右のいずれかがマップ境界)"""
#     if r==0 or (r+h==rows):
#         return True
#     if c==0 or (c+w==cols):
#         return True
#     return False

# #########################################################
# # 小さい部屋(浴室,脱衣所,トイレ,玄関,ホール)を外壁に面して配置
# #########################################################

# def place_small_room_on_wall(madori, room_code, rooms, max_tries=50):
#     """
#     指定した小部屋を壁際(外周)に面するよう優先的に配置
#     """
#     rows, cols = madori.shape
#     room = rooms[room_code]
#     min_s, max_s = room["size"]
#     import numpy as np

#     def get_layer_coords(layer):
#         top, left = layer, layer
#         bottom=rows-1-layer
#         right=cols-1-layer
#         coords=[]
#         if top>bottom or left>right:
#             return coords
#         # 上辺
#         for cc in range(left,right+1):
#             coords.append((top,cc))
#         # 右辺
#         for rr in range(top+1,bottom):
#             coords.append((rr,right))
#         # 下辺
#         if bottom>top:
#             for cc in range(right,left-1,-1):
#                 coords.append((bottom,cc))
#         # 左辺
#         if right>left:
#             for rr in range(bottom-1,top,-1):
#                 coords.append((rr,left))
#         return coords

#     for _ in range(max_tries):
#         layer=0
#         placed=False
#         while True:
#             perimeter = get_layer_coords(layer)
#             if not perimeter:
#                 break

#             for (r,c) in perimeter:
#                 if min_s>max_s:
#                     return False
#                 if min_s==max_s:
#                     h=w=min_s
#                 else:
#                     h=np.random.randint(min_s,max_s)
#                     w=np.random.randint(min_s,max_s)

#                 if r+h<=rows and c+w<=cols:
#                     if utils.is_space_available(madori, r,c,h,w):
#                         # 壁に面している？
#                         if is_on_outer_wall(rows,cols,r,c,h,w):
#                             if utils.check_adjacency(madori, r,c,h,w, room["adjacent_to"]):
#                                 # 配置
#                                 madori[r:r+h, c:c+w] = room_code
#                                 placed=True
#                                 break
#             if placed:
#                 return True
#             layer+=1
#     return False

# def place_small_room_set_on_wall(madori, codeA, codeB, rooms, max_tries=50):
#     """
#     2部屋(例: B, ut)を両方外壁に面しつつ相互に隣接
#     """
#     rows, cols = madori.shape
#     for _ in range(max_tries):
#         # 先にcodeAを外壁に
#         okA = place_small_room_on_wall(madori, codeA, rooms, max_tries=1)
#         if not okA:
#             continue

#         # codeA配置領域
#         placed_pos=[]
#         for rr in range(rows):
#             for cc in range(cols):
#                 if madori[rr,cc]==codeA:
#                     placed_pos.append((rr,cc))
#         if not placed_pos:
#             continue

#         min_r = min(p[0] for p in placed_pos)
#         min_c = min(p[1] for p in placed_pos)
#         max_r = max(p[0] for p in placed_pos)
#         max_c = max(p[1] for p in placed_pos)
#         hA=(max_r-min_r)+1
#         wA=(max_c-min_c)+1

#         roomB=rooms[codeB]
#         placedB=False
#         import numpy as np
#         for _2 in range(max_tries):
#             if roomB["size"][0]<roomB["size"][1]:
#                 bh = np.random.randint(roomB["size"][0], roomB["size"][1])
#                 bw = np.random.randint(roomB["size"][0], roomB["size"][1])
#             else:
#                 bh=bw=roomB["size"][0]

#             # Bを(Aの周囲±1)で外壁に面するように
#             rr = np.random.randint(max(min_r-bh,0), min(max_r+2, rows-bh+1))
#             cc = np.random.randint(max(min_c-bw,0), min(max_c+2, cols-bw+1))

#             if rr+bh<=rows and cc+bw<=cols:
#                 if utils.is_space_available(madori, rr, cc, bh, bw):
#                     if is_on_outer_wall(rows,cols, rr,cc,bh,bw):
#                         if utils.check_adjacency(madori, rr, cc, bh, bw, [codeA]):
#                             madori[rr:rr+bh, cc:cc+bw]=codeB
#                             placedB=True
#                             break
#         if placedB:
#             return True
#         else:
#             # 撤去
#             madori[madori==codeA]="."
#     return False

# #########################################################
# # 階段(s)を中央
# #########################################################

# def place_stair_center(madori, room_code, rooms, tries=50):
#     rows, cols = madori.shape
#     rcenter = rows//2
#     ccenter = cols//2
#     import numpy as np

#     coords=[]
#     maxdist=max(rcenter,rows-1-rcenter, ccenter,cols-1-ccenter)
#     for dist in range(maxdist+1):
#         for dr in range(-dist, dist+1):
#             for dc in range(-dist, dist+1):
#                 rr=rcenter+dr
#                 cc=ccenter+dc
#                 if 0<=rr<rows and 0<=cc<cols:
#                     coords.append((rr,cc))

#     min_s,max_s = rooms[room_code]["size"]
#     for _ in range(tries):
#         for (r,c) in coords:
#             if min_s>max_s:
#                 return False
#             if min_s==max_s:
#                 h=w=min_s
#             else:
#                 h=np.random.randint(min_s,max_s)
#                 w=np.random.randint(min_s,max_s)
#             if (r+h<=rows) and (c+w<=cols):
#                 if utils.is_space_available(madori, r,c,h,w):
#                     if utils.check_adjacency(madori, r,c,h,w, rooms[room_code]["adjacent_to"]):
#                         madori[r:r+h, c:c+w] = room_code
#                         return True
#     return False

# #########################################################
# # LDK + r を「大きさ未定」で余りスペースに配置
# #########################################################

# def place_ldk_and_rooms(madori, rooms, tries=30):
#     """
#     小さい部屋、廊下後の大空間を使って
#     L, D, K, r を大きめに配置する(サイズ制限ほぼなし)
#     """
#     rows,cols=madori.shape
#     import numpy as np

#     big_codes=["L","D","K","r"]

#     def place_big_area_code(code):
#         for _ in range(tries):
#             r0=np.random.randint(0,rows)
#             c0=np.random.randint(0,cols)
#             # 下方向にどこまで '.' が連続しているか
#             max_h=0
#             for h in range(rows-r0+1):
#                 if r0+h>rows:
#                     break
#                 # h行ぶん見てすべて'.'かチェック
#                 for x in range(c0, cols):
#                     if madori[r0+h-1, x]!=".":
#                         break
#                 else:
#                     max_h=h
#                     continue
#                 break
#             max_w=0
#             for w in range(cols-c0+1):
#                 if c0+w>cols:
#                     break
#                 # w列ぶん見てすべて'.'かチェック
#                 for y in range(r0, r0+max_h):
#                     if madori[y, c0+w-1]!=".":
#                         break
#                 else:
#                     max_w=w
#                     continue
#                 break

#             if max_h>0 and max_w>0:
#                 if utils.is_space_available(madori, r0,c0,max_h,max_w):
#                     madori[r0:r0+max_h, c0:c0+max_w] = code
#                     return True
#         return False

#     for code in big_codes:
#         place_big_area_code(code)

# #########################################################
# # 一般的な小部屋配置 or セット配置 (既存)
# #########################################################

# def place_corridor(madori, rooms):
#     """小さい部屋後に廊下を敷いて接続"""
#     unconnected=utils.find_unconnected_rooms(madori, rooms)
#     for (r1,c1),(r2,c2) in unconnected:
#         path=utils.find_path(madori,(r1,c1),(r2,c2))
#         if path:
#             for (rr,cc) in path:
#                 if madori[rr,cc]==".":
#                     madori[rr,cc]="co"

# def generate_madori_rule_based(rows, cols):
#     """
#     最終的なメイン:
#       1. 浴室(B) & 脱衣所(ut) (外壁)
#       2. トイレ(t) (外壁)
#       3. 玄関(e) & ホール(H) (外壁)
#       4. 廊下(co)
#       5. 階段(s) 中央
#       6. LDK + r(大部屋) で余りを埋める
#       7. 必須部屋全部配置できるまでリトライ
#     """
#     MAX_ATTEMPTS=100
#     needed_codes = ["B","ut","t","e","H","s","L","D","K"]  # 必須

#     def check_all_placed(m):
#         for cd in needed_codes:
#             if cd not in m:
#                 return False
#         return True

#     for attempt in range(1, MAX_ATTEMPTS+1):
#         print(f"--- Attempt {attempt} ---")
#         madori=initialize_madori(rows,cols)

#         # 1) B+ut(外壁)
#         ok_but = place_small_room_set_on_wall(madori,"B","ut", config.ROOMS)
#         if not ok_but:
#             continue

#         # 2) トイレ(t) (外壁)
#         ok_t = place_small_room_on_wall(madori,"t", config.ROOMS)
#         if not ok_t:
#             continue

#         # 3) 玄関(e)+ホール(H) (外壁)
#         ok_eh = place_small_room_set_on_wall(madori,"e","H", config.ROOMS)
#         if not ok_eh:
#             continue

#         # 4) 廊下
#         place_corridor(madori, config.ROOMS)

#         # 5) 階段(s) 中央
#         ok_s = place_stair_center(madori,"s", config.ROOMS)
#         if not ok_s:
#             continue

#         # 6) 大きな空きスペースに LDK + r
#         place_ldk_and_rooms(madori, config.ROOMS, tries=30)

#         # 追加の廊下
#         place_corridor(madori, config.ROOMS)

#         # 判定
#         if check_all_placed(madori):
#             return madori

#     print("Failed to place all mandatory rooms after max attempts.")
#     return initialize_madori(rows,cols)