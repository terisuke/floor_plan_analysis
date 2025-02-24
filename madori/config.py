ROOMS = {
    "L":  {"name": "リビング",   "size": (3, 7),  "adjacent_to": ["D", "K"]},
    "D":  {"name": "ダイニング", "size": (2, 6),  "adjacent_to": ["L", "K"]},
    "K":  {"name": "キッチン",   "size": (2, 5),  "adjacent_to": ["L", "D"]},
    # 例: 風呂は常に(2,2)
    "B":  {"name": "風呂",       "size": (2, 2),  "adjacent_to": ["ut"]},
    # 脱衣所は(2,4)
    "ut": {"name": "脱衣所",     "size": (2, 4),  "adjacent_to": ["B"]},
    # トイレ(1,3)
    "t":  {"name": "トイレ",     "size": (1, 3),  "adjacent_to": []},
    # ホール(1,6)
    "H":  {"name": "ホール",     "size": (1, 6),  "adjacent_to": []},
    # 玄関(1,8)
    "e":  {"name": "玄関",       "size": (1, 8),  "adjacent_to": ["H"]},
    # クローゼット(1,6)
    "c":  {"name": "クローゼット","size": (1, 6), "adjacent_to": []},
    # 廊下
    "co": {"name": "廊下",       "size": (1, 1),  "adjacent_to": []},
    # 部屋(2,13)
    "r":  {"name": "部屋",       "size": (2, 13), "adjacent_to": []},
    # 階段(1,5)
    "s":  {"name": "階段",       "size": (1, 5),  "adjacent_to": ["H"]},
}