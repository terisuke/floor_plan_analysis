ROOMS = {
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