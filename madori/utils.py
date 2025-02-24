import numpy as np

def is_space_available(madori, row, col, height, width):
    return np.all(madori[row:row+height, col:col+width] == ".")

def check_adjacency(madori, row, col, height, width, adjacent_to):
    if not adjacent_to:
        return True
    rows, cols = madori.shape
    for r in range(row-1, row+height+1):
        for c in range(col-1, col+width+1):
            if (r<0 or r>=rows or c<0 or c>=cols or
                (row<=r<row+height and col<=c<col+width)):
                continue
            if madori[r, c] in adjacent_to:
                return True
    return False

def find_unconnected_rooms(madori, rooms):
    rows, cols = madori.shape
    room_positions = {}
    unconnected_rooms = []

    for r in range(rows):
        for c in range(cols):
            code = madori[r,c]
            if code!="." and code!="co": # coと.は無視
                if code not in room_positions:
                    room_positions[code] = []
                room_positions[code].append((r,c))

    print(f"room_positions: {room_positions}")
    room_codes = list(room_positions.keys())

    for i in range(len(room_codes)):
        for j in range(i+1, len(room_codes)):
            code1 = room_codes[i]
            code2 = room_codes[j]
            print(f"Checking connection between {code1} and {code2}")
            connected=False
            for pos1 in room_positions[code1]:
                for pos2 in room_positions[code2]:
                    print(f"  Checking positions: {pos1} and {pos2}")
                    if is_connected(madori, pos1, pos2):
                        connected=True
                        print(f"  {code1} and {code2} are connected")
                        break
                if connected:
                    break
            if not connected:
                print(f"  {code1} and {code2} are NOT connected")
                unconnected_rooms.append((room_positions[code1][0], room_positions[code2][0]))

    return unconnected_rooms

def is_connected(madori, pos1, pos2):
    """
    部屋(pos1) と 部屋(pos2) が '.'または'co' を通じて繋がるか
    """
    rows, cols = madori.shape

    def get_adjacent_walkables(r,c):
        result=[]
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            rr,cc=r+dr,c+dc
            if 0<=rr<rows and 0<=cc<cols:
                if madori[rr, cc] in ('.','co'):
                    result.append((rr,cc))
        return result

    start_positions = get_adjacent_walkables(pos1[0], pos1[1])
    goal_positions  = set(get_adjacent_walkables(pos2[0], pos2[1]))
    if not start_positions or not goal_positions:
        return False

    visited=set(start_positions)
    queue=list(start_positions)
    while queue:
        cur=queue.pop(0)
        if cur in goal_positions:
            return True
        r,c=cur
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            rr,cc=r+dr,c+dc
            if 0<=rr<rows and 0<=cc<cols:
                if (rr,cc) not in visited and madori[rr,cc] in ('.','co'):
                    visited.add((rr,cc))
                    queue.append((rr,cc))
    return False

def find_path(madori, start, goal):
    """
    A*で '.' を通して最短経路。
    ※ 'co' も通したいなら get_neighborsで ('.','co') を許可。
    """
    rows, cols = madori.shape

    def heuristic(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set={start}
    came_from={}
    gscore={start:0}
    fscore={start:heuristic(start,goal)}

    while open_set:
        current=min(open_set, key=lambda x:fscore.get(x,float('inf')))
        if current==goal:
            return reconstruct_path(came_from,current)
        open_set.remove(current)
        for neigh in get_neighbors(madori, current):
            tg=gscore.get(current,float('inf'))+1
            if tg<gscore.get(neigh,float('inf')):
                came_from[neigh]=current
                gscore[neigh]=tg
                fscore[neigh]=tg+heuristic(neigh,goal)
                if neigh not in open_set:
                    open_set.add(neigh)

    return None

def get_neighbors(madori, pos):
    """
    '.' だけ通す。 'co' も通したいなら下記条件を if madori[nr,nc] in ('.','co'): に変更
    """
    rows, cols = madori.shape
    r,c=pos
    neighbors=[]
    for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        rr,cc=r+dr,c+dc
        if 0<=rr<rows and 0<=cc<cols:
            if madori[rr,cc]=='.':
                neighbors.append((rr,cc))
    return neighbors

def reconstruct_path(came_from, current):
    path=[current]
    while current in came_from:
        current=came_from[current]
        path.append(current)
    return path[::-1]