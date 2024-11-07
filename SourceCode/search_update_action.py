from asyncio.windows_events import NULL
from multiprocessing import parent_process
import os
from enum import Enum
import math
from queue import PriorityQueue
from queue import Queue
from typing import Tuple
import tracemalloc
#from memory_profiler import profile

tracemalloc.start()

class Vector:
    def __init__(self, *components):
        if len(components) == 1 and isinstance(components[0], (list, tuple)):
            self.components = tuple(components[0])
        else:
            self.components = tuple(components)
    def __repr__(self):
         return f"Vector({', '.join(map(str, self.components))})"
    def __add__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must be of the same dimension.")
        return Vector(*[a + b for a, b in zip(self.components, other.components)])
    
    def __sub__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must be of the same dimension.")
        return Vector(*[a - b for a, b in zip(self.components, other.components)])
    
    def __mul__(self, scalar):
        return Vector(*[scalar * x for x in self.components])
    def __hash__(self):
        return hash(self.components)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def dot(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must be of the same dimension.")
        return sum(a * b for a, b in zip(self.components, other.components))
    
    def magnitude_square(self):
        return sum(x ** 2 for x in self.components)

    def magnitude(self):
        return math.sqrt(self.magnitude_square())
    
    def __eq__(self, other):
        return self.components == other.components
    
    def __len__(self):
        return len(self.components)
    
    def __getitem__(self, index):
        return self.components[index]
    def __lt__(self, other):
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must be of the same dimension.")
        for com1, com2 in zip(self.components, other.components):
            if (com1 == com2):
                continue
            return (com1 < com2)

class Hungarian:
    INF = float('inf')

    def __init__(self, n):
        self.n = n
        self.c = [[self.INF] * n for _ in range(n)]
        self.fx = [0] * n
        self.fy = [0] * n
        self.mX = [-1] * n
        self.mY = [-1] * n
        self.trace = [-1] * n
        self.q = [0] * (n + 10)
        self.arg = [0] * n
        self.d = [0] * n
        self.start = -1
        self.finish = -1

    def add_edge(self, u, v, cost):
        self.c[u][v] = min(self.c[u][v], cost)

    def solve(self):
        max_iterations = 2 * self.n  # Or any other reasonable limit
        iteration_count = 0

        for i in range(self.n):
            self.start = i
            self.init_bfs()
            while self.finish == -1:
                iteration_count += 1
                if iteration_count > max_iterations:
                    return self.INF
                self.find_aug_path()
                if self.finish == -1:
                    self.subx_addy()
            self.enlarge()
        
        total_cost = sum(self.c[i][self.mX[i]] for i in range(self.n) if self.mX[i] != -1)
        return total_cost if all(x != -1 for x in self.mX) else "No complete matching found"


    def get_c(self, i, j):
        return self.c[i][j] - self.fx[i] - self.fy[j]

    def init_bfs(self):
        self.trace = [-1] * self.n
        self.ql = self.qr = 0
        self.q[self.qr] = self.start
        self.qr += 1
        for j in range(self.n):
            self.d[j] = self.get_c(self.start, j)
            self.arg[j] = self.start
        self.finish = -1

    def find_aug_path(self):
        while self.ql < self.qr:
            i = self.q[self.ql]
            self.ql += 1
            for j in range(self.n):
                if self.trace[j] == -1:
                    w = self.get_c(i, j)
                    if w == 0:
                        self.trace[j] = i
                        if self.mY[j] == -1:
                            self.finish = j
                            return
                        self.q[self.qr] = self.mY[j]
                        self.qr += 1
                    elif self.d[j] > w:
                        self.d[j] = w
                        self.arg[j] = i

    def subx_addy(self):
        delta = min(self.d[j] for j in range(self.n) if self.trace[j] == -1)

        self.fx[self.start] += delta
        for j in range(self.n):
            if self.trace[j] != -1:
                self.fy[j] -= delta
                self.fx[self.mY[j]] += delta
            else:
                self.d[j] -= delta

        for j in range(self.n):
            if self.trace[j] == -1 and self.d[j] == 0:
                self.trace[j] = self.arg[j]
                if self.mY[j] == -1:
                    self.finish = j
                    return
                self.q[self.qr] = self.mY[j]
                self.qr += 1

    def enlarge(self):
        while self.finish != -1:
            i = self.trace[self.finish]
            nxt = self.mX[i]
            self.mX[i] = self.finish
            self.mY[self.finish] = i
            self.finish = nxt


class PrioritizedItem:
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item

    def __lt__(self, other):
        return self.priority < other.priority
    def getItem(self):
        return self.item

import time
time_start = time.time()

inputWall = {}
inputSwitch = {}
inputStone = {}
inputAres = {}
fileNames = []
mapWall = {}
deadLock = {}
shortest_dist = {}

wall = []
lineCnt = 0
wordSet = set()
wordSetCnt = 0
wordMap = {}

def is_movable_cell(fileName, vec):
    if (vec[0] < 0 or vec[0] >= len(mapWall[fileName])):
        return False
    if (vec[1] < 0 or vec[1] >= len(mapWall[fileName][vec[0]])):
        return False
    #print(f"is_movable_cell: {vec}, {not mapWall[fileName][vec[0]][vec[1]]}")
    return not mapWall[fileName][vec[0]][vec[1]]

def in_inputWall(fileName, vec):
    if (vec[0] < 0 or vec[0] >= len(mapWall[fileName])):
        return False
    if (vec[1] < 0 or vec[1] >= len(mapWall[fileName][0])):
        return False
    return mapWall[fileName][vec[0]][vec[1]]
def is_deadLock(fileName, vec):
    if (vec[0] < 0 or vec[0] >= len(deadLock[fileName])):
        return False
    if (vec[1] < 0 or vec[1] >= len(deadLock[fileName][0])):
        return False
    return deadLock[fileName][vec[0]][vec[1]]
class Direction(Enum):
    Up = Vector([-1, 0])
    Left = Vector([0, -1])
    Down = Vector([1, 0])
    Right = Vector([0, 1])

test_dir = '../official_test/'
test_list = ['demo.txt']
for x in sorted(os.listdir(test_dir)):
    if x.startswith("random_047") or x in test_list:
        content = open(test_dir + x, 'r').read()
        fileName = x[:-4]
        fileNames.append(fileName)
        inputWall[fileName] = []
        inputSwitch[fileName] = []
        inputStone[fileName] = {}

        lines = content.split('\n')

        height_map = len(lines[1:])
        width_map = max([len(line) for line in lines[1:]])
        mapWall[fileName] = [[False] * width_map for _ in range(height_map)]
        deadLock[fileName] = [[False] * width_map for _ in range(height_map)]

        weightQueue = Queue()
        for number in lines[0].split():
            weightQueue.put(int(number))

        for i in range(1, len(lines)):
            for j in range(0, len(lines[i])):
                if (lines[i][j] == '#'):
                   inputWall[fileName].append(Vector([i - 1,j]))
                elif (lines[i][j] == '$'):
                    inputStone[fileName][Vector([i - 1, j])] = weightQueue.get()
                elif (lines[i][j] == '@'):
                    inputAres[fileName] = Vector([i - 1,j])
                elif (lines[i][j] == '.'):
                    inputSwitch[fileName].append(Vector([i - 1,j]))
                elif (lines[i][j] == '*'):
                    inputStone[fileName][Vector([i - 1, j])] = weightQueue.get()
                    inputSwitch[fileName].append(Vector([i - 1,j]))
                elif (lines[i][j] == '+'):
                    inputSwitch[fileName].append(Vector([i - 1,j]))
                    inputAres[fileName] = Vector([i - 1,j])

        for vec in inputWall[fileName]:
            mapWall[fileName][vec[0]][vec[1]] = True
            deadLock[fileName][vec[0]][vec[1]] = True

        # if a stone toward a dead end then it is a dead lock
        for i in range(height_map):
            for j in range(width_map):
                queue = Queue()
                queue.put(Vector([i, j]))
                while not queue.empty():
                    vec = queue.get()
                    if is_deadLock(fileName, vec) or vec in inputSwitch[fileName] or vec == inputAres[fileName]:
                        continue
                    near_deadLock = [is_deadLock(fileName, vec + direct.value) for direct in Direction]
                    if sum(near_deadLock) >= 3:
                        deadLock[fileName][vec[0]][vec[1]] = True
                        for direct in Direction:
                            queue.put(vec + direct.value)

        # corner not on a switch is a deadLock
        new_deadLock = []
        for i in range(height_map):
            for j in range(width_map):
                if Vector([i, j]) in inputSwitch[fileName]:
                    continue
                myDirection = [Direction.Up,  Direction.Left, Direction.Down, Direction.Right]
                wall_near = [is_deadLock(fileName, Vector([i, j]) + direct.value) for direct in myDirection]
                for k in range(4):
                    if wall_near[k] and wall_near[k - 1]:
                        new_deadLock.append((i, j))
                        break
        for i, j in new_deadLock:
            deadLock[fileName][i][j] = True

        # x in the below sample is a near-wall deadLock:
        #   #   #
        #   #xxx#
        #   #####
        # If there is no switch in the near-wall deadLock then it is a deadLock
        direct_move_wall = []
        direct_move_wall.append((Direction.Left.value, Direction.Up.value))
        direct_move_wall.append((Direction.Left.value, Direction.Down.value))
        direct_move_wall.append((Direction.Down.value, Direction.Left.value))
        direct_move_wall.append((Direction.Down.value, Direction.Right.value))
        new_deadLock = []
        for i in range(height_map):
            for j in range(width_map):
                vec = Vector([i, j])
                if is_deadLock(fileName, vec):
                    continue
                for direct_move_base, direct_wall in direct_move_wall:
                    border = []
                    for direct_move in [Vector([0, 0]) - direct_move_base, direct_move_base]:
                        cur = vec
                        while 0 <= cur[0] and cur[0] < height_map and 0 <= cur[1] and cur[1] < width_map:
                            if is_deadLock(fileName, cur):
                                border.append(cur)
                                break
                            cur += direct_move
                    if len(border) < 2:
                        continue

                    flag = True
                    cur = border[0] + direct_move_base
                    while cur != border[1]:
                        if cur in inputSwitch[fileName] or not is_deadLock(fileName, cur + direct_wall):
                            flag = False
                            break
                        cur += direct_wall
                    if flag:
                        cur = border[0] + direct_move_base
                        while cur != border[1]:
                            new_deadLock.append(cur)
                            cur += direct_wall
        for i, j in new_deadLock:
            deadLock[fileName][i][j] = True

        # shortest_dist is the shortest distance between pairs non-wall cells
        # Beside walls, we will also block 1 cell
        # For each blocked cell choice, we will calculate the shortest distance between pairs non-wall cells
        # We also need to calculate for each choice of starting direction at start cell
        # shortest_dist[file_name][blocked_i, blocked_j][start_i, start_j][start_direction][end_i, end_j]
        

        # Compute shortest distances for all pairs of non-wall cells, taking into account blocking a single cell
        shortest_dist[fileName] = {}
        directions = [Direction.Up, Direction.Left, Direction.Down, Direction.Right]
        
        def is_valid_cell(vec, blocked_vec=None):
            """Checks if a cell is valid for traversal."""
            if not is_movable_cell(fileName, vec):
                #print(f"{vec} is not movable")
                return False
            if in_inputWall(fileName, vec) or (blocked_vec is not None and vec == blocked_vec):
                return False
            return True
        
        def bfs_shortest_path(start_vec, blocked_vec):
            """Performs BFS to find the shortest path lengths from start_vec, considering blocked_vec."""
            dist = {}
            queue = Queue()
            queue.put(start_vec)
            dist[start_vec] = 0
            
            while not queue.empty():
                current = queue.get()
                for direction in directions:
                    next_vec = current + direction.value
                    if is_valid_cell(next_vec, blocked_vec) and next_vec not in dist:
                        dist[next_vec] = dist[current] + 1
                        queue.put(next_vec)
            
            return dist
        
        # Calculate shortest distances while considering each cell as blocked one by one
        for blocki in range(height_map):
            for blockj in range(width_map):
                blocked_vec = Vector([blocki, blockj])
                shortest_dist[fileName][blocked_vec] = {}
                if not is_valid_cell(blocked_vec):
                    continue
                for direction in directions:
                    start_vec = blocked_vec + direction.value
                #print(f"Calculating shortest distances for {fileName} with blocked cell at ({i}, {j})")
                    if not is_valid_cell(start_vec):
                        continue
                    dist = bfs_shortest_path(start_vec, blocked_vec)
                    shortest_dist[fileName][blocked_vec][start_vec] = dist

def get_shortest_dist(fileName, blocked_vec, start_vec, end_vec):
    # print(f"Finding shortest distance from {start_vec} to {end_vec} in {fileName} with blocked cell at {blocked_vec}")
    # Check if the input is valid
    if (fileName not in shortest_dist):
        # print(f"File not found in shortest distances: {fileName}")
        return float('inf')
    if blocked_vec not in shortest_dist[fileName]:
        # print(f"Blocked vector not found in shortest distances: {blocked_vec}")
        return float('inf')
    if start_vec not in shortest_dist[fileName][blocked_vec]:
        # print(f"Start vector not found in shortest distances: {start_vec}")
        return float('inf')
    
    # Check if the end vector is in the computed distances
    if end_vec not in shortest_dist[fileName][blocked_vec][start_vec]:
        # print(f"End vector not found in computed distances: {end_vec}")
        return float('inf')
    
    # Return the shortest distance from start to end
    return shortest_dist[fileName][blocked_vec][start_vec][end_vec]
            # print('------------')



# Complete execution or integration logic can be added here
# E.g., process test cases or further analyze the results



dist_list = {}

def get_pushing_stone_cost(fileName, i, stone_pos, switch_pos, direction):
    if fileName not in dist_list:
        dist_list[fileName] = {}
    if i not in dist_list[fileName]:
        dist_list[fileName][i] = {}
    if stone_pos not in dist_list[fileName][i]:
        dist_list[fileName][i][stone_pos] = {}
    if len(dist_list[fileName][i][stone_pos].keys()) != 0:
        if (switch_pos, direction) in dist_list[fileName][i][stone_pos]:
            return dist_list[fileName][i][stone_pos][switch_pos, direction]
        else:
            return float('inf')
    for id, _w in enumerate(inputStone[fileName].values()):
        if id == i:
            w = _w
    start_stone_pos = stone_pos
    queue = PriorityQueue()
    dist_list[fileName][i][start_stone_pos] = {}

    # Initialize with current stone position and all directions
    for direction in Direction:
        ares_pos = start_stone_pos - direction.value
        if is_valid_cell(ares_pos):
            queue.put(PrioritizedItem(0, (start_stone_pos, direction)))
            dist_list[fileName][i][start_stone_pos][start_stone_pos, direction] = 0
    #print(f"Calculating distances for {fileName} with stone {i} at {start_stone_pos}")
    while not queue.empty():
        top_queue = queue.get()
        current_stone_pos, current_direction = top_queue.item
        current_cost = top_queue.priority
        ares_pos = current_stone_pos - current_direction.value
        #print(f"current_stone_pos: {current_stone_pos}, current_direction: {current_direction}, current_cost: {current_cost} ares_pos: {ares_pos}")

        # Explore all possible moves
        for new_direction in Direction:
            next_stone_pos = current_stone_pos + new_direction.value
            next_ares_pos = current_stone_pos - new_direction.value
            #print(f"{is_valid_cell(next_stone_pos)}, {is_valid_cell(next_ares_pos)}")
            if (is_valid_cell(next_stone_pos) and is_valid_cell(next_ares_pos)):
                ares_move_cost = get_shortest_dist(fileName, current_stone_pos, ares_pos, next_ares_pos)
                # print(f"Attempting next_stone_pos: {next_stone_pos}, new_direction: {new_direction} next_ares_pos: {next_ares_pos} ares_pos: {ares_pos} -> ares_next_pos: {next_ares_pos} ares_move_cost: {ares_move_cost}")
                if ares_move_cost == float('inf'):
                    continue

                action_cost = ares_move_cost + w + 1
                new_cost = current_cost + action_cost
                
                if ((next_stone_pos, new_direction) not in dist_list[fileName][i][start_stone_pos] or new_cost < dist_list[fileName][i][start_stone_pos][next_stone_pos, new_direction]):
                    # print(f"next_stone_pos: {next_stone_pos}, new_direction: {new_direction}, ares_move_cost: {ares_move_cost}, action_cost: {action_cost}")
                    
                    dist_list[fileName][i][start_stone_pos][next_stone_pos, new_direction] = new_cost
                    #print(f"next_stone_pos: {next_stone_pos}, new_direction: {new_direction}, action_cost: {action_cost}")
                    queue.put(PrioritizedItem(new_cost, (next_stone_pos, new_direction)))       

    if (switch_pos, direction) in dist_list[fileName][i][start_stone_pos]:
        return dist_list[fileName][i][start_stone_pos][switch_pos, direction]
    else:
        return float('inf')
         

class State:
    def __init__(self, name, ares, stones):
        self.ares = ares
        self.stones = dict(sorted(stones.copy().items()))
        self.name = name

    def __str__(self):
        res = self.name + '\n'
        for stone in self.stones:
            res += str(stone) + ": " + str(self.stones[stone]) + '\n'
        res += "ares: " + str(self.ares)
        return res
    
    def isGoal(self):
        for switch in inputSwitch[self.name]:
            if (switch not in self.stones):
                return False
        return True
    
    def neighbors(self):
        result = []
        for stone in self.stones():
            for direct in Direction:
                newState = self.neighbor(stone, direct.value)
                if (newState != NULL):
                    result.append(newState)
            return result

    def neighbor(self, stone, enum):
        if ((stone + enum) in self.stones or in_inputWall(self.name, stone + enum) or (stone - enum) not in self.reached):
            return NULL
        newStateStones = self.stones.copy()
        newStateStones.pop(stone)
        newStateStones[stone + enum] = self.stones[stone]

        return State(self.name, stone, newStateStones)

    def reverseNeighbor(self, enum):
        newStateStone = self.stones.copy()
        if ((self.ares + enum) in self.stones):
            newStateStone.pop(self.ares + enum)
            newStateStone[self.ares] = self.stones[self.ares + enum]
        return State(self.name, self.ares - enum, newStateStone)

    def actionCost(self, stone, enum):
        return len(self.reached[stone - enum]) + self.stones[stone] + 1
    def actionPath(self, stone, direct):
        return self.reached[stone - direct.value] + str(direct)[10]

    def extractPath(self):
        start = time.time()
        self.reached = {}
        queue = PriorityQueue()
        queue.put(PrioritizedItem('', self.ares))
        self.reached[self.ares] = ''

        while not queue.empty():
            top = queue.get()
            for direct in Direction:
                if ((space:=(top.item + direct.value)) in self.stones or in_inputWall(self.name, space)):
                    continue
                if space not in self.reached or len(self.reached[space]) > len(top.priority) + 1:
                    self.reached[space] = (prio:=(top.priority + str(direct)[10].lower()))
                    queue.put(PrioritizedItem(prio, space))
        end = time.time()
        #print("Check reached done. Time taken: ", end - start)

    def heuristic(self):
        n = len(inputSwitch[self.name])
        total_cost = 0

        # Calculate minimum cost to match all stones to the switches
        # using Hungarian algorithm and Manhattan distance
        #print('------------------')
        hungarian = Hungarian(n)
        for i, (stone_pos, w) in enumerate(self.stones.items()):  # Iterate over stones
            for j, switch_pos in enumerate(inputSwitch[self.name]):  # Iterate over switches
                min_cost = float('inf')
                for direction in Direction:
                    min_cost = min(min_cost, get_pushing_stone_cost(self.name, i, stone_pos, switch_pos, direction))
                if min_cost != float('inf'):
                    
                    hungarian.add_edge(i, j, min_cost)

            #total_cost += min_cost
        total_cost += hungarian.solve()

        # # Distance from player to the nearest stone
        # total_cost += min([get_shortest_dist(self.name, None, self.ares, stone_pos) for stone_pos in self.stones.keys()])

        # # Penalty for deadlock
        # for (stone_pos, w) in self.stones.items():
        #     # Penalty +infinity if stone in corner and not on a switch
        #     if is_deadLock(self.name, stone_pos) and stone_pos not in inputSwitch[self.name]:
        #         total_cost += float('inf')
        #     ## Penalty +infinity if 2 stones are adjacent and are not on switches and cannot be moved
        #     f = stone_pos in inputSwitch[self.name]
        #     for stone_pos2 in self.stones.keys():
        #         vec_dist = stone_pos - stone_pos2
        #         if vec_dist.magnitude_square() == 1 and (not f or stone_pos2 not in inputSwitch[self.name]):
        #             direction = Vector([vec_dist[1], vec_dist[0]])
        #             if (in_inputWall(self.name, stone_pos + direction) or in_inputWall(self.name, stone_pos - direction)) and (in_inputWall(self.name, stone_pos2 + direction) or in_inputWall(self.name, stone_pos2 - direction)):
        #                 total_cost += float('inf')

        return total_cost

class SearchNode:
    def __init__(self, state, parent, cost, path):
        self.state = state
        self.cost = cost
        self.path = path
        if (parent != NULL):
           self.cost += parent.cost
           self.path = parent.path + self.path

    def priority_value(self):
        return self.state.heuristic() + self.cost

    def children(self):
        result = []
        self.state.extractPath()
        for stone in self.state.stones:
            for direct in Direction:
                newState = self.state.neighbor(stone, direct.value)
                if (newState != NULL):
                    result.append(SearchNode(newState, self, self.state.actionCost(stone, direct.value), self.state.actionPath(stone, direct)))
        return result

    def printTree(self):
        parent = self.state
        for state in reversed(self.path):
            print(parent)
            parent = self.getParent(parent, state.lower())

        print("cost: ", self.cost)
    def getParent(self, parent, char):
        enum = None
        if (char == 'd'):
            enum = Direction.Down.value
        elif (char == 'u'):
            enum = Direction.Up.value
        elif (char == 'l'):
            enum = Direction.Left.value
        elif (char == 'r'):
            enum = Direction.Right.value
        return parent.reverseNeighbor(enum)
            
#@profile
class SolverBFS:
    def __init__(self, state):
        self.rootNode = SearchNode(state, NULL, 0, '')
        self.reached = {}
        self.reached[str(state)] = 0
        self.node_number = 0

        self.priorityQueue = PriorityQueue()
        self.priorityQueue.put(PrioritizedItem(self.rootNode.priority_value(), self.rootNode))

    def expand(self):
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            self.node_number += 1
            if (self.node_number % 100 != 0): 
                print ("\033[A                             \033[A")
                print(self.node_number)
            #   print(top.path)    
            for child in top.children():
                if (child.state.isGoal()):
                    #child.printTree()
                    print(child.path)
                    print("Node: ", self.node_number)
                    print("Cost: ", child.cost) 
                    return
                if (str(child.state) not in self.reached.keys() or self.reached[str(child.state)] > child.cost):
                    self.reached[str(child.state)] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.priority_value(), child))
        print("FAILURE")

class SolverUCS:
    def __init__(self, state):
        self.rootNode = SearchNode(state, NULL, 0, '')
        self.reached = {}
        self.reached[str(state)] = 0
        self.node_number = 0
        self.priorityQueue = PriorityQueue()
        self.priorityQueue.put(PrioritizedItem(self.rootNode.cost, self.rootNode))

    def expand(self):
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            if (top.state.isGoal()):
                #top.printTree()
                print(top.path)
                print("Node: ", self.node_number)
                return
            print ("\033[A                             \033[A")
            self.node_number += 1
            print(self.node_number)
            for child in top.children():
                if (str(child.state) not in self.reached.keys() or self.reached[str(child.state)] > child.cost):
                    self.reached[str(child.state)] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.cost, child))
        print("FAILURE")

for fileName in fileNames:
    solver = SolverBFS(State(fileName, inputAres[fileName], inputStone[fileName]))
    solver.expand()
time_end = time.time()
print('time cost', time_end-time_start)
current, peak = tracemalloc.get_traced_memory()

print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage: {peak / 10**6:.2f} MB")

# Stop tracing memory allocation
tracemalloc.stop()