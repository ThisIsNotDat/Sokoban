from asyncio.windows_events import NULL
from multiprocessing import parent_process
import psutil
import os
from enum import Enum
import math
from queue import PriorityQueue
from queue import Queue
from typing import Tuple
from memory_profiler import profile
import tracemalloc
import time

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


inputWall = {}
inputSwitch = {}
inputStone = {}
inputAres = {}
fileNames = []
mapWall = {}
deadLock = {}

wall = []
lineCnt = 0
wordSet = set()
wordSetCnt = 0
wordMap = {}

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
test_list = []#'diff-rating_01.txt', 'diff-rating_02.txt', 'diff-rating_03.txt', 'diff-rating_04.txt']
for x in sorted(os.listdir(test_dir)):
    if x.startswith("random_046") or x in test_list:
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
        for i in range(self.n):
            self.fx[i] = min(self.c[i])

        for j in range(self.n):
            self.fy[j] = self.c[0][j] - self.fx[0]
            for i in range(1, self.n):
                self.fy[j] = min(self.fy[j], self.c[i][j] - self.fx[i])

        for i in range(self.n):
            self.start = i
            self.init_bfs()
            while self.finish == -1:
                self.find_aug_path()
                if self.finish == -1:
                    self.subx_addy()
            self.enlarge()

        total_cost = sum(self.c[i][self.mX[i]] for i in range(self.n))
        return total_cost

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
        for direct in Direction:
            newState = self.neighbor(direct.value)
            if (newState != NULL):
                result.append(newState)
        return result

    def neighbor(self, enum):
        if in_inputWall(self.name, self.ares + enum):
            return NULL
 
        if ((self.ares + enum) not in self.stones):
            return State(self.name, self.ares + enum, self.stones)

        if ((self.ares + 2*enum) in self.stones or in_inputWall(self.name, self.ares + 2*enum)):
            return NULL

        newStateStones = self.stones.copy()
        newStateStones.pop(self.ares + enum)
        newStateStones[self.ares + 2*enum] = self.stones[self.ares + enum]

        return State(self.name, self.ares + enum, newStateStones)
    def reverseNeighbor(self, enum):
        newStateStone = self.stones.copy()
        if ((self.ares + enum) in self.stones):
            newStateStone.pop(self.ares + enum)
            newStateStone[self.ares] = self.stones[self.ares + enum]
        return State(self.name, self.ares - enum, newStateStone)

    def actionCost(self, enum):
        if (self.ares + enum) in self.stones and (self.ares + 2*enum) not in self.stones and not in_inputWall(self.name, self.ares + 2*enum):
            return self.stones[self.ares + enum] + 1
        return 1

    def heuristic(self):
        n = len(inputSwitch[self.name])
        total_cost = 0

        # Calculate minimum cost to match all stones to the switches
        # using Hungarian algorithm and Manhattan distance
        hungarian = Hungarian(n)
        for i, (stone_pos, w) in enumerate(self.stones.items()): # stone
            #min_cost = float('inf')
            for j in range(n): # switch
                cost = (stone_pos - inputSwitch[self.name][j]).magnitude_square()
                cost *= (w + 1)
                #min_cost = min(min_cost, cost)
                hungarian.add_edge(i, j, cost)
            #total_cost += min_cost
        total_cost += hungarian.solve()

        # Distance from player to the nearest stone
        total_cost += min((self.ares - stone).magnitude_square() for stone in self.stones)

        # Penalty for deadlock
        for (stone_pos, w) in self.stones.items():
            # Penalty +infinity if stone in corner and not on a switch
            if is_deadLock(self.name, stone_pos) and stone_pos not in inputSwitch[self.name]:
                total_cost += float('inf')
            ## Penalty +infinity if 2 stones are adjacent and are not on switches and cannot be moved
            f = stone_pos in inputSwitch[self.name]
            for stone_pos2 in self.stones:
                vec_dist = stone_pos - stone_pos2
                if vec_dist.magnitude_square() == 1 and (not f or stone_pos2 not in inputSwitch[self.name]):
                    direction = Vector([vec_dist[1], vec_dist[0]])
                    if (in_inputWall(self.name, stone_pos + direction) or in_inputWall(self.name, stone_pos - direction)) and (in_inputWall(self.name, stone_pos2 + direction) or in_inputWall(self.name, stone_pos2 - direction)):
                        total_cost += float('inf')

        return total_cost

class SearchNode:
    def __init__(self, state, parent, cost, direct):
        self.state = state
        self.cost = cost
        self.path = ''
        if (parent != NULL):
           self.cost += parent.cost
           direction = str(direct)[10]
           if (cost == 1):
               direction = direction.lower()
           self.path = parent.path + direction

    def priority_value(self):
        return self.state.heuristic() + self.cost

    def children(self):
        result = []
        for direct in Direction:
            newState = self.state.neighbor(direct.value)
            if (newState != NULL):
                result.append(SearchNode(newState, self, self.state.actionCost(direct.value), direct))
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
            

class PrioritizedItem:
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item

    def __lt__(self, other):
        return self.priority < other.priority
    def getItem(self):
        return self.item

class SolverBFS:
    def __init__(self, state):
        self.rootNode = SearchNode(state, NULL, 0, NULL)
        self.reached = {}
        self.reached[str(state)] = 0
        self.node_number = 0

        self.priorityQueue = PriorityQueue()
        self.priorityQueue.put(PrioritizedItem(self.rootNode.priority_value(), self.rootNode))

    #@profile
    def expand(self):
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            print ("\033[A                             \033[A")
            self.node_number += 1
            if (self.node_number % 1000 == 0): 
                print(top.path)    
            print(self.node_number)
            for child in top.children():
                if (child.state.isGoal()):
                    child.printTree()
                    print(child.path)
                    print("Node: ", self.node_number)
                    return
                if (str(child.state) not in self.reached or self.reached[str(child.state)] > child.cost):
                    self.reached[str(child.state)] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.priority_value(), child))
        print("FAILURE")

class SolverUCS:
    def __init__(self, state):
        self.rootNode = SearchNode(state, NULL, 0, NULL)
        self.reached = {}
        self.reached[str(state)] = 0
        self.node_number = 0
        self.priorityQueue = PriorityQueue()
        self.priorityQueue.put(PrioritizedItem(self.rootNode.cost, self.rootNode))

    def expand(self):
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            if (top.state.isGoal()):
                top.printTree()
                print(top.path)
                print("Node: ", self.node_number)
                return
            print ("\033[A                             \033[A")
            self.node_number += 1
            print(self.node_number)
            for child in top.children():
                if (str(child.state) not in self.reached or self.reached[str(child.state)] > child.cost):
                    self.reached[str(child.state)] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.cost, child))
        print("FAILURE")

#process = psutil.Process(os.getpid())
start_time = time.time()
if __name__ == "__main__":
    for fileName in fileNames:
        solver = SolverBFS(State(fileName, inputAres[fileName], inputStone[fileName]))
        solver.expand()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

#memory_usage = process.memory_info().rss / 1024 ** 2
#print(f"Memory usage: {memory_usage:.2f} MB")