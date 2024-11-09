from asyncio.windows_events import NULL
from logging import raiseExceptions
from multiprocessing import parent_process
import os
from enum import Enum
import math
from queue import PriorityQueue
from queue import Queue
from typing import Tuple
import tracemalloc
import sys
import json
#from memory_profiler import profile
import numpy as np
from scipy.optimize import linear_sum_assignment

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
        self.mat = np.full((n, n), self.INF)
        self.mX = np.full(n, -1)

    def add_edge(self, u, v, cost):
        self.mat[u][v] = cost
        #self.mat[u][v] = min(self.mat[u][v], cost)

    def solve(self):
        try:
            row_ind, col_ind = linear_sum_assignment(self.mat)
            self.mX = col_ind
            total_cost = self.mat[row_ind, col_ind].sum()
            return total_cost
        except ValueError:
            return self.INF
            #print(self.mat)
            #return "No complete matching found"


class PrioritizedItem:
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item

    def __lt__(self, other):
        return self.priority < other.priority
    def getItem(self):
        return self.item

class Input():
    inputWall = {}
    inputSwitch = {}
    inputStone = {}
    inputAres = {}
    fileNames = []
    mapWall = {}
    shortest_dist = {}
    @staticmethod
    def is_movable_cell(fileName, vec):
        if (vec[0] < 0 or vec[0] >= len(Input.mapWall[fileName])):
            return False
        if (vec[1] < 0 or vec[1] >= len(Input.mapWall[fileName][vec[0]])):
            return False
        #print(f"is_movable_cell: {vec}, {not mapWall[fileName][vec[0]][vec[1]]}")
        return not Input.mapWall[fileName][vec[0]][vec[1]]
    @staticmethod
    def in_inputWall(fileName, vec):
        if (vec[0] < 0 or vec[0] >= len(Input.mapWall[fileName])):
            return False
        if (vec[1] < 0 or vec[1] >= len(Input.mapWall[fileName][vec[0]])):
            return False
        return Input.mapWall[fileName][vec[0]][vec[1]]
    @staticmethod
    def is_valid_cell(fileName, vec, blocked_vec=None):
            """Checks if a cell is valid for traversal."""
            if not Input.is_movable_cell(fileName, vec):
                #print(f"{vec} is not movable")
                return False
            if Input.in_inputWall(fileName, vec) or (blocked_vec is not None and vec == blocked_vec):
                return False
            return True
    @staticmethod
    def bfs_shortest_path(fileName, start_vec, blocked_vec):
        """Performs BFS to find the shortest path lengths from start_vec, considering blocked_vec."""
        dist = {}
        queue = Queue()
        queue.put(start_vec)
        dist[start_vec] = 0
            
        while not queue.empty():
            current = queue.get()
            for direction in Direction:
                next_vec = current + direction.value
                if Input.is_valid_cell(fileName, next_vec, blocked_vec) and next_vec not in dist:
                    dist[next_vec] = dist[current] + 1
                    queue.put(next_vec)
            
        return dist
    @staticmethod
    def readFile(fileName):
        content = open(fileName, 'r').read()
        Input.fileNames.append(fileName[:-4])
        Input.inputWall[fileName] = []
        Input.inputSwitch[fileName] = []
        Input.inputStone[fileName] = {}

        lines = content.split('\n')

        height_map = len(lines[1:])
        width_map = max([len(line) for line in lines[1:]])
        Input.mapWall[fileName] = [[False] * width_map for _ in range(height_map)]

        weightQueue = Queue()
        for number in lines[0].split():
            weightQueue.put(int(number))

        for i in range(1, len(lines)):
            for j in range(0, len(lines[i])):
                if (lines[i][j] == '#'):
                   Input.inputWall[fileName].append(Vector([i - 1,j]))
                elif (lines[i][j] == '$'):
                    Input.inputStone[fileName][Vector([i - 1, j])] = weightQueue.get()
                elif (lines[i][j] == '@'):
                    Input.inputAres[fileName] = Vector([i - 1,j])
                elif (lines[i][j] == '.'):
                    Input.inputSwitch[fileName].append(Vector([i - 1,j]))
                elif (lines[i][j] == '*'):
                    Input.inputStone[fileName][Vector([i - 1, j])] = weightQueue.get()
                    Input.inputSwitch[fileName].append(Vector([i - 1,j]))
                elif (lines[i][j] == '+'):
                    Input.inputSwitch[fileName].append(Vector([i - 1,j]))
                    Input.inputAres[fileName] = Vector([i - 1,j])

        for vec in Input.inputWall[fileName]:
            Input.mapWall[fileName][vec[0]][vec[1]] = True


        # shortest_dist is the shortest distance between pairs non-wall cells
        # Beside walls, we will also block 1 cell
        # For each blocked cell choice, we will calculate the shortest distance between pairs non-wall cells
        # We also need to calculate for each choice of starting direction at start cell
        # shortest_dist[file_name][blocked_i, blocked_j][start_i, start_j][start_direction][end_i, end_j]
        

        # Compute shortest distances for all pairs of non-wall cells, taking into account blocking a single cell
        Input.shortest_dist[fileName] = {}
        directions = [Direction.Up, Direction.Left, Direction.Down, Direction.Right]
        
        # Calculate shortest distances while considering each cell as blocked one by one
        for blocki in range(height_map):
            for blockj in range(width_map):
                blocked_vec = Vector([blocki, blockj])
                Input.shortest_dist[fileName][blocked_vec] = {}
                if not Input.is_valid_cell(fileName, blocked_vec):
                    continue
                for direction in directions:
                    start_vec = blocked_vec + direction.value
                #print(f"Calculating shortest distances for {fileName} with blocked cell at ({i}, {j})")
                    if not Input.is_valid_cell(fileName, start_vec):
                        continue
                    dist = Input.bfs_shortest_path(fileName, start_vec, blocked_vec)
                    Input.shortest_dist[fileName][blocked_vec][start_vec] = dist

class Direction(Enum):
    Up = Vector([-1, 0])
    Left = Vector([0, -1])
    Down = Vector([1, 0])
    Right = Vector([0, 1])

def get_shortest_dist(fileName, blocked_vec, start_vec, end_vec):
    # print(f"Finding shortest distance from {start_vec} to {end_vec} in {fileName} with blocked cell at {blocked_vec}")
    # Check if the input is valid
    if (fileName not in Input.shortest_dist):
        # print(f"File not found in shortest distances: {fileName}")
        return float('inf')
    if blocked_vec not in Input.shortest_dist[fileName]:
        # print(f"Blocked vector not found in shortest distances: {blocked_vec}")
        return float('inf')
    if start_vec not in Input.shortest_dist[fileName][blocked_vec]:
        # print(f"Start vector not found in shortest distances: {start_vec}")
        return float('inf')
    
    # Check if the end vector is in the computed distances
    if end_vec not in Input.shortest_dist[fileName][blocked_vec][start_vec]:
        # print(f"End vector not found in computed distances: {end_vec}")
        return float('inf')
    
    # Return the shortest distance from start to end
    return Input.shortest_dist[fileName][blocked_vec][start_vec][end_vec]
            # print('------------')



# Complete execution or integration logic can be added here
# E.g., process test cases or further analyze the results



dist_list = {}

def get_pushing_stone_cost(fileName, w, stone_pos, switch_pos, end_direction):
    if fileName not in dist_list:
        dist_list[fileName] = {}
    if w not in dist_list[fileName]:
        dist_list[fileName][w] = {}
    if stone_pos not in dist_list[fileName][w]:
        dist_list[fileName][w][stone_pos] = {}
    if len(dist_list[fileName][w][stone_pos].keys()) != 0:
        if (switch_pos, end_direction) in dist_list[fileName][w][stone_pos]:
            return dist_list[fileName][w][stone_pos][switch_pos, end_direction]
        else:
            return float('inf')
    start_stone_pos = stone_pos
    queue = PriorityQueue()
    dist_list[fileName][w][start_stone_pos] = {}

    # Initialize with current stone position and all directions
    for direction in Direction:
        ares_pos = start_stone_pos - direction.value
        if Input.is_valid_cell(fileName, ares_pos):
            queue.put(PrioritizedItem(0, (start_stone_pos, direction)))
            dist_list[fileName][w][start_stone_pos][start_stone_pos, direction] = 0
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
            if (Input.is_valid_cell(fileName, next_stone_pos) and Input.is_valid_cell(fileName, next_ares_pos)):
                ares_move_cost = get_shortest_dist(fileName, current_stone_pos, ares_pos, next_ares_pos)
                # print(f"Attempting next_stone_pos: {next_stone_pos}, new_direction: {new_direction} next_ares_pos: {next_ares_pos} ares_pos: {ares_pos} -> ares_next_pos: {next_ares_pos} ares_move_cost: {ares_move_cost}")
                if ares_move_cost == float('inf'):
                    continue

                action_cost = ares_move_cost + w + 1
                new_cost = current_cost + action_cost
                
                if ((next_stone_pos, new_direction) not in dist_list[fileName][w][start_stone_pos] or new_cost < dist_list[fileName][w][start_stone_pos][next_stone_pos, new_direction]):
                    # print(f"next_stone_pos: {next_stone_pos}, new_direction: {new_direction}, ares_move_cost: {ares_move_cost}, action_cost: {action_cost}")
                    
                    dist_list[fileName][w][start_stone_pos][next_stone_pos, new_direction] = new_cost
                    #print(f"next_stone_pos: {next_stone_pos}, new_direction: {new_direction}, action_cost: {action_cost}")
                    queue.put(PrioritizedItem(new_cost, (next_stone_pos, new_direction)))  

    if (switch_pos, end_direction) in dist_list[fileName][w][start_stone_pos]:
        return dist_list[fileName][w][start_stone_pos][switch_pos, end_direction]
    else:
        return float('inf')
         

class State:
    def __init__(self, name, ares, stones):
        self.ares = ares
        #self.stones = dict(sorted(stones.copy().items()))
        self.stones = stones.copy()
        self.name = name

    def __str__(self):
        temp_stones = dict(sorted(self.stones.copy().items()))
        res = self.name + '\n'
        for stone in temp_stones:
            res += str(stone) + ": " + str(temp_stones[stone]) + '\n'
        res += "ares: " + str(self.ares)
        return res
    
    def isGoal(self):
        for switch in Input.inputSwitch[self.name]:
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
        if ((stone + enum) in self.stones or Input.in_inputWall(self.name, stone + enum) or (stone - enum) not in self.reached):
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
                if ((space:=(top.item + direct.value)) in self.stones or Input.in_inputWall(self.name, space)):
                    continue
                if space not in self.reached or len(self.reached[space]) > len(top.priority) + 1:
                    self.reached[space] = (prio:=(top.priority + str(direct)[10].lower()))
                    queue.put(PrioritizedItem(prio, space))
        end = time.time()
        #print("Check reached done. Time taken: ", end - start)

    def heuristic(self):
        n = len(Input.inputSwitch[self.name])
        total_cost = 0

        # Calculate minimum cost to match all stones to the switches
        # using Hungarian algorithm and Manhattan distance
        #print('------------------')
        hungarian = Hungarian(n)
        for i, (stone_pos, w) in enumerate(self.stones.items()):  # Iterate over stones
            for j, switch_pos in enumerate(Input.inputSwitch[self.name]):  # Iterate over switches
                min_cost = float('inf')
                for direction in Direction:
                    min_cost = min(min_cost, get_pushing_stone_cost(self.name, w, stone_pos, switch_pos, direction))
                #if min_cost != float('inf'):
                hungarian.add_edge(i, j, min_cost)

            #total_cost += min_cost
        total_cost += hungarian.solve()

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
class Solver():
    def __init__(self, state):
        self.rootNode = SearchNode(state, NULL, 0, '')
        self.reached = {}
        self.reached[str(state)] = 0
        self.node_number = 0
        self.priorityQueue = PriorityQueue()
    def expand(self):
        raise NotImplementedError("Subclasses must implement this method")
class SolverA(Solver):
    def expand(self):
        self.priorityQueue.put(PrioritizedItem(self.rootNode.priority_value(), self.rootNode))
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            if (top.state.isGoal()):
                return {
                    'node': top.path,
                    'node_number': self.node_number,
                    'cost': top.cost
                }
            self.node_number += 1
            # if (self.node_number % 100 != 0): 
            #     print ("\033[A                             \033[A")
            #     print(self.node_number)
            #   print(top.path)    
            for child in top.children():
                str_child = str(child.state)
                if (str_child not in self.reached.keys() or self.reached[str_child] > child.cost):
                    self.reached[str_child] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.priority_value(), child))
        print("FAILURE")
class SolverBFS(Solver):
    def expand(self):
        self.priorityQueue.put(PrioritizedItem(self.rootNode.cost, self.rootNode))
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            self.node_number += 1
            #if (self.node_number % 100 != 0): 
                #print ("\033[A                             \033[A")
                #print(self.node_number)
                #print(top.path)    
            for child in top.children():
                if (child.state.isGoal()):
                    return {
                        'node': child.path,
                        'node_number': self.node_number,
                        'cost': child.cost
                    }
                str_child = str(child.state)
                if (str_child not in self.reached.keys() or self.reached[str_child] > child.cost):
                    self.reached[str_child] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.cost, child))
        print("FAILURE")
class SolverDFS(Solver):
    def expand(self):
        self.priorityQueue.put(PrioritizedItem(-self.rootNode.cost, self.rootNode))
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            self.node_number += 1
            # if (self.node_number % 100 != 0): 
                # print ("\033[A                             \033[A")
                # print(self.node_number)
            #   print(top.path)    
            for child in top.children():
                if (child.state.isGoal()):
                    return {
                        'node': child.path,
                        'node_number': self.node_number,
                        'cost': child.cost
                    }
                str_child = str(child.state)
                if (str_child not in self.reached.keys() or self.reached[str_child] > child.cost):
                    self.reached[str_child] = child.cost
                    self.priorityQueue.put(PrioritizedItem(-child.cost, child))
        print("FAILURE")
class SolverUCS(Solver):
    def expand(self):
        self.priorityQueue.put(PrioritizedItem(self.rootNode.cost, self.rootNode))

        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            if (top.state.isGoal()):
                return {
                    'node': top.path,
                    'node_number': self.node_number,
                    'cost': top.cost
                }
            # print ("\033[A                             \033[A")
            self.node_number += 1
            # print(self.node_number)
            for child in top.children():
                str_child = str(child.state)
                if (str_child not in self.reached.keys() or self.reached[str_child] > child.cost):
                    self.reached[str_child] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.cost, child))
        print("FAILURE")

def output_result(result, input_file, type, output_file=None):
    if output_file is None:
        current_dir = os.path.dirname(os.path.realpath(input_file))
        if not os.path.exists(os.path.join(current_dir, 'output')):
            os.makedirs(os.path.join(current_dir, 'output'))
        base = os.path.basename(input_file)
        output_file = os.path.join(current_dir, 'output', base.split('.')[0] + f'_box_{type}.json')
        output_file = output_file.replace('*', 'star')

    print(result)
    print(f"Saving in {output_file}")
    with open(output_file, 'w') as f:
        result['time'] = time_end - time_start
        result['current_memory'] = current
        result['peak_memory'] = peak
        json.dump(result, f, indent=4)

import time
import argparse
time_start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file', required=True)
parser.add_argument('--type', help='type of search', required=True)
parser.add_argument('--output', help='output file', required=False, default=None)
args = parser.parse_args()
input_file = args.input
input_type = args.type
output_file = args.output

Input.readFile(input_file)
start_state = State(input_file, Input.inputAres[input_file], Input.inputStone[input_file])
if (input_type == 'BFS'):
    solver = SolverBFS(start_state)
elif (input_type == 'DFS'):
    solver = SolverDFS(start_state)
elif (input_type == 'A*'):
    solver = SolverA(start_state)
elif (input_type == 'UCS'):
    solver = SolverUCS(start_state)
result = solver.expand()

time_end = time.time()



print('time cost', time_end-time_start)
current, peak = tracemalloc.get_traced_memory()

print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage: {peak / 10**6:.2f} MB")

output_result(result, input_file, input_type, output_file)

# Stop tracing memory allocation
tracemalloc.stop()