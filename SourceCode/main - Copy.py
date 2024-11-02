from asyncio.windows_events import NULL
from multiprocessing import parent_process
import os
from enum import Enum
import math
from queue import PriorityQueue
from queue import Queue
from typing import Tuple
from memory_profiler import profile

from click import File

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
    
    def magnitude(self):
        return math.sqrt(sum(x ** 2 for x in self.components))
    
    def __eq__(self, other):
        return self.components == other.components
    
    def __len__(self):
        return len(self.components)
    
    def __getitem__(self, index):
        return self.components[index]
    def __lt__(self, other):
        for i, num in enumerate(self.components):
            if num == other.components[i]:
                continue
            return (num < other.components[i]) 
        return False

inputWall = {}
inputSwitch = {}
inputStone = {}
inputAres = {}
fileNames = []

wall = []
lineCnt = 0
wordSet = set()
wordSetCnt = 0
wordMap = {}

test_dir = '../TestCases/'
for x in os.listdir(test_dir):
    if x.endswith("3.txt"):
        content = open(test_dir + x, 'r', encoding='utf-16-le').read()
        if content.startswith('\ufeff'):
            content = content[1:]
        fileName = x[:-4]
        fileNames.append(fileName)
        inputWall[fileName] = {}
        inputSwitch[fileName] = {}
        inputStone[fileName] = {}

        lines = content.split('\n')
        print(lines)
        weightQueue = Queue()

        for number in lines[0].split(' '):
            print(number)
            if (number.isnumeric()):
               weightQueue.put(int(number))

        for i in range(1, len(lines)):
            print(lines[i])
            for j in range(0, len(lines[i])):
                if (lines[i][j] == '#'):
                   inputWall[fileName][Vector([i - 1,j])] = None
                elif (lines[i][j] == '$'):
                    inputStone[fileName][Vector([i - 1, j])] = weightQueue.get()
                elif (lines[i][j] == '@'):
                    inputAres[fileName] = Vector([i - 1,j])
                elif (lines[i][j] == '.'):
                    inputSwitch[fileName][Vector([i - 1,j])] = None
                elif (lines[i][j] == '*'):
                    inputStone[fileName][Vector([i - 1, j])] = weightQueue.get()
                    inputSwitch[fileName][Vector([i - 1,j])] = None
                elif (lines[i][j] == '+'):
                    inputSwitch[fileName][Vector([i - 1,j])] = None
                    inputAres[fileName] = Vector([i - 1,j])

        print("Input Successful")



class Direction(Enum):
    Up = Vector([-1, 0])
    Down = Vector([1, 0])
    Left = Vector([0, -1])
    Right = Vector([0, 1])


class State:
    def __init__(self, name, ares, stones):
        self.ares = ares
        self.stones = dict(sorted(stones.copy().items()))
        self.dist = {}
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
        for stone in self.stones:
            for direct in Direction:
                newState = self.neighbor(stone, direct.value)
                if (newState != NULL):
                    result.append(newState)
        return result

    def neighbor(self, stone, enum):
        if ((stone + enum) in inputWall[self.name] or (stone - enum) in inputWall[self.name] or (stone + enum) in self.stones or (stone + enum) in self.stones or (stone - enum) not in self.dist):
            return NULL

        newStateStone = self.stones.copy()
        newStateStone.pop(stone)
        newStateStone[stone + enum] = self.stones[stone]

        newState = State(self.name, stone, newStateStone)

        return newState

    def playerNeighbors(self):
        queue = PriorityQueue()
        self.dist[self.ares] = 0
        queue.put(PrioritizedItem(0, self.ares))
        while (not queue.empty()):
            top = queue.get()
            for direct in Direction:
                newSpace = top.getItem() + direct.value
                if (newSpace in self.stones or newSpace in inputWall[self.name]):
                    continue
                if (newSpace not in self.dist or self.dist[newSpace] > top.getPriority() + 1):
                   #     print(queue.qsize(), newSpace)
                    self.dist[newSpace] = top.getPriority() + 1
                    queue.put(PrioritizedItem(top.getPriority() + 1, newSpace))

    def actionCost(self, stone, enum):
        return self.dist[stone - enum] + self.stones[stone] + 1

    def heuristic(self):
        return 0

class SearchNode:
    def __init__(self, state, parent, cost, direct):
        self.state = state
        self.parent = parent
        self.cost = cost
        if (parent != NULL):
           self.cost += parent.cost
           self.direction = str(direct)[10]
           if (cost == 1):
               self.direction = self.direction.lower()

    def priority_value(self):
        return self.state.heuristic() + self.cost

    def children(self):
        result = []
        for stone in self.state.stones:
            for direct in Direction:
                newState = self.state.neighbor(stone, direct.value)
                if (newState != NULL):
                    result.append(SearchNode(newState, self, self.state.actionCost(stone, direct.value), direct))
        return result

    def printTree(self):
        if (self.parent != NULL):
            self.parent.printTree()
        print(self.state)
        print("cost: ", self.cost)
    def path(self):
        if (self.parent == NULL):
            return ""
        return self.parent.path() + self.direction
            

class PrioritizedItem:
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item

    def __lt__(self, other):
        return self.priority < other.priority
    def getItem(self):
        return self.item
    def getPriority(self):
        return self.priority

@profile
class SolverBFS:
    def __init__(self, state):
        self.rootNode = SearchNode(state, NULL, 0, NULL)
        self.reached = {}
        self.reached[str(state)] = 0
        self.node_number = 0

        self.priorityQueue = PriorityQueue()
        self.priorityQueue.put(PrioritizedItem(self.rootNode.priority_value(), self.rootNode))

    def expand(self):
        while (not self.priorityQueue.empty()):
            top = self.priorityQueue.get().getItem()
            print ("\033[A                             \033[A")
            self.node_number += 1
            print(self.node_number)
            top.state.playerNeighbors()

            for child in top.children():
                if (child.state.isGoal()):
                    child.printTree()
                    print(child.path())
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
                print(top.path())
                print("Node: ", self.node_number)
                return
            self.node_number += 1
            for child in top.children():
                if (str(child.state) not in self.reached or self.reached[str(child.state)] > child.cost):
                    self.reached[str(child.state)] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.cost, child))
        print("FAILURE")


for fileName in fileNames:
    solver = SolverBFS(State(fileName, inputAres[fileName], inputStone[fileName]))
    solver.expand()
        