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
    if x.endswith(".txt"):
        content = open(test_dir + x, 'r').read()
        fileName = x[:-4]
        fileNames.append(fileName)
        inputWall[fileName] = []
        inputSwitch[fileName] = []
        inputStone[fileName] = {}

        lines = content.split('\n')
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



class Direction(Enum):
    Up = Vector([-1, 0])
    Down = Vector([1, 0])
    Left = Vector([0, -1])
    Right = Vector([0, 1])


class State:
    def __init__(self, name, ares, stones):
        self.ares = ares
        self.stones = stones.copy()
        self.name = name

    def __str__(self):
        res = self.name + '\n'
        for stone in self.stones:
            res += str(stone) + ": " + str(self.stones[stone]) + '\n'
        res += "ares: " + str(self.ares)
        return res
    
    def isGoal(self):
        for switch in inputSwitch[self.name]:
            if (switch not in self.stones.keys()):
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
        if ((self.ares + enum) in inputWall[self.name]):
            return NULL

        newState = State(self.name, self.ares + enum, self.stones)
        if ((self.ares + enum) not in self.stones.keys()):
            return newState

        if ((self.ares + 2*enum) in self.stones.keys() or (self.ares + 2*enum) in inputWall[self.name]):
            return NULL

        newState.stones.pop(newState.ares)
        
        newState.stones[newState.ares + enum] = self.stones[newState.ares]

        return newState

    def actionCost(self, enum):
        if ((self.ares + enum) in self.stones.keys() and (self.ares + 2*enum) not in self.stones.keys() and (self.ares + 2*enum) not in inputWall[self.name]):
            return self.stones[self.ares + enum] + 1
        return 1

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
        for direct in Direction:
            newState = self.state.neighbor(direct.value)
            if (newState != NULL):
                result.append(SearchNode(newState, self, self.state.actionCost(direct.value), direct))
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
            self.node_number += 1
            for child in top.children():
                if (child.state.isGoal()):
                    child.printTree()
                    print(child.path())
                    print("Node: ", self.node_number)
                    return
                if (str(child.state) not in self.reached.keys() or self.reached[str(child.state)] > child.cost):
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
                if (str(child.state) not in self.reached.keys() or self.reached[str(child.state)] > child.cost):
                    self.reached[str(child.state)] = child.cost
                    self.priorityQueue.put(PrioritizedItem(child.cost, child))
        print("FAILURE")


for fileName in fileNames:
    solver = SolverBFS(State(fileName, inputAres[fileName], inputStone[fileName]))
    solver.expand()
        