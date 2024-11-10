# Source: https://www.inspiredpython.com/course/create-tower-defense-game/tower-defense-game-finite-state-automata-state-machines
import enum
import pygame
import subprocess
import logging
import json
import pygame_gui
import os
import signal

from src.map import Map
from src.settings import DESIRED_FPS, SECOND_PER_FRAME, \
    WIDTH, HEIGHT, TEST_FOLDER


class StateList(enum.Enum):
    """
    Enum for the Game's State Machine. Every state represents a
    known game state for the game engine.
    """
    game_playing = "game_playing"
    main_menu = "main_menu"
    quitting = "quitting"
    initialized = "initialized"
    initializing = "initializing"


class StateError(Exception):
    """
    Raised if the game is in an unexpected game state at a point
    where we expect it to be in a different state. For instance, to
    start the game loop we must be initialized.
    """


class State:
    def __init__(self):
        self.next_state = None
        self.manager = pygame_gui.UIManager(
            (WIDTH, HEIGHT),
            theme_path="src/assets/theme.json",)
        print(f"Initializing State: {self.__class__.__name__}")

    def update(self, events, dt):
        """Update the state based on input events and game logic."""
        for event in events:
            if event.type == pygame.QUIT:
                self.next_state = StateList.quitting
            self.manager.process_events(event)
        self.manager.update(dt)

    def draw(self, screen):
        """Draw the state to the screen."""
        self.manager.draw_ui(screen)
        # print(f"draw gui {self.manager.get_root_container()}")

    def enter_state(self):
        """Optional: Initialize or reset the state when entering."""
        pass

    def exit_state(self):
        """Optional: Clean up the state when exiting."""
        pass

    def loop(self, screen):
        """The main game loop. This is where the game logic is executed."""
        clock = pygame.time.Clock()
        while self.next_state is None:
            # Calculate delta time in seconds
            delta_time = clock.tick(DESIRED_FPS) / 1000.0
            while delta_time - SECOND_PER_FRAME > 0:
                events = pygame.event.get()
                # Pass delta time to update method
                self.update(events, SECOND_PER_FRAME)
                delta_time -= SECOND_PER_FRAME
            screen.fill((0, 0, 0))  # Clear the screen
            self.draw(screen)
            pygame.display.set_caption(f"Sokoban - FPS: {clock.get_fps()}")
            pygame.display.flip()


class MainMenu(State):
    def __init__(self):
        super().__init__()
        self.go_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (WIDTH // 2 - 96, HEIGHT // 2 - 24), (192, 48)),
            text="GO",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )

    def update(self, events, dt):
        super().update(events, dt)
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RETURN:  # Start game on Enter
                    self.next_state = StateList.game_playing
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.go_button:
                        self.next_state = StateList.game_playing

    def draw(self, screen):
        super().draw(screen)
        # pygame.display.set_caption("Sokoban - Main Menu")


class GamePlay(State):

    def __init__(self):
        super().__init__()
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(
            format=format, level=logging.INFO, datefmt="%H:%M:%S")
        self.solving_process = None
        self.file_name = None
        self.test_paths = []
        self.current_map = 7
        self.load_test_paths()
        self.refresh = False
        self.solve_state = "unsolved"
        self.gCost = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 0), (192, 48)),
            text=f"Cost: {0:03}",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gStep = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 48), (192, 48)),
            text=f"Steps: {0:03}",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gPush = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 96), (192, 48)),
            text=f"Push: {0:03}",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gMapLabel = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 164), (192, 48)),
            text="Choose Map",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gChooseMap = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((0, 212), (192, 192)),
            item_list=[path.split("/")[-1] for path in self.test_paths],
            manager=self.manager,
            container=self.manager.get_root_container(),
            default_selection=self.test_paths[self.current_map].split("/")[-1],
        )
        self.gAlgoLabel = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 404), (192, 48)),
            text="Algorithm",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gAlgorithm = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((0, 452), (192, 32*4)),
            item_list=['DFS', 'BFS', 'A*', 'UCS'],
            manager=self.manager,
            container=self.manager.get_root_container(),
            default_selection="A*",
        )
        self.gMoveLabel = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 580), (192, 48)),
            text="Transition",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gMove = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((0, 628), (192, 33*2)),
            item_list=['Ares', 'Box'],
            manager=self.manager,
            container=self.manager.get_root_container(),
            default_selection="Ares",
        )
        self.gSolveButton = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((0, 414), (192, 48)),
            text="Solve",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gPlayButton = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((0, 704), (96, 48)),
            text="Play",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )
        self.gResetButton = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((96, 704), (96, 48)),
            text="Reset",
            manager=self.manager,
            container=self.manager.get_root_container(),
        )

    def load_test_paths(self):
        for file in os.listdir(TEST_FOLDER):
            if file.endswith(".txt"):
                self.test_paths.append(os.path.join(TEST_FOLDER, file))
                print(f"Found test file: {file}")

    def load_map(self, map_file):
        print("Loading map", map_file)
        with open(map_file, "r") as f:
            content = f.read().split("\n")
            weights = content[0].strip().split(" ")
            print("Weights:", weights)
            self.map = Map(content[1:-1], weights)
            # call a new thread to solve the map
        self.file_name = map_file.split("/")[-1].split(".")[0]
        self.map_file = map_file
        self.refresh = True
        self.change_solve_state("unsolved")
        print("Loaded map", map_file)

    def change_solve_state(self, state):
        print("Change solve state to", state)
        self.solve_state = state
        if self.solve_state == "unsolved":
            self.gSolveButton.show()
            self.refresh = True
            self.gAlgoLabel.hide()
            self.gAlgorithm.hide()
            self.gMoveLabel.hide()
            self.gMove.hide()
            self.gSolveButton.set_text("Solve")
            self.gPlayButton.hide()
            self.gResetButton.hide()
            # self.gSolveButton.rebuild()
            # self.gAlgorithm.rebuild()
            # self.gAlgoLabel.rebuild()
        elif self.solve_state == "solving":
            self.gSolveButton.set_text("Solving")
        elif self.solve_state == "finished":
            self.refresh = True
            self.gSolveButton.hide()
            self.gMoveLabel.show()
            self.gMove.show()
            self.gAlgoLabel.show()
            self.gAlgorithm.show()
            self.gPlayButton.show()
            self.gResetButton.show()
            # self.gSolveButton.rebuild()

    def updateGUI(self):
        self.gCost.set_text(f"Cost: {self.map.cost:03}")
        self.gStep.set_text(f"Steps: {self.map.steps:03}")
        self.gPush.set_text(f"Push: {self.map.push_weight:03}")

    def toggle_play(self):
        self.map.toggle_play()
        if self.map.playing:
            self.gPlayButton.set_text("Pause")
        else:
            self.gPlayButton.set_text("Play")

    def reset(self):
        self.map.reset()
        self.gPlayButton.set_text("Play")

    def process_event(self, event):
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:  # Start game on Enter
                self.next_state = StateList.quitting
            if self.solve_state == "finished":
                if event.key == pygame.K_r:
                    # R to reset
                    self.reset()
                elif event.key == pygame.K_SPACE:
                    self.toggle_play()
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.\
                    UI_SELECTION_LIST_NEW_SELECTION:
                if event.ui_element == self.gChooseMap:
                    self.change_map(os.path.join(
                        TEST_FOLDER, event.text))
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.gSolveButton:
                    self.solve_map(self.map_file)
                if event.ui_element == self.gPlayButton:
                    self.toggle_play()
                if event.ui_element == self.gResetButton:
                    self.reset()

    def update(self, events, dt):
        for event in events:
            self.process_event(event)
        if self.solving_process is not None:
            if self.solving_process.poll() is not None:
                logging.info("Solving process finished")
                self.read_solution()
                self.solving_process = None
                self.change_solve_state("finished")
        self.map.update(events, dt)
        self.updateGUI()
        super().update(events, dt)

    def change_map(self, new_map):
        if isinstance(new_map, int):
            self.current_map = new_map
        else:
            self.current_map = self.test_paths.index(new_map)
        self.load_map(self.test_paths[self.current_map])
        self.kill_solving_process()

    def draw(self, screen):
        if self.refresh:
            print("Refresh")
            self.map.draw(screen, full=True)
            self.refresh = False
        self.map.draw(screen)
        super().draw(screen)
        # pygame.display.set_caption("Sokoban - Visualization")

    def solve_map(self, map_file):
        if self.solve_state != "unsolved":
            logging.info("A solving process is running")
            return
        self.change_solve_state("solving")
        self.map.reset()
        logging.info(f"Solving map {map_file}")
        self.solving_process = subprocess.Popen(
            f'python search.py --input {map_file} --type A*', shell=True)

    def read_solution(self):
        with open(os.path.join(TEST_FOLDER, f"output/{self.file_name}_ares_Astar.json"), "r") as f:
            data = json.load(f)
            self.map.load_moves(data["node"])
            print("Solution:", data["node"])

    def loop(self, screen):
        """The main game loop. This is where the game logic is executed."""
        clock = pygame.time.Clock()
        self.load_map(self.test_paths[self.current_map])
        while self.next_state is None:
            # Calculate delta time in seconds
            delta_time = clock.tick(DESIRED_FPS) / 1000.0
            # while delta_time - SECOND_PER_FRAME >= 0:
            events = pygame.event.get()
            # Pass delta time to update method
            self.update(events, delta_time)
            # delta_time -= SECOND_PER_FRAME
            self.draw(screen)
            pygame.display.set_caption(
                f"Sokoban Visualization - FPS: {clock.get_fps()}")
            pygame.display.flip()

    def exit_state(self):
        super().exit_state()
        self.kill_solving_process()

    def kill_solving_process(self):
        if self.solving_process is not None:
            self.solving_process.terminate()
            self.solving_process.wait()
            self.solving_process = None
            print("Kill solve process")

    def enter_state(self):
        return super().enter_state()


class Initializing(State):
    pass


class Initialized(State):
    pass


class Quitting(State):
    def loop(self, screen):
        return
