# Source: https://www.inspiredpython.com/course/create-tower-defense-game/tower-defense-game-finite-state-automata-state-machines
import enum
import pygame
import subprocess
import logging
import json
import pygame_gui

from src.map import Map
from src.settings import DESIRED_FPS, SECOND_PER_FRAME, WIDTH, HEIGHT
from src.gui import TextBoxWithCaption


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
            (WIDTH, HEIGHT), theme_path="assets/theme.json")
        print(f"Initializing State: {self.__class__.__name__}")

    def update(self, events, dt):
        """Update the state based on input events and game logic."""
        for event in events:
            if event.type == pygame.QUIT:
                self.next_state = StateList.quitting
            self.manager.process_events(event)

    def draw(self, screen):
        """Draw the state to the screen."""
        self.manager.draw_ui(screen)

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

    def update(self, events, dt):
        super().update(events, dt)
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RETURN:  # Start game on Enter
                    self.next_state = StateList.game_playing

    def draw(self, screen):
        super().draw(screen)
        screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 74)
        text = font.render("Press Enter to Start", True, (255, 255, 255))
        screen.blit(text, (100, 250))
        # pygame.display.set_caption("Sokoban - Main Menu")


class GamePlay(State):

    def __init__(self):
        super().__init__()
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(
            format=format, level=logging.INFO, datefmt="%H:%M:%S")
        self.solving_process = None
        self.file_name = None
        self.gCost = TextBoxWithCaption(
            pygame.Rect((10, 10), (200, 50)),
            self.manager,
            "100",
            "gCost",
            "gCost",
        )

    def load_map(self, map_file):
        with open(map_file, "r") as f:
            content = f.read().split("\n")
            self.map = Map(content[1:-1])
            # call a new thread to solve the map
        self.file_name = map_file.split("/")[-1].split(".")[0]
        self.solve_map(map_file)

    def update(self, events, dt):
        super().update(events, dt)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Start game on Enter
                    self.next_state = StateList.quitting
        if self.solving_process is not None:
            if self.solving_process.poll() is not None:
                logging.info("Solving process finished")
                self.read_solution()
                self.solving_process = None
        self.map.update(events, dt)

    def draw(self, screen):
        super().draw(screen)
        # pygame.display.set_caption("Sokoban - Visualization")
        self.map.draw(screen)

    def solve_map(self, map_file):
        logging.info(f"Solving map {map_file}")
        self.solving_process = subprocess.Popen(
            f'python search.py --input {map_file} --type A*', shell=True)

    def read_solution(self):
        with open(f"./TestCases/output/{self.file_name}_ares_Astar.json", "r") as f:
            data = json.load(f)
            self.map.load_moves(data["node"])
            print("Solution:", data["node"])

    def loop(self, screen):
        """The main game loop. This is where the game logic is executed."""
        clock = pygame.time.Clock()
        self.map.draw(screen, full=True)
        while self.next_state is None:
            # Calculate delta time in seconds
            delta_time = clock.tick(DESIRED_FPS) / 1000.0
            while delta_time - SECOND_PER_FRAME > 0:
                events = pygame.event.get()
                # Pass delta time to update method
                self.update(events, SECOND_PER_FRAME)
                delta_time -= SECOND_PER_FRAME
            self.draw(screen)
            pygame.display.set_caption(f"Sokoban - FPS: {clock.get_fps()}")
            pygame.display.flip()


class Initializing(State):
    pass


class Initialized(State):
    pass


class Quitting(State):
    def loop(self, screen):
        return
