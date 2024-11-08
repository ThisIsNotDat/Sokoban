# Source: https://www.inspiredpython.com/course/create-tower-defense-game/tower-defense-game-finite-state-automata-state-machines
import enum
import pygame

from src.map import Map


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
        print(f"Initializing State: {self.__class__.__name__}")

    def update(self, events, dt):
        """Update the state based on input events and game logic."""
        raise NotImplementedError

    def draw(self, screen):
        """Draw the state to the screen."""
        raise NotImplementedError

    def enter_state(self):
        """Optional: Initialize or reset the state when entering."""
        pass

    def exit_state(self):
        """Optional: Clean up the state when exiting."""
        pass


class MainMenu(State):
    def __init__(self):
        super().__init__()

    def update(self, events, dt):
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RETURN:  # Start game on Enter
                    self.next_state = StateList.game_playing

    def draw(self, screen):
        screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 74)
        text = font.render("Press Enter to Start", True, (255, 255, 255))
        screen.blit(text, (100, 250))
        pygame.display.set_caption("Sokoban - Main Menu")


class GamePlay(State):

    def __init__(self):
        super().__init__()

    def load_map(self, map_file):
        with open(map_file, "r") as f:
            content = f.read().split("\n")
            self.map = Map(content[1:-1])

    def update(self, events, dt):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Start game on Enter
                    self.next_state = StateList.quitting
        self.map.update(events, dt)

    def draw(self, screen):
        screen.fill((240, 165, 59))
        pygame.display.set_caption("Sokoban - Visualization")
        self.map.draw(screen)


class Initializing(State):
    pass


class Initialized(State):
    pass


class Quitting(State):
    pass
