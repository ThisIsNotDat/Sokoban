from src.sprites import load_sprites, SPRITES, IMAGE_SPRITES
from dataclasses import dataclass, field
from src.settings import SCREENRECT, DESIRED_FPS, SECOND_PER_FRAME
import pygame

from src.state import State, StateList, StateError, MainMenu, GamePlay, \
    Initializing, Quitting, Initialized


@dataclass
class GameLoop:
    pass


@dataclass
class SokobanGame:

    screen: pygame.Surface
    screen_rect: pygame.Rect
    fullscreen: bool
    state: State = field(init=False)
    bg_color: pygame.Color
    states: dict = field(init=False, default_factory=dict)

    @classmethod
    def create(cls, fullscreen=False):
        game = cls(
            screen=None,
            screen_rect=SCREENRECT,
            fullscreen=fullscreen,
            bg_color=pygame.Color("black"),
        )
        game.init()
        return game

    def set_state(self, new_state):
        print(f"Setting state to {new_state}")
        self.state = self.states[new_state]

    def change_state(self):
        if self.state.next_state:
            self.state.exit_state()
            self.set_state(self.state.next_state)
            if isinstance(self.state, GamePlay):
                self.state.load_map("TestCases/input_1.txt")
            self.state.enter_state()

    def update(self, events, dt):
        is_quitting = False
        for event in events:
            if event.type == pygame.QUIT:
                self.state.next_state = StateList.quitting
                is_quitting = True
        if not is_quitting:
            self.state.update(events, dt)
        self.change_state()

    def draw(self):
        self.state.draw(self.screen)

    def assert_state_is(self, *expected_states: str):
        """
        Asserts that the game engine is one of
        `expected_states`. If that assertions fails, raise
        `StateError`.
        """
        if self.state.class_name() not in expected_states:
            raise StateError(
                f"Expected the game state to be one of \
                    {expected_states} not {self.state}"
            )

    def loop(self):
        clock = pygame.time.Clock()
        while True:
            # Calculate delta time in seconds
            delta_time = clock.tick(DESIRED_FPS) / 1000.0
            while delta_time - SECOND_PER_FRAME > 0:
                events = pygame.event.get()
                # Pass delta time to update method
                self.update(events, SECOND_PER_FRAME)
                delta_time -= SECOND_PER_FRAME
            self.screen.fill((0, 0, 0))  # Clear the screen
            if self.is_quitting():
                break
            self.draw()
            pygame.display.set_caption(f"Sokoban - FPS: {clock.get_fps()}")
            pygame.display.flip()
        self.quit()

    def quit(self):
        pygame.quit()

    def start_game(self):
        self.set_state(StateList.main_menu)
        self.loop()

    def init(self):
        self.states = {
            StateList.main_menu: MainMenu(),
            StateList.game_playing: GamePlay(),
            StateList.initializing: Initializing(),
            StateList.initialized: Initialized(),
            StateList.quitting: Quitting(),
        }
        self.set_state(StateList.initializing)
        self.init_screen()
        load_sprites()
        self.set_state(StateList.initialized)

    def init_screen(self):
        pygame.init()
        window_style = pygame.HWSURFACE | pygame.DOUBLEBUF | (
            pygame.FULLSCREEN if self.fullscreen else 0)
        # We want 32 bits of color depth
        bit_depth = pygame.display.mode_ok(
            self.screen_rect.size, window_style, 32)
        screen = pygame.display.set_mode(
            self.screen_rect.size, window_style, bit_depth)
        screen.fill(self.bg_color)
        pygame.mixer.pre_init(
            frequency=44100,
            size=-16,
            # N.B.: 2 here means stereo, not the number of channels to
            # use in the mixer
            channels=2,
            buffer=256,
        )
        pygame.font.init()
        self.screen = screen

    def is_quitting(self):
        return isinstance(self.state, Quitting)
