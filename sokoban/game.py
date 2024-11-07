from dataclasses import dataclass, field
import pygame

from sokoban.state import GameState, StateError
from sokoban.sprites import import_image, load_sprites, SPRITES, IMAGE_SPRITES

DESIRED_FPS = 30
WIDTH = 800
HEIGHT = 600
SCREENRECT = pygame.Rect(0, 0, WIDTH, HEIGHT)


@dataclass
class GameLoop:
    pass


@dataclass
class SokobanGame:

    screen: pygame.Surface
    screen_rect: pygame.Rect
    fullscreen: bool
    state: GameState
    bg_color: pygame.Color

    game_menu: GameLoop = field(init=False, default=None)

    @classmethod
    def create(cls, fullscreen=False):
        game = cls(
            screen=None,
            screen_rect=SCREENRECT,
            fullscreen=fullscreen,
            state=GameState.initializing,
            bg_color=pygame.Color("black"),
        )
        game.init()
        return game

    def set_state(self, new_state):
        print(f"Setting state to {new_state}")
        self.state = new_state

    def assert_state_is(self, *expected_states: GameState):
        """
        Asserts that the game engine is one of
        `expected_states`. If that assertions fails, raise
        `StateError`.
        """
        if self.state not in expected_states:
            raise StateError(
                f"Expected the game state to be one of \
                    {expected_states} not {self.state}"
            )

    def loop(self):
        while self.state != GameState.quitting:
            if self.state == GameState.main_menu:
                # pass control to the game menu's loop
                self.game_menu.loop()
            elif self.state == GameState.map_editing:
                # ... etc ...
                pass
            elif self.state == GameState.game_playing:
                # ... etc ...
                pass
        self.quit()

    def quit(self):
        pygame.quit()

    def start_game(self):
        self.assert_state_is(GameState.initialized)
        self.set_state(GameState.main_menu)
        self.loop()

    def init(self):
        self.assert_state_is(GameState.initializing)
        pygame.init()
        window_style = pygame.FULLSCREEN if self.fullscreen else 0
        # We want 32 bits of color depth
        bit_depth = pygame.display.mode_ok(
            self.screen_rect.size, window_style, 32)
        screen = pygame.display.set_mode(
            self.screen_rect.size, window_style, bit_depth)
        screen.fill(self.bg_color)
        pygame.mixer.pre_init(
            frequency=44100,
            size=32,
            # N.B.: 2 here means stereo, not the number of channels to
            # use in the mixer
            channels=2,
            buffer=512,
        )
        pygame.font.init()
        self.screen = screen
        self.game_menu = GameMenu(game=self)
        for sprite_index, sprite_name in SPRITES.items():
            img = import_image(sprite_name)
            for flipped_x in (True, False):
                for flipped_y in (True, False):
                    new_img = pygame.transform.flip(
                        img, flip_x=flipped_x, flip_y=flipped_y)
                    IMAGE_SPRITES[(flipped_x, flipped_y,
                                   sprite_index)] = new_img
        self.set_state(GameState.initialized)

    def load_sprites(self):
        pass


@dataclass
class GameLoop:
    game: SokobanGame

    def handle_events(self):
        """
        Sample event handler that ensures quit events and normal
        event loop processing takes place. Without this, the game will
        hang, and repaints by the operating system will not happen,
        causing the game window to hang.
        """
        for event in pygame.event.get():
            if (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ) or event.type == pygame.QUIT:
                self.set_state(GameState.quitting)
            # Delegate the event to a sub-event handler `handle_event`
            self.handle_event(event)

    def loop(self):
        while self.state != GameState.quitting:
            self.handle_events()

    def handle_event(self, event):
        """
        Handles a singular event, `event`.
        """

    # Convenient shortcuts.
    def set_state(self, new_state):
        self.game.set_state(new_state)

    @property
    def screen(self):
        return self.game.screen

    @property
    def state(self):
        return self.game.state


class GameMenu(GameLoop):
    def loop(self):
        clock = pygame.time.Clock()
        self.screen.blit(IMAGE_SPRITES[(False, False, "tileset")], (0, 0))
        while self.state == GameState.main_menu:
            self.handle_events()
            pygame.display.flip()
            pygame.display.set_caption(f"FPS {round(clock.get_fps())}")
            clock.tick(DESIRED_FPS)


class GameEditing(GameLoop):
    pass
