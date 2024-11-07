# Source: https://www.inspiredpython.com/course/create-tower-defense-game/tower-defense-game-finite-state-automata-state-machines
import enum


class GameState(enum.Enum):
    """
    Enum for the Game's State Machine. Every state represents a
    known game state for the game engine.
    """

    # Unknown state, indicating possible error or misconfiguration.
    unknown = "unknown"
    # The state the game engine would rightly be set to before
    # anything is initialized or configured.
    initializing = "initializing"
    # The game engine is initialized: pygame is configured, the sprite
    # images are loaded, etc.
    initialized = "initialized"
    # The game engine is in map editing mode
    map_editing = "map_editing"
    # The game engine is in game playing mode
    game_playing = "game_playing"
    # The game engine is in the main menu
    main_menu = "main_menu"
    # The game engine is rendering the game ended screen.
    game_ended = "game_ended"
    # The game engine is exiting and is unwinding
    quitting = "quitting"


class StateError(Exception):
    """
    Raised if the game is in an unexpected game state at a point
    where we expect it to be in a different state. For instance, to
    start the game loop we must be initialized.
    """
