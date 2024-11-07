# Source: https://www.inspiredpython.com/course/create-tower-defense-game/tower-defense-game-basic-game-template#loading-assets
import importlib.resources
import pygame

SPRITES = {
    "tileset": "tileset.png",
    # "game_logo": "game_logo.png",
    # "wall": "wall.png",
    # "sokoban": "sokoban.png",
    # "background": "background.png",
    # "floor": "floor.png",
    # "switch": "switch.png",
}

IMAGE_SPRITES = {}


def load(module_path, name):
    return importlib.resources.path(module_path, name)


def import_image(asset_name: str):
    with load("sokoban.assets", asset_name) as resource:
        return pygame.image.load(resource).convert_alpha()


# need to call this function at least once to load the sprites
def load_sprites():
    for sprite_index, sprite_name in SPRITES.items():
        img = import_image(sprite_name)
        for flipped_x in (True, False):
            for flipped_y in (True, False):
                new_img = pygame.transform.flip(
                    img, flip_x=flipped_x, flip_y=flipped_y)
                IMAGE_SPRITES[(flipped_x, flipped_y, sprite_index)] = new_img


class Tilesheet:
    def __init__(self, image, tile_width, tile_height):
        self.image = image
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tiles = []
        self.load_tiles()

    def load_tiles(self):
        for y in range(0, self.image.get_height(), self.tile_height):
            for x in range(0, self.image.get_width(), self.tile_width):
                self.tiles.append(self.image.subsurface(
                    (x, y, self.tile_width, self.tile_height)))

    def get_tile(self, index):
        return self.tiles[index]

    def get_tile_by_pos(self, x, y):
        return self.tiles[y * (self.image.get_width() // self.tile_width) + x]

    def get_tile_count(self):
        return len(self.tiles)
