# Source: https://www.inspiredpython.com/course/create-tower-defense-game/tower-defense-game-basic-game-template#loading-assets
import importlib.resources
import pygame

from src.settings import TILESIZE

SPRITES = {
    "tileset": "tileset.png",
    "peach": "peach.png",
}

IMAGE_SPRITES = {}

MAP_SQUARES = {
    'wall': 2,
    'wall_bottom_left': 1,
    'wall_bottom': 0,
    'wall_bottom_paint': 7,
    'target_unreached': 3,
    'target_reached': 4,
    'box': 20,
    'water': 15,
    'water_left': 16,
    'grass': 5,
    'grass_left': 6,
}
MAP_SPRITES = {}


def load(module_path, name):
    return importlib.resources.path(module_path, name)


def import_image(asset_name: str):
    with load("src.assets.images", asset_name) as resource:
        return pygame.image.load(resource).convert_alpha()


# need to call this function at least once to load the sprites
def load_sprites():
    for sprite_index, sprite_name in SPRITES.items():
        img = import_image(sprite_name)
        print(f"Adding sprite: {sprite_index}")
        for flipped_x in (True, False):
            for flipped_y in (True, False):
                new_img = pygame.transform.flip(
                    img, flip_x=flipped_x, flip_y=flipped_y)
                print(f"Adding sprite: {flipped_x}, {
                      flipped_y}, {sprite_index}")
                IMAGE_SPRITES[(flipped_x, flipped_y, sprite_index)] = new_img

    tiles = Tilesheet(
        IMAGE_SPRITES[(False, False, "tileset")], TILESIZE, TILESIZE)
    for sprite_name, sprite_index in MAP_SQUARES.items():
        MAP_SPRITES[sprite_name] = tiles.get_tile(sprite_index)


class Tilesheet:
    def __init__(self, image, tile_width, tile_height):
        self.image = image
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tiles = []
        self.load_tiles()

    def size(self):
        return len(self.tiles)

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

    def draw(self, screen, index, x, y):
        screen.blit(self.get_tile(index),
                    (x * self.tile_width, y * self.tile_height))
