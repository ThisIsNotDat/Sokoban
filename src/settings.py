import pygame

DESIRED_FPS = 60
SCALE_FACTOR = 1

SECOND_PER_FRAME = 1 / DESIRED_FPS
WIDTH = round(1504 * SCALE_FACTOR)
HEIGHT = round(800 * SCALE_FACTOR)
SCREENRECT = pygame.Rect(0, 0, WIDTH, HEIGHT)
TILESIZE = round(32 * SCALE_FACTOR)
PEACH_HEIGHT = round(40 * SCALE_FACTOR)

# the smaller the faster the animation, 0.5 means 1 action takes 0.5 seconds
ANIMATION_SPEED = 0.25

TEST_FOLDER = "./maps/"
