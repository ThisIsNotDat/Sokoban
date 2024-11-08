import pygame

DESIRED_FPS = 24
SECOND_PER_FRAME = 1 / DESIRED_FPS
WIDTH = 800
HEIGHT = 400
SCREENRECT = pygame.Rect(0, 0, WIDTH, HEIGHT)
SCALE_FACTOR = 0.5
TILESIZE = round(32 * SCALE_FACTOR)
PEACH_HEIGHT = round(40 * SCALE_FACTOR)

# the smaller the faster the animation, 0.5 means 1 action takes 0.5 seconds
ANIMATION_SPEED = 0.5
