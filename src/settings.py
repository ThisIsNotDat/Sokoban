import pygame

DESIRED_FPS = 60
SECOND_PER_FRAME = 1 / DESIRED_FPS
WIDTH = 1504
HEIGHT = 800
SCREENRECT = pygame.Rect(0, 0, WIDTH, HEIGHT)
TILESIZE = 32

# the smaller the faster the animation, 0.5 means 1 action takes 0.5 seconds
ANIMATION_SPEED = 0.5
