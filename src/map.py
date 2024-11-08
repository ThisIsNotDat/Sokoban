import pygame.sprite

from src.sprites import IMAGE_SPRITES, MAP_SPRITES, Tilesheet
from src.settings import WIDTH, HEIGHT, TILESIZE, ANIMATION_SPEED


class Map():
    def __init__(self, map_str):
        self.map = map_str
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.padding_map()
        self.load_map_sprites()

    def padding_map(self):
        self.padding_left = (WIDTH - self.width * TILESIZE) // TILESIZE // 2
        self.padding_top = (HEIGHT - self.height * TILESIZE) // TILESIZE // 2
        self.padding_right = (WIDTH - self.width *
                              TILESIZE) // TILESIZE - self.padding_left
        self.padding_bottom = (HEIGHT - self.height *
                               TILESIZE) // TILESIZE - self.padding_top
        print(f"top: {self.padding_top}, bottom: {
              self.padding_bottom}, height: {self.height}")
        for y in range(len(self.map)):
            self.map[y] = ' '*self.padding_left + \
                self.map[y] + ' '*self.padding_right
        self.width = len(self.map[0])
        for _ in range(self.padding_top):
            self.map.insert(0, ' ' * WIDTH)
        for _ in range(self.padding_bottom):
            self.map.append(' ' * WIDTH)
        self.height = len(self.map)
        print(f"Map size: {self.width}x{self.height}")

    def load_map_sprites(self):
        self.map_sprites = pygame.sprite.Group()
        self.target_sprites = pygame.sprite.Group()
        self.box_sprites = pygame.sprite.Group()
        self.peach = None
        for y, row in enumerate(self.map):
            for x, tile in enumerate(row):
                if tile == '#':
                    if y < self.height-1 and self.map[y+1][x] == '#':
                        self.map_sprites.add(MapBlock(x, y, 'wall'))
                    elif x < self.width-1 and y < self.height-1 \
                            and self.map[y][x+1] == '#' \
                            and self.map[y+1][x+1] == '#':
                        self.map_sprites.add(
                            MapBlock(x, y, 'wall_bottom_left'))
                    else:
                        if self.my_random(x, y, 7) < 4:
                            self.map_sprites.add(MapBlock(x, y, 'wall_bottom'))
                        else:
                            self.map_sprites.add(
                                MapBlock(x, y, 'wall_bottom_paint'))
                elif y >= self.padding_top \
                        and y < self.height-self.padding_bottom \
                        and x >= self.padding_left \
                        and x < self.width-self.padding_right:
                    if x < self.height-1 and self.map[y][x+1] == '#':
                        self.map_sprites.add(MapBlock(x, y, 'grass_left'))
                    else:
                        self.map_sprites.add(MapBlock(x, y, 'grass'))
                else:
                    if x < self.height-1 and self.map[y][x+1] == '#':
                        self.map_sprites.add(MapBlock(x, y, 'water_left'))
                    else:
                        self.map_sprites.add(MapBlock(x, y, 'water'))
        for y, row in enumerate(self.map):
            for x, tile in enumerate(row):
                if tile == '$':
                    self.box_sprites.add(BoxSprite(x, y))
                elif tile == '.':
                    self.target_sprites.add(TargetSprite(x, y, False))
                elif tile == '*':
                    self.target_sprites.add(TargetSprite(x, y, True))
                    self.box_sprites.add(BoxSprite(x, y))
                elif tile == '@':
                    self.peach = Peach(x, y)
                elif tile == '+':
                    self.peach = Peach(x, y)
                    self.target_sprites.add(TargetSprite(x, y, False))
        self.peach.load_actions("rrurruuurrDDRluulldRurrrdrddlUllU")

    def draw(self, screen):
        self.map_sprites.draw(screen)
        self.target_sprites.draw(screen)
        self.box_sprites.draw(screen)
        self.peach.draw(screen)
        # for y, row in enumerate(self.map):
        #     for x, tile in enumerate(row):
        #         if tile == '$':
        #             self.tiles.draw(screen, 20, x, y)

    def update(self, events, dt):
        self.map_sprites.update(self)
        self.target_sprites.update(self)
        self.box_sprites.update(self, dt)
        self.peach.update(dt)
        for target in self.target_sprites:
            target.set_reached(False)
            for box in self.box_sprites:
                if box.rect.x == target.rect.x and box.rect.y == target.rect.y:
                    target.set_reached(True)
                    break

    def my_random(self, x, y, lim):
        return (x*2333+y)*13 % lim


class MapBlock(pygame.sprite.Sprite):
    def __init__(self, x, y, block_name):
        super().__init__()
        # Load image with transparency
        self.image = MAP_SPRITES[block_name]
        # set position
        self.rect = self.image.get_rect()
        self.rect.x = x*TILESIZE
        self.rect.y = y*TILESIZE
        self._name = block_name

    @property
    def name(self):
        return self._name


class BoxSprite(MapBlock):
    def __init__(self, x, y):
        super().__init__(x, y, 'box')
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE
        self.target_rect = self.rect
        self.speed = ANIMATION_SPEED  # second per animation
        self.velocity = (0, 0)  # pixel per second
        self.countdown = self.speed

    def update(self, map, dt):
        # assert not self.collision(map)
        self.countdown -= dt
        print(f"countdown: {self.countdown}")
        if self.countdown <= 0:
            self.countdown = self.speed
            self.rect = self.target_rect
            self.velocity = (0, 0)
            self.move('down')
        else:
            self.rect = self.rect.move(self.velocity[0] * dt,
                                       self.velocity[1] * dt)

    def move(self, direction):
        if direction == 'up':
            self.target_rect = self.rect.move(0, -TILESIZE)
        elif direction == 'down':
            self.target_rect = self.rect.move(0, TILESIZE)
        elif direction == 'left':
            self.target_rect = self.rect.move(-TILESIZE, 0)
        else:
            self.target_rect = self.rect.move(TILESIZE, 0)
        self.velocity = ((self.target_rect.x - self.rect.x) / self.speed,
                         (self.target_rect.y - self.rect.y) / self.speed)

    def collision(self, map):
        for block in map.map_sprites:
            if block.name.startswith("wall") \
                    and pygame.sprite.collide_rect(self, block):
                return True

        for box in map.box_sprites:
            if box != self and pygame.sprite.collide_rect(self, box):
                return True
        return False


class TargetSprite(MapBlock):
    def __init__(self, x, y, reached=False):
        super().__init__(x, y,
                         'target_unreached' if not reached
                         else 'target_reached')

    def set_reached(self, reached):
        if reached:
            self.image = MAP_SPRITES['target_reached']
        else:
            self.image = MAP_SPRITES['target_unreached']


class Peach(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.tiles = Tilesheet(
            IMAGE_SPRITES[(False, False, 'peach')], TILESIZE, 40)
        self.tile_idx = 0
        self.direction = 0  # 0: down, 1: left, 2: right, 3: up
        self.image = self.tiles.get_tile(self.tile_idx)
        self.rect = self.image.get_rect()
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE - TILESIZE // 2
        self.speed = ANIMATION_SPEED  # 1 tile per second = 32 pixels per second
        self.target_rect = self.rect
        self.moved = (0, 0)
        self.actions_buffer = []

    def load_actions(self, actions):
        for action in actions:
            if action.lower() == 'u':
                self.actions_buffer.append(('up', action.isupper()))
            elif action.lower() == 'd':
                self.actions_buffer.append(('down', action.isupper()))
            elif action.lower() == 'l':
                self.actions_buffer.append(('left', action.isupper()))
            elif action.lower() == 'r':
                self.actions_buffer.append(('right', action.isupper()))

    def update(self, dt):
        new_rect = self.rect.move(self.moved)
        if new_rect.x == self.target_rect.x \
                and new_rect.y == self.target_rect.y:
            if self.moved != (0, 0):
                self.rect = self.target_rect
                self.tile_idx = self.stand_still()
                self.image = self.tiles.get_tile(self.tile_idx)
                self.moved = (0, 0)
            elif self.actions_buffer:
                self.move(self.actions_buffer.pop(0))
            # print(f"stand still: {self.tile_idx}")
        else:
            old_moved = self.moved
            self.moved = (self.moved[0] +
                          round((self.target_rect.x - self.rect.x)
                                * self.speed * dt),
                          self.moved[1] +
                          round((self.target_rect.y - self.rect.y)
                                * self.speed * dt))
            if (old_moved[0] + old_moved[1]) // 4 \
                    != (self.moved[0] + self.moved[1]) // 4:
                self.tile_idx = self.next_tile(self.direction, self.tile_idx)
                self.image = self.tiles.get_tile(self.tile_idx)

    def draw(self, screen):
        screen.blit(self.image, self.rect.move(self.moved))

    def next_tile(self, direction, tile_idx):
        if direction == 0:
            return (tile_idx + 1) % 3
        elif direction == 1:
            return (tile_idx + 1) % 3 + 3
        elif direction == 2:
            return (tile_idx + 1) % 3 + 6
        else:
            return (tile_idx + 1) % 3 + 9

    def move(self, direction):
        action, pushing = direction
        if action == "down":
            self.direction = 0
            self.target_rect = self.rect.move(0, TILESIZE)
        elif action == "left":
            self.direction = 1
            self.target_rect = self.rect.move(-TILESIZE, 0)
        elif action == "right":
            self.direction = 2
            self.target_rect = self.rect.move(TILESIZE, 0)
        else:
            self.direction = 3
            self.target_rect = self.rect.move(0, -TILESIZE)
        self.tile_idx = self.next_tile(self.direction, self.tile_idx)
        self.image = self.tiles.get_tile(self.tile_idx)

    def stand_still(self):
        if self.direction == 0:
            return 1
        elif self.direction == 1:
            return 4
        elif self.direction == 2:
            return 7
        else:
            return 10

    def is_moving(self):
        return self.rect != self.target_rect
