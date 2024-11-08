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
        self.playing = False

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

    def reset(self):
        self.load_map_sprites()
        self.playing = False

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
        for event in events:
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r:
                    self.reset()
                    print("Reset")
                elif event.key == pygame.K_SPACE:
                    self.playing = not self.playing
                    print(f"Playing: {self.playing}")

        if not self.playing:
            return

        self.map_sprites.update(self)
        self.target_sprites.update(self)
        self.box_sprites.update(self, dt)
        self.peach.update(self, dt)
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
        assert not self.collision(map)
        if self.velocity != (0, 0):
            self.countdown -= dt

        if self.countdown <= 0:
            self.countdown = self.speed
            self.rect = self.target_rect
            self.velocity = (0, 0)
        else:
            self.rect = self.rect.move(self.velocity[0] * dt,
                                       self.velocity[1] * dt)

    def move(self, direction):
        if self.velocity != (0, 0):
            self.rect = self.target_rect

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
        self.tile_idx = 1
        self.direction = "down"
        self.image = self.tiles.get_tile(self.tile_idx)
        self.rect = self.image.get_rect()
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE - TILESIZE // 2
        self.speed = ANIMATION_SPEED  # 1 tile per second = 32 pixels per second
        self.target_rect = self.rect
        self.velocity = (0, 0)  # pixel per second
        self.countdown = 0
        self.animation_dt = 0
        self.actions_buffer = []
        self.pushing = False

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

    def update(self, map, dt):
        if self.velocity:
            self.countdown -= dt
            self.animation_dt += dt
        if self.countdown <= 0:
            self.reset_and_next_action()
        elif self.velocity != (0, 0):
            self.move_and_change_tile(dt)
            self.check_box_collision(map)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def next_tile(self, direction, tile_idx):
        if direction == "down":
            return (tile_idx + 1) % 3
        elif direction == "left":
            return (tile_idx + 1) % 3 + 3
        elif direction == "right":
            return (tile_idx + 1) % 3 + 6
        else:
            return (tile_idx + 1) % 3 + 9

    def move(self, action, transition=True):
        if self.velocity != (0, 0):
            self.rect = self.target_rect

        direction, pushing = action
        self.direction = direction
        self.pushing = pushing
        if direction == "down":
            self.target_rect = self.rect.move(0, TILESIZE)
        elif direction == "left":
            self.target_rect = self.rect.move(-TILESIZE, 0)
        elif direction == "right":
            self.target_rect = self.rect.move(TILESIZE, 0)
        else:
            self.target_rect = self.rect.move(0, -TILESIZE)
        if transition:
            self.velocity = ((self.target_rect.x - self.rect.x) / self.speed,
                             (self.target_rect.y - self.rect.y) / self.speed)
            self.tile_idx = self.next_tile(self.direction, self.tile_idx)
        else:
            self.rect = self.target_rect
            self.tile_idx = self.stand_still()
        self.image = self.tiles.get_tile(self.tile_idx)

    def stand_still(self):
        if self.direction == "down":
            return 1
        elif self.direction == "left":
            return 4
        elif self.direction == "right":
            return 7
        else:
            return 10

    def is_moving(self):
        return self.rect != self.target_rect

    def reset_and_next_action(self):
        self.countdown = self.speed
        self.rect = self.target_rect
        if len(self.actions_buffer) > 0:
            self.move(self.actions_buffer.pop(0))
        else:
            self.tile_idx = self.stand_still()
            self.image = self.tiles.get_tile(self.tile_idx)
            self.velocity = (0, 0)

    def move_and_change_tile(self, dt):
        self.rect = self.rect.move(self.velocity[0] * dt,
                                   self.velocity[1] * dt)
        if self.animation_dt >= self.speed/6:
            self.tile_idx = self.next_tile(self.direction, self.tile_idx)
            self.image = self.tiles.get_tile(self.tile_idx)
            self.animation_dt = 0

    def check_box_collision(self, map):
        for box in map.box_sprites:
            if pygame.sprite.collide_rect(self, box):
                if self.pushing and box.velocity == (0, 0):
                    box.move(self.direction)
                    break
