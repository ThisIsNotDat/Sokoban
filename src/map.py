import pygame.sprite

from src.sprites import IMAGE_SPRITES, MAP_SPRITES, Tilesheet
from src.settings import WIDTH, HEIGHT, TILESIZE, ANIMATION_SPEED, PEACH_HEIGHT
from src.fonts import FONTS


class Map():
    def __init__(self, map_str, weights=None):
        self.map = map_str
        self.height = len(self.map)
        self.width = max(len(row) for row in self.map)
        self.padding_map()
        self.load_map_sprites(weights)
        self.playing = False
        self.moves = ""
        self.weights = weights

    @property
    def cost(self):
        return self.peach.cost

    @property
    def steps(self):
        return self.peach.steps

    @property
    def push_weight(self):
        return self.peach.push_weight

    def load_moves(self, moves):
        assert self.peach is not None, "Peach not found"
        self.peach.load_actions(moves)
        self.moves = moves

    def padding_map(self):
        for y in range(self.height):
            self.map[y] = self.map[y].ljust(self.width)
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

    def in_grass_zone(self, x, y):  # x, y is the coordinate in the map
        return x >= self.padding_left and x < self.width-self.padding_right \
            and y >= self.padding_top and y < self.height-self.padding_bottom

    def load_map_sprites(self, weights=None):
        self.map_sprites = pygame.sprite.LayeredUpdates()
        self.non_water_sprites = pygame.sprite.Group()
        self.target_sprites = pygame.sprite.Group()
        self.box_sprites = pygame.sprite.Group()
        self.peach = None
        for y, row in enumerate(self.map):
            for x, tile in enumerate(row):
                if tile == '#':
                    if y < self.height-1 and self.map[y+1][x] == '#':
                        wall = MapBlock(x, y, 'wall')
                    elif x < self.width-1 and y < self.height-1 \
                            and self.map[y][x+1] == '#' \
                            and self.map[y+1][x+1] == '#':
                        wall = MapBlock(x, y, 'wall_bottom_left')
                    else:
                        if self.my_random(x, y, 7) < 4:
                            wall = MapBlock(x, y, 'wall_bottom')
                        else:
                            wall = MapBlock(x, y, 'wall_bottom_paint')
                    self.map_sprites.add(wall)
                    self.non_water_sprites.add(wall)
                elif y >= self.padding_top \
                        and y < self.height-self.padding_bottom \
                        and x >= self.padding_left \
                        and x < self.width-self.padding_right:
                    if x < self.height-1 and self.map[y][x+1] == '#':
                        grass = MapBlock(x, y, 'grass_left')
                    else:
                        grass = MapBlock(x, y, 'grass')
                    self.map_sprites.add(grass)
                    self.non_water_sprites.add(grass)
                else:
                    if y > 0 and self.in_grass_zone(x, y-1):
                        self.map_sprites.add(MapBlock(x, y, 'water_top_grass'))
                    elif x < self.height-1 and self.in_grass_zone(x+1, y):
                        self.map_sprites.add(MapBlock(x, y, 'water_left'))
                    else:
                        self.map_sprites.add(MapBlock(x, y, 'water'))
        weight_id = 0
        for y, row in enumerate(self.map):
            for x, tile in enumerate(row):
                if tile == '$':
                    box_sprite = BoxSprite(x, y, int(weights[weight_id]))
                    weight_id += 1
                    self.box_sprites.add(box_sprite)
                    self.map_sprites.add(box_sprite, layer=1)
                elif tile == '.':
                    target_sprite = TargetSprite(x, y, False)
                    self.target_sprites.add(target_sprite)
                    self.map_sprites.add(target_sprite)
                elif tile == '*':
                    target_sprite = TargetSprite(x, y, True)
                    box_sprite = BoxSprite(x, y, int(weights[weight_id]))
                    weight_id += 1
                    self.target_sprites.add(target_sprite)
                    self.box_sprites.add(box_sprite)
                    self.map_sprites.add(target_sprite)
                    self.map_sprites.add(box_sprite, layer=1)
                elif tile == '@':
                    self.peach = Peach(x, y)
                    self.map_sprites.add(self.peach, layer=2)
                elif tile == '+':
                    self.peach = Peach(x, y)
                    target_sprite = TargetSprite(x, y, True)
                    self.target_sprites.add(target_sprite)
                    self.map_sprites.add(target_sprite)
                    self.map_sprites.add(self.peach, layer=2)

    def reset(self):
        self.load_map_sprites(self.weights)
        self.load_moves(self.moves)
        self.playing = False
        print("Reset")

    def draw(self, screen, full=False):
        if full:
            screen.fill((0, 0, 0))
            self.map_sprites.draw(screen)
        else:
            self.non_water_sprites.draw(screen)
            self.target_sprites.draw(screen)
            self.box_sprites.draw(screen)

        self.peach.draw(screen)

    def toggle_play(self):
        self.playing = not self.playing
        print(f"Playing: {self.playing}")

    def update(self, events, dt):
        if self.peach is not None and self.peach.solving:
            self.peach.update(self, dt)
            return
        if not self.playing:
            return

        for box in self.box_sprites:
            if box.velocity != (0, 0):
                box.update(self, dt)
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
    def __init__(self, x, y, weight=1):
        super().__init__(x, y, 'box')
        self.image = self.image.copy()
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE
        self.target_rect = self.rect
        self.speed = ANIMATION_SPEED  # second per animation
        self.velocity = (0, 0)  # pixel per second
        self.countdown = self.speed
        self.float_position = [self.rect.x, self.rect.y]
        self.add_weight(weight)

    def add_weight(self, weight):
        print(f"Adding weight {weight}")
        self.weight = weight
        # Set up font
        # Use None for default font or specify a font path
        font = FONTS['default']
        text_surface = font.render(
            str(weight), True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=self.image.get_rect().center)

        # Blit the text onto the image
        self.image.blit(text_surface, text_rect)

    def match_target(self):
        # print("Matching target Box")
        self.rect = self.target_rect
        self.float_position = [self.rect.x, self.rect.y]

    def update(self, map, dt):
        assert not self.collision(map)
        if self.velocity != (0, 0):
            self.countdown -= dt

        # print(f"dt = {dt}, countdown = {self.countdown}, rect = {
        #       self.rect}, target = {self.target_rect}")
        if self.countdown <= 0:
            self.countdown = self.speed
            self.match_target()
            self.velocity = (0, 0)
        else:
            self.float_position[0] += self.velocity[0] * dt
            self.float_position[1] += self.velocity[1] * dt
            self.rect.x = round(self.float_position[0])
            self.rect.y = round(self.float_position[1])

    def move(self, direction):
        if self.velocity != (0, 0):
            self.match_target()

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
            IMAGE_SPRITES[(False, False, 'peach')], TILESIZE, PEACH_HEIGHT)
        self.tile_idx = 1
        self.direction = "down"
        self.image = self.tiles.get_tile(self.tile_idx)
        self.rect = self.image.get_rect()
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE - TILESIZE // 2
        self.speed = ANIMATION_SPEED  # 1 tile per second = 32 pixels per s
        self.target_rect = self.rect
        self.velocity = (0, 0)  # pixel per second
        self.countdown = 0
        self.animation_dt = 0
        self.actions_buffer = []
        self.pushing = False
        self.float_positon = [self.rect.x, self.rect.y]
        self.cost = 0
        self.steps = 0
        self.push_weight = 0
        self.solving = False

    @property
    def name(self):
        return "peach"

    def load_actions(self, actions):
        self.actions_buffer = []
        for action in actions:
            if action.lower() == 'u':
                self.actions_buffer.append(('up', action.isupper()))
            elif action.lower() == 'd':
                self.actions_buffer.append(('down', action.isupper()))
            elif action.lower() == 'l':
                self.actions_buffer.append(('left', action.isupper()))
            elif action.lower() == 'r':
                self.actions_buffer.append(('right', action.isupper()))
        self.tile_idx = 2
        self.image = self.tiles.get_tile(self.tile_idx)

    def update(self, map, dt):
        if self.solving:
            self.animation_dt += dt
            if self.animation_dt >= self.speed:
                self.tile_idx = (self.tile_idx + 1) % 12
                # print(self.tile_idx)
                self.image = self.tiles.get_tile(self.tile_idx)
                self.animation_dt = 0
            return
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

    def match_target(self):
        self.rect = self.target_rect
        self.float_position = [self.rect.x, self.rect.y]

    def move(self, action, transition=True):
        if self.velocity != (0, 0):
            self.match_target()

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
            # print(f"Moving {direction} with speed {self.velocity}")
            self.tile_idx = self.next_tile(self.direction, self.tile_idx)
        else:
            self.match_target()
            self.tile_idx = self.stand_still()
        self.image = self.tiles.get_tile(self.tile_idx)
        self.cost += 1
        self.steps += 1
        if not self.pushing:
            self.push_weight = 0

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
        self.match_target()
        self.pushing = False
        if len(self.actions_buffer) > 0:
            self.move(self.actions_buffer.pop(0))
        else:
            self.tile_idx = self.stand_still()
            self.image = self.tiles.get_tile(self.tile_idx)
            self.velocity = (0, 0)
            self.pushing = False

    def move_and_change_tile(self, dt):
        self.float_position[0] += self.velocity[0] * dt
        self.float_position[1] += self.velocity[1] * dt
        self.rect.x = round(self.float_position[0])
        self.rect.y = round(self.float_position[1])
        if self.animation_dt >= self.speed/4:
            self.tile_idx = self.next_tile(self.direction, self.tile_idx)
            self.image = self.tiles.get_tile(self.tile_idx)
            self.animation_dt = 0

    def check_box_collision(self, map):
        for block in map.map_sprites:
            if block.name.startswith("wall") \
                    and self.collision_box().colliderect(block.rect):
                raise Exception("Peach hit the wall")
        for box in map.box_sprites:
            if self.collision_box().colliderect(box.rect):
                if self.pushing and box.velocity == (0, 0):
                    box.move(self.direction)
                    self.cost += box.weight
                    self.push_weight = box.weight
                    break

    def collision_box(self):
        return pygame.Rect(self.rect.x, self.rect.y + TILESIZE // 2,
                           TILESIZE, TILESIZE)
