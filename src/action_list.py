from src.settings import ANIMATION_SPEED


class ActionList:
    def __init__(self):
        self.actions = []
        self.playing = False
        self.current_action = 0
        self.seconds_per_action = ANIMATION_SPEED
        self.countdown = 0

    def add_to_last_group(self, action):
        if len(self.actions) == 0:
            self.actions.append([action])
        self.actions[-1].append(action)

    def update(self, dt):
        if not self.playing:
            return
        self.countdown -= dt
        if self.countdown <= 0:
            self.countdown = self.seconds_per_action
            self.execute(self.current_action)
            self.current_action += 1
            if self.current_action >= len(self.actions):
                self.playing = False

    def execute(self, id):
        assert id < len(self.actions) and id >= 0, "Invalid action id"
        for action in self.actions[id]:
            action[0].move(action[1])

    def pause(self):
        self.playing = False

    def play(self):
        self.playing = True

    def reset(self):
        self.playing = False
        self.current_action = 0
        self.countdown = self.seconds_per_action
        for action in self.actions[::-1]:
            for obj, direction in action:
                obj.reset(direction)
