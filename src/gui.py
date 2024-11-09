import pygame_gui
import pygame

from src.fonts import FONTS


class TextBoxWithCaption(pygame_gui.elements.UIPanel):
    def __init__(self, rect, manager, text, caption, object_id, container=None, starting_height=0):
        super().__init__(rect, manager=manager, container=container)
        self.caption = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((0, 0), (rect.width, 20)),
            text=caption,
            manager=manager,
            container=self,
        )
        self.text = pygame_gui.elements.UITextBox(
            html_text=text,
            relative_rect=pygame.Rect((0, 20), (rect.width, 20)),
            manager=manager,
            object_id=object_id,
            container=self,
        )
        self.text.starting_height = starting_height
        # self.text.font = FONTS["default"]
        # self.caption.font = FONTS["default"]

    def get_text(self):
        return self.text.get_text()

    def set_text(self, text):
        self.text.set_text(text)

    def process_event(self, event):
        self.text.process_event(event)

    def update(self, time_delta):
        self.text.update(time_delta)

    def draw(self, surface):
        self.caption.draw(surface)
        self.text.draw(surface)

    def set_active(self, active):
        self.text.set_active(active)
