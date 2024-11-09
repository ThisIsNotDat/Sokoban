import pygame

FONT_PATHS = {
    "default": "src/assets/fonts/VCR_OSD_MONO_1.001.ttf",
}

FONTS = {}


def load_fonts():
    for font_name, font_path in FONT_PATHS.items():
        print(f"Loading font: {font_name}")
        FONTS[font_name] = pygame.font.Font(font_path, 16)
