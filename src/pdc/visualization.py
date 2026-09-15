from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pdc.config import VisualizationConfig
from pdc.pipeline_types import PersonAnalysis

Box = Tuple[int, int, int, int]

PERSON_COLOR = (245, 248, 252, 225)
PERSON_FILL = (245, 248, 252, 12)
FACE_COLOR = (91, 214, 255, 255)
BADGE_BORDER = (255, 255, 255, 42)
BADGE_ACCENT = (91, 214, 255, 235)
BADGE_BACKGROUND = (11, 15, 23, 226)
BADGE_SHADOW = (0, 0, 0, 88)
TEXT_COLOR = (255, 255, 255, 255)
FEMALE_COLOR = (255, 138, 176, 255)
MALE_COLOR = (111, 197, 255, 255)
_FONT_CACHE = {}

_FEMALE_TOKENS = {"female", "woman", "girl", "lady"}
_MALE_TOKENS = {"male", "man", "boy", "guy"}


def _font_has_glyphs(font: ImageFont.FreeTypeFont, text: str) -> bool:
    for char in text:
        if char == " ":
            continue
        if font.getmask(char).getbbox() is None:
            return False
    return True


def get_visualization_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    cache_key = (size, bold)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]
    font_paths = (
        [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, size)
            break
        except Exception:
            continue
    font = font or ImageFont.load_default()
    _FONT_CACHE[cache_key] = font
    return font


def resolve_gender_symbol(label: str) -> Optional[bool]:
    tokens = set(re.findall(r"[a-z]+", label.strip().lower()))
    if tokens & _FEMALE_TOKENS:
        return True
    if tokens & _MALE_TOKENS:
        return False
    return None


def get_age_bin(age: float, age_bins: dict) -> str:
    age = float(age)
    for label, ceiling in age_bins.items():
        if ceiling is None or age <= ceiling:
            return label
    return list(age_bins.keys())[-1]


def rectangles_overlap(box_a: Box, box_b: Box, margin: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 + margin < bx1 or ax1 - margin > bx2 or ay2 + margin < by1 or ay1 - margin > by2)


def is_inside_image(box: Box, image_width: int, image_height: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 >= 0 and y1 >= 0 and x2 <= image_width and y2 <= image_height


def draw_rounded_box(draw: ImageDraw.ImageDraw, box: Box, radius: int, outline, width: int) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, outline=outline, width=width)


def draw_face_brackets(draw: ImageDraw.ImageDraw, box: Box, color, width: int = 2, corner_length: int = 14, radius: int = 5) -> None:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    box_width, box_height = x2 - x1, y2 - y1
    corner_length = int(min(corner_length, box_width * 0.30, box_height * 0.30))
    corner_length = max(corner_length, 5)
    segments = [
        [(x1 + radius, y1), (x1 + corner_length, y1)],
        [(x1, y1 + radius), (x1, y1 + corner_length)],
        [(x2 - corner_length, y1), (x2 - radius, y1)],
        [(x2, y1 + radius), (x2, y1 + corner_length)],
        [(x1, y2 - corner_length), (x1, y2 - radius)],
        [(x1 + radius, y2), (x1 + corner_length, y2)],
        [(x2 - corner_length, y2), (x2 - radius, y2)],
        [(x2, y2 - corner_length), (x2, y2 - radius)],
    ]
    for segment in segments:
        draw.line(segment, fill=color, width=width)


def draw_female_symbol(draw: ImageDraw.ImageDraw, center_x: float, center_y: float, radius: float, color, width: int) -> None:
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        outline=color,
        width=width,
    )
    stem_bottom = center_y + radius * 1.85
    draw.line([(center_x, center_y + radius * 0.85), (center_x, stem_bottom)], fill=color, width=width)
    cross_y = center_y + radius * 1.5
    cross_half = radius * 0.72
    draw.line([(center_x - cross_half, cross_y), (center_x + cross_half, cross_y)], fill=color, width=width)


def draw_male_symbol(draw: ImageDraw.ImageDraw, center_x: float, center_y: float, radius: float, color, width: int) -> None:
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        outline=color,
        width=width,
    )
    diagonal = math.cos(math.radians(45))
    shaft_start = (center_x + radius * 0.88 * diagonal, center_y - radius * 0.88 * diagonal)
    shaft_end = (center_x + radius * 2.0 * diagonal, center_y - radius * 2.0 * diagonal)
    draw.line([shaft_start, shaft_end], fill=color, width=width)
    arrow_length = radius * 0.5
    for angle in (0, 90):
        rad = math.radians(45 + angle)
        tip = (
            shaft_end[0] + arrow_length * math.cos(rad),
            shaft_end[1] - arrow_length * math.sin(rad),
        )
        draw.line([shaft_end, tip], fill=color, width=width)


def find_best_badge_position(person_box: Box, face_box: Optional[Box], badge_width: int, badge_height: int, image_width: int, image_height: int, margin: int = 6) -> Tuple[int, int]:
    px1, py1, px2, py2 = person_box
    candidates = [
        (px1, py1 - badge_height - margin),
        (px2 - badge_width, py1 - badge_height - margin),
        (px1, py2 + margin),
        (px2 - badge_width, py2 + margin),
        (px1 + margin, py1 + margin),
        (px2 - badge_width - margin, py1 + margin),
    ]
    for x, y in candidates:
        candidate_box = (x, y, x + badge_width, y + badge_height)
        if not is_inside_image(candidate_box, image_width, image_height):
            continue
        if face_box is not None and rectangles_overlap(candidate_box, face_box, margin=4):
            continue
        return x, y
    for x, y in candidates:
        candidate_box = (x, y, x + badge_width, y + badge_height)
        if is_inside_image(candidate_box, image_width, image_height):
            return x, y
    return (max(0, min(px1, image_width - badge_width)), max(0, min(py1, image_height - badge_height)))


def render_annotations(image_bgr: np.ndarray, analysis_results: List[PersonAnalysis], config: VisualizationConfig) -> Image.Image:
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(canvas, 'RGBA')

    image_width, image_height = canvas.size
    base_dim = min(image_width, image_height)
    scale = max(0.8, min(3.2, base_dim / 720.0))

    badge_font_size = int(round(max(13, min(36, base_dim * 0.024))))
    badge_font = get_visualization_font(badge_font_size, bold=True)

    person_line_width = max(2, min(8, int(round(config.person_line_width * scale))))
    face_line_width = max(2, min(8, int(round(config.face_line_width * scale))))

    person_radius = max(10, int(round(base_dim * 0.012)))
    shadow_offset = max(2, int(round(2 * scale)))
    badge_margin = max(config.badge_margin, int(round(base_dim * 0.008)))

    accent_width = max(3, int(round(badge_font_size * 0.14)))
    symbol_radius = badge_font_size * 0.34
    symbol_stroke = max(2, int(round(symbol_radius * 0.3)))
    padding_y = max(8, int(round(badge_font_size * 0.44)))
    side_pad = int(round(badge_font_size * 0.42))
    symbol_gap = int(round(badge_font_size * 0.32))
    text_gap = int(round(badge_font_size * 0.3))
    right_pad = int(round(badge_font_size * 0.58))

    separator = "·" if _font_has_glyphs(badge_font, "·") else "-"

    for result in analysis_results:
        person_box = result.person.box

        x1, y1, x2, y2 = person_box
        draw.rounded_rectangle([x1, y1, x2, y2], radius=person_radius, fill=PERSON_FILL)
        draw_rounded_box(
            draw=draw,
            box=person_box,
            radius=person_radius,
            outline=PERSON_COLOR,
            width=person_line_width,
        )

        face_box = result.face.box if result.face is not None else None

        if face_box is not None:
            draw_face_brackets(
                draw=draw,
                box=face_box,
                color=FACE_COLOR,
                width=face_line_width,
                corner_length=max(10, int(round(base_dim * 0.018))),
                radius=max(4, int(round(base_dim * 0.004))),
            )

        show_symbol = False
        if result.analysis is not None:
            raw_label = config.gender_labels.get(result.analysis.category_index, f'Class {result.analysis.category_index}')
            gender_symbol = resolve_gender_symbol(raw_label)
            display_age = result.smoothed_age if result.smoothed_age is not None else result.analysis.estimated_age
            age_label = get_age_bin(display_age, config.age_bins)
            if gender_symbol is not None:
                badge_text = f"{display_age:.0f}" if config.age_display_mode == "exact" else age_label
                show_symbol = True
            else:
                age_text = f"{display_age:.0f}" if config.age_display_mode == "exact" else age_label
                badge_text = f'{raw_label} {separator} {age_text}'
        else:
            badge_text = 'Analysis unavailable'
            gender_symbol = None

        visibility_class = (result.metadata or {}).get("visibility_class", "BO")
        badge_text = f"{visibility_class} {separator} {badge_text}"

        text_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        badge_height = text_height + padding_y * 2
        if show_symbol:
            symbol_span = symbol_radius * 2.9
            badge_height = max(badge_height, int(round(symbol_span + badge_font_size * 0.5)))
            fit_radius = (badge_height * 0.5 - symbol_stroke * 1.5) / 1.45
            symbol_radius = min(symbol_radius, fit_radius)
            symbol_slot = int(round(symbol_radius * 2 + side_pad))
            text_offset_x = accent_width + symbol_slot + text_gap
        else:
            symbol_slot = 0
            text_offset_x = accent_width + side_pad

        badge_width = text_offset_x + text_width + right_pad

        badge_x, badge_y = find_best_badge_position(
            person_box=person_box,
            face_box=face_box,
            badge_width=badge_width,
            badge_height=badge_height,
            image_width=image_width,
            image_height=image_height,
            margin=badge_margin,
        )

        radius = max(8, min(badge_height // 2, int(round(badge_height * 0.45))))

        draw.rounded_rectangle(
            [badge_x, badge_y + shadow_offset, badge_x + badge_width, badge_y + badge_height + shadow_offset],
            radius=radius,
            fill=BADGE_SHADOW,
        )

        draw.rounded_rectangle(
            [badge_x, badge_y, badge_x + badge_width, badge_y + badge_height],
            radius=radius,
            fill=BADGE_BACKGROUND,
            outline=BADGE_BORDER,
            width=1,
        )

        draw.rounded_rectangle(
            [badge_x, badge_y, badge_x + accent_width, badge_y + badge_height],
            radius=radius,
            fill=BADGE_ACCENT,
        )

        if show_symbol:
            symbol_center_x = badge_x + accent_width + symbol_slot / 2
            symbol_center_y = badge_y + badge_height / 2 + symbol_stroke * 0.5
            if gender_symbol:
                draw_female_symbol(draw, symbol_center_x, symbol_center_y, symbol_radius, FEMALE_COLOR, symbol_stroke)
            else:
                draw_male_symbol(draw, symbol_center_x, symbol_center_y, symbol_radius, MALE_COLOR, symbol_stroke)

        text_center_x = badge_x + text_offset_x + text_width / 2
        text_center_y = badge_y + badge_height / 2
        draw.text(
            (text_center_x, text_center_y),
            badge_text,
            font=badge_font,
            fill=TEXT_COLOR,
            anchor='mm',
        )

    return canvas
