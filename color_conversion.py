# Amelia Sinclaire 2024
import numpy as np


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def bgr_to_rgb(value):
    return value[::-1]


def rgb_to_bgr(value):
    return value[::-1]


def rgb_to_hex(value):
    r_q = value[0] // 16
    r_r = value[0] % 16
    g_q = value[1] // 16
    g_r = value[1] % 16
    b_q = value[2] // 16
    b_r = value[2] % 16

    def map_hex(val):
        val = int(val)
        match val:
            case 10:
                return 'A'
            case 11:
                return 'B'
            case 12:
                return 'C'
            case 13:
                return 'D'
            case 14:
                return 'E'
            case 15:
                return 'F'
            case _:
                return str(val)
    return f'#{map_hex(r_q)}{map_hex(r_r)}{map_hex(g_q)}{map_hex(g_r)}{map_hex(b_q)}{map_hex(b_r)}'


def rgb_to_hsv(value):
    r, g, b = value
    r /= 255.0
    g /= 255.0
    b /= 255.0
    cmax = max(r, max(g, b))
    cmin = min(r, min(g, b))
    delta = cmax - cmin
    if delta == 0:
        hue = 0
    elif cmax == r:
        hue = ((g - b) / delta)
    elif cmax == g:
        hue = ((b - r) / delta) + 2
    else:
        hue = ((r - g) / delta) + 4
    if cmax == 0:
        sat = 0
    else:
        sat = (delta / cmax) * 100.0
    val = cmax * 100.0
    hue *= 60
    return [hue, sat, val]


def hsv_to_rgb(value):
    h, s, v = value
    s /= 100.0
    v /= 100.0
    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
        temp = [c, x, 0]
    elif 60 <= h < 120:
        temp = [x, c, 0]
    elif 120 <= h < 180:
        temp = [0, c, x]
    elif 180 <= h < 240:
        temp = [0, x, c]
    elif 240 <= h < 300:
        temp = [x, 0, c]
    else:
        temp = [c, 0, x]
    return [(temp[0] + m) * 255.0, (temp[1] + m) * 255.0, (temp[2] + m) * 255.0]


if __name__ == '__main__':
    rgb = [100, 63, 34]
    hsv = rgb_to_hsv(rgb)
    rgb_2 = hsv_to_rgb(hsv)
    hex = rgb_to_hex(rgb)
    rgb_3 = hex_to_rgb(hex)
    bgr = rgb_to_bgr(rgb)
    rgb_4 = bgr_to_rgb(bgr)
    print(f'{rgb=}, {hsv=}, {rgb_2=}, {hex=}, {rgb_3=}, {bgr=}, {rgb_4=}')
