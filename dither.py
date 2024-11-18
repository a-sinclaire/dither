import collections
import itertools
import operator
from tqdm import tqdm
import random
import time
from copy import deepcopy
from itertools import product
from functools import lru_cache
from multiprocessing import Pool
from statistics import mean

import cv2
import numpy as np
from numpy.typing import NDArray

from color_conversion import rgb_to_hsv, hex_to_rgb, rgb_to_hex, hsv_to_rgb, bgr_to_rgb, rgb_to_bgr

grayscale_palette = ['#000000', '#333333', '#555555', '#777777', '#999999', '#BBBBBB', '#DDDDDD', '#FFFFFF']
binary_palette = ['#000000', '#FFFFFF']
two_bit_palette = ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#FFFFFF']
rgb_palette = ['#FF0000', '#00FF00', '#0000FF']


def angle_difference(ang1: float, ang2: float) -> float:
    """:return: diff of two angles (shortest dist)"""
    diff = ((ang2 - ang1 + 180) % 360) - 180
    return diff + 360 if (diff < -180) else diff


def limit_pixel(rgb: [int | float]) -> [float]:
    for idx, c in enumerate(rgb):
        if c < 0:
            rgb[idx] = 0
        if c > 255:
            rgb[idx] = 255
    return rgb


@lru_cache
def closest_hexes(rgb_color: tuple, hex_palette: tuple,
                  h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> [str]:
    h, s, v = rgb_to_hsv(rgb_color)
    hsv_palette = [rgb_to_hsv(hex_to_rgb(x)) for x in hex_palette]

    distances = []
    for (h_p, s_p, v_p) in hsv_palette:
        h_diff = abs(angle_difference(h, h_p)) / 360
        s_diff = abs(s - s_p) / 100
        v_diff = abs(v - v_p) / 100
        dist = (h_weight * h_diff) + (s_weight * s_diff) + (v_weight * v_diff)
        distances.append(dist)

    return [x for _, x in sorted(zip(distances, hex_palette), key=lambda pair: pair[0])]


def closest_rgb(rgb_color: [int], hex_palette: [str],
                h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> [int]:
    return hex_to_rgb(closest_hexes(tuple(rgb_color), tuple(hex_palette), h_weight, s_weight, v_weight)[0])


def closest_hex(rgb_color: [int], hex_palette: [str],
                h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> [int]:
    return closest_hexes(tuple(rgb_color), tuple(hex_palette), h_weight, s_weight, v_weight)[0]


def floyd_steinberg_dither(image: NDArray, hex_palette: list[str],
                           h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> NDArray:
    image = image.copy()
    hex_palette = hex_palette.copy()

    # Floyd-Steinberg Dithering
    height, width, depth = image.shape
    image = np.array(image, dtype=np.float32)
    for row in range(height):
        for col in range(width):
            old_rgb_pixel = image[row][col].copy()
            new_rbg_pixel = closest_rgb(old_rgb_pixel, hex_palette, h_weight, s_weight, v_weight)
            image[row][col] = np.asarray(new_rbg_pixel, dtype=np.float32)
            quant_error = [o - n for o, n in zip(old_rgb_pixel, new_rbg_pixel)]
            if col < width - 1:
                image[row, col + 1] += [q * (7 / 16) for q in quant_error]
                image[row, col + 1] = limit_pixel(image[row, col + 1])
            if row < height - 1 and col > 0:
                image[row + 1][col - 1] += [q * (3 / 16) for q in quant_error]
                image[row + 1][col - 1] = limit_pixel(image[row + 1][col - 1])
            if row < height - 1:
                image[row + 1][col + 0] += [q * (5 / 16) for q in quant_error]
                image[row + 1][col + 0] = limit_pixel(image[row + 1][col + 0])
            if row < height - 1 and col < width - 1:
                image[row + 1][col + 1] += [q * (1 / 16) for q in quant_error]
                image[row + 1][col + 1] = limit_pixel(image[row + 1][col + 1])
    image = np.array(image / np.max(image, axis=(0, 1)) * 255, dtype=np.uint8)
    return image


def combine(hex1: str, hex2: str) -> str:
    a = (hex_to_rgb(hex1))
    b = (hex_to_rgb(hex2))
    avg_rgb = [mean([x[0], x[1]]) for x in zip(a, b)]
    return rgb_to_hex(avg_rgb)


def combine_many(hexes: list[str]) -> str:
    rgbs = [hex_to_rgb(a) for a in hexes]
    rs = [x[0] for x in rgbs]
    gs = [x[1] for x in rgbs]
    bs = [x[2] for x in rgbs]
    avg_rgb = [mean(rs), mean(gs), mean(bs)]
    return rgb_to_hex(avg_rgb)


def meeley_dither_2(image: NDArray, hex_palette: list[str],
                    h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> NDArray:
    image = deepcopy(image)
    hex_palette = hex_palette.copy()

    parent_palette = generate_parents(hex_palette, 2)

    height, width, depth = image.shape
    for row in range(height):
        for col in range(width):
            # image[row][col] = (closest_rgb(image[row][col], parent_palette.keys(), h_weight, s_weight, v_weight))
            # continue

            best_hex = closest_hex(image[row][col], parent_palette.keys(), h_weight, s_weight, v_weight)
            parents = parent_palette[best_hex]

            if row % 2 == 0 and col % 2 == 0 or row % 2 == 1 and col % 2 == 1:
                image[row][col] = hex_to_rgb(parents[0])
            else:
                image[row][col] = hex_to_rgb(parents[1])
    return image


def meeley_dither_4(image: NDArray, hex_palette: list[str],
                    h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> NDArray:
    image = deepcopy(image)
    hex_palette = hex_palette.copy()

    parent_palette = generate_parents(hex_palette, 4)

    height, width, depth = image.shape
    for row in range(height):
        for col in range(width):
            # image[row][col] = closest_rgb(image[row][col], parent_palette.keys(), h_weight, s_weight, v_weight)
            # continue

            best_hex = closest_hex(image[row][col], parent_palette.keys(), h_weight, s_weight, v_weight)
            parents = parent_palette[best_hex]

            if row % 2 == 0 and col % 2 == 0:
                image[row][col] = hex_to_rgb(parents[0])
            elif row % 2 == 1 and col % 2 == 1:
                image[row][col] = hex_to_rgb(parents[1])
            elif row % 2 == 0 and col % 2 == 1:
                image[row][col] = hex_to_rgb(parents[2])
            elif row % 2 == 1 and col % 2 == 0:
                image[row][col] = hex_to_rgb(parents[3])
    return image


def generate_parents(hex_palette: [str], n: int) -> [str]:
    extended_hex_palette = itertools.combinations_with_replacement(hex_palette, n)

    parent_palette = {}
    for p in extended_hex_palette:
        parent_palette[combine_many(p)] = p

    sorted_p = sorted(parent_palette.items(), key=operator.itemgetter(0))
    parent_palette = collections.OrderedDict(sorted_p)
    return parent_palette


# This function works off a probabilistic method for filling in colors since we could have any n
# Note: it will be extremely slow for high values of n
# it took 26s on a 128x128 image with n=8 and a palette of 8 colors
# would be faster if put on gpu (not implemented)
def meeley_dither_n(image: NDArray, hex_palette: list[str], n: int,
                    h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> NDArray:
    image = deepcopy(image)
    hex_palette = hex_palette.copy()

    parent_palette = generate_parents(hex_palette, n)

    height, width, depth = image.shape
    for row in tqdm(range(height)):
        for col in range(width):
            # image[row][col] = closest_rgb(image[row][col], parent_palette.keys(), h_weight, s_weight, v_weight)
            # continue
            best_hex = closest_hex(image[row][col], parent_palette.keys(), h_weight, s_weight, v_weight)
            parents = parent_palette[best_hex]

            image[row][col] = hex_to_rgb(random.choice(parents))  # random method
    return image


def constrain_image(image: NDArray, hex_palette: list[str],
                    h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> NDArray:
    image = deepcopy(image)
    hex_palette = hex_palette.copy()
    return np.apply_along_axis(func1d=closest_rgb, axis=2, arr=image, hex_palette=hex_palette, h_weight=h_weight,
                               s_weight=s_weight, v_weight=v_weight).astype(np.uint8)


def parallel_constrain(image: NDArray, hex_palette: list[str],
                       h_weight: float = 1, s_weight: float = 1, v_weight: float = 1) -> NDArray:
    image = deepcopy(image)
    hex_palette = hex_palette.copy()
    pool = Pool(4)
    height, width, depth = image.shape
    for row in range(height):
        image[row] = (pool.starmap(closest_rgb, [[p, hex_palette, h_weight, s_weight, v_weight] for p in image[row]]))
    pool.close()
    pool.join()
    return image


def results():
    image = cv2.imread('unit.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_weight = 1
    s_weight = 1
    v_weight = 1
    palette = rgb_palette

    cv2.imwrite('results/constrain.png', cv2.cvtColor(parallel_constrain(image, palette, h_weight, s_weight, v_weight), cv2.COLOR_RGB2BGR))
    cv2.imwrite('results/floyd_ste.png', cv2.cvtColor(floyd_steinberg_dither(image, palette, h_weight, s_weight, v_weight), cv2.COLOR_RGB2BGR))
    cv2.imwrite('results/meeley_d2.png', cv2.cvtColor(meeley_dither_2(image, palette, h_weight, s_weight, v_weight), cv2.COLOR_RGB2BGR))
    cv2.imwrite('results/meeley_d4.png', cv2.cvtColor(meeley_dither_4(image, palette, h_weight, s_weight, v_weight), cv2.COLOR_RGB2BGR))


def main():
    image = cv2.imread('color_test.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    image = meeley_dither_n(image, two_bit_palette, 8)
    print(f'Completed in {time.time() - start_time}s')

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # print(combine('#FFFF00', '#0000FF'))
    # print(combine('#00FF00', '#00FF80'))
    main()
    # results()
