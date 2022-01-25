"""
**************************************

Created in 21.01.2022
by Aiyyskhan Alekseev

https://github.com/AiyyArt

https://opensea.io/collection/aiyyart-collection

ETH: 0x4e6c76f938d941e5b4bf1a11e3fd20f311e59df6

timirkhan@gmail.com

**************************************
"""

# *** path to PNG file ***
PDF_LOAD_PATH = "data/NeuralNet_G000.png" #C:/.../labirint_0/data/NeuralNet_G000.png" 

__author__ = "Aiyyskhan Alekseev"
__version__ = "0.1.1"


import math
import numpy as np
from PIL import Image
import pygame

from settings import *
import nn_1pl as nn
from player import Player
from map_file_lev0_2 import get_map
from drawing import Drawing
from ray_casting import RayCast

VAL = np.array([-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])

def png2arr(path, val):
    _w_ids = (np.array(Image.open(path)) / 255.0) * 8.0
        
    w_ids = np.rot90(_w_ids, axes=(2,0))

    w_id_0 = np.rot90(w_ids[0], axes=(0,1))
    w_id_1 = np.rot90(w_ids[1], k=2, axes=(0,1))
    w_id_2 = np.rot90(w_ids[2], k=3, axes=(0,1))

    w_id = np.around((w_id_0 + w_id_1 + w_id_2) / 3.0).astype(np.uint8)

    i_w_id = w_id[:5, :50]
    h_w_id = w_id[5:, :50]
    o_w_id = w_id[5:, 50:53]

    return (
        val[i_w_id], 
        val[h_w_id], 
        val[o_w_id]
    )

class Game:
    def __init__(self):
        self.color_r = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
        self.color_g = np.random.randint(50, 220) # np.linspace(10, 200).astype(int)
        self.color_b = np.random.randint(50, 220) # np.linspace(200, 10).astype(int)

        pygame.init()
        pygame.display.set_caption('*** Labirint 1 ***')
        self.sc = pygame.display.set_mode((WIDTH, HEIGHT))
        self.sc_map = pygame.Surface((WIDTH // MAP_SCALE, HEIGHT // MAP_SCALE))
        self.clock = pygame.time.Clock()
        self.drawing = Drawing(self.sc, self.sc_map)

        self.road_coords = set()
        self.finish_coords = set()
        self.wall_coord_list = list()
        self.world_map, self.collision_walls = get_map(TILE)
        for coord, signature in self.world_map.items():
            if signature == "1":
                self.wall_coord_list.append(coord)
            elif signature == "2":
                self.finish_coords.add(coord)
            elif signature == ".":
                self.road_coords.add(coord)

    def player_setup(self):
        # self.loading()

        self.player = Player(self.sc, self.collision_walls, self.finish_coords)

        self.player.color = (self.color_r, self.color_g, self.color_b)
        self.player.init_angle = math.pi + (math.pi/2)
        self.player.rays = RayCast(self.world_map)

        weights = png2arr(PDF_LOAD_PATH, VAL)
        
        self.player.brain = nn.Ganglion_numpy(weights)
        self.player.test_mode = True

        self.player.setup()

    def game_event(self):
        self.player.movement()
        self.player.draw()
        
        for x, y in self.wall_coord_list:
            pygame.draw.rect(self.sc, WALL_COLOR_1, (x, y, TILE, TILE), 2)
        for x, y in self.finish_coords:
            pygame.draw.rect(self.sc, WALL_COLOR_2, (x, y, TILE, TILE), 2)

        # self.drawing.info(0, 0, 1, self.clock)

    def run(self):
        self.player_setup()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
            self.sc.fill(BLACK)

            self.game_event()

            pygame.display.flip()
            self.clock.tick()

if __name__ == "__main__":
    game = Game()
    game.run()