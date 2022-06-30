import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np                  #pacote NumPy para cálculos numéricos

pygame.init()
font = pygame.font.Font('games-italic.ttf',25)
image = pygame.image.load('bg2.jpg') 
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

random_color=list(np.random.choice(range(255),size=3))
random_color2=list(np.random.choice(range(255),size=3))
random_color3=list(np.random.choice(range(255),size=3))

# rgb colors
BRANCO = (255,255,255)
COR_COMIDA = (random_color)
COR_QUAD = (random_color2)
COR_STROKE = (random_color3)
COR_FUNDO = (0,0,0)


DIMENSAO_QUAD = 20
SPEED = 50

class JogoCobraIA:

    def __init__(self):

        self.display = pygame.display.set_mode((480,480))
        pygame.display.set_caption('Cobra')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        #ordena reinicializacao cabeca, corpo, ponto e comida
        self.direction = Direction.RIGHT

        self.cabeca = Point(480/2, 480/2)
        self.cobra = [self.cabeca,
                      Point(self.cabeca.x-DIMENSAO_QUAD, self.cabeca.y),
                      Point(self.cabeca.x-(2*DIMENSAO_QUAD), self.cabeca.y)]

        self.pontuacao = 0
        self.comida = None
        self._lugar_comida()
        self.frame_iteration = 0


    def _lugar_comida(self):
        #local da comida e sua modificacao
        x = random.randint(0, (480-DIMENSAO_QUAD )//DIMENSAO_QUAD )*DIMENSAO_QUAD
        y = random.randint(0, (480-DIMENSAO_QUAD )//DIMENSAO_QUAD )*DIMENSAO_QUAD
        self.comida = Point(x, y)
        if self.comida in self.cobra:
            self._lugar_comida()

    def play_step(self, acao):
        self.frame_iteration += 1
        #1. ver entrada do jogo
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        #2. movimento
        self._move(acao) # atualiza cabeca
        self.cobra.insert(0, self.cabeca)
        
        # 3 verifica se o jogo acabou
        recompensa = 0 #valor de recompensa
        fim_de_jogo = False
        if self._ocorre_colisao() or self.frame_iteration > 100*len(self.cobra):
            fim_de_jogo = True
            recompensa = -10 #valor de recompensa
            return recompensa, fim_de_jogo, self.pontuacao

        # 4.adc novos alimentos ou movendo
        if self.cabeca == self.comida:
            self.pontuacao += 1
            recompensa = 10
            self._lugar_comida()
            pygame.mixer.music.load('rapai.wav')
            pygame.mixer.music.play()
        else:
            self.cobra.pop()
        
        # 5. atualiza interface
        self._update_ui()
        self.clock.tick(60)
        # 6. retorna game over e pontuacao
        return recompensa, fim_de_jogo, self.pontuacao


    def _ocorre_colisao(self, pt=None):
        if pt is None:
            pt = self.cabeca
        # atinge o limite
        if pt.x > 480 - DIMENSAO_QUAD or pt.x < 0 or pt.y > 480 - DIMENSAO_QUAD or pt.y < 0:
            pygame.mixer.music.load('Tome.wav')
            pygame.mixer.music.play()
            return True
        # atinge a si mesma
        if pt in self.cobra[1:]:
            pygame.mixer.music.load('ui.wav')
            pygame.mixer.music.play()
            return True

        return False


    def _update_ui(self):
        #self.display.fill(COR_FUNDO) 
        self.display.blit(image, (0,0))
        
        
        for pt in self.cobra:
            pygame.draw.rect(self.display, (0,0,0), pygame.Rect(pt.x, pt.y, DIMENSAO_QUAD, DIMENSAO_QUAD))
            pygame.draw.rect(self.display, COR_STROKE, pygame.Rect(pt.x+2, pt.y+2, 16, 16))

        #pygame.draw.rect(self.display, (255,255,255), pygame.Rect(self.comida.x, self.comida.y, DIMENSAO_QUAD+1, DIMENSAO_QUAD+1))
        pygame.draw.polygon(self.display,(255,0,0),  
            [(self.comida.x+1, self.comida.y+8),
            (self.comida.x+2, self.comida.y+4), 
            (self.comida.x+4, self.comida.y+2),  
            (self.comida.x+5, self.comida.y+2), 
            (self.comida.x+6, self.comida.y+2), 
            (self.comida.x+10, self.comida.y+6), 
            (self.comida.x+14, self.comida.y+2), 
            (self.comida.x+15, self.comida.y+2),
            (self.comida.x+16, self.comida.y+2), 
            (self.comida.x+18, self.comida.y+4), 
            (self.comida.x+19, self.comida.y+8), 
            (self.comida.x+10, self.comida.y+19)])
        pygame.draw.polygon(self.display,(252,138,0,99),  
            [(self.comida.x+1, self.comida.y+8),
            (self.comida.x+2, self.comida.y+6),
            (self.comida.x+10, self.comida.y+6),
            (self.comida.x+18, self.comida.y+6), 
            (self.comida.x+19, self.comida.y+8), 
            (self.comida.x+10, self.comida.y+19)])
        pygame.draw.polygon(self.display,(255,241,48,100),  
            [(self.comida.x+1, self.comida.y+8), 
            (self.comida.x+19, self.comida.y+8), 
            (self.comida.x+10, self.comida.y+19)])
        pygame.draw.polygon(self.display,(0,255,0,100),  
            [(self.comida.x+2, self.comida.y+10), 
            (self.comida.x+18, self.comida.y+10), 
            (self.comida.x+10, self.comida.y+19)])
        pygame.draw.polygon(self.display,(0,2,255,100),  
            [(self.comida.x+4, self.comida.y+13), 
            (self.comida.x+16, self.comida.y+13), 
            (self.comida.x+10, self.comida.y+19)])
        pygame.draw.polygon(self.display,(153,0,255,100),  
            [(self.comida.x+6, self.comida.y+15), 
            (self.comida.x+14, self.comida.y+15), 
            (self.comida.x+10, self.comida.y+19)])


        text = font.render("Pontos: " + str(self.pontuacao), True, BRANCO)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, acao):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(acao, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(acao, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.cabeca.x
        y = self.cabeca.y
        if self.direction == Direction.RIGHT:
            x += DIMENSAO_QUAD
        elif self.direction == Direction.LEFT:
            x -= DIMENSAO_QUAD
        elif self.direction == Direction.DOWN:
            y += DIMENSAO_QUAD
        elif self.direction == Direction.UP:
            y -= DIMENSAO_QUAD

        self.cabeca = Point(x, y)