import pygame
import random 
import time
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('games-italic.ttf',25)
pygame.mixer.init()

class Direction(Enum):    #Classe base para criar constantes enumeradas
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point','x,y')

random_color=list(np.random.choice(range(255),size=4))
random_color2=list(np.random.choice(range(255),size=4))
random_color3=list(np.random.choice(range(255),size=4))

#cores rgb
BRANCO = (255, 255, 255)
#COR_COMIDA = pygame.image.load('mata.jpg')
COR_COMIDA = (random_color)
COR_QUAD = (random_color2)
COR_STROKE = (random_color3)

DIMENSAO_QUAD = 20
SPEED = 23

class jogo_cobra:

    def __init__(self):
    
        self.display = pygame.display.set_mode((480, 480)) #inicialização da tela
        
        pygame.display.set_caption('Cobra')
        self.clock = pygame.time.Clock()

        pygame.display.set_mode((480, 480))
        bg_img = pygame.image.load('bg2.jpg')
        bg_img = pygame.transform.scale(bg_img,(480, 480))    
        
        self.direction = Direction.RIGHT

        self.cabeca = Point(480/2, 480/2)
        self.cobra = [self.cabeca, Point(self.cabeca.x-DIMENSAO_QUAD,self.cabeca.y),Point(self.cabeca.x-(2*DIMENSAO_QUAD),self.cabeca.y)]
        self.pontuacao = 0
        self.comida = None
        self._lugar_comida()
    
    def _lugar_comida(self):
        x = random.randint(0, (480-DIMENSAO_QUAD) // DIMENSAO_QUAD)*DIMENSAO_QUAD
        y = random.randint(0, (480-DIMENSAO_QUAD) // DIMENSAO_QUAD)*DIMENSAO_QUAD
        self.comida = Point(x,y)
        if self.comida in self.cobra:
            
            self._lugar_comida()
    
    def play_step(self):
        #1. coleta entrada do user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        #2. movimentação
        self._move(self.direction) 
        self.cobra.insert(0,self.cabeca)

        #3. checar se o jogo acabou
        fim_de_jogo = False
        if self._ocorre_colisao():
            fim_de_jogo = True
            return fim_de_jogo, self.pontuacao
        
        #4 posicao da nova comida/mover comida
        if self.cabeca == self.comida:
            self.pontuacao +=1
            self._lugar_comida()
            pygame.mixer.music.load('rapai.wav')
            pygame.mixer.music.play()
        else:
            self.cobra.pop()

        #5 atualizando interface do usuario (UI) e pontuacao
        self._update_ui()
        self.clock.tick(SPEED)

        #6. retornar fim de jogo e pontucao
        return fim_de_jogo, self.pontuacao
    
    def _ocorre_colisao(self):
        #atinge o limite
        if self.cabeca.x > 480 - DIMENSAO_QUAD or self.cabeca.x < 0 or self.cabeca.y > 480 - DIMENSAO_QUAD or self.cabeca.y < 0:
            pygame.mixer.music.load('Tome.wav')
            pygame.mixer.music.play()
            return True

        #atinge o proprio corpo
        if self.cabeca in self.cobra[1:]:
            pygame.mixer.music.load('ui.wav')
            pygame.mixer.music.play()
            return True

        return False
    
    def _update_ui(self):
        background = pygame.display.set_mode((480, 480)) 
        image = pygame.image.load('bg2.jpg') 
        background.blit(image, (0,0))

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

        #pygame.Surface.blit('mata.jpg',self.display,pygame.Rect(self.comida.x, self.comida.y, DIMENSAO_QUAD, DIMENSAO_QUAD))

        text = font.render("Pontos: " + str(self.pontuacao), True, BRANCO)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.cabeca.x
        y = self.cabeca.y
        if direction == Direction.RIGHT:
            x += DIMENSAO_QUAD
        elif direction == Direction.LEFT:
            x -= DIMENSAO_QUAD
        elif direction == Direction.DOWN:
            y += DIMENSAO_QUAD
        elif direction == Direction.UP:
            y -= DIMENSAO_QUAD
        
        self.cabeca = Point(x,y)

if __name__ == '__main__':
    game = jogo_cobra()

    # game loop
    while True:
        
        fim_de_jogo, pontuacao = game.play_step()
        
        if fim_de_jogo == True:
            time.sleep(1)
            break
        
    print('Pontuacao final:', pontuacao)
    
    pygame.quit()
    