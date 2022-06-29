import torch
import random
import numpy as np
from collections import deque
from jogo import JogoCobraIA, Direction, Point
from modelo_busca import QNet, QTrainer
from Grafico import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_jogos = 0
        self.epsilon = 0 # aleatoriedade
        self.gamma = 0.9 # taxa de desconto
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_estado(self, jogo):
        cabeca = jogo.cobra[0]
        point_l = Point(cabeca.x - 20, cabeca.y)
        point_r = Point(cabeca.x + 20, cabeca.y)
        point_u = Point(cabeca.x, cabeca.y - 20)
        point_d = Point(cabeca.x, cabeca.y + 20)
        
        dir_l = jogo.direction == Direction.LEFT
        dir_r = jogo.direction == Direction.RIGHT
        dir_u = jogo.direction == Direction.UP
        dir_d = jogo.direction == Direction.DOWN

        estado = [
            # Perigo direto
            (dir_r and jogo._ocorre_colisao(point_r)) or 
            (dir_l and jogo._ocorre_colisao(point_l)) or 
            (dir_u and jogo._ocorre_colisao(point_u)) or 
            (dir_d and jogo._ocorre_colisao(point_d)),

            # Perigo à direita
            (dir_u and jogo._ocorre_colisao(point_r)) or 
            (dir_d and jogo._ocorre_colisao(point_l)) or 
            (dir_l and jogo._ocorre_colisao(point_u)) or 
            (dir_r and jogo._ocorre_colisao(point_d)),

            # Perigo à esquerda
            (dir_d and jogo._ocorre_colisao(point_r)) or 
            (dir_u and jogo._ocorre_colisao(point_l)) or 
            (dir_r and jogo._ocorre_colisao(point_u)) or 
            (dir_l and jogo._ocorre_colisao(point_d)),
            
            # Mover direção
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #local da comida 
            jogo.comida.x < jogo.cabeca.x,  #comida à esquerda
            jogo.comida.x > jogo.cabeca.x,  #comida à direita
            jogo.comida.y < jogo.cabeca.y,  #comida para cima
            jogo.comida.y > jogo.cabeca.y  #comida para baixo
            ]

        return np.array(estado, dtype=int)

    def remember(self, estado, acao, recompensa, prox_estado, pronto):
        self.memory.append((estado, acao, recompensa, prox_estado, pronto)) # popleft se MAX_MEMORY for alcançado

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # lista de tuplas
        else:
            mini_sample = self.memory

        estados, acaos, recompensas, prox_estados, prontos = zip(*mini_sample)
        self.trainer.train_step(estados, acaos, recompensas, prox_estados, prontos)
        #for estado, acao, recompensa, nexrt_estado, pronto in mini_sample:
        #self.trainer.train_step(estado, acao, recompensa, prox_estado, pronto)

    def train_short_memory(self, estado, acao, recompensa, prox_estado, pronto):
        self.trainer.train_step(estado, acao, recompensa, prox_estado, pronto)

    def get_acao(self, estado):
        # movimentos aleatórios: exploração
        self.epsilon = 80 - self.n_jogos
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            estado0 = torch.tensor(estado, dtype=torch.float)
            prediction = self.model(estado0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_pontos = []
    plot_medias = []
    total_ponto = 0
    recorde = 0
    agent = Agent()
    jogo = JogoCobraIA()
    while True:
        # voltar no estado antigo
        estado_old = agent.get_estado(jogo)

        # mexa-se
        final_move = agent.get_acao(estado_old)

        # executa o movimento e obtém um novo estado
        recompensa, pronto, pontuacao = jogo.play_step(final_move)
        estado_new = agent.get_estado(jogo)

        # treina memória curta
        agent.train_short_memory(estado_old, final_move, recompensa, estado_new, pronto)

        # lembrar
        agent.remember(estado_old, final_move, recompensa, estado_new, pronto)

        if pronto:
            # treina memória longa, resultado do gráfico
            jogo.reset()
            agent.n_jogos += 1
            agent.train_long_memory()

            if pontuacao > recorde:
                recorde = pontuacao
                agent.model.save()

            print('Jogo', agent.n_jogos, 'Ponto', pontuacao, 'Recorde:', recorde)

            plot_pontos.append(pontuacao)
            total_ponto += pontuacao
            media = total_ponto / agent.n_jogos
            plot_medias.append(media)
            plot(plot_pontos, plot_medias)

if __name__ == '__main__':
    train()