import torch                        # ajuda criar e treinar redes neurais
import torch.nn as nn               # blocos de construção básicos para gráfico; 
import torch.optim as optim         # pacote que implementa vários algoritmos de otimização.
import torch.nn.functional as F     # realizam operações aritméticas, não as camadas que possuem parâmetros treináveis, como pesos e termos de polarização.
import os                           # fornece uma maneira portátil de usar a funcionalidade dependente do sistema operacional. 

class QNet(nn.Module):              # polinômio de terceira ordem, treinado para y=sin(x) a partir de -π minimizando a distância euclidiana ao quadrado.
    def __init__(self, valor_entrada, valor_oculto, valor_saida):
        super().__init__()          # retorna um objeto temporário da superclasse e isso permite o acesso a todos os seus métodos à sua classe filha.
        self.linear1 = nn.Linear(valor_entrada, valor_oculto) # aplica uma transformação linear aos dados de entrada
        self.linear2 = nn.Linear(valor_oculto, valor_saida)
    
    def forward(self, x):           #semelhante ao método call, mas com atributos registrados. Usado para chamar diretamente um método na classe quando um nome de instância é chamado
        x = F.relu(self.linear1(x)) # f(x) = max(0,x) função de ativação linear retificada (RELU) é uma função linear por partes que,
                                    # se a entrada positiva ex. x, a saída será x, se nao sera 0 
        x = self.linear2(x)
        return x
    
class QTrainer:
    def __init__(self, model,lr,gamma):
        self.lr = lr                # Regressão Logística é encontrar uma relação entre as características e a probabilidade de um determinado resultado
        self.gamma = gamma          #calcula numericamente o valor gama do número que é passado na função 0,9
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #Adam() (Otimização Estocástica) é uma técnica que implementa a taxa de aprendizagem adaptativa
        self.criterion = nn.MSELoss() #nn.MSELoss() cria um critério que mede o erro quadrático médio entre cada elemento na entrada x e y.
    
    def train_step(self, estado, acao, recompensa, prox_estado, pronto):
        estado = torch.tensor(estado, dtype=torch.float) #tensor() é uma matriz multidimensional contendo elementos de um único tipo de dados.
        prox_estado = torch.tensor(prox_estado, dtype=torch.float)
        acao = torch.tensor(acao, dtype=torch.long)
        recompensa = torch.tensor(recompensa, dtype=torch.float)

        if len(estado.shape) == 1:
            estado = torch.unsqueeze(estado,0) #unsqueeze() retorna um novo tensor com uma dimensão de tamanho um inserido na posição especificada.
            prox_estado = torch.unsqueeze(prox_estado,0)
            acao = torch.unsqueeze(acao,0)
            recompensa = torch.unsqueeze(recompensa,0)
            pronto = (pronto, )

    #1: definindo valores para Q

        pred = self.model(estado)

        alvo = pred.clone()
        for idx in range(len(pronto)):
            Q_novo = recompensa[idx]
            if not pronto[idx]:
                Q_novo = recompensa[idx] + self.gamma * torch.max(self.model(prox_estado[idx])) #max() retorna o valor máximo de todos os elementos no tensor de entrada.
            
            alvo[idx][torch.argmax(acao[idx]).item()] = Q_novo #argmax() retorna os índices do valor máximo de todos os elementos no tensor de entrada.

        self.optimizer.zero_grad() #Define os gradientes otimizadas para zero.
        loss = self.criterion(alvo,pred)
        loss.backward()             #Calcula o gradiente do tensor atual.

        self.optimizer.step()       #O step é o intervalo entre cada elemento numérico.
