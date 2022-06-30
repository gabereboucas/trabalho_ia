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
    
    def forward(self, x):  #semelhante ao método call, mas com atributos registrados. Usado para chamar diretamente um método na classe quando um nome de instância é chamado
        x = F.relu(self.linear1(x)) # f(x) = max(0,x) função de ativação linear retificada (RELU) é uma função linear por partes que,
        # se a entrada positiva ex. x, a saída será x, se nao sera 0 
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'): #salva arquivo em path o que permite sua visuazilacao e armazena o arquivo
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(),file_name)
    
class QTrainer:
    def __init__(self, model,lr,gamma):
        self.lr = lr  # Regressão Logística é encontrar uma relação entre as características e a probabilidade de um determinado resultado
        self.gamma = gamma #calcula numericamente o valor gama do número que é passado na função 0,9
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, estado, acao, recompensa, prox_estado, pronto):
        estado = torch.tensor(estado, dtype=torch.float)
        prox_estado = torch.tensor(prox_estado, dtype=torch.float)
        acao = torch.tensor(acao, dtype=torch.long)
        recompensa = torch.tensor(recompensa, dtype=torch.float)

        if len(estado.shape) == 1:
            estado = torch.unsqueeze(estado,0)
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
                Q_novo = recompensa[idx] + self.gamma * torch.max(self.model(prox_estado[idx]))
            
            alvo[idx][torch.argmax(acao[idx]).item()] = Q_novo

        self.optimizer.zero_grad()
        loss = self.criterion(alvo,pred)
        loss.backward()

        self.optimizer.step()



