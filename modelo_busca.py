import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class QNet(nn.Module):
    def __init__(self, valor_entrada, valor_oculto, valor_saida):
        super().__init__()
        self.linear1 = nn.Linear(valor_entrada, valor_oculto)
        self.linear2 = nn.Linear(valor_oculto, valor_saida)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(),file_name)
    
class QTrainer:
    def __init__(self, model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
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



