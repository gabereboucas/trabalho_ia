import matplotlib.pyplot as plt         
import numpy          #pacote NumPy para cálculos numéricos
import matplotlib  # usado com o NumPy para criação de gráficos e visualizações de dados
from IPython import display
from matplotlib import pyplot

plt.ion() # Matplotlib é uma biblioteca de visualização em Python para gráficos 2D de matrizes.

plt.style.use('dark_background') # tema de gráfico dark melhor visualizacão
plt.figure('Grafico de desempenho')
plt.show() # função no módulo pyplot da biblioteca matplotlib é usada para exibir todas as figuras

def plot(pontos, media):
    display.clear_output(wait=False) # armazena em buffer os últimos n eventos. Sempre que o buffer muda,pode limpar a saída da célula e imprimir o buffer novamente.
    plt.clf() # são usados ​​para limpar a figura atual, usado antes de definir um gráfico para que os anteriores não interfiram nos próximos.
    plt.title('Treinamento')
    plt.xlabel('Quantidade de jogos')
    plt.ylabel('Pontuação')
    plt.plot(pontos)
    plt.plot(media)
    plt.ylim(ymin=0) # limite de eixo y
    plt.text(len(pontos)-1, pontos[-1], str(pontos[-1]))
    plt.text(len(media)-1, media[-1], str(media[-1]))
    plt.show(block=False)
    plt.pause(.1) #pausa para nao haver erros
