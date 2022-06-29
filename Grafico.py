import matplotlib.pyplot as plt
from IPython import display
import numpy
import matplotlib
from matplotlib import pyplot

plt.ion()

plt.style.use('dark_background')

plt.show()

def plot(pontos, media):
    display.clear_output(wait=False)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Treinamento')
    plt.xlabel('Quantidade de jogos')
    plt.ylabel('Pontuação')
    plt.plot(pontos)
    plt.plot(media)
    plt.ylim(ymin=0)
    plt.text(len(pontos)-1, pontos[-1], str(pontos[-1]))
    plt.text(len(media)-1, media[-1], str(media[-1]))
    plt.show(block=False)
    plt.pause(.1)
