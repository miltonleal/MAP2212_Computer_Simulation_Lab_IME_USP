### EP1 - MAP 2212
### Milton Leal Neto - NUSP: 8973974

import random
import math
import time

def calcula_proporcao(n): #calcula proporção de pontos dentro do círculo

  ponto_dentro_circulo = 0 #inicia contador

  for i in range(n): #gera pontos aleatórios
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    if (x*x + y*y)**0.5 <= 1: #verifica se caiu dentro ou fora do círculo
      ponto_dentro_circulo += 1 #atualiza contador

  proporcao = ponto_dentro_circulo/n #devolve a proporção

  return proporcao

def calcula_n(n_inicial): #calcula qual o valor "ótimo" de n para a simulação

    #dado um n inicial, calcula a proporção da amostra piloto
    proporcao_estimada = calcula_proporcao(n_inicial)

    #calcula a variância amostral da proporção
    variancia_amostral = (proporcao_estimada*(1-proporcao_estimada))/n_inicial

    #calcula o desvio padrão amostral da proporção
    dp_amostral = variancia_amostral**(1/2)

    #calcula o n "ótimo" considerando um intervalo de confiança de 95% e acurácia de 0.0005
    n_otimo = n_inicial * ((1.96*dp_amostral) / (0.0005 * proporcao_estimada)) ** 2

    return math.trunc(n_otimo)

def main(): #programa principal

    random.seed(27) #fixa uma seed

    n_inicial = 10000
    n_final = calcula_n(n_inicial)

    print("Número de pontos aleatórios =", n_final)

    t0 = time.time() #calcula o tempo de execução

    for i in range(1,11): #imprime 10 rodadas do programa
        print(i, ") Pi é aproximadamente = ", 4*(calcula_proporcao(n_final)))

    t1 = time.time()

    print("Tempo de execução =", t1 - t0)

main()
