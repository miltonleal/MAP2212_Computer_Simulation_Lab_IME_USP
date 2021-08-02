### EP 5 - Laboratório de Computação e Simulação
### Alunos: Lucka de Godoy Gianvechio e Milton Leal

### Tempo estimado para rodar o programa: 3 a 5 minutos

import numpy as np
from scipy.special import gamma

def calcula_n_final(x, y, seed=42):  # calcula número de thetas que serão gerados

    n_ini = 1000  # número inicial de thetas para amostra piloto

    v1_s = []  # lista que vai guardar a média entre os primeiros dois f(theta)

    for i in range(100):  # realiza 100 repetições do experimento

        fs = calcula_f(x, y, n_ini, i)  # calcula a f_theta

        # tira a média entre os dois primeiros f_theta
        # que já foram ordenados na função calcula_f
        v1_s.append((fs[0] + fs[1]) / 2)

    vr = np.var(v1_s, ddof=1)  # calcula a variância da média dos dois primeiros f_thetas

    # calcula o n_final
    n_final = int((1.96 ** 2) * vr / (0.0005 ** 2))

    return n_final

def matriz_de_covariancia(alfa):  # calcula a matriz de covariância Sigma

    M = np.array([[0.0, 0.0] for k in range(2)])

    for i in range(2):
        for j in range(2):

            if i == j:
                M[i, j] = alfa[i] * (sum(alfa) - alfa[i]) / \
                          ((sum(alfa) ** 2) * (sum(alfa) + 1))  # cálculo das variâncias
                continue

            M[i, j] = -alfa[i] * alfa[j] / \
                      ((sum(alfa) ** 2) * (sum(alfa) + 1))  # cálculo das covariâncias

    return M

def calcula_f_indicadora(t, alfa):  # calcula a f da Dirichlet e avalia condição

    # verifica se o ponto a ser calculado está ou não no domínio da f
    if (t < 0).any(): return 0.0

    # realiza o cálculo da função Dirichlet
    produto = np.array([])
    theta = t.copy()
    a = alfa.copy()
    for i in range(len(theta)):
        produto = np.append(produto, np.power(theta[i], a[i] - 1))

    return np.prod(produto)

def gera_dir(n=100, alfa=[5, 5, 5], seed=42):

    np.random.seed(seed)

    burnin = 1000  # desconsidera os 1000 primeiros valores gerados

    pontos = np.array([[1 / 3, 1 / 3, 1 / 3]])  # ponto inicial da cadeia de Markov

    k = 0  # contador de pontos aceitos

    # multiplica matriz de covariância por constante ótima
    M = matriz_de_covariancia(alfa) * (2.38 ** 2) / 2

    while len(pontos) < burnin + n:

        p = np.random.multivariate_normal([0, 0], M, 1)[0]  # gera da Normal Multivariada

        ponto_atual = np.array \
            ([pontos[-1][0] + p[0], pontos[-1][1] + p[1], 1 - (pontos[-1][0] + p[0] + pontos[-1][1] + p[1])])

        # algoritmo de aceitação de Metropolis
        ac = min(1, calcula_f_indicadora(ponto_atual, alfa) /
                 calcula_f_indicadora(pontos[-1], alfa))

        if ac >= np.random.uniform():
            k += 1
            pontos = np.append(pontos, np.array(ponto_atual, ndmin=2), axis=0)
            continue

        pontos = np.append(pontos, np.array(pontos[-1], ndmin=2), axis=0)

    return pontos[burnin:]

def calcula_f(x, y, n, seed=42):
    a = x + y  # soma os vetores de entrada x e y

    thetas = gera_dir(n, a, seed)  # gera thetas de uma dirichelt

    produtorio = np.array([])

    # calcula o produtório da f_theta
    for i in range(len(thetas)):
        produtorio = np.append(produtorio, np.prod(np.power(thetas[i], a - 1)))

    # calcula a constante de normalização
    c = 1 / ((np.prod(gamma(a))) / (gamma(sum(a))))

    # multiplica a constante pela resultado do produtório
    f_thetas = c * produtorio

    # retorna os f_thetas ordenados
    return np.sort(f_thetas)

def estima_W(f_ord, v):
    # determina quantos pontos estão abaixo de um determinado v
    n = np.searchsorted(f_ord, v=v)

    # tamanho do vetor que guarda os f_thetas ordenados
    N = len(f_ord)

    # retorna a estimativa
    return n / N

def main():
    print("*** EP 5 - Aproximação da massa de probabilidade a posteriori de uma função "
          "utilizando um gerador criado com MCMC ***\n")

    continuar = True

    while continuar:  # loop principal do programa que executa até o usuário sair

        x = np.array([])
        y = np.array([])

        # solicita entra do vetor x
        for i in range(3):
            x = np.append(x, int(input(f"Entre com x{i}: ")))

        # solicita entra do vetor y
        for i in range(3):
            y = np.append(y, int(input(f"Entre com y{i}: ")))

        # cálculo do n_final
        n_de_thetas = calcula_n_final(x, y)

        print("Número de thetas que serão gerados: ", n_de_thetas)

        # calcula a f_theta e ordena os resultados
        fs_ordenados = calcula_f(x, y, n_de_thetas, seed=100)

        # função que retorna a estimativa da função W(v)
        U = lambda v: estima_W(fs_ordenados, v)

        continuar_v = True

        # loop secundário que executa até o usuário decidir trocar os vetores x e y
        while continuar_v:

            # solicita um valor de v entre 0 e o sup(f_theta)
            v = float(input(f"\nEntre com o valor de v de 0 a {fs_ordenados[-1]}: "))

            # retorna a estimativa na tela
            print(f"U({v})= {U(v)}")

            print("\nGostaria de calcular uma nova estimativa da U(v)? (s/n)")

            sn_v = input().lower()

            if sn_v == 'n': continuar_v = False

            print()

        print('Gostaria de inserir novos vetores de entrada para gerar uma nova U(v)? (s/n)')

        sn = input().lower()

        if sn == 'n': continuar = False

        print()

    print("\n\n *** FIM DO PROGRAMA *** \n\n")

main()