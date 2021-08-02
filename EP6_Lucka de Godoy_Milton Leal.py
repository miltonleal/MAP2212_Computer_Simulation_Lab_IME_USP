### EP 6 - Laboratório de Computação e Simulação
### Alunos: Lucka de Godoy Gianvechio e Milton Leal

import numpy as np
from scipy.special import gamma
from scipy.stats import chi2
from scipy.optimize import minimize_scalar

def calcula_n_final(alfa, seed=42):  # calcula número de thetas que serão gerados

    n_ini = 1000  # número inicial de thetas para amostra piloto

    v1_s = []  # lista que vai guardar a média entre os primeiros dois f(theta)

    for i in range(100):  # realiza 100 repetições do experimento

        fs = calcula_f(alfa, n_ini, i)  # calcula a f_theta

        # tira a média entre os dois primeiros f_theta
        # que já foram ordenados na função calcula_f
        v1_s.append((fs[0] + fs[1]) / 2)

    vr = np.var(v1_s, ddof=1)  # calcula a variância da média dos dois primeiros f_thetas

    # calcula o n_final
    n_final = int((1.96 ** 2) * vr / (0.0005 ** 2))
    #print(n_final)

    return n_final

def calcula_f(alfa, n, seed=42):

    np.random.seed(seed=seed)

    thetas = np.random.dirichlet(alfa, n)  # gera thetas de uma dirichelt

    produtorio = np.array([])

    # calcula o produtório da f_theta
    for i in range(len(thetas)):
        produtorio = np.append(produtorio, np.prod(np.power(thetas[i], alfa - 1)))

    # calcula a constante de normalização
    c = 1 / ((np.prod(gamma(alfa))) / (gamma(sum(alfa))))

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

def dirichilet(theta, alfa):

    produtorio = np.array([])

    # calcula o produtório da f_theta
    for i in range(len(theta)):
        produtorio = np.append(produtorio, np.power(theta[i], alfa[i] - 1))

    f = np.prod(produtorio)

    return f

def sev(t, h, z): #calcula o e-valor padronizado (SEV)

    ev_barra = 1 - z #define o ev_barra como 1 - ev

    df = t - h #graus de liberdade

    qq = chi2.cdf(chi2.ppf(ev_barra, t), df) #função QQ

    return 1 - qq #retorna QQ_barra

def main():

    print("*** Full Bayesian Significance Test ***".center(82))
    print()

    t = 2 #dimensão do espaço paramétrico
    h = 1 #dimensão do espaço paramétrico da hipótese nula

    decision = 0.05 #nível de significância do teste

    vetores_x = [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10],
                 [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18],
                 [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
                 [5, 9], [5, 10], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7]]

    vetores_x = np.array([np.array([x[0], 20 - x[0] - x[1], x[1]]) for x in vetores_x])

    vetores_y = [[1,1,1], [0,0,0]]

    sevs = []

    ### Espaço da Hipótese Nula

    s_star = lambda theta1: np.array([theta1,
                                      1 - theta1 - np.power((1 - np.sqrt(theta1)), 2),
                                      np.power((1 - np.sqrt(theta1)), 2)])

    print("X1 \t X3 \t Y \t\t     EV       \t SEV       \t Decisão (alfa = 5%)")
    print("-" * 84)

    for y in vetores_y:
        for x in vetores_x:

            alfa = x + y

            # calcula a constante de normalização
            c = 1 / ((np.prod(gamma(alfa))) / (gamma(sum(alfa))))

            if alfa[2] == 0 and y == [0, 0, 0]: #condição para casos anômalos quando x3 = 0 e y = 0

                print(f"{str(x[0]).zfill(2)} \t "
                      f"{str(x[2]).zfill(2)} \t {y} \t NA      \t NA       \t NA")
                continue

            #otimizador do máximo da f dentro do espaço paramétrico da hipótese nula
            sol = minimize_scalar(fun=lambda p: -dirichilet(s_star(p), alfa)*c,
                                  bounds=(0.0, 1.0), method='Bounded')


            n_final = calcula_n_final(alfa) #calcula o n_final para cada conjunto de dados

            ev = estima_W(calcula_f(alfa, n_final), v=-sol.fun) #calcula o e-valor

            sevs.append(sev(t, h, ev)) #lista com os e-valores padronizados

            #printa os resultados e toma a decisão se rejeita ou não a hipótese nula
            print(
                f"{str(x[0]).zfill(2)} \t "
                f"{str(x[2]).zfill(2)} \t "
                f"{y} \t {ev:6,.4f} \t "
                f"{sevs[-1]:6,.4f} \t "
                f"{'Rejeita H0' * int(sevs[-1] <= decision) + 'Não Rejeita H0' * int(sevs[-1] > decision)}")

main()
