### EP 3 - Laboratório de Computação e Simulação
### Alunos: Lucka de Godoy Gianvechio e Milton Leal

### O código abaixo necessita das bibliotecas chaospy, numpy, pandas e matplotlib

### O código está dividido em dois blocos.

### O primeiro bloco gera as estimativas da integral que queremos calcular

### O segundo bloco gera uma série de gráficos que estão no relatório

### O segundo bloco está comentado e não vai rodar direto.

#################
import matplotlib.pyplot as plt
import numpy as np
import chaospy as cp
import pandas as pd
#################

## Primeiro bloco
## Tempo de execução: em torno de 60 segundos
def main(): #função de chamada principal

    #parâmetros de RG e CPF para para o cálculo da integral
    rg = 0.508662102
    cpf = 0.43898402827

    func = lambda x: np.exp(-rg * x) * np.cos(cpf * x) #integral que queremos calcular

    #cria a base de dados para o dataframe que contém as estimativas da integral
    data = {'Crude': crude(func),
            'Hit.Miss': hitmiss(func),
            "Imp.Sampling": imp_sampling(func),
            "C.Variate": control_variate(func)}

    df = pd.DataFrame(data,
                      columns=['Crude',
                               'Hit.Miss',
                               "Imp.Sampling",
                               "C.Variate"],
                      index=["Pseudo-Random",
                             "Quasi-Random Halton",
                             "Quasi-Random Sobol"])

    print(df) #

def crude(func): #calcula as estimativas no método Crude

    n_crude = 467811 #n obtido no EP 2

    # Pseudo random
    pseudo = np.array([np.mean(func(np.random.uniform(0, 1, n_crude)))])

    # Quasi_Halton
    halton = np.array([np.mean(func(cp.create_halton_samples(n_crude, 1)[0]))])

    # Quasi_Sobol
    sobol = np.array([np.mean(func(cp.create_sobol_samples(n_crude, 1)[0]))])

    results_crude = [np.round(pseudo, 5), np.round(halton, 5), np.round(sobol, 5)]

    return results_crude

def hitmiss(func): #calcula as estimativas no método Hit or Miss

    n_hitmiss = 4830000 #n obtido no EP 2

    pseudo, halton, sobol = [], [], []

    # Pseudo random
    vec_xy_pseudo = np.random.uniform(0, 1, (2, n_hitmiss))
    pseudo.append(np.mean((vec_xy_pseudo[1] <= func(vec_xy_pseudo[0]))))

    # Quasi_Halton
    vec_xy_halton = cp.create_halton_samples(n_hitmiss, 2)
    halton.append(np.mean((vec_xy_halton[1] <= func(vec_xy_halton[0]))))

    # Quasi_Sobol
    vec_xy_sobol = cp.create_sobol_samples(n_hitmiss, 2)
    sobol.append(np.mean((vec_xy_sobol[1] <= func(vec_xy_sobol[0]))))

    results_hitmiss = [np.round(pseudo, 5), np.round(halton, 5), np.round(sobol, 5)]

    return results_hitmiss

def imp_sampling(func): #calcula as estimativas no método Importance Sampling

    n_sampling = 555 #n obtido no EP 2

    #função auxiliar
    g = lambda x: (100 / 77) * (-0.46 * x + 1)

    #inversa da função auxiliar
    Ginv = lambda x: (-(100 / 77) +
                      np.sqrt((100 / 77) **
                              2 - 4 * ((-23 / 77)
                                       * (-x)))) / (-46 / 77)

    pseudo, halton, sobol = [], [], []

    # Pseudo Random
    sim = Ginv(np.random.uniform(0, 1, n_sampling))
    pseudo.append(np.mean(func(sim) / g(sim)))

    # Quasi_Halton
    sim = Ginv(cp.create_halton_samples(n_sampling, 1))
    halton.append(np.mean(func(sim) / g(sim)))

    # Quasi_Sobol
    sim = Ginv(cp.create_sobol_samples(n_sampling, 1))
    sobol.append(np.mean(func(sim) / g(sim)))

    results_sampling = [np.round(pseudo, 5), np.round(halton, 5), np.round(sobol, 5)]

    return results_sampling

def control_variate(func): #calcula as estimativas no método Importance Sampling

    #função de controle
    g = lambda x: 1 - x

    #calcula a constante c
    c = lambda vec: - np.cov(func(vec), g(vec))[0, 1] / np.var(g(vec))

    # Estimativa inicial da constante
    vec = np.array(cp.create_halton_samples(100, 1)[0])
    C = c(vec)

    pseudo, halton, sobol = [], [], []

    n_control = 552 #n obtido no EP 2

    # Pseudo Random
    vec_pseudo = np.array(np.random.uniform(0, 1, n_control))
    pseudo.append(np.mean(func(vec_pseudo) + C * (g(vec_pseudo) - 1 / 2)))

    # Quasi_Halton
    vec_halton = np.array(cp.create_halton_samples(n_control, 1)[0])
    halton.append(np.mean(func(vec_halton) + C * (g(vec_halton) - 1 / 2)))

    # Quasi_Sobol
    vec_sobol = np.array(cp.create_sobol_samples(n_control, 1)[0])
    sobol.append(np.mean(func(vec_sobol) + C * (g(vec_sobol) - 1 / 2)))

    results_control = [np.round(pseudo, 5), np.round(halton, 5), np.round(sobol, 5)]

    return results_control

main()

## Segundo bloco
## Tempo de execução: em torno de 10 minutos

def main_plot():

    rg = 0.508662102
    cpf = 0.43898402827

    func = lambda x: np.exp(-rg * x) * np.cos(cpf * x)

    space = np.logspace(1, np.log10(500000), 100, base=10)

    space_hit = np.logspace(1, np.log10(4800000), 100)

    space_sampling_control = np.logspace(1, np.log10(10000), 100)

    titulos = ["Método de Monte Carlo Crude",
               "Método de Monte Carlo Hit or Miss",
               "Método de Monte Carlo Importance Sampling",
               "Método de Monte Carlo Control Variate"]

    arquivos = ["crude.png", "hitmiss.png", "imp_sampling.png", "control_variate.png"]

    plotagem(crude_plot(func,space), titulos[0], arquivos[0], space)

    plotagem(hitmiss_plot(func, space_hit), titulos[1], arquivos[1], space_hit)

    plotagem(imp_sampling_plot(func, space_sampling_control), titulos[2], arquivos[2], space_sampling_control)

    plotagem(control_variate_plot(func, space_sampling_control), titulos[3], arquivos[3], space_sampling_control)

def crude_plot(funcao, logspace):

    #Pseudo random
    pseudo = np.array([np.mean(funcao(np.random.uniform(0,1,int(n)))) for n in logspace])

    #Quasi_Halton
    halton = np.array([np.mean(funcao(cp.create_halton_samples(int(n),1)[0])) for n in logspace])

    #Quasi_Sobol
    sobol = np.array([np.mean(funcao(cp.create_sobol_samples(int(n), 1)[0])) for n in logspace])

    results = [pseudo, halton, sobol]

    return results

def hitmiss_plot(funcao, logspace):

    pseudo, halton, sobol = [], [], []

    for n in logspace:

        vec_xy_pseudo = np.random.uniform(0, 1, (2, int(n)))
        pseudo.append(np.mean((vec_xy_pseudo[1] <= funcao(vec_xy_pseudo[0])).astype('int')))

        vec_xy_halton = cp.create_halton_samples(int(n), 2)
        halton.append(np.mean((vec_xy_halton[1] <= funcao(vec_xy_halton[0])).astype('int')))

        vec_xy_sobol = cp.create_sobol_samples(int(n), 2)
        sobol.append(np.mean((vec_xy_sobol[1] <= funcao(vec_xy_sobol[0])).astype('int')))

    results = [pseudo, halton, sobol]

    return results

def imp_sampling_plot(funcao, logspace):

    g = lambda x: (100 / 77) * (-0.46 * x + 1)
    Ginv = lambda x: (-(100 / 77) +
                      np.sqrt((100 / 77) **
                              2 - 4 * ((-23 / 77)
                                       * (-x)))) / (-46 / 77)

    pseudo = []
    halton = []
    sobol = []

    # Pseudo Random
    for n in logspace:
        sim = Ginv(np.random.uniform(0, 1, int(n)))
        pseudo.append(np.mean(funcao(sim) / g(sim)))

        sim = Ginv(cp.create_halton_samples(int(n), 1))
        halton.append(np.mean(funcao(sim) / g(sim)))

        sim = Ginv(cp.create_sobol_samples(int(n), 1))
        sobol.append(np.mean(funcao(sim) / g(sim)))

    results = [pseudo, halton, sobol]

    return results

def control_variate_plot(funcao, logspace):

    g = lambda x: 1 - x
    c = lambda vec: - np.cov(funcao(vec), g(vec))[0, 1] / np.var(g(vec))

    # Estimativa inicial da constante
    vec = np.array(cp.create_halton_samples(100, 1)[0])
    C = c(vec)

    pseudo = []
    halton = []
    sobol = []

    # Quasi
    for n in logspace:

        vec_pseudo = np.array(np.random.uniform(0, 1, int(n)))
        pseudo.append(np.mean(funcao(vec_pseudo) + C * (g(vec_pseudo) - 1 / 2)))

        vec_halton = np.array(cp.create_halton_samples(int(n), 1)[0])
        halton.append(np.mean(funcao(vec_halton) + C * (g(vec_halton) - 1 / 2)))

        vec_sobol = np.array(cp.create_sobol_samples(int(n), 1)[0])
        sobol.append(np.mean(funcao(vec_sobol) + C * (g(vec_sobol) - 1 / 2)))

    results = [pseudo, halton, sobol]

    return results

def plotagem(vectors, title, arquivo, space):

    plt.figure(figsize=(15,10) )
    plt.plot(space, vectors[0], c='goldenrod', label = "Pseudo-Random")
    plt.plot(space, vectors[1], c='teal', label = "Quasi-Random Halton")
    plt.plot(space, vectors[2], c='fuchsia', label = "Quasi-Random Sobol")
    plt.plot(space, np.full((len(space), 1), 0.761983), c='k', label = "Valor real da integral")
    plt.legend(prop={'size': 15})
    plt.xscale("log")
    plt.title(title, fontsize=15)
    plt.xlabel("Número de pontos", fontsize=15)
    plt.ylabel("Valor da integral", fontsize=15)
    plt.tick_params(labelsize = 15)
    plt.savefig(arquivo)
    plt.show()

def plot_padroes():

    fig, ax = plt.subplots(1, 3)
    s = 2
    n = 1000

    xy = np.random.uniform(0, 1, (2, n))
    ax[0].scatter(xy[0], xy[1], s=s)
    ax[0].set_title("Pseudo-Random", fontsize=15)
    xy = cp.create_halton_samples(n, 2)
    ax[1].scatter(xy[0], xy[1], s=s)
    ax[1].set_title("Quasi-Random Halton", fontsize=15)
    xy = cp.create_sobol_samples(n, 2)
    ax[2].scatter(xy[0], xy[1], s=s)
    ax[2].set_title("Quasi-Random Sobol", fontsize=15)
    fig.set_figwidth(20)
    plt.savefig("padroes.png")
    plt.show()

#main_plot()
#plot_padroes()




