"""
Fornece funções comuns para manipulação de arquivos e dados nas análises de espectroscopia gamma.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy.integrate import simps


data_dir = '/projects/14c3d9b6-c022-4309-a4b7-f0496c815d29/dados/'
pars_ics = [0.00078468, 2.36246, 12.4706]
pars_python = [0.000663857983, 2.43873365, -1.38995037]
err_pars_python = [0.0000375338450, 0.0250880598, 3.44567325]


def load_roi_info(arquivo):
    """
    Retorna 6-upla com dados das regiões de interesse detectadas pelo programa
    de captura e registradas em arquivo.
    arquivo: (str) caminho para arquivo .tsv com dados da coleta

    retorna:
        low, high, gross, net, fwhms, centroids (6-uplas de np.ndarray de floats)
    """
    reading = False

    with open(arquivo, 'r')as f, open('temp', 'w') as out_f:
        for line in f:
            if "FWHM" in line:
                reading = True
                continue
            if reading and len(line) >= 2:
                print(line.strip('\n'), file=out_f)
            elif reading and len(line) < 2:
                break
    low = np.loadtxt('temp', delimiter='\t', skiprows=0, usecols=(1,), ndmin=1)
    high = np.loadtxt('temp', delimiter='\t', skiprows=0, usecols=(2,), ndmin=1)
    gross = np.loadtxt('temp', delimiter='\t', skiprows=0, usecols=(3,), ndmin=1)
    net = np.loadtxt('temp', delimiter='\t', skiprows=0, usecols=(4,), ndmin=1)
    fwhms = np.loadtxt('temp', delimiter='\t', skiprows=0, usecols=(5,), ndmin=1)
    centroids = np.loadtxt('temp', delimiter='\t', skiprows=0, usecols=(6,), ndmin=1)

    # convertendo low e high e gross para inteiros
    low, high, gross = map(lambda x: x.astype(int), [low, high, gross])

    return low, high, gross, net, fwhms, centroids


def load_live_time(arquivo):
    """
    Carrega o tempo de exposição do espectro em arquivo.
    """
    with open(arquivo, 'r') as f:
        for line in f:
            if "Live Time" in line:
                live_time = float(line.split('\t')[-1])
                break
    return live_time


def load_lines_and_intensities(amostra, intensidade_cut):
    """
    Retorna lista com valores padrão das transições e intensidades do material amostra
    obtidas de http://www.nucleide.org/DDEP_WG/DDEPdata_by_A.htm
    amostra: (str) material e.g.: "na22", "ba133"
    intensidade_cut: (float) corte de intensidade (desprezar linhas com intensidade menor que este valor)

    retorna:
    dupla de listas (energias (em keV), intensidades (em %))
    """
    data_f = os.path.join(data_dir, 'linhas_espectrais/', amostra + '.txt')
    skip_line = 1
    with open(data_f, 'r') as f:
        for line in f:
            if 'Energy ' in line:
                break
            skip_line += 1
    energies = np.loadtxt(data_f, delimiter=';', skiprows=skip_line, usecols=(0,), comments='=')
    intensities = np.loadtxt(data_f, delimiter=';', skiprows=skip_line, usecols=(2,), comments='=')
    energies = energies[intensities > intensidade_cut]
    intensities = intensities[intensities > intensidade_cut]
    return energies, intensities


def load_standard_lines(amostra, intensidade_cut=0):
    """
    Retorna lista com valores padrão das transições do material amostra,
    obtidas de http://www.nucleide.org/DDEP_WG/DDEPdata_by_A.htm
    amostra: (str) material e.g.: "na22", "ba133"
    intensidade_cut: (float) corte de intensidade (desprezar linhas com intensidade menor que este valor)

    retorna:
    lista de energias (em keV)
    """
    energies, _ = load_lines_and_intensities(amostra, intensidade_cut)
    return energies


def calibracao_ics(c):
    """Dado um canal retorna a energia associada (na calibracao do ICS)"""
    return calibracao(c, pars_ics)


def calibracao(x, pars):
    """
    Dada uma lista pars retorna o polinômio com coeficientes em pars avaliado em c, onde
    pars[0] corresponde ao coeficiente do monômio de maior grau

    pars: lista de floats
    c: float ou np.ndarray

    retorna: energy (mesmo tipo de c)
    """
    degree = len(pars)
    exps = np.arange(len(pars)-1, -1, -1)
    try:
        x_pows = np.array([k ** exps for k in x])
    except TypeError:
        x_pows = x ** exps
    try:
        return np.sum(pars_ics * x_pows, axis=1)
    except ValueError:
        return np.sum(pars_ics * x_pows)


def load_data(arquivo):
    """
    Retorna dados (canais e contagens) em arquivo
    arquivo: (str) caminho para arquivo .tsv contendo as medições

    retorna:
    dupla (canais, contagens) de arrays do numpy com os canais e as respectivas contagens
    """
    line_number = 1
    data_delimiter = 'Data:'
    with open(arquivo, 'r') as f:
        for line in f:
            if data_delimiter in line:
                break
            else:
                line_number += 1
    canais = np.loadtxt(arquivo, delimiter='\t', skiprows=line_number, usecols=(0,))
    contagens = np.loadtxt(arquivo, delimiter='\t', skiprows=line_number, usecols=(1,))
    return (canais, contagens)


def load_peaks(arquivo, amostra, pars=None, cut=0, return_channel=False):
    """
    Retorna fwhms e centroides das linhas detectadas pelo programa de aquisição
    e registradas no espectro em arquivo, descartando linhas cujos valores 
    são muito diferentes dos valores padrão 
    (obtidos de http://www.nucleide.org/DDEP_WG/DDEPdata_by_A.htm).

    arquivo (str): caminho até arquivo .tsv contendo dados adquiridos
    amostra (str): elemento referente ao espectro (e.g. "Na22" ou "Cs137")
    pars (list of float): parâmetros do ajuste polinomial (sendo pars[0] o coeficiente de maior grau)
    return_channel (bool): retornar canais ou energias
    cut (float): desconsiderar linhas com intensidade menor que cut

    retorna: fwhms, centroids
    dupla de arrays do numpy contendo energias em keV(ou canais) dos fwhms e centroids das linhas medidas.
    """
    if pars is None:
        pars = pars_ics
    _, _, _, _, fwhms, centroids = load_roi_info(arquivo)
    lefts = centroids - fwhms
    rights = centroids + fwhms
    centroid_energies = calibracao_ics(centroids)
    left_energies = calibracao_ics(lefts)
    right_energies = calibracao_ics(rights)
    fwhm_energies = right_energies - left_energies

    centroids_final = []
    fwhms_final = []
    for centroid, fwhm in zip(centroid_energies, fwhm_energies):
        if np.any(np.abs(centroid - load_standard_lines(amostra, cut)) < 15) and centroid > 30:
            centroids_final.append(centroid)
            fwhms_final.append(fwhm)

    if return_channel:
        return fwhms, centroids
    return fwhms_final, centroids_final

def get_peak_points(arquivo, amostra, pars=None, cut=0, return_channel=False):
    """
    Retorna par (x, y, xerr) com as coordenadas dos picos de amostra presentes em arquivo
    """
    centroids, centroids_errs = get_peak_energies_w_uncertainty(arquivo, pars=None, return_channel=True)
    channels, counts = load_data(arquivo)

    indexes = channels.searchsorted(centroids)
    err_x = centroids_errs
    peak_y = counts[indexes]
    peak_x = centroids

    if return_channel is False:
        if pars is None:
            pars = pars_ics
        peak_x = calibracao(peak_x, pars)
    return peak_x, peak_y, err_x



def smooth(data, scale):
    """
    Suaviza dados medidos realizando uma média ponderada ao redor de cada ponto,
    com pesos gaussianos. scale define o desvio padrão da gaussiana.
    """
    b = gaussian(100, scale)
    return filters.convolve1d(data, b/b.sum())


def load_corrected_counts(arquivo):
    lows, highs, _, net_counts, _, _ = load_roi_info(arquivo)

    canais, contagens = load_data(arquivo)
    corrected_counts = []
    for low, high, net_count in zip(lows, highs, net_counts):
        total_area = simps(contagens[low:high], canais[low:high])
        baseline_area = simps([contagens[low], contagens[high]], dx=(high-low))
        correction_factor = 1 - baseline_area / total_area
        corrected_count = correction_factor * net_count
        corrected_counts.append(corrected_count)
    return corrected_counts


def gaussian_with_baseline(x, mu, sgm, A, a, b):
    return (A * np.exp(-((x - mu) / sgm) ** 2) + a * x + b )

def get_peak_energies_w_uncertainty(arquivo, pars=None, return_channel=False):
    if pars is None:
        pars = pars_ics
    lows, highs, _, _, fwhms, old_centroids = load_roi_info(arquivo)

    channels, counts = load_data(arquivo)

    centroids = []
    centroids_err = []
    for low, high, fwhm, guess_centroid in zip(lows, highs, fwhms, old_centroids):
        if low < 0 or high < 0 or fwhm < 0 or guess_centroid < 0:
            continue
        guess_std = fwhm / 2.355
        guess_a = (counts[high] - counts[low]) / (high - low)
        guess_b = counts[low] - guess_a * low
        pvar, pcov = curve_fit(gaussian_with_baseline, channels[low:high], counts[low:high],
                               p0 = (guess_centroid, guess_std, 100000, guess_a, guess_b))
        perr = np.sqrt(np.diag(pcov))
        centroid, centroid_err = pvar[0], perr[0]
        if not return_channel:
            centroid = calibracao(centroid, pars)
            centroid_left = centroid - centroid_err
            centroid_right = centroid + centroid_err
            centroid_left = calibracao(centroid_left, pars)
            centroid_right = calibracao(centroid_right, pars)
            centroid_err = centroid_right - centroid_left
        centroids.append(centroid)
        centroids_err.append(centroid_err) 
    return np.array(centroids), np.array(centroids_err)



def load_net_count_rate(arquivo):
    """
    Carrega e calcula a atividade das linhas espectrais de arquivo.
    """
    lows, highs, _, net_counts, _, _ = load_roi_info(arquivo)

    canais, contagens = load_data(arquivo)
    corrected_counts = []
    for low, high, net_count in zip(lows, highs, net_counts):
        total_area = simps(contagens[low:high], canais[low:high])
        baseline_area = simps([contagens[low], contagens[high]], dx=(high-low))
        correction_factor = 1 - baseline_area / total_area
        corrected_count = correction_factor * net_count
        corrected_counts.append(corrected_count)
    return np.array(corrected_counts)/load_live_time(arquivo)


def filter_peaks(picos, isotopos, cut=0.0, tolerance=15.0):
    """
    Retorna picos presentes em picos cujas energias estao na lista
    padrao de linhas espectrais dos elementos em isotopos com intensidade
    superior a cut, dentro da tolerancia tolerance (em keV)
    (valores padrao obtidos de http://www.nucleide.org/DDEP_WG/DDEPdata_by_A.htm).

    picos (float): energias dos picos a filtrar (em keV)
    isotopos (list of str): lista de isotopos a buscar (e.g. "Na22")
    cut (float): desconsiderar linhas com intensidade menor que cut
        padrao: 0.0
    tolerance (float): tolerancia no valor da energia das linhas (em keV)
        padrao: 15.0

    retorna
        lista de picos suficientemente proximos de algum pico tabelado 
        para algum dos isotopos
    """
    picos_final = []
    indexes = []
    search_lines = []
    search_intensities = []
    search_isotopes = []
    testemunhas_final = {}
    for isotopo in isotopos:
        isotope_lines = list(load_standard_lines(isotopo, cut))
        _, isotope_intensities = map(list, load_lines_and_intensities(isotopo, cut))
        search_lines.extend(isotope_lines)
        search_intensities.extend(isotope_intensities)
        search_isotopes.extend([isotopo for line in isotope_lines])
    for i, pico in enumerate(picos):
        close_search_lines = np.abs(pico - search_lines) < tolerance
        if np.any(close_search_lines) and pico > 30:
            picos_final.append(pico)
            indexes.append(i)
            testemunhas = [
                (line, isotope, intensity) 
                for line, isotope, intensity, close in zip(search_lines, search_isotopes, search_intensities, close_search_lines)
                if close
            ]
            testemunhas_final[pico] = testemunhas

    return picos_final, indexes, testemunhas_final