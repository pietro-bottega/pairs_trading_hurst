
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 00:31:21 2025

@author: Renato, Natália, Júlia, Mateus e Alana.

Test_mode
na linha 30 altere de True para False para processamento integral
na linha 338 se necessário alterar número de pares pós-SSD para número maior que 2.000 porém aumenta esforço de processamento
também na linha 340 é possível alterar o número de pares pós-SSD no Test_mode

Modo de operação:
USE_SSD_ECM = True para usar SSD+ADF+ECM (código Modificado)
USE_SSD_ECM = False para usar apenas ADF (como código do Bruno)

"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from tqdm import tqdm
import itertools
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração do pandas para exibir todas as colunas e linhas
pd.set_option('display.max_columns', None)  # Mostra todas as colunas
pd.set_option('display.max_rows', None)     # Mostra todas as linhas
pd.set_option('display.width', None)        # Largura do display automática
pd.set_option('display.max_colwidth', None) # Mostra conteúdo completo das células
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.width', 1000)  # Aumenta a largura do display


from pathlib import Path
os.chdir(Path("C:/users/rgaye/tcc/"))  # Ajuste para o seu diretório

# =========================
# CONFIGURAÇÃO DO MODO DE OPERAÇÃO --- com SSD + ADF + ECM = True     Só ADF = False 
# =========================
USE_SSD_ECM = True  # True para usar SSD+ADF+ECM (código modificado), False para usar apenas ADF (código do Bruno)


# =========================
# CONFIGURAÇÃO DO TEST MODE
# =========================
TEST_MODE = False  # False para rodar o modo completo e True para Test_mode

if TEST_MODE:
    print("Rodando em modo de teste (rápido)")
    MAX_PERIODS = 1
    MAX_PAIRS = 20
    MAX_ASSETS = 100
    MAX_DAYS = 1000  #número de dias deve ser compatível com número de períodos
else:
    MAX_PERIODS = None
    MAX_PAIRS = 20
    MAX_ASSETS = None
    MAX_DAYS = None

s1218 = 1
years = 2015 - 1990 + 0.5
stop_loss = float('-inf')
duration_limit = 50

# MQO para encontrar o coeficiente de cointegração e criando a serie do spread
def OLS(data_ticker1, data_ticker2):
    spread = sm.OLS(data_ticker1, data_ticker2)
    spread = spread.fit()
    return data_ticker1 + (data_ticker2 * -spread.params[0]), spread.params[0]

# ADF test
def ADF(spread):
    return ts.adfuller(spread) # H0: Raiz unitária.

def ADF_test(data_ticker1, data_ticker2):
    ols = OLS(data_ticker1, data_ticker2)
    spread = ols[0]
    gamma = ols[1]
    return ts.adfuller(spread), gamma  # Substituímos ADF(spread) por ts.adfuller(spread)

# Implementar Teste SSD
def calculate_ssd(price1, price2):
    """
    Calcula a Soma dos Desvios Quadrados (SSD) entre dois ativos normalizados pelo primeiro dia.
    Quanto menor o SSD, mais próximo é o movimento dos preços.
    """
    # Normalização pelo primeiro dia
    norm_price1 = price1 / price1.iloc[0]
    norm_price2 = price2 / price2.iloc[0]

    # Calcula SSD dos preços normalizados
    ssd = np.sum((norm_price1 - norm_price2) ** 2)
    return ssd

# Nova função para estimar o ECM
def estimate_ecm(y, x, gamma):
    """
    Estima o Modelo de Correção de Erros (ECM) para o par (y, x).

    Parâmetros:
    y: Série dependente (log de preços do ativo 1)
    x: Série independente (log de preços do ativo 2)
    gamma: Coeficiente de cointegração estimado na regressão OLS

    Retorna:
    model: Modelo ECM estimado
    ecm_stats: Dicionário com estatísticas relevantes do ECM
    """
    try:
        # 1. Calcule o spread (resíduo da relação de cointegração)
        spread = y - gamma * x

        # 2. Calcule as primeiras diferenças
        dy = y.diff().dropna()
        dx = x.diff().dropna()
        spread_lag = spread.shift(1).dropna()

        # Ajuste os índices para alinhar as séries
        min_len = min(len(dy), len(dx), len(spread_lag))
        dy = dy[-min_len:]
        dx = dx[-min_len:]
        spread_lag = spread_lag[-min_len:]

        # 3. Monte o DataFrame para regressão
        df_ecm = pd.DataFrame({
            'dy': dy.values,
            'dx': dx.values,
            'spread_lag': spread_lag.values
        })

        # 4. Ajuste o modelo ECM: Δyₜ = α + βΔxₜ + γ(spread_{t-1}) + εₜ
        X = sm.add_constant(df_ecm[['dx', 'spread_lag']])
        model = sm.OLS(df_ecm['dy'], X).fit()

        # 5. Extraia estatísticas relevantes
        ecm_stats = {
            'adjustment_coef': model.params['spread_lag'],
            'adjustment_pvalue': model.pvalues['spread_lag'],
            'short_term_coef': model.params['dx'],
            'short_term_pvalue': model.pvalues['dx'],
            'r_squared': model.rsquared
        }

        return model, ecm_stats

    except Exception as e:
        print(f"Erro ao estimar ECM: {e}")
        return None, None

# Função para encontrar pares cointegrados usando SSD+ADF+ECM
def find_cointegrated_pairs_ssd(data, top_ssd_pairs=100, alpha=0.05):
    try:
        print(f"Finding cointegrated pairs for shape {np.shape(data)}")
        n = data.shape[1]
        keys = list(data.columns)
        pvalue_matrix = np.ones((n, n))
        gammas_matrix = np.ones((n, n))
        total_pairs = n * (n - 1) // 2

        print("Fase 1: Calculando SSD para todos os pares possíveis...")
        ssd_results = []
        for i, j in tqdm(itertools.combinations(range(n), 2), total=total_pairs, desc="Calculando SSD"):
            S1 = keys[i]
            S2 = keys[j]
            ssd = calculate_ssd(data[S1], data[S2])
            ssd_results.append((S1, S2, i, j, ssd))

        ssd_results.sort(key=lambda x: x[4])
        selected_pairs = ssd_results[:top_ssd_pairs]

        ssd_dict = {(x[0], x[1]): x[4] for x in selected_pairs}

        print(f"Fase 2: Testando cointegração e ECM nos {top_ssd_pairs} pares com menor SSD...")
        results = []
        for index, (S1, S2, i, j, ssd) in enumerate(tqdm(selected_pairs, desc="Testando cointegração")):
            # Teste ADF
            adf_result = ADF_test(data[S1], data[S2])
            gamma = adf_result[1]
            adf_pvalue = adf_result[0][1]

            # Teste ECM
            if USE_SSD_ECM:
                ecm_model, ecm_stats = estimate_ecm(data[S1], data[S2], gamma)
                ecm_pvalue = ecm_stats['adjustment_pvalue'] if ecm_stats is not None else None

                # Status OK somente se ambos os testes passarem (p-value < 0.05)
                status = "OK" if (adf_pvalue < alpha and ecm_pvalue is not None and ecm_pvalue < alpha) else ""
            else:
                ecm_pvalue = None
                # Status OK somente se o teste ADF passar (p-value < 0.05)
                status = "OK" if (adf_pvalue < alpha) else ""

            results.append({
                'SSD_Rank': index + 1,
                'SSD_value': round(ssd, 4),
                'Pair': f"{S1}-{S2}",
                'Gamma': gamma,
                'Coint_P-Value': round(adf_pvalue, 4),
                'ECM_P-Value': round(ecm_pvalue, 4) if ecm_pvalue is not None else None,
                'Status': status
            })

        coint_pairs_df = pd.DataFrame(results)

        # Reordena as colunas
        coint_pairs_df = coint_pairs_df[['Pair', 'SSD_value', 'SSD_Rank', 'Coint_P-Value', 'ECM_P-Value', 'Status', 'Gamma']]

        print("\nResultados dos Testes Estatísticos:\n")
        print(coint_pairs_df.to_string(index=False))

        return pvalue_matrix, gammas_matrix, ssd_dict, coint_pairs_df
    except Exception as e:
        print(e)
        return None, None, None, pd.DataFrame()

# Função para encontrar pares cointegrados usando apenas ADF (como no código do Bruno)
def find_cointegrated_pairs_adf(data):
    try:
        print(f"Finding cointegrated pairs for shape {np.shape(data)}")
        n = data.shape[1]
        pvalue_matrix = np.ones((n, n))
        gammas_matrix = np.ones((n, n))
        keys = list(data.columns)
        total_pairs = n * (n - 1) // 2  # Total de pares únicos

        with tqdm(total=total_pairs, desc="Calculando p-valores") as pbar:
            for i in range(n):
                for j in range(i+1, n):
                    S1 = keys[i]
                    S2 = keys[j]
                    result = ADF_test(data[S1], data[S2])
                    gammas_matrix[i, j] = result[1]  # gamma
                    pvalue = result[0][1]  # pvalue
                    pvalue_matrix[i, j] = pvalue
                    pbar.update(1)
            # Nenhum print dentro do loop!

        return pvalue_matrix, gammas_matrix
    except Exception as e:
        print(e)
        return None, None
        return None, None

# Ordenando os melhores pares (como no código 2)
def top_coint_pairs(data, pvalue_matrix, gamma, alpha, n): 
    #alpha = nivel de significancia para o teste ADF
    #n = top n ativos com o menor pvalue    
    alpha_filter = np.where(pvalues < alpha)
    pvalues_f = pvalues[alpha_filter] # pvalores menores que alpha
    stock_a = data.columns[list(alpha_filter)[0]] # relacionando o pvalor com a ação A
    stock_b = data.columns[list(alpha_filter)[1]] # relacionando o pvalor com a ação B
    gammas_f = gammas[alpha_filter] # relacionando o pvalor com o gamma
    N = len(list(alpha_filter[0])) # quantidade de pares cointegrados

    d = []
    for i in range(N):
        pair_dict = {
            'Stock A': stock_a[i],
            'Stock B': stock_b[i],
            'P-Values': pvalues_f[i],
            'Gamma': gammas_f[i]
        }
        d.append(pair_dict)

    return pd.DataFrame(d).sort_values(by="P-Values").iloc[:n,]

# Calcula os retornos da carteira e armazenando em um data frame
def calculate_profit(pair, spread, threshold, par1, par2, resumo, semester, gamma):
    date_format = "%Y-%m-%d"
    log_ret = spread.diff()  # log return eh o incremento
    dias = spread.index
    z_score = (spread-spread.mean())/spread.std()
    z_score.plot()  # Plotar o z-score (como no código 2)
    portfolio_return = []
    pos = 0  # 0: sem posição aberta
    dias_abertura = []
    dias_fechamento = []

    count = 0
    dia_abertura = 0
    dia_fechamento = 0

    closing_threshold = 0.5

    for i in z_score.index:
        if (z_score[i] > threshold) and (pos == 0):
            pos = -1
            count += 1
            dia_abertura = dias[i - dias[0]]
            retornos_op = []

        elif (z_score[i] < -threshold)  and (pos == 0):
            pos = 1
            count += 1
            dia_abertura = dias[i - dias[0]]
            retornos_op = []

        else:
            if (pos == 1) and (z_score[i] >= -closing_threshold or sum(retornos_op) < stop_loss):
                portfolio_return.append(log_ret[i]*pos)
                pos = 0

                dia_fechamento = dias[i - dias[0]]
                delta_dias = dia_fechamento - dia_abertura
                if sum(retornos_op) < stop_loss:
                    retorno_op = stop_loss
                else:
                    retornos_op.append(log_ret[i]*pos)
                    retorno_op = pd.Series(retornos_op).sum()

                Rpair[i-1, pair] = log_ret[i]*pos

                resumo.append([count, semester, dia_abertura, dia_fechamento, delta_dias, retorno_op, par1, par2, True])

            elif (pos == -1) and (z_score[i] <= closing_threshold or sum(retornos_op) < stop_loss):
                portfolio_return.append(log_ret[i]*pos)
                pos = 0

                dia_fechamento = dias[i - dias[0]]
                delta_dias = dia_fechamento - dia_abertura
                if sum(retornos_op) < stop_loss:
                    retorno_op = stop_loss
                else:
                    retornos_op.append(log_ret[i]*pos)
                    retorno_op = pd.Series(retornos_op).sum()

                Rpair[i-1, pair] = log_ret[i]*pos

                resumo.append([count, semester, dia_abertura, dia_fechamento, delta_dias, retorno_op, par1, par2, True])

            elif (pos == 1) and (z_score[i] < -closing_threshold):
                portfolio_return.append(log_ret[i]*pos)
                retornos_op.append(log_ret[i]*pos)
                Rpair[i-1, pair] = log_ret[i]*pos

            elif (pos == -1) and (z_score[i] > closing_threshold):
                portfolio_return.append(log_ret[i]*pos)
                retornos_op.append(log_ret[i]*pos)
                Rpair[i-1, pair] = log_ret[i]*pos

            else:
                if pos != 0:
                    dia_fechamento = dias[i - dias[0]]
                    delta_dias = dia_fechamento - dia_abertura
                    retornos_op.append(log_ret[i]*pos)
                    retorno_op = pd.Series(retornos_op).sum()
                    Rpair[i-1, pair] = log_ret[i]*pos
                    resumo.append([count, semester, dia_abertura, dia_fechamento, delta_dias, retorno_op, par1, par2, True])
                pos = 0

    if pos != 0:
        pos = 0
        dia_fechamento = i
        delta_dias = dia_fechamento - dia_abertura
        retorno_op = pd.Series(retornos_op).sum()
        resumo.append([count, semester, dia_abertura, dia_fechamento, delta_dias, retorno_op, par1, par2, False])

    total_ret = pd.Series(portfolio_return).sum()
    return total_ret, resumo

# Função para calcular o expoente de Hurst (do código 2)
def get_hurst_exponent(time_series):
    # Definindo o intervalo de taus
    max_tau = round(len(time_series)/4)
    taus = range(2, max_tau)

    # Calculando a variável k
    k = [np.std(np.subtract(time_series[tau:], time_series[:-tau])) for tau in taus]

    'To calculate the Hurst exponent, we first calculate the standard deviation of the differences between a series and its lagged version, for a range of possible lags.'

    # Calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(taus), np.log(k), 1)

    'We then estimate the Hurst exponent as the slope of the log-log plot of the number of lags versus the mentioned standard deviations.'

    return reg[0]

# Carregamento dos dados
print("\nCarregando dados...")
prices = pd.read_csv("cointegration_data/Pt_cointegration.csv")
returns = pd.read_csv("cointegration_data/Rt_cointegration.csv")
periods = pd.read_csv("cointegration_data/Periods.csv", header=None)
ticker2 = pd.read_csv("cointegration_data/ticker2.csv", header=None)
ticker_b = pd.read_csv("cointegration_data/ticker_b.csv", header=None)

# Aplicar limites do modo teste
if TEST_MODE:
    if MAX_ASSETS is not None:
        prices = prices.iloc[:, :MAX_ASSETS]
        returns = returns.iloc[:, :MAX_ASSETS]
        print(f"Número de ativos reduzido para {MAX_ASSETS}")
    if MAX_DAYS is not None:
        prices = prices.iloc[:MAX_DAYS, :]
        returns = returns.iloc[:MAX_DAYS, :]
        print(f"Número de dias reduzido para {MAX_DAYS}")

print("\nDimensões dos dados carregados:")
print(f"Preços: {prices.shape}")
print(f"Retornos: {returns.shape}")

# Conversão dos dados para log dos preços
log_data = np.log(prices)

# Inicialização de variáveis
no_pairs = MAX_PAIRS if TEST_MODE else 20
days, num_assets = np.shape(returns)
first_traininig = int(periods.iloc[0, 3])
Rpair = np.zeros((days, no_pairs))

resumos = []
alpha = 0.05
threshold = 1

read_csv = False
past_days = 0

# Configuração para SSD (apenas usado se USE_SSD_ECM = True)
if TEST_MODE:
    top_ssd_pairs = 200  # Menor número para testes
else:
    top_ssd_pairs = 500  # Número maior para produção

# Definir número de períodos
total_periods = int(years * 2 - 2)
if TEST_MODE and MAX_PERIODS is not None:
    total_periods = min(total_periods, MAX_PERIODS)
print(f"\nTotal de períodos a serem analisados: {total_periods}")

# Nova estrutura para armazenar resultados do ECM (apenas usado se USE_SSD_ECM = True)
if USE_SSD_ECM:
    ecm_results = []
    ecm_columns = [
        'Period', 'Stock A', 'Stock B', 'Gamma',
        'ECM_Adjustment_Coef', 'ECM_Adjustment_PValue',
        'ECM_ShortTerm_Coef', 'ECM_ShortTerm_PValue',
        'ECM_R_Squared'
    ]

    # Criar diretório para resultados do ECM se não existir
    ecm_dir = "cointegration_results/threshold_1/ecm_results"
    if not os.path.exists(ecm_dir):
        os.makedirs(ecm_dir)

# Loop principal
for big_loop in range(0, total_periods):
    print(f"\nIniciando período {big_loop+1}/{total_periods} | Past days: {past_days}")

    twelve_months = int(periods.iloc[big_loop, 3])
    six_months = int(periods.iloc[big_loop + 2, 0])

    # Limpeza das ações não listadas no período
    listed1 = log_data.iloc[past_days, :] > 0
    listed2 = log_data.iloc[past_days+int(twelve_months+six_months*(s1218 == 1))-1, :] > 0
    listed = np.multiply(listed1, listed2)
    listed_num = np.sum(listed)
    listed_indexes = np.where(listed > 0)[0]
    listed_stocks = log_data.columns[listed_indexes]

    try:
        [D, ia, ib] = np.intersect1d(
            ticker2.iloc[:, big_loop], ticker2.iloc[:, big_loop+1], return_indices=True)

        ic = np.isin(D, ticker2.iloc[:, big_loop+2])

        Dic_unique_sorted, B_idx = np.unique(D[ic], return_index=True)

        listed_union = np.intersect1d(listed_stocks, Dic_unique_sorted)

        index_listed2 = [log_data.columns.get_loc(i) for i in listed_union if i in log_data]
        index_listed2.sort()

        if os.path.exists(f"cointegration_data/pvalues_semester_{big_loop}.csv"):
            print("Lendo arquivos CSV existentes...")
            pvalues = np.genfromtxt(f"cointegration_data/pvalues_semester_{big_loop}.csv", delimiter=',')
            gammas = np.genfromtxt(f"cointegration_data/gammas_semester_{big_loop}.csv", delimiter=',')
            # Inicializar coint_pairs_df como DataFrame vazio
            coint_pairs_df = pd.DataFrame()
        else:
            print("Gerando novos arquivos CSV...")
            if USE_SSD_ECM:
                # Modo completo: SSD + ADF + ECM
                pvalues, gammas, ssd_dict, coint_pairs_df = find_cointegrated_pairs_ssd(
                    log_data.iloc[past_days:(past_days+twelve_months), index_listed2],
                    top_ssd_pairs=top_ssd_pairs
                )
            else:
                # Modo simples: apenas ADF (como no código 2)
                pvalues, gammas = find_cointegrated_pairs_adf(
                    log_data.iloc[past_days:(past_days+twelve_months), index_listed2]
                )

        print(f"\nAnálise de cointegração concluída para o período {big_loop+1}")

        # Processamento dos pares cointegrados
        if USE_SSD_ECM:
            # Modo completo: filtrar por Status e ordenar por SSD
            if not coint_pairs_df.empty:
                coint_pairs_df = coint_pairs_df[coint_pairs_df['Status'] == 'OK'].copy()
                coint_pairs_df = coint_pairs_df.sort_values('SSD_value').head(no_pairs).copy()
                coint_pairs_df = coint_pairs_df.reset_index(drop=True)

                # Criar coluna de numeração começando em 1
                to_show = coint_pairs_df.copy()
                to_show['Número'] = to_show.index + 1
                print(f"\nPares cointegrados selecionados para o período {big_loop+1}:\n")
                print(to_show[['Número', 'Pair', 'SSD_value', 'SSD_Rank', 'Coint_P-Value', 'ECM_P-Value', 'Gamma']].to_string(index=False))
        else:
            # Modo simples: usar top_coint_pairs (como no código 2)
            try:
                coint_pairs_df = top_coint_pairs(
                    log_data.iloc[past_days:(past_days+twelve_months), index_listed2], 
                    pvalues, gammas, alpha, no_pairs
                )
                coint_pairs_df['P-Values'] = coint_pairs_df['P-Values'].apply(lambda x: f"{x:.8f}")
                # Adiciona a coluna de ranking
                coint_pairs_df['Ranking'] = range(1, len(coint_pairs_df) + 1)
        # Reseta o índice e adiciona 1
                coint_pairs_df = coint_pairs_df.reset_index(drop=True)
                coint_pairs_df.index = coint_pairs_df.index + 1
                print(f"Found top cointegrated pairs big_loop {big_loop}")
                print(coint_pairs_df.to_string())  # Adicionado para imprimir os pares
            except Exception as e:
                print(e)
                print("Exception ao achar top pares cointegrados")
                continue

        # Processamento dos pares selecionados
        if USE_SSD_ECM:
            # Modo completo: usar a estrutura do Pair
            for i in range(0, coint_pairs_df.shape[0]):
                try:
                    pair = coint_pairs_df.iloc[i]['Pair']
                    S1_name, S2_name = pair.split('-')
                    gamma_1_2 = coint_pairs_df.iloc[i]['Gamma']

                    # Dados para o período de formação
                    S1_formation = log_data[S1_name].iloc[past_days:(past_days+twelve_months)]
                    S2_formation = log_data[S2_name].iloc[past_days:(past_days+twelve_months)]

                    # Estimar ECM (apenas se USE_SSD_ECM = True)
                    if USE_SSD_ECM:
                        ecm_model, ecm_stats = estimate_ecm(S1_formation, S2_formation, gamma_1_2)
                        ecm_pvalue = ecm_stats['adjustment_pvalue'] if ecm_stats is not None else None

                    # Dados para o período de trading
                    S1 = log_data[S1_name].iloc[(past_days+twelve_months):(past_days+twelve_months+six_months)]
                    S2 = log_data[S2_name].iloc[(past_days+twelve_months):(past_days+twelve_months+six_months)]

                    spread = S1 - gamma_1_2*S2
                    ret, resumos = calculate_profit(i, spread, threshold, S1_name, S2_name, resumos, big_loop, gamma_1_2)
                except KeyError as e:
                    print(f"Erro ao processar par {pair}: {e}")
                    continue
        else:
            # Modo simples: usar a estrutura do Stock A/Stock B (como no código 2)
            for i in range(0, coint_pairs_df.shape[0]):
                S1_name = coint_pairs_df.iloc[i, 0]  # Stock A
                S2_name = coint_pairs_df.iloc[i, 1]  # Stock B
                gamma_1_2 = coint_pairs_df.iloc[i, 3]  # Gamma

                S1 = log_data[S1_name].iloc[(past_days+twelve_months):(past_days+twelve_months+six_months)]
                S2 = log_data[S2_name].iloc[(past_days+twelve_months):(past_days+twelve_months+six_months)]

                # Spread
                spread = S1 - gamma_1_2*S2
                # Pegando o resultado da estratégia
                ret, resumos = calculate_profit(i, spread, threshold, S1_name, S2_name, resumos, big_loop, gamma_1_2)
    except Exception as e:
        print(f"Erro no processamento do período {big_loop+1}: {e}")

    past_days = past_days + periods.iloc[big_loop, 0]

# Salvando os resultados tradicionais
print("\nSalvando resultados...")
cols = ['Operação', 'Semestre', 'Abertura', 'Fechamento', 'Dias', 'Retorno total', 'Ticker 1', 'Ticker 2', 'Converged']
df = pd.DataFrame(resumos, columns=cols)
df['Index'] = df['Ticker 1'].astype(str) + '-' + df['Ticker 2'].astype(str) + '-' + df['Operação'].astype(str)
df['Retorno total - exp'] = np.exp(df['Retorno total'])
df.to_csv("cointegration_results/threshold_1/operations.csv", sep=',', index=False)

# Salvando os resultados do ECM (apenas se USE_SSD_ECM = True)
#if USE_SSD_ECM:
#   df_ecm = pd.DataFrame(ecm_results, columns=ecm_columns)
#    df_ecm.to_csv(f"{ecm_dir}/ecm_analysis.csv", index=False)
#    print(f"Resultados do ECM salvos em {ecm_dir}/ecm_analysis.csv")

pd.DataFrame(Rpair).to_csv("cointegration_results/threshold_1/Rpair.csv", header=None, index=False)
daily_returns = np.sum(Rpair, axis=1)
pd.DataFrame(daily_returns).to_csv("cointegration_results/threshold_1/daily_returns.csv", header=None, index=False)

print("\nProcessamento concluído!")
