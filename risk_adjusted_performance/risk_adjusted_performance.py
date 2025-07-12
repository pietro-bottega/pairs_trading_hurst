import pandas as pd
import numpy as np

###################
# DISTANCE
###################

# Carrega os dados
df_returns = pd.read_csv("../distance_results/Rp_ew_cc.csv")  # 'Return' e 'Semester'
df_rf = pd.read_csv("../distance_results/risk_free.csv")      # 'Return' e 'Semester'

# Renomeia a coluna da taxa livre de risco
df_rf = df_rf.rename(columns={"Return": "RiskFree"})

# Faz o merge
df = pd.merge(df_returns, df_rf, on="Semester")

# Função para calcular métricas
def calcular_metricas_completas(returns, risk_free, target=0.0):
    if len(returns) == 0 or returns.std() == 0:
        return pd.Series({m: np.nan for m in [
            "Sharpe", "Sortino", "Kappa_3", "Omega", "VaR_95", "CVaR_95",
            "Max_Drawdown", "Calmar", "Sterling", "Burke"
        ]})
    
    rf = risk_free.mean()
    excess_return = returns.mean() - rf
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdowns = cumulative / peak - 1

    downside = returns[returns < target]
    downside_std = np.sqrt((downside**2).mean())
    lpm3 = ((np.maximum(target - returns, 0))**3).mean()
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    max_dd = drawdowns.min()

    threshold = -0.10
    worst_dds = drawdowns[drawdowns < threshold]
    mean_dd = abs(worst_dds.mean()) if not worst_dds.empty else np.nan
    squared_dd_sum = (drawdowns[drawdowns < 0] ** 2).sum()
    burke_denom = np.sqrt(squared_dd_sum)

    return pd.Series({
        "Sharpe": excess_return / returns.std(),
        "Sortino": excess_return / downside_std,
        "Kappa_3": excess_return / (lpm3**(1/3)),
        "Omega": ((returns[returns > target] - target).sum()) / ((target - returns[returns < target]).sum()),
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Max_Drawdown": max_dd,
        "Calmar": (returns.mean() / abs(max_dd)) if max_dd != 0 else np.nan,
        "Sterling": (returns.mean() / mean_dd) if mean_dd != 0 else np.nan,
        "Burke": (returns.mean() / burke_denom) if burke_denom != 0 else np.nan
    })

# Loop por semestre
metricas_list = []
for semestre in df["Semester"].unique():
    grupo = df[df["Semester"] == semestre]
    metricas = calcular_metricas_completas(grupo["Return"], grupo["RiskFree"])
    metricas["Semester"] = semestre
    metricas_list.append(metricas)

# Junta tudo
distance_metricas_por_semestre = pd.DataFrame(metricas_list)

# Visualiza
distance_metricas_por_semestre = distance_metricas_por_semestre[["Semester"] + [col for col in distance_metricas_por_semestre.columns if col != "Semester"]]

# OUTPUT RESULTS
distance_metricas_por_semestre.to_csv(f"../distance_results/distance_risk_adjusted_measures.csv")

###################
# COINTEGRATION
###################

# Carregar os dados
df = pd.read_csv("../cointegration_results/operations_SSD_ECM.csv")  

# Garantir tipos corretos
df["Retorno total"] = pd.to_numeric(df["Retorno total"], errors="coerce")
df["Dias"] = pd.to_numeric(df["Dias"], errors="coerce")
df["Semestre"] = pd.to_numeric(df["Semestre"], errors="coerce")

# Retorno diário
df["Retorno diário"] = df["Retorno total"] / df["Dias"]

# Parâmetros
risk_free_rate = 0.0
target_return = 0.0
confidence_level = 0.95

# Função para calcular métricas de risco
def calcular_metricas(daily_returns):
    if len(daily_returns) < 2:
        return [np.nan]*10

    sharpe = (daily_returns.mean() - risk_free_rate) / daily_returns.std()

    downside = daily_returns[daily_returns < target_return]
    sortino = (daily_returns.mean() - target_return) / downside.std() if downside.std() > 0 else np.nan

    downside_m3 = ((target_return - downside) ** 3).mean()
    kappa_3 = (daily_returns.mean() - target_return) / (downside_m3 ** (1/3)) if downside_m3 > 0 else np.nan

    pos = daily_returns[daily_returns > target_return] - target_return
    neg = target_return - daily_returns[daily_returns < target_return]
    omega = pos.sum() / neg.sum() if neg.sum() > 0 else np.nan

    var = np.percentile(daily_returns, 100 * (1 - confidence_level))
    cvar = daily_returns[daily_returns <= var].mean()

    cumulative = (1 + daily_returns).cumprod()
    max_run = cumulative.cummax()
    drawdown = (cumulative - max_run) / max_run
    max_dd = drawdown.min()

    ann_return = daily_returns.mean() * 252
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    dd_threshold = drawdown[drawdown < -0.10]
    avg_dd = dd_threshold.abs().mean()
    sterling = ann_return / avg_dd if avg_dd > 0 else np.nan

    burke_den = np.sqrt(np.sum(drawdown[drawdown < 0] ** 2))
    burke = ann_return / burke_den if burke_den > 0 else np.nan

    return [sharpe, sortino, kappa_3, omega, var, cvar, max_dd, calmar, sterling, burke]

# Inicializa DataFrame de resultados
resultados = []

# Agrupar por semestre e calcular métricas
for semestre, grupo in df.groupby("Semestre"):
    retornos = grupo["Retorno diário"].dropna()
    metrics = calcular_metricas(retornos)
    resultados.append([semestre] + metrics)

# Converter para DataFrame
colunas = ["Semestre", "Sharpe Ratio", "Sortino Ratio", "Kappa 3", "Omega Ratio", 
           "VaR (95%)", "CVaR (95%)", "Maximum Drawdown", "Calmar Ratio", 
           "Sterling Ratio", "Burke Ratio"]

cointegration_df_metricas = pd.DataFrame(resultados, columns=colunas)

# Ordenar por semestre
cointegration_df_metricas = cointegration_df_metricas.sort_values(by="Semestre").reset_index(drop=True)

cointegration_df_metricas.to_csv(f"../cointegration_results/cointegration_risk_adjusted_measures.csv")
