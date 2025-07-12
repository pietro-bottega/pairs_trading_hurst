import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import scipy.stats
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning) #remove python standard warning and have clear output


periods = pd.read_csv("../distance_data/Periods.csv", header=None)
Pt = pd.read_csv("../distance_data/Pt_formatted.csv", header=0)
Rt = pd.read_csv("../distance_data/Rt_formatted.csv", header=0)
Rm = pd.read_csv("../distance_data/Rm.csv", header=None)
Rf = pd.read_csv("../distance_data/Rf.csv", header=None)
Vt = pd.read_csv("../distance_data/Vt.csv", header=None)
ticker2 = pd.read_csv("../distance_data/ticker2.csv", header=None)
ticker_b = pd.read_csv("../distance_data/ticker_b.csv", header=None)

# --------------------------------------
# Defining parameters for analysis
# --------------------------------------

days, num_assets = np.shape(Rt)

print(f"Days: {days} | Assets: {num_assets}")

daylag = 0
wi_update = 1
years = 2015 - 1990 + 0.5

no_pairs = 10 # number of pairs to be selected and used in operations
candidate_pool_size = 500 # We will first select the 500 best pairs using the distance method.

trading_costs = 0 # absolute trading cost
percentage_costs = 0.002 # buy/sell (percentage cost for opening and closing pairs: 0.001, 0.002, for example)
trade_req = 0 # set whether (0) or not (2) positive trading volume is required for opening/closing a pair
Stop_loss = 0.95 # Choose how much loss we are willing to accept on a given pair, compared to 1, i.e, 0.93 = 7% stop loss

if Stop_loss != float('-inf'):
    stop_dir = 100 - (Stop_loss * 100)
Stop_gain = float('inf') # Choose how much gain we are willing to accept on a given pair, compared to 1, i.e 1.10 = 10% stop gain, set float('inf') if no limit for gain
s1218 = 1  # listing req. (look ahead): 12+6 months (=1)

opening_threshold = 1.5
closing_threshold = 0.75

duration_limit = float('inf')

avg_price_dev = np.zeros(
    (days-sum(periods.iloc[0:2, 0].to_list()), no_pairs*2))
# 12 months are w/o price deviations. The first 12 months are formation period

# --------------------------------------
# Creating blank matrices and arrays to store
# --------------------------------------

# Operations array
operations = []

# Keeps track of the return of each pair
first_traininig = int(periods.iloc[0, 3])
Rpair = np.zeros((days-first_traininig, no_pairs))
Rp_ew_cc = pd.DataFrame(np.zeros((days-first_traininig, 2)),
                        columns=['Return', 'Semester'])
Rp_vw_fi = pd.DataFrame(np.zeros((days-first_traininig, 2)),
                        columns=['Return', 'Semester'])
ret_acum_df = pd.DataFrame(
    np.zeros((days-first_traininig, 4)), columns=['CC', 'FI', 'RMRF', 'SEMESTER'])
RmRf = np.zeros((days-first_traininig, 1))
risk_free = pd.DataFrame(np.zeros((days-first_traininig, 2)),
                        columns=['Return', 'Semester'])

periods_with_open_pair = 0  # number of periods with pairs opened
periods_without_open_pair = 0  # number of periods without pairs opened
pairs_number = 0
pair_open = 0
days_open = np.zeros((no_pairs*10000, 1))
# measures number of days each pair open; bad programming, but we do not know how many pairs we get
# measures number of times pairs opened in each pair per 6 month period
no_pairs_opened = np.zeros((int(years*2-2), no_pairs))

counter = 0  # Keeps track of the days in the main loop

asset_frequency_counter = Counter() # later used to track how many times each stock is selected

# --------------------------------------
# Defining functions for Hurst Exponent 
# --------------------------------------

def calculate_rs_for_lag(time_series, lag):
    """
    Calculate the Rescaled Range (R/S) for a specific lag.

    Args:
        time_series (array-like): The time series data.
        lag (int): The lag to use for the calculation.

    Returns:
        float: The Rescaled Range (R/S) value.
    """
    # 1. O procedimento a seguir é realizado para diversos subperíodos ou partições dentro da série original
    sub_series_list = [time_series[i:i+lag] for i in range(0, len(time_series), lag)]

    # 2. Cada série temporal é convertida em uma série de totais acumulados dos seus desvios em relação à média.
    mean_adjusted_cumulative_sums = []
    for sub_series in sub_series_list:
        mean = np.mean(sub_series)
        mean_adjusted_series = sub_series - mean
        cumulative_sum = np.cumsum(mean_adjusted_series)
        mean_adjusted_cumulative_sums.append(cumulative_sum)

    # 3. Cálculo da estatística "Rescaled Range" ou R/S
    ranges = []
    std_devs = []
    for i in range(len(sub_series_list)): 
        # 3.1. O "Range" (R) é então definido como a diferença entre o valor máximo e o mínimo desta série acumulada.
        ranges.append(np.max(mean_adjusted_cumulative_sums[i]) - np.min(mean_adjusted_cumulative_sums[i]))
        # 3.2. O "Standard Deviation" (S) ou desvio padrão é calculado para cada série
        std_devs.append(np.std(sub_series_list[i]))

    # 4. O valor de R é normalizado ao ser dividido por S das observações da série original.
    avg_rescaled_range = np.mean([ranges[i] / std_devs[i] for i in range(len(ranges)) if std_devs[i] > 0])

    return avg_rescaled_range

def calculate_hurst_rs(time_series):
    """
    Calculate the Hurst exponent of a time series using Rescaled Range (R/S) analysis.

    Args:
        time_series (array-like): The time series data.

    Returns:
        float: The Hurst exponent.
    """
    n = len(time_series)
    if n < 20:
        raise ValueError("Time series must have at least 20 data points.")

    # 5. Criar uma lista de partições
    lags = range(2, n)

    # 6. Calcular o R/S para cada uma das partições:
    rescaled_ranges = [calculate_rs_for_lag(time_series, lag) for lag in lags]

    # 7. Estimativa via Regressão Linear
    # A relação entre os valores de R/S e os seus respectivos tamanhos de partição (n) segue uma lei de potência.
    # Ao aplicar logaritmos a essa relação, o expoente de Hurst é finalmente estimado por meio de uma regressão linear simples
    # Correspondendo ao coeficiente angular da reta que ajusta o logaritmo de R/S em função do logaritmo de n.
    hurst_exponent = np.polyfit(np.log(lags), np.log(rescaled_ranges), 1)[0]

    return hurst_exponent

# ----------------------------------------------------
# Start of Main Loop - Creating Price Index
# ----------------------------------------------------

big_loop = 0
i = 0

while big_loop < (years * 2 - 2):
    twelve_months = int(periods.iloc[big_loop, 3])
    six_months = int(periods.iloc[big_loop + 2, 0])

    print(
        f"\n================ Period {big_loop} ================")
    print(
        f"Formation Period: {twelve_months} days | Trading Period: {six_months} days")
    

    # ----------------------------------------------------
    # Create price index IPt by setting first Pt>0 to 1
    # ----------------------------------------------------

    # Preallocate a zeros matrix with the size of the Formation + Trading period
    # IPt = Indexed Price at time t
    IPt = np.zeros((int(twelve_months + six_months), num_assets))

    print("Generating Assets Price Index")

    for j in tqdm(range(0, num_assets)):
        m = 0
        for i2 in range(0, int(twelve_months+six_months)):  # same here
            if not math.isnan(Pt.iloc[i+i2, j]) and m == 0:
                IPt[i2, j] = 1
                m = 1
            elif not math.isnan(Pt.iloc[i+i2, j]) and m == 1:
                IPt[i2, j] = IPt[i2-1, j] * (1 + Rt.iloc[i+i2, j])

    pd.DataFrame(IPt).to_csv("IPt.csv", header=None,
                             index=False, na_rep='NULL')

    listed1 = IPt[0, :] > 0
    listed2 = IPt[int(twelve_months+six_months*(s1218 == 1))-1, :] > 0
    listed = np.multiply(listed1, listed2)

    listed_num = np.sum(listed)
    listed_indexes = np.where(listed > 0)[0]
    listed_stocks = Pt.columns[listed_indexes]

    [D, ia, ib] = np.intersect1d(
        ticker2.iloc[:, big_loop], ticker2.iloc[:, big_loop+1], return_indices=True)

    ic = np.isin(D, ticker2.iloc[:, big_loop+2])

    Dic_unique_sorted, B_idx = np.unique(D[ic], return_index=True)

    listed_union = np.intersect1d(listed_stocks, Dic_unique_sorted)

    index_listed2 = [Pt.columns.get_loc(i) for i in listed_union if i in Pt]
    index_listed2.sort()

    no_listed2 = len(index_listed2)
    print(f"Number of listed stocks in period: {no_listed2}")

    # ----------------------------------------------------
    # Add filters (if needed)
    # ----------------------------------------------------
    no_comp = np.transpose(sum(np.transpose(IPt > 0)))
    print(f'Period {big_loop}')

    # ----------------------------------------------------
    # Pair selection with Hurst Exponent
    # ----------------------------------------------------
    print("--- Starting Stage 1: Pre-selecting candidates with Distance Method (SSE) ---")
    
    # --- STAGE 1: Fast pre-selection using Sum of Squared Errors (SSE) ---

    # Use np.inf as a sentinel for pairs that are not calculated or already selected.
    sse = np.full((no_listed2, no_listed2), np.inf) 
    for j in tqdm(range(no_listed2 - 1), desc="Stage 1: Calculating SSE"):
        for k in range(j + 1, no_listed2):
            sse[j, k] = np.sum(np.power(IPt[0:int(twelve_months), index_listed2[j]] - IPt[0:int(twelve_months), index_listed2[k]], 2))

    # Find the top 'candidate_pool_size' pairs with the lowest SSE
    candidate_pairs_indices = []
    sse_copy = sse.copy()

    # Ensure we don't try to select more candidates than there are pairs
    num_to_select = min(candidate_pool_size, (no_listed2 * (no_listed2 - 1)) // 2)
    if num_to_select > 0:
        for _ in range(num_to_select):
            # Find the minimum SSE value that is not infinity
            min_sse_val = np.nanmin(sse_copy)
            if np.isinf(min_sse_val):
                break # Stop if no finite values are left
            
            ro_list, co_list = np.where(sse_copy == min_sse_val)
            ro, co = ro_list[0], co_list[0]
            
            candidate_pairs_indices.append({'ro': ro, 'co': co})
            
            # Mark this pair as selected by setting its SSE to infinity
            sse_copy[ro, co] = np.inf

    print(f"--- Stage 1 Complete: {len(candidate_pairs_indices)} candidate pairs selected. ---")
    print("--- Starting Stage 2: Calculating Hurst Exponent for candidate pairs... ---")

    # --- STAGE 2: Calculate Hurst Exponent ONLY for the pre-selected candidates ---
    hurst_results = []
    for pair_idx in tqdm(candidate_pairs_indices, desc="Stage 2: Calculating Hurst"):
        j = pair_idx['ro']
        k = pair_idx['co']
        
        try:
            spread_series = IPt[0:int(twelve_months), index_listed2[j]] - IPt[0:int(twelve_months), index_listed2[k]]
            spread_diff = np.diff(spread_series)

            if np.std(spread_diff) > 1e-6: # Check for non-zero variance
                h_exponent = calculate_hurst_rs(spread_diff)
                hurst_results.append({'ro': j, 'co': k, 'hurst': h_exponent})
        except Exception:
            continue # Skip if Hurst calculation fails for any reason

    print("--- Stage 2 Complete. Finalizing pair selection... ---")

    # --- FINAL SELECTION: Sort candidates by Hurst and select the top 'no_pairs' ---
    hurst_results.sort(key=lambda x: x['hurst'])
    final_selection = hurst_results[:no_pairs]

    print("\n--- Top Selected Pairs for this Period (Ranked by Hurst Exponent) ---")
    if not final_selection:
        print("No pairs were selected for this period.")
    else:
        for idx, selection in enumerate(final_selection):
            ro = selection['ro']
            co = selection['co']
            hurst_val = selection['hurst']
            ticker1_name = Pt.columns[index_listed2[ro]]
            ticker2_name = Pt.columns[index_listed2[co]]
            # Print rank, pair tickers, and the formatted Hurst Exponent
            print(f"{idx + 1:2d}. Pair: {ticker1_name:>7}-{ticker2_name:<7} | Hurst: {hurst_val:.4f}")
    print("------------------------------------------------------------------\n")

    pairs = []
    for selection in final_selection:
        ro = selection['ro']
        co = selection['co']
        pairs.append({
            "s1_col": int(ro), 
            "s2_col": int(co),
            "s1_ticker": Pt.columns[index_listed2[ro]], 
            "s2_ticker": Pt.columns[index_listed2[co]]
        })

    # This will be used to create final analysis in the end
    for pair_data in pairs:
        asset_frequency_counter[pair_data['s1_ticker']] += 1
        asset_frequency_counter[pair_data['s2_ticker']] += 1

    if len(pairs) < no_pairs:
        print(f"Warning: Only {len(pairs)} pairs were selected after Hurst filtering.")

    # ----------------------------------------------------
    # Calculate returns during the 6 month period
    # ----------------------------------------------------

    count_temp = counter

    print(f"counter value: {counter} | counter_temp = {count_temp}")

    print(
        f"Portfolio period from days {i+twelve_months} to {i+twelve_months+six_months-1}")
    for p in range(0, no_pairs):
        first_col = index_listed2[pairs[p]['s1_col']]
        second_col = index_listed2[pairs[p]['s2_col']]
        counter = count_temp
        pairs_opened = 0
        new_pairs_opened = 0
        lag = 0
        can_trade = True
        last_operation = 0


        std_limit = np.std(IPt[0:twelve_months, first_col] -
                           IPt[0:twelve_months, second_col])  # standard deviation

        print(f"Std limit: {std_limit}")
        spread = IPt[0:twelve_months, first_col] - IPt[0:twelve_months, second_col]
        #print(f"Spread series: {spread}")

        # Fixed volatility estimated in the 12 months period. Doing the calculation one pair at a time
        # Presets all variables for each pair
        Rcum = np.zeros((twelve_months, 1))
        counter_ret = 1
        Rcum_ret = [1]

        wi = []

        for j in range(i+twelve_months, i+twelve_months+six_months-1):  # portfolio period
            # Defining the period as from the first day of the twe_month to the last day of the twe_month

            """ if Rcum_ret[-1] < 0.85:
                print(f"pair_no {p}")
                print(f"pairs_opened: {pairs_opened}")
                print(f"pair indexes: ({first_col},{second_col})")
                break """

            if daylag == 0:  # w/o one day delay
                if not can_trade:
                    if (last_operation == 1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) <= closing_threshold*std_limit) or (last_operation == -1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) >= -closing_threshold*std_limit):
                        can_trade = True

                if pairs_opened == -1:  # pairs opened: long 1st, short 2nd stock
                    # print("Pair is long on 1st")
                    # If a sign to open has been given, then calcule the returns
                    """ print(f"Weights: w1 {wi[0]} w2 {wi[1]}")
                    print(f"Return indexes {j},{second_col} | Returns r1 {Rt.iloc[j, first_col]} | r2 {Rt.iloc[j, second_col]}")
                    print(f"Calculated return {np.multiply(Rt.iloc[j, first_col], wi[0]) - np.multiply(Rt.iloc[j, second_col], wi[1])}")
                     """
                    Rpair[counter, p] = np.multiply(
                        Rt.iloc[j, first_col], wi[0]) - np.multiply(Rt.iloc[j, second_col], wi[1])
                    # Rpair is the return of each pair.
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    lag = lag + 1  # used for paying tc

                    if wi_update == 1:  # The weight of each asset in the pair is updated.
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])

                        # print(f"Updated weights w1 {wi[0]} - w2 {wi[1]}")

                elif pairs_opened == 1:  # pairs opened: short 1st, long 2nd stock
                    # print("Pair is short on 1st")
                    """ print(f"Weights: w1 {wi[0]} w2 {wi[1]}")
                    print(
                        f"Return indexes {j},{second_col} | Returns r1 {Rt.iloc[j, first_col]} - weighted {np.multiply(Rt.iloc[j, first_col], wi[0])} r2 {Rt.iloc[j, second_col]} - weighted {np.multiply(Rt.iloc[j, second_col], wi[1])}")
                    print(
                        f"Calculated return {np.multiply(-Rt.iloc[j, first_col], wi[0]) + np.multiply(Rt.iloc[j, second_col], wi[1])}")
                     """
                    Rpair[counter, p] = np.multiply(
                        -Rt.iloc[j, first_col], wi[0]) + np.multiply(Rt.iloc[j, second_col], wi[1])
                    # print(f"Rpair value {Rpair[counter, p]}")
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    lag = lag + 1

                    if wi_update == 1:
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])
                        # print(f"Updated weights w1 {wi[0]} - w2 {wi[1]}")

                else:
                    Rpair[counter, p] = 0  # closed (this code not necessary)

                if ((pairs_opened == 1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) <= closing_threshold*std_limit) or (counter_ret > duration_limit) or (Rcum_ret[-1] < Stop_loss) or (Rcum_ret[-1] >= Stop_gain)
                        or (pairs_opened == -1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) >= -closing_threshold*std_limit)) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    #print(f"Closing position {pairs_opened}. Price diff: {IPt[j-i, first_col]-IPt[j-i, second_col]}")
                    
                    converged = True

                    if Rcum_ret[-1] < Stop_loss:
                        Rcum_ret[-1] = Stop_loss
                        converged = False
                        can_trade = False
                        last_operation = pairs_opened

                    if counter_ret > duration_limit:
                        converged = False
                        can_trade = False
                        last_operation = pairs_opened

                    pairs_opened = 0  # close pairs: prices cross
                    # when pair is closed reset lag (it serves for paying tc)
                    lag = 0
                    # add a marker for closing used to calc length of the "open-period"
                    avg_price_dev[counter, no_pairs+p] = 1
                    # Includes trading cost in the last day of trading, due to closing position
                    Rpair[counter, p] = Rpair[counter, p] - percentage_costs

                    print(
                        f"Pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} closed. Days: {counter_ret} | Return: {Rcum_ret[-1]}")
                    operations.append({
                        "Semester": big_loop,
                        "Days": counter_ret,
                        "S1": pairs[p]['s1_ticker'],
                        "S2": pairs[p]['s2_ticker'],
                        "Pair": f"{pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']}",
                        "Return": Rcum_ret[-1],
                        "Converged": converged,
                        "Count day": counter
                    })

                    counter_ret = 1

                elif can_trade and (pairs_opened == 0) and (+IPt[j-i, first_col]-IPt[j-i, second_col] > opening_threshold*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    #print(f"Opening short position. Price diff: {IPt[j-i, first_col]-IPt[j-i, second_col]} | Threshold: {1.5*std_limit}")
                    if pairs_opened == 0:  # record dev (and time) at open
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1
                        avg_price_dev[counter, p] = 2*(+IPt[j-i, first_col] - IPt[j-i, second_col])/(
                            IPt[j-i, first_col] + IPt[j-i, second_col])

                    # print(
                    #    f"Trading short pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} on day {j}")
                    pairs_opened = 1  # open pairs
                    lag = lag + 1  # - Lag was 0. On the next loop C will be paid
                    wi = [1, 1]

                elif can_trade and (pairs_opened == 0) and (IPt[j-i, first_col]-IPt[j-i, second_col] < -opening_threshold*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    #print(f"Opening long position. Price diff: {IPt[j-i, first_col]-IPt[j-i, second_col]} | Threshold: {-1.5*std_limit}")
                    if pairs_opened == 0:  # record dev (and time) at open
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1
                        avg_price_dev[counter, p] = 2*(-IPt[j-i, first_col] + IPt[j-i, second_col])/(
                            IPt[j-i, first_col] + IPt[j-i, second_col])
                    # print(
                    #    f"Trading long pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} on day {j}")
                    pairs_opened = -1  # open pairs
                    lag = lag + 1
                    wi = [1, 1]

                counter += 1

            elif daylag == 1:
                if pairs_opened == -1:  # pairs opened: long 1st, short 2nd stock
                    Rpair[counter, p] = (+Rt.iloc[j, first_col] * wi[0] -
                                         Rt.iloc[j, second_col] * wi[1]) - (lag == 2)*trading_costs
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    if wi_update == 1:
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])

                elif pairs_opened == 1:  # pairs opened: short 1st, long 2nd stock
                    Rpair[counter, p] = (-Rt.iloc[j, first_col] * wi[0] +
                                         Rt.iloc[j, second_col] * wi[1]) - (lag == 2)*trading_costs
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    if wi_update == 1:
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])

                else:
                    Rpair[counter, p] = 0  # closed (this code not necessary)

                pairs_opened = new_pairs_opened

                if (pairs_opened == +1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) <= 0
                    or Rcum_ret[-1] <= Stop_loss
                    or (Rcum_ret[-1] >= Stop_gain)
                        or pairs_opened == -1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) >= 0) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):

                    new_pairs_opened = 0  # close prices: prices cross
                    # If the pairs are open and the spread is smaller than the
                    # threshold, close the position
                    lag = 0
                    # see above, marker
                    avg_price_dev[counter+1, no_pairs+p] = 1

                    if wi_update == 1:
                        Rpair[counter, p] = Rpair[counter, p] - \
                            trading_costs - percentage_costs

                elif (+IPt[j-i, first_col]-IPt[j-i, second_col] > 2.0*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    new_pairs_opened = 1  # open pairs
                    # If the difference between the prices are larger than
                    # the limit, and there is volume, open the position (short 1st, long 2nd)
                    lag = lag + 1
                    if pairs_opened == 0:
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1

                elif (-IPt[j-i, first_col]+IPt[j-i, second_col] > 2.0*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    new_pairs_opened = -1  # open pairs
                    # If the difference between the prices are larger than
                    # the limit, and there is volume, open the position (short 2nd, long 1st)
                    lag = lag + 1
                    if pairs_opened == 0:  # - If the pair was closed, reset accumulated return matrix and counter
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1

                if new_pairs_opened == +1 and lag == 1:
                    avg_price_dev[counter, p] = 2*(+IPt[j-i, first_col] - IPt[j-i, second_col])/(
                        IPt[j-i, first_col] + IPt[j-i, second_col])
                    lag = lag + 1
                    wi = [1, 1]

                elif new_pairs_opened == -1 and lag == 1:

                    avg_price_dev[counter, p] = 2*(-IPt(j-i, first_col) + IPt(
                        j-i, second_col))/(IPt(j-i, first_col) + IPt(j-i, second_col))
                    lag = lag + 1
                    wi = [1, 1]

                counter += 1

        if pairs_opened != 0:
            print(
                f"Pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} did not converged, Days: {counter_ret} | Return: {Rcum_ret[-1]}")
            # Includes trading cost in the last day of trading, due to closing position
            Rpair[counter-1, p] = Rpair[counter-1, p] - \
                trading_costs - percentage_costs
            avg_price_dev[counter-1, no_pairs+p] = 1
            operations.append(
                {
                    "Semester": big_loop,
                    "Days": counter_ret,
                    "S1": pairs[p]['s1_ticker'],
                    "S2": pairs[p]['s2_ticker'],
                    "Pair": f"{pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']}",
                    "Return": Rcum_ret[-1],
                    "Converged": False,
                    "Count day": counter
                }
            )

    # print(f"Rpair len: {len(Rpair)} | value: {Rpair}")
    pd.DataFrame(Rpair).to_csv("Rpair.csv", header=None, index=False)

    # Using 2 Weighting Schemes - Fully Invested and Committed Capital
    # ------------------------------------------------------------
    # Calculate portfolio returns (ew, vw) out of percentage Rpair
    # ------------------------------------------------------------

    print(f"Calculating returns from day {counter-six_months+1} to {counter}")

    Rpair_temp = Rpair[counter-six_months+1:counter+1]
    # Ret_acum = np.zeros((six_months, 4))
    #
    # eq-weighted average on committed cap.; weights reset to "one" (or any equal weight) after each day
    #
    wi = np.ones((1, no_pairs))
    np.append(wi, np.cumprod(1+Rpair_temp))
    # print(f"Wi shape: {np.shape(wi)}")

    #
    # ew-weighted, committed cap.; weights "restart" every 6 month period;
    # (each portfolio gets 1 dollar at the beginning)
    # print(f"Rpair_temp len: {np.shape(Rpair_temp)} value: {Rpair_temp}")

    pd.DataFrame(Rpair_temp[:six_months]).to_csv(
        "rpair_limited.csv", header=None, index=False, na_rep='NULL')
    pd.DataFrame(np.sum(np.multiply(wi, Rpair_temp[:six_months]), axis=1)).to_csv(
        "row_sum.csv", header=None, index=False)
    Rp_ew_cc['Return'][counter-six_months+1:counter+1] = np.nansum(
        np.multiply(wi, Rpair_temp[:six_months]), axis=1) / np.sum(wi)

    # print(f"Rp_ew_cc len: {np.shape(Rp_ew_cc)} | value: {Rp_ew_cc}")

    ret_acum_df['CC'][counter-six_months+1:counter+1] = np.cumprod(
        1 + (Rp_ew_cc['Return'][counter-six_months+1:counter+1]))[:six_months]
    # [MaxDD,~] = maxdrawdown(ret2tick(Rp_ew_cc(counter-Six_mo:counter-1,:)));
    # MDDw(big_loop,2) = MaxDD;
    # Sortino = sortinoratio(Rp_ew_cc(counter-Six_mo:counter-1,:),0);
    # Sortino_w(big_loop,2) = Sortino;

    #
    # vw-weighted, fully invested; weights "restart" from 1 every time a new pair is opened;
    # Capital divided between open portfolios.

    pd.DataFrame(avg_price_dev).to_csv(
        "avg_price_dev.csv", header=None, index=False)

    # indicator for days when pairs open
    pa_open = np.zeros((six_months, no_pairs))
    for i2 in range(0, no_pairs):
        pa_opened_temp = 0
        temp_lag = 0

        for i1 in range(0, six_months):
            if pa_opened_temp == 1 and daylag == 0:  # opening period not included, closing included
                pa_open[i1, i2] = 1
                days_open[pairs_number, 0] = days_open[pairs_number, 0] + 1

            if pa_opened_temp == 1 and daylag == 1 and temp_lag == 1:
                pa_open[i1, i2] = 1
                days_open[pairs_number, 0] = days_open[pairs_number, 0] + 1

            if pa_opened_temp == 1 and daylag == 1 and temp_lag == 0:
                temp_lag = 1

            if avg_price_dev[counter-six_months+i1+1, i2] != 0:
                pa_opened_temp = 1
                pairs_number = pairs_number + 1

            if avg_price_dev[counter-six_months+i1+1, no_pairs+i2] != 0:
                pa_opened_temp = 0
                temp_lag = 0

    # pd.DataFrame(pa_open).to_csv("pa_open.csv", header=None, index=False)

    wi2 = np.multiply(wi, pa_open)

    for i2 in range(0, six_months):  # takes care in a situation where no pairs are open
        if sum(pa_open[i2, :]) == 0:
            wi2[i2, 0:no_pairs] = 0.2 * np.ones((1, no_pairs))
            pa_open[i2, 0:no_pairs] = np.ones((1, no_pairs))

    # pd.DataFrame(wi2).to_csv("wi2.csv", header=None, index=False)

    Rp_vw_fi['Return'][counter-six_months+1:counter+1] = np.divide(np.nansum(np.multiply(
        wi2, Rpair_temp[:six_months+1, :]), axis=1), np.sum(wi2, axis=1))

    ret_acum_df['FI'][counter-six_months+1:counter +
                      1] = np.cumprod(1+(Rp_vw_fi['Return'][counter-six_months+1:counter+1]))
    # [MaxDD,~] = maxdrawdown(ret2tick(Rp_vw_fi(counter-Six_mo:counter-1,:)));
    # MDDw(big_loop,4) = MaxDD;
    # Sortino = sortinoratio(Rp_vw_fi(counter-Six_mo:counter-1,:),0);
    # Sortino_w(big_loop,4) = Sortino;

    RmRf[counter-six_months+1:counter +
                   1] = (Rm.iloc[counter-six_months+1:counter + 1, :] - Rf.iloc[counter-six_months+1:counter + 1, :]).to_numpy()

    risk_free['Return'][counter-six_months+1:counter + 1] = (Rf.iloc[counter-six_months+1:counter + 1, 0])

    ret_acum_df['RMRF'][counter-six_months+1:counter+1] = np.cumprod(
        1+(Rm.iloc[:six_months, 0]-Rf.iloc[:six_months, 0]).to_numpy())

    ret_acum_df['SEMESTER'][counter-six_months+1:counter+1] = int(big_loop)
    Rp_ew_cc['Semester'][counter-six_months+1:counter+1] = int(big_loop)
    Rp_vw_fi['Semester'][counter-six_months+1:counter+1] = int(big_loop)
    risk_free['Semester'][counter-six_months+1:counter+1] = int(big_loop)

    if Stop_loss == float("-inf"):
        Rp_ew_cc.to_csv("../hurst_results/Rp_ew_cc.csv", index=False)
        Rp_vw_fi.to_csv("../hurst_results/Rp_vw_fi.csv", index=False)
        pd.DataFrame(RmRf).to_csv("../hurst_results/RmRf.csv", header=None, index=False)
        risk_free.to_csv("../hurst_results/risk_free.csv")
        ret_acum_df.to_csv("../hurst_results/ret_acum_df.csv")
    else:
        Rp_ew_cc.to_csv(f"../hurst_results/Rp_ew_cc.csv", index=False)
        Rp_vw_fi.to_csv(f"../hurst_results/Rp_vw_fi.csv", index=False)
        pd.DataFrame(RmRf).to_csv(f"../hurst_results/RmRf.csv", header=None, index=False)
        risk_free.to_csv(f"../hurst_results/risk_free.csv")
        ret_acum_df.to_csv(f"../hurst_results/ret_acum_df.csv")

    for i2 in range(0, no_pairs):
        if sum(avg_price_dev[counter-six_months+1:counter+1, i2] != 0) != 0:
            periods_with_open_pair = periods_with_open_pair + 1
            no_pairs_opened[big_loop, i2] = no_pairs_opened[
                big_loop, p] + sum(avg_price_dev[counter-six_months+1:counter+1, i2] != 0)
        else:
            periods_without_open_pair = periods_without_open_pair + 1

    
    i = i + periods.iloc[big_loop, 0]

    big_loop = big_loop + 1

operations_df = pd.DataFrame(operations)
if Stop_loss == float('-inf'):
    operations_df.to_csv(f"../hurst_results/duration_limit/operations.csv")
else:
    operations_df.to_csv(f"../hurst_results/operations.csv")

##################################################################
# FINAL ANALYSIS AND CHART GENERATION
##################################################################

print("\n\n========================================================")
print("           FINAL PORTFOLIO ANALYSIS")
print("========================================================")

# --- Define Helper Functions for Performance Metrics ---

def calculate_sortino_ratio(returns, risk_free_rate=0):
    """Calculates the Sortino Ratio."""
    target_return = 0
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.nan
    expected_return = returns.mean()
    downside_std = downside_returns.std()
    sortino = (expected_return - risk_free_rate) / downside_std
    return sortino * np.sqrt(252) # Annualize

def calculate_max_drawdown(returns):
    """Calculates the Maximum Drawdown."""
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

# --- Part 1: Generate Data for "Descriptive Statistics" Table ---

print("\n--- Descriptive Statistics for the Most Traded Assets (Table 2) ---")

# Define how many top assets to analyze
top_n_assets = 11 
most_common_assets = asset_frequency_counter.most_common(top_n_assets)

stats_data = []
for asset_ticker, count in most_common_assets:
    # Get the full return series for the asset
    asset_returns = Rt[asset_ticker].dropna()
    
    # Calculate stats
    mean_ret = asset_returns.mean()
    std_dev = asset_returns.std()
    sharpe_ratio = (mean_ret / std_dev) * np.sqrt(252) # Assuming Rf=0 for individual asset SR
    min_ret = asset_returns.min()
    max_ret = asset_returns.max()
    
    stats_data.append({
        "Symbol": asset_ticker,
        "Mean": f"{mean_ret:.4f}",
        "Std Dev": f"{std_dev:.4f}",
        "SR": f"{sharpe_ratio:.4f}",
        "Min": f"{min_ret*100:.2f}",
        "Max": f"{max_ret*100:.2f}"
    })

stats_df = pd.DataFrame(stats_data)
print(stats_df.to_string(index=False))

stats_df.to_csv(f"../hurst_results/final_analysis_stats.csv")

# --- Part 2: Generate Data for "Excess Returns" Table ---

print("\n\n--- Performance Metrics of Trading Strategies (Table 3) ---")

# Consolidate the portfolio returns, dropping any non-trading periods
cc_returns = Rp_ew_cc[Rp_ew_cc['Return'] != 0]['Return'].dropna()
fi_returns = Rp_vw_fi[Rp_vw_fi['Return'] != 0]['Return'].dropna()
rf_returns = risk_free[risk_free.index.isin(cc_returns.index)]['Return'].dropna()

# Align risk-free rates with portfolio returns
cc_excess_returns = cc_returns - rf_returns
fi_excess_returns = fi_returns - rf_returns

# Calculate metrics for both strategies
metrics = {}
strategies = {
    "Committed Capital": (cc_returns, cc_excess_returns),
    "Fully Invested": (fi_returns, fi_excess_returns)
}

for name, (returns, excess_returns) in strategies.items():
    if len(returns) > 0:
        # Annualized values
        ann_mean_return = returns.mean() * 252
        ann_std_dev = returns.std() * np.sqrt(252)
        ann_sharpe_ratio = (excess_returns.mean() * 252) / ann_std_dev
        
        metrics[name] = {
            "Mean Return (%)": f"{ann_mean_return * 100:.3f}",
            "Sharpe Ratio": f"{ann_sharpe_ratio:.3f}",
            "Sortino Ratio": f"{calculate_sortino_ratio(returns, rf_returns.mean()):.3f}",
            "t-statistic": f"{scipy.stats.ttest_1samp(returns, 0).statistic:.3f}",
            "% of negative excess returns": f"{ (excess_returns < 0).sum() / len(excess_returns) * 100:.2f}",
            "MDD (%)": f"{calculate_max_drawdown(returns) * 100:.2f}",
            "Annualized STD (%)": f"{ann_std_dev * 100:.3f}",
            "Skewness": f"{returns.skew():.3f}",
            "Kurtosis": f"{returns.kurtosis():.3f}", # Pandas kurtosis is excess kurtosis (0 is normal)
            "Minimum Daily Ret (%)": f"{returns.min() * 100:.3f}",
            "Maximum Daily Ret (%)": f"{returns.max() * 100:.3f}"
        }
    else:
        metrics[name] = {k: "N/A" for k in strategies["Committed Capital"][1]}


performance_df = pd.DataFrame(metrics)
print(performance_df)

performance_df.to_csv(f"../hurst_results/final_analysis_performance.csv")

# --------------------
# Generating chart
# --------------------
import matplotlib.pyplot as plt

print("\n--- Generating Cumulative Excess Returns Plot ---")

# We will plot only the "Committed Capital" strategy returns.
# First, get the series of non-zero daily returns for this strategy.
cc_returns = Rp_ew_cc[Rp_ew_cc['Return'] != 0]['Return'].dropna()

# Get the corresponding risk-free rates for the same days
if not cc_returns.empty:
    rf_returns = risk_free.loc[cc_returns.index, 'Return'].dropna()

    # Calculate the excess returns
    cc_excess_returns = cc_returns - rf_returns

    # Calculate the cumulative product of the excess returns
    cumulative_cc = (1 + cc_excess_returns).cumprod()

    # Use a simple numerical series (0, 1, 2, ...) for the x-axis
    x_axis_data = range(len(cumulative_cc))

    # --- Create the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(x_axis_data, cumulative_cc, label='Committed Capital', linewidth=2, color='C0') # 'C0' is default blue

    # Add titles and labels for clarity
    ax.set_title('Cumulative Excess Returns (Committed Capital)', fontsize=16)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Cumulative Excess Returns', fontsize=12)
    ax.legend(fontsize=12)

    # Set a baseline at 1.0 for reference
    ax.axhline(y=1, color='grey', linestyle='--', linewidth=1)
    
    print("Displaying plot...")
    plt.show()

    fig = ax.get_figure()
    fig.savefig('./hurst_cumm_return.png')
    
else:
    print("No trading data available to generate the plot.")

print("\n========================================================")
print("                 ANALYSIS COMPLETE")
print("========================================================")