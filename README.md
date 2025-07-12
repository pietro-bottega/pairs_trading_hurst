# Pairs trading

`requirements.yml` has all libraries needed execute code.

#### 1. Using conda, run in CLI
```console
conda env create -f environment.yaml
```
<br>
Or manually install main libraries needed:
<br>- numpy
<br>- pandas
<br>- tqdm
<br>- statsmodels
<br>- matplotlib
<br>- seaborn
<br>
<br> with:

```console
conda install "library from list above"
```

#### 2. Run the distance method with:
```console
cd distance_code # going inside a folder
python distance_gatev_v2.py
cd .. # going back to main folder
```
<br> Main output at [operations.csv](https://github.com/pietro-bottega/pairs_trading_crisis/blob/master/distance_results/operations.csv)

#### 3. Run the cointegration method with:
```console
cd cointegration_code
python cointegration_with_SSD_ECM.py
cd ..
```
<br> Main output at [operations_SSD_ECM.csv](https://github.com/pietro-bottega/pairs_trading_crisis/blob/master/cointegration_results/operations_SSD_ECM.csv)

#### 4. Run calculations of return by subperiod (crisis vs. non-crisis):
```console
cd crisis_analysis
python operations_crisis_classification.py
cd ..
```
<br> View results in [operations.csv](https://github.com/pietro-bottega/pairs_trading_crisis/tree/master/crisis_analysis/crisis_subperiods_comparison.png)

#### 4. Run calculations of risk adjusted performance:
```console
cd risk_adjusted_performance
python risk_adjusted_performance.py
cd ..
```
<Br>View results in [cointegration_risk_adjusted_measures.csv](https://github.com/pietro-bottega/pairs_trading_crisis/blob/master/cointegration_results/cointegration_risk_adjusted_measures.csv)
<br>View results in [distance_risk_adjusted_measures.csv](https://github.com/pietro-bottega/pairs_trading_crisis/blob/master/distance_results/distance_risk_adjusted_measures.csv)


## Listing changes
from original [bkalil7/tcc](https://github.com/bkalil7/tcc/tree/main)

### Performing ECM and SSD before cointegration (task 2)

Added `cointegration_with_SSD_ECM.py`, incorporating ECM and SSD filters to select cointegrated pairs. It will output:
1. `daily_returns_SSD_ECM.csv`
2. `Rpair_SSD_ECM.csv`
3. `operations_SSD_ECM.csv`

<br> It also had transactional cost added.
<br> Defined by user input on the beginning:
```python
percentage_cost = 0.002
```
<br> And discounted from pairs return:
```python
Rpair[i-1, pair] = log_ret[i]*pos - percentage_costs
```

<br>Following steps were adjusted to take results from this cointegration method as input (mainly operations with SSD, ECM).

### Measuring risk adjusted performance (task 4)

Added `risk_adjusted_performance/` directory.
<br>`risk_adjusted_analysis.ipynb`: notebook with draft to calculate measeures and view dataframe with them calculated per semester
<br>`risk_adjusted_performance.py`: code to calculate risk adjusted measures based on results:
- Outputs `distance_results/distance_risk_adjusted_measures.csv`
- Outputs `cointegration_results/cointegration_risk_adjusted_measures.csv`


### Classify operations by subperiod crisis vs. non-crisis (task 5)

`distance_gatev_v2.py`

Differences from `distance_gatev.py`:
1. Fixed errors on writing results in `distance_results/`
2. Added "Count day" into big_loop while statement, when it appends to the operation.csv file, using `counter` variable:

```python
operations.append({
    "Semester": big_loop,
    "Days": counter_ret,
    "S1": pairs[p]['s1_ticker'],
    "S2": pairs[p]['s2_ticker'],
    "Pair": f"{pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']}",
    "Return": Rcum_ret[-1],
    "Converged": converged,
    "Count day": counter # here is the line I added
    })
```

#### Directory `crisis_analysis/`

Created directory
<br> Added `bear_markets.csv` file with periods of bear markets in histort, based on Hartford Funds reseach
<br> Added `crisis_analysis.ipynb` notebook to do data manipulation, later incorporated on `operations_crisis_classification.py`
    
`distance_code/operations_crisis_classification.py`
- Added this .py script classifying operations from pairs trading into bear market period
- Outputs `distance_results/operations_crisis_classified.csv`
- Outputs `cointegration_results/operations_crisis_classified.csv`
        
Additional files for support:        
- Outputs `distance_data/period_crisis_classification.csv`
- Outputs `cointegration_data/period_crisis_classification.csv`
