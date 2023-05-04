"""The module provides functions to compute quantities relevant to financial
portfolios, e.g. a weighted average, which is the expected value/return, a
weighted standard deviation (volatility), and the Sharpe ratio.
"""


import numpy as np
import pandas as pd
import quantstats as qs


def cagr(returns, rf=0, nperiods=252):            
    
    years = (returns.index[-1] - returns.index[0]).days / 365.
    
    if rf>0:
        excess_ret = qs.utils.to_excess_returns(returns, rf, nperiods)
        ret = qs.stats.comp(excess_ret)    
    else:
        ret = qs.stats.comp(returns)    

    res = abs(ret + 1.0) ** (1.0 / years) - 1

    if isinstance(ret, pd.DataFrame):
        res = pd.Series(res)
        res.index = ret.columns
    return res

def cvar(returns, sigma=1, confidence=0.95, prepare_returns=True):
    if prepare_returns:
        returns = qs.utils._prepare_returns(returns)    
    var = qs.stats.var(returns, sigma, confidence=0.95, prepare_returns=False)
    c_var = returns[returns < var].values.mean()
    res = c_var if ~np.isnan(c_var) else var
    
    if isinstance(returns, pd.DataFrame):
        res = pd.Series(res)
        res.index = returns.columns
    return res

def max_drawdown(prices): 
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def df_alloc(alloc_list):

    rows = []  
    for data in alloc_list:
        data_row = data['Class']
        time = data['Asset_Class']
        
        for row in data_row:
            row['Asset_Class']= time
            rows.append(row)
    
    # using data frame
    df = pd.DataFrame(rows)
    #print(df.Allocation.sum())
    

    df_allocate = df.groupby(['Name']).sum()
    df_allocate['Allocation'] = df_allocate['Allocation']/100

    df_allocate= df_allocate[df_allocate['Allocation'] != 0]
    df_allocate.sort_values(by=['Name'])

    df_allocate.reset_index(inplace = True)
    
    
    print("Alloc_Total_%:", df_allocate['Allocation'].sum()*100)


    #df_full = df.groupby(['Name']).sum()
    #df_full.reset_index(inplace = True)

    return df_allocate


def weighted_mean(means, weights):
    """Computes the weighted mean/average, or in the case of a
    financial portfolio, it can be used for the Expected Return
    of said portfolio.

    :Input:
     :means: ``numpy.ndarray``/``pd.Series`` of mean/average values
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted mu: ``numpy.ndarray``: ``(np.sum(means*weights))``
    """
    if not isinstance(weights, (pd.Series, np.ndarray)):
        raise ValueError("weights is expected to be a pandas.Series/np.ndarray")
    if not isinstance(means, (pd.Series, np.ndarray)):
        raise ValueError("means is expected to be a pandas.Series/np.ndarray")
    return np.sum(means * weights)


def weighted_std(cov_matrix, weights):
    """Computes the weighted standard deviation, or Volatility of
    a portfolio, which contains several stocks.

    :Input:
     :cov_matrix: ``numpy.ndarray``/``pandas.DataFrame``, covariance matrix
     :weights: ``numpy.ndarray``/``pd.Series`` of weights

    :Output:
     :weighted sigma: ``numpy.ndarray``:
         ``np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))``
    """
    if not isinstance(weights, (pd.Series, np.ndarray)):
        raise ValueError("weights is expected to be a pandas.Series, np.array")
    if not isinstance(cov_matrix, (np.ndarray, (np.ndarray, pd.DataFrame))):
        raise ValueError(
            "cov_matrix is expected to be a numpy.ndarray/pandas.DataFrame"
        )
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))



def sharpe_ratio(exp_return, volatility, risk_free_rate=0.005):
    """Computes the Sharpe Ratio

    :Input:
     :exp_return: ``int``/``float``, Expected Return of a portfolio
     :volatility: ``int``/``float``, Volatility of a portfolio
     :risk_free_rate: ``int``/``float`` (default= ``0.005``), risk free rate

    :Output:
     :sharpe ratio: ``float`` ``(exp_return - risk_free_rate)/float(volatility)``
    """
    if not isinstance(
        exp_return, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("exp_return is expected to be an integer or float.")
    if not isinstance(
        volatility, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("volatility is expected to be an integer or float.")
    if not isinstance(
        risk_free_rate, (int, float, np.int32, np.int64, np.float32, np.float64)
    ):
        raise ValueError("risk_free_rate is expected to be an integer or float.")
    return (exp_return - risk_free_rate) / float(volatility)


def annualised_portfolio_quantities(
    weights, means, cov_matrix, risk_free_rate=0.005, freq=252
):
    """Computes and returns the expected annualised return, volatility
    and Sharpe Ratio of a portfolio.

    :Input:
     :weights: ``numpy.ndarray``/``pd.Series`` of weights
     :means: ``numpy.ndarray``/``pd.Series`` of mean/average values
     :cov_matrix: ``numpy.ndarray``/``pandas.DataFrame``, covariance matrix
     :risk_free_rate: ``float`` (default= ``0.005``), risk free rate
     :freq: ``int`` (default= ``252``), number of trading days, default
         value corresponds to trading days in a year

    :Output:
     :(Expected Return, Volatility, Sharpe Ratio): tuple of those
         three quantities
    """
    if not isinstance(freq, int):
        raise ValueError("freq is expected to be an integer.")
    #expected_return = weighted_mean(means, weights) * freq
    #volatility = weighted_std(cov_matrix, weights) * np.sqrt(freq)    

    expected_return = means.values@weights.T                
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(freq)
    sharpe = sharpe_ratio(expected_return, volatility, risk_free_rate)

    return (expected_return, volatility, sharpe)



