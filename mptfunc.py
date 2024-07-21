import yfinance as yf
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
import datetime as dt
def get_data(stocks, start, end):
    stocks_yf_form = [f"{stock_id}.KS" for stock_id in stocks]
    stock_df = yf.download(" ".join(stocks_yf_form), start=start, end=end, interval='1d')
    stock_df = stock_df['Close']
    returns = stock_df.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def portfolio_performance(weights, meanReturns, covMatrix, days):
    returns = np.sum(meanReturns * weights) * days
    std = np.sqrt(
        np.dot(weights.T, np.dot(covMatrix, weights))
    ) * np.sqrt(days)
    return returns, std


def negative_sharp_ratio(weights, meanReturns, covMatrix, days, riskFreeRate=0):
    pReturns, pStd = portfolio_performance(weights, meanReturns, covMatrix, days)
    return - (pReturns - riskFreeRate) / pStd


def max_sharp_ratio(meanReturns, covMatrix, days, riskFreeRate=0.0, constraintSet=(0, 1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, days, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for _ in range(numAssets))
    result = scipy.optimize.minimize(negative_sharp_ratio, [1. / numAssets] * numAssets, args=args,
                                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_variance(weights, meanReturns, covMatrix, days):
    return portfolio_performance(weights, meanReturns, covMatrix, days)[1]


def minimize_variance(meanReturns, covMatrix, days, constraintSet=(0, 1)):
    """Minimize the portfolio variance by altering the
     weights/allocation of assets in the portfolio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, days)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for _ in range(numAssets))
    result = scipy.optimize.minimize(portfolio_variance, [1. / numAssets] * numAssets, args=args,
                                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_return(weights, meanReturns, covMatrix, days):
    return portfolio_performance(weights, meanReturns, covMatrix, days)[0]


def efficient_opt(meanReturns, covMatrix, days, returnTarget, constraintSet=(0, 1)):
    """For each returnTarget, we want to optimise the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, days)

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, meanReturns, covMatrix, days) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = scipy.optimize.minimize(portfolio_variance, numAssets * [1. / numAssets], args=args, method='SLSQP',
                                     bounds=bounds, constraints=constraints)
    return effOpt

def calculated_results(max_sharp_ratio_result, min_val_result, mean_returns, cov_matrix, days, resolution, riskFreeRate=0.0, constraintSet=(0, 1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_returns, maxSR_std = portfolio_performance(max_sharp_ratio_result.x, mean_returns, cov_matrix, days)
    # Min Volatility Portfolio
    minVol_returns, minVol_std = portfolio_performance(min_val_result.x, mean_returns, cov_matrix, days)

    Point = namedtuple("Point", "std er")
    max_sharp_ratio_point = Point(maxSR_std, maxSR_returns)
    min_val_point = Point(minVol_std, minVol_returns)

    # Efficient Frontier
    efficient_std = []
    efficient_weights = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, resolution)
    for target in tqdm(targetReturns, desc="Efficient Frontier Calculating...."):
        efficient_result = efficient_opt(mean_returns, cov_matrix, days, target)
        efficient_std.append(efficient_result.fun)
        efficient_weights.append(efficient_result.x.reshape(1, -1))

    efficient_std_df = pd.DataFrame({
        "er": targetReturns,
        "std": efficient_std,
    })
    efficient_weights_df = pd.DataFrame(np.concatenate(efficient_weights, axis=0), columns=mean_returns.index)

    efficient_df = pd.concat([efficient_std_df, efficient_weights_df], axis=1)

    return max_sharp_ratio_point, min_val_point, efficient_df

def plot_data(max_sharp_point, min_val_point ,efficient_frontier_df, outdir, start, end, days):
    """Return a graph ploting the min vol, max sr and efficient frontier"""
    plt.figure(figsize=(10, 10))
    sns.set(style="whitegrid")
    plt.title("Efficient Frontier")
    # plot max_sharp_point
    sns.scatterplot(x=[max_sharp_point.std], y=[max_sharp_point.er], label="Max Sharp Point", s=200)
    # plot min_vol_point
    sns.scatterplot(x=[min_val_point.std], y=[min_val_point.er], label="Min Variance Point", s=200)

    # plot efficient frontier line
    sns.lineplot(efficient_frontier_df, x="std", y="er", label="Efficient Frontier", linestyle="--", c="black", weights=3)
    # plot cal line if is_cal is True

    plt.xlabel("Risk(standard devaition)")
    plt.ylabel("Expected Return(%)")
    plt.legend()
    plt.savefig(outdir / f"efficient_frontier_{dt.date.strftime(start, format='%Y%m%d')}_{dt.date.strftime(end, format='%Y%m%d')}_{days}d.png")

def save_efficient_list(stock_list, efficient_df:pd.DataFrame, outdir, start, end, days):
    name_dict = {}
    for stock in stock_list:
        stock_id_yf_form = stock + ".KS"
        ticker = yf.Ticker(stock_id_yf_form)
        long_name = ticker.info['longName']
        name_dict[stock_id_yf_form] = long_name
    efficient_df['er'] = efficient_df.er.apply(lambda x: round(x, ndigits=2) * 100)
    efficient_df['std'] = efficient_df['std'].apply(lambda x: round(x, ndigits=5))
    efficient_df.iloc[:, 2:] = np.round(efficient_df.iloc[:, 2:] * 100, decimals=2)
    efficient_df.columns = ["기대수익률(%)", "표준편차", *[name_dict[stock_id] + "(%)" for stock_id in efficient_df.columns[2:]]]
    efficient_df.to_excel(outdir / f"efficient_frontier_{dt.date.strftime(start, format='%Y%m%d')}_{dt.date.strftime(end, format='%Y%m%d')}_{days}d.xlsx")




