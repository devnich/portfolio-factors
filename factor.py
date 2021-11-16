# Import standard libraries
import re
import pprint
from pathlib import Path

# Import numerical computing libraries
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint

# Install the factor_analyzer library with:
#   conda install -c ets factor_analyzer

from factor_analyzer import FactorAnalyzer
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity as bartlett
# from factor_analyzer.factor_analyzer import calculate_kmo as kmo

# Configure file paths
raw_path = Path('data/raw')
processed_path = Path('data/processed')
results_path = Path('results')

# Wrap text when printing long collections in the terminal
pp = pprint.PrettyPrinter(width=100, compact=True, indent=2)

# Display DataFrames without truncation
pd.set_option("display.max_columns", None, "precision", 2)


#------------------------------------------------
# Data processesing
#------------------------------------------------
def detect_colinear(criterion=0.9):
    """Find the set of potentially colinear items in the correlation matrix."""

    for filename in raw_path.glob(''.join(['asset_cor_', '[0-9]'*4, '.csv'])):
        if filename.is_file():
            print(filename)
            data = pd.read_csv(filename, index_col="Ticker")

            # Extract the correlation columns from the DataFrame
            corr = data[data.index]

            # Set diagonal to NaN
            np.fill_diagonal(corr.values, np.nan)

            # Filter by criterion
            filtered = corr.where(corr > criterion)

            # Drop empty columns
            reduced = filtered.dropna(axis=1, how='all')

            # Insert asset class name
            reduced = pd.concat([data["Name"], reduced], axis=1)

            # Save the processed file
            year = re.findall('[0-9]+', filename.parts[-1])[0]
            newfile = processed_path.joinpath(''.join(["colinear_", year, ".csv"]))
            reduced.to_csv(newfile)


def generate_covariances():
    """Create covariance matrices from correlations and standard deviations."""

    for filename in raw_path.glob(''.join(['asset_cor_', '[0-9]'*4, '.csv'])):
        if filename.is_file():
            print(filename)
            data = pd.read_csv(filename, index_col="Ticker")

            # Extract the correlation columns from the DataFrame
            corr = data[data.index]
            # Standard deviations are formatted as text strings in the input
            # file; clean this up:
            sd = data['Annualized Standard Deviation'].str.replace('%', '').astype(np.float64)
            # To reshape a Series, drill down to the values of the underlying array
            cov = corr * sd * sd.values.reshape(-1, 1)

            # Test the correctness of our transformation
            # NB: The diagonal of the covariance matrix is 1 * sd * sd by construction
            diag = cov.values.diagonal()
            var = sd * sd
            assert all(diag.round(2) == var.values.round(2))

            # Save the processed file
            year = re.findall('[0-9]+', filename.parts[-1])[0]
            newfile = processed_path.joinpath(''.join(["asset_cov_", year, ".csv"]))
            cov.to_csv(newfile)

            print(newfile, cov.shape)


#------------------------------------------------
# Calculate Diversification Ratio
#------------------------------------------------
def diversification_ratio(w, cov, sd):
    """Estimate the diversification ratio from an asset covariance matrix and weights.

    w:   Series of weights indexed by ticker
    cov: covariance matrix (array) of tickers
    sd:  array of ticker standard deviations"""

    # Diversification ratio:
    dr = np.dot(w, sd) / np.sqrt(np.linalg.multi_dot([w, cov, w]))

    # Multiply diversification ratio by -1 so that minimize() works properly
    return -1*dr


def diversification_inputs(weights, asset_cov):
    """Generate the inputs for calculating the diversification_ratio.

    weights:   1xn array of weights, indexed by ticker
    asset_cov: nxn array of covariances, indexed by ticker"""

    df = pd.read_csv(asset_cov, index_col="Ticker")
    w_df = pd.read_csv(weights, index_col="Ticker")

    # Extract weights and bounds
    w = w_df["Weights"]
    bounds = [i for i in zip(w_df['lower'], w_df['upper'])]

    # Extract covariances for the set of tickers in weights
    cov_df = df[w.index].loc[w.index]

    # Weight vector must match covariance matrix
    assert len(cov_df) == len(w)
    # Weights must sum to 1
    assert w.sum().round(3) == 1

    cov = cov_df.values
    sd = np.sqrt(cov.diagonal())

    return w, cov, sd, bounds, w_df


def optimize_diversification(w, cov, sd, bounds, leverage, short):
    """Optimize the weights for the diversification_ratio() objective function."""

    # minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None,
    #          bounds=None, constraints=(), tol=None, callback=None, options=None)

    # Number of independent bets in the original portfolio
    dr = diversification_ratio(w, cov, sd)
    bets = dr * dr

    # Allow leverage (i.e. weight total can be > 1)
    if leverage:
        constraint = None
    else:
        # Constrain weights to sum to 1
        constraint = LinearConstraint(np.ones(len(w)), lb=1, ub=1)

    # Allow long-short portfolios (i.e. individual weights can be < 0)
    if short:
        bounds = None

    # Optimize objective function
    res = minimize(diversification_ratio, w.values, args=(cov, sd),
                   constraints=constraint, bounds=bounds)

    # Number of independent bets in the optimized  portfolio
    bets_out = res.fun * res.fun

    # Optimized portfolio weights
    w_out = pd.Series(res.x, w.index)

    return bets, bets_out, w_out, res


def fit_portfolio(weights="portfolio_weights_2006.csv",
                  asset_cov="asset_cov_2006.csv",
                  leverage=False, short=False):

    # Set input paths. If the asset_cov file does not exist, create it in
    # data/processed using the generate_covariances() function.
    weight_path = raw_path.joinpath(weights)
    cov_path = processed_path.joinpath(asset_cov)

    # Fit portfolio
    w, cov, sd, bounds, w_df = diversification_inputs(weights=weight_path, asset_cov=cov_path)
    bets, bets_out, w_out, res = optimize_diversification(w, cov, sd, bounds, leverage, short)

    # Output to screen
    print("Portfolio Bets:", bets.round(2))
    print()
    print("Optimized Portfolio")
    print(w_out.where(w_out > 0.005).dropna().round(2).to_string().strip("Ticker\n"))
    print("Total:", w_out.sum().round(2))
    print("Optimal Bets: ", round(bets_out, 2))
    print("Difference:   ", np.subtract(bets_out, bets).round(2))

    # Save output
    w_df["Fitted Weights"] = w_out.round(3)

    year = re.findall('[0-9]+', weight_path.parts[-1])[0]
    newfile = results_path.joinpath(''.join(['fit', '_', year, ".csv"]))
    w_df.to_csv(newfile)

# TO DO:
# 1. weighted return
# 2. portfolio sharpe

#------------------------------------------------
# Asset factor analysis
#------------------------------------------------
def factor(rotation="varimax", n=5):
    """Run an exploratory factor analysis with n factors."""

    ffits = {}

    for filename in raw_path.glob(''.join(['asset_cor_', '[0-9]'*4, '.csv'])):
        if filename.is_file():
            print(filename)

            data = pd.read_csv(filename, index_col="Ticker")

            # Extract the correlation columns from the DataFrame
            corr = data[data.index]

            fa = FactorAnalyzer(rotation=rotation, is_corr_matrix=True, n_factors=n)
            fa.fit(corr)

            # Print explained variance
            fvar = fa.get_factor_variance()
            print("Variables:", corr.shape)
            print("Proportional explained variance:")
            print(fvar[1].round(3))
            print("Cumulative explained variance:")
            print(fvar[2].round(3))

            # Create factor DataFrame
            factor_labels = [''.join(['F', str(i)]) for i in range(1, n+1)]
            df = pd.DataFrame(data=fa.loadings_.round(2),
                              index=data.index,
                              columns=factor_labels)

            # Create factor R^2 DataFrame
            r2_labels = ['_'.join([i, 'r2']) for i in factor_labels]
            var = pd.DataFrame(data=np.square(fa.loadings_).round(2),
                               index=data.index,
                               columns=r2_labels)

            # Concatenate name, factor loadings, R^2
            df = pd.concat([data["Name"], df, var], axis=1)

            df["Communality"] = fa.get_communalities().round(2)

            ## Calculate Sharpe ratio: : (R_i - R_shy)/SD_i
            # Convert SHY single string to float
            risk_free_r = np.float64(data.loc['SHY', 'Annualized Return'].replace('%', ''))

            # Convert Series to floats
            r = data['Annualized Return'].str.replace('%', '').astype(np.float64)
            sd = data['Annualized Standard Deviation'].str.replace('%', '').astype(np.float64)

            # Calculate Sharpe ratio
            sharpe = (r - risk_free_r)/sd
            df["Sharpe"] = sharpe.round(2)

            # Save the processed file
            year = re.findall('[0-9]+', filename.parts[-1])[0]
            newfile = results_path.joinpath(''.join([rotation, '_', str(n), '_', year, ".csv"]))
            df.to_csv(newfile)

            ffits[str(year)] = fa

    return ffits
