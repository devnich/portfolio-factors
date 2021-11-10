import pprint
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# conda install -c ets factor_analyzer
from factor_analyzer import FactorAnalyzer
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity as bartlett
# from factor_analyzer.factor_analyzer import calculate_kmo as kmo

# Wrap text when printing long collections in the terminal
pp = pprint.PrettyPrinter(width=100, compact=True, indent=2)

# Display DataFrames without truncation
pd.set_option("display.max_columns", None, "precision", 2)

def detect_colinear(fname="asset_cor_2009.csv", criterion=0.9):
    """Find the set of potentially colinear items in the correlation matrix."""

    data = pd.read_csv(fname, index_col="Ticker")
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
    parts = fname.split("asset_cor_")
    newfile = '_'.join(["colinear", parts[1]])
    reduced.to_csv(newfile)

def diversification_ratio(weights, fname="asset_cor_2009_cov.csv"):
    """Estimate the diversification ratio from an asset covariance matrix and weights."""

    # TODO: pass in a list of tickers and weights to estimate that portfolio's diversification ratio

    data = pd.read_csv(fname, index_col="Ticker")

    # weights is a dictionary of ticker names and weights
    w = pd.Series(weights)

    # Covariance matrix must be square
    assert data.shape[0] == data.shape[1]
    # Weights must sum to 1
    assert w.sum() == 1

    cov = data.values
    sd = np.sqrt(cov.diagonal())

    # Create dummy array of for testing; sum(w) == 1.0
    # w = np.full(len(cov), 1/len(cov))

    ## Test with a dummy cov matrix
    # w = np.array([0.33, 0.33, 0.34])
    #
    #---- Test 1: Variance = 1.0 ----
    # cov = np.array([[1.0, 0.5, 0.5],
    #                 [0.5, 1.0, 0.5],
    #                 [0.5, 0.5, 1.0]])
    # sd = np.ones(3)
    #
    #---- Test 2: Variance = 2.0 ----
    # cov = np.array([[2.0, 1.0, 1.0],
    #                 [1.0, 2.0, 1.0],
    #                 [1.0, 1.0, 2.0]])
    # sd = np.array([1.414, 1.414, 1.414])

    dr = np.dot(w, sd) / np.sqrt(np.linalg.multi_dot([w, cov, w]))

    # Number of independent bets is the square of the Diversification Ratio
    bets = dr * dr

    return data, sd, dr, bets


def factor(fname="asset_cor_2009.csv", rotation="varimax", n=5):
    """Run an exploratory factor analysis and save the output."""

    data = pd.read_csv(fname, index_col="Ticker")

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

    # Create factor R2 DataFrame
    r2_labels = ['_'.join([i, 'r2']) for i in factor_labels]
    var = pd.DataFrame(data=np.square(fa.loadings_).round(2),
                       index=data.index,
                       columns=r2_labels)

    # Concatenate name, factor loadings, R2
    df = pd.concat([data["Name"], df, var], axis=1)

    df["Communality"] = fa.get_communalities().round(2)

    # Calculate Sharpe ratio: (R_i - R_shy)/SD_i
    # Convert single value to float
    risk_free_r = np.float64(data.loc['SHY', 'Annualized Return'].replace('%', ''))

    # Convert Series to floats
    r = data['Annualized Return'].str.replace('%', '').astype(np.float64)
    sd = data['Annualized Standard Deviation'].str.replace('%', '').astype(np.float64)

    sharpe = (r - risk_free_r)/sd
    df["Sharpe"] = sharpe.round(2)

    # Save the processed file
    parts = fname.split(".csv")
    newfile = ''.join([parts[0], '_', rotation, '_', str(n), '.csv'])
    df.to_csv(newfile)

    return fa, df

def gen_covariance_files():
    """Create covariance matrices from correlations and standard deviations."""
    path = Path()
    for filename in path.glob(''.join(['asset_cor_', '[0-9]'*4, '.csv'])):
        if filename.is_file():
            print(filename)
            data = pd.read_csv(filename, index_col="Ticker")

            # Get all the columns whose label matches a row label
            corr = data[data.index]
            # standard deviations are entered as text percentages in original file
            sd = data['Annualized Standard Deviation'].str.replace('%', '').astype(np.float64)
            # NB: To reshape a Series, drill down to the values of the underlying array
            cov = corr * sd * sd.values.reshape(-1, 1)

            # Test the correctness of our transformation
            # NB: The diagonal of the covariance matrix is 1 * sd * sd by construction
            diag = cov.values.diagonal()
            var = sd * sd
            assert all(diag.round(2) == var.values.round(2))

            # Save the processed file
            parts = filename.name.split(".csv")
            newfile = ''.join([parts[0], "_cov.csv"])
            cov.to_csv(newfile)

            print(newfile, cov.shape)

def run_pca(fname="asset_cor_2009_processed.csv"):
    """Given a covariance matrix, derive the principal components from scratch."""

    cov = pd.read_csv(fname, index_col="Ticker")

    # PCA: Extract eigenvalues and eigenvectors
    values, vectors = np.linalg.eig(cov)
    component_var = [round((i/sum(values)) * 100, 2) for i in sorted(values, reverse=True)]
    loadings = vectors.T.dot(cov.T)

    # Purpose of transpose?
    print(loadings.T)

    # Convert to dataframe
    df = pd.DataFrame(loadings, index=cov.index, columns=cov.columns)
    # Z-scale?
