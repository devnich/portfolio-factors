import pprint
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# conda install -c ets factor_analyzer
from factor_analyzer import FactorAnalyzer

#---------------------------------
#  Factor Analysis references
#---------------------------------
# https://journals.sagepub.com/doi/full/10.1177/0095798418771807

#---------------------------------
#  Using Factor Analyzer package
#---------------------------------
# https://factor-analyzer.readthedocs.io/en/latest/factor_analyzer.html
#
# Loadings of variables onto factors
#     fa.loadings_
# Proportion of variable's variance that is shared (i.e. non-unique, explained by other variables)
#     fa.get_communalities()
# Proportion of variable's variance that is unique; inverse of get_communalities()
#     fa.get_uniquenesses()

#---------------------------------
#  PCA references
#---------------------------------
# https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/

#---------------------------------
#  Notes on covariance
#---------------------------------
# cor(x,y) = cov(x,y)/(sd(x) * sd(y))
# cov(x,y) = cor(x,y) * sd(x) * sd(y)

#---------------------------------
#  Notes on NumPy arrays
#---------------------------------
# Vertical vector:
#     v.reshape(-1, 1)
# Horizontal vector:
#     v or v.reshape(1, -1)
# Item multiplication across:
#     v * array
# Item multiplication down:
#     v.reshape(-1, 1) * array
# Matrix multiplication (row x column):
#     v @ array

# Wrap text when printing long collections in the terminal
pp = pprint.PrettyPrinter(width=100, compact=True, indent=2)

# Display DataFrames without truncation
pd.set_option("display.max_columns", None, "precision", 2)

def factor(fname="asset_cor_2009.csv", rotation="varimax", n=5):
    """Run an exploratory factor analysis and save the output."""

    data = pd.read_csv(fname, index_col="Ticker")

    # Extract the correlation columns from the DataFrame
    corr = data[data.index]

    fa = FactorAnalyzer(rotation=rotation, is_corr_matrix=True, n_factors=n)
    fa.fit(corr)

    # Print explained variance
    fvar = fa.get_factor_variance()
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
    r2_labels = ['_'.join([i, 'R2']) for i in factor_labels]
    var = pd.DataFrame(data=np.square(fa.loadings_).round(2),
                       index=data.index,
                       columns=r2_labels)

    # Insert asset class name
    df = pd.concat([data["Name"], df, var], axis=1)

    df["Communality"] = fa.get_communalities().round(2)

    # Save the processed file
    parts = fname.split(".csv")
    newfile = ''.join([parts[0], '_', rotation, '.csv'])
    df.to_csv(newfile)

    return fa, df

# Factor plot visualization:
# https://rpubs.com/danmirman/plotting_factor_analysis

# Then look at value/quality screeners, cf https://twitter.com/soloprosperity/status/1450228819273551876

def gen_covariance_files():
    """Create covariance matrices from correlations and standard deviations."""
    path = Path()
    for filename in path.glob(''.join(['asset_cor_', '[0-9]'*4, '.csv'])):
        if filename.is_file():
            data = pd.read_csv(filename, index_col="Ticker")

            # Get all the columns whose label matches a row label
            corr = data[data.index]
            sd = data['Annualized Standard Deviation']
            # NB: To reshape a Series, drill down to the values of the underlying array
            cov = corr * sd * sd.values.reshape(-1, 1)

            # Test the correctness of our transformation
            # NB: The diagonal of the covariance matrix is 1 * sd * sd by construction
            diag = cov.values.diagonal()
            var = sd * sd
            assert all(diag.round(2) == var.values.round(2))

            # Save the processed file
            parts = filename.name.split(".csv")
            newfile = ''.join([parts[0], "_processed.csv"])
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
