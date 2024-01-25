# %%
import argparse
from scipy.stats import wilcoxon , binom , binomtest

import numpy as np
import pandas as pd



def signtest(A, B, alternative='two-sided'):

    # Let's assume these are your two datasets
    A = np.array(A, dtype=np.int32)
    B = np.array(B, dtype=np.int32)

    # Calculate the differences
    diff = A - B

    # Count the number of positive and negative differences
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)

    # The test statistic is the smaller of these two counts
    n = min(n_pos, n_neg)

    # The Sign Test is essentially a binomial test with p=0.5
    #p_value = 2 * binom.cdf(n, n_pos + n_neg, 0.5)

    return binomtest(n, n_pos + n_neg, 0.5, alternative=alternative)





# %%

if __name__ == "__main__": #True:
    
    parser = argparse.ArgumentParser(description='Conduct hypothesis testing on two sets of predictions')
    parser.add_argument('--Apath', help=f'path of predictions for model A')
    parser.add_argument('--Bpath', help='path of predictions for model B')

    arg = parser.parse_args()

    A = pd.read_csv(arg.Apath)
    B = pd.read_csv(arg.Bpath)


    transformA = A[['predict', 'truth']].apply(lambda x : 1 if x['predict'] == x['truth'] else 0, axis=1)
    transformB = B[['predict', 'truth']].apply(lambda x : 1 if x['predict'] == x['truth'] else 0, axis=1)


    print("Wilcoxon signed-rank test")
    print("Two-sided: ", wilcoxon(transformA, transformB, alternative='two-sided'))
    print("Greater: ", wilcoxon(transformA, transformB, alternative='greater'))
    print("Less: ", wilcoxon(transformA, transformB, alternative='less'))

    print()

    print("Sign Test")
    print(f"Two-sided: {signtest(transformA, transformB, alternative='two-sided')}")
    print(f"Greater: {signtest(transformA, transformB, alternative='greater')}")
    print(f"Less: {signtest(transformA, transformB, alternative='less')}")
    print()


    print("Binomial test if A is better than chance")
    print()
    p = 0.5
    n = len(transformA)

    A_success = np.sum(transformA)

    print(f"Two-sided: {binomtest(A_success, n, p, alternative='two-sided')}") # 'greater', or 'less'.
    print(f"Greater: {binomtest(A_success, n, p, alternative='greater')}")

    print()
    

    print("Binomial test if B is better than chance")

    print()
    B_success = np.sum(transformB)

    print(f"Two-sided: {binomtest(B_success, n, p, alternative='two-sided')}") # 'greater', or 'less'.
    print(f"Greater: {binomtest(B_success, n, p, alternative='greater')}")




# %%
