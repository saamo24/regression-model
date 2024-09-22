import pandas as pd
import numpy as np
from itertools import combinations
from pprint import pprint

df = pd.read_csv("matrix.csv") #reading variables
x = df.to_numpy()
AVRGs = np.average(x, axis=0) # calculatenig averages
print(f"Averages \n {AVRGs}")
sigma = np.std(x, axis = 0) # calculating sigmas
print(f"Sigmas \n {sigma}")
r_corr = (pd.DataFrame(x, columns=['Y', 'X1', 'X2', 'X3', 'X4'])).corr() # creating correlation matrix
print(f"Correlacion Matrix\n{r_corr}")
r_corr = r_corr.to_numpy()
Q = np.linalg.inv(r_corr) # inversing the matrix
print(f"Inversed Q matrix \n {Q}")

F_CRITERIA = 3.8 

all_indices = tuple(range(r_corr.shape[0]))

# all possible combinations
all_combs = {
    i: [
        (indices[1:], np.array(r_corr[np.ix_(indices,indices)]))
        for indices in combinations(all_indices, i)
        if indices[0] == 0
    ]
    for i in range(2, r_corr.shape[0])
}
print()
pprint(all_combs)

# inversing all possible combinations
inverted = {
    key: [
        (indices, np.linalg.inv(comb))
        for (indices, comb) in matrix
    ]
    for key, matrix in all_combs.items()
}

# calculating R coefficients for each submatrixes
r_sqrs = {}
for key, matrix in inverted.items():
    tmp = np.array([(1 - 1 / comb[0, 0]) for (_, comb) in matrix])
    max_id = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)[0]
    r_sqrs[key] = max_id, matrix[max_id][0], tmp[max_id]

r_sqrs[5] = (0, (all_indices[1:]), (1 - (1 / Q[0, 0])))
print("> "*10)
pprint(r_sqrs)
# calculating Fishers coefficient
for m in range(4, 1, -1):
    F = ((r_sqrs[m + 1][2] - r_sqrs[m][2]) * (25 - m - 1)) / (1 - r_sqrs[m + 1][2])
    if F >= F_CRITERIA:
        break

# checkoing F coefficient significance
if m == 5:
    best_regr_model = Q
else:
    best_regr_model = inverted[m+1][r_sqrs[m+1][0]][1]
print("> "*20)
pprint(best_regr_model)

columns = r_sqrs[m+1][1]
pprint(columns)
print("> "*20)

# calculateing Alphas
alphas = [
        - best_regr_model[0, i] / best_regr_model[0, 0]
        for i in range(1, best_regr_model.shape[0])
]
print("Alphas")
pprint(alphas)
#calculateing Betas
betas = [
        alphas[i] * sigma[0] / sigma[col]
        for i, col in enumerate(columns)
]
print("Betas")
pprint(betas)

# calculateing sum of average of betas
sum_of_beta_avg = sum([
        betas[i] * AVRGs[col]
        for i, col in enumerate(columns)
])

result = AVRGs[0] - sum_of_beta_avg

print("_ "*20)
#CREATING THE MODEL
print('-'*5,"THE MODEL", '-'*5)
print(f"Y = {result:.6f}{''.join([f'{betas[i]:+.6f}*X{col}' for i, col in enumerate(columns)])}")
# print("Y = 47.46 + 1.7X1 + 0.67X2 - 0.26X3")
print("_ "*20)