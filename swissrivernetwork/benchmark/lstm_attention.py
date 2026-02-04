##
#Attention Models?


## Double Loss Attention Model
# Full Scale Graphlet
#
#
# 1) create prediction based on lstm(at, e_k) -> \hat{wt}
# 2) create refined prediction based on: lstm(at, e_k, all other \hat{wt})
# 2) create refined prediction based on: a_ij, lstm(at, e_k, a_kj*\hat{wt}) -> \hat{wt}
#
#
#
# 1) create prediction based on lstm(at, e_k) -> h_k -> \hat{wt}
# 2) Refine prediction based on lstm(at, e_k, \Sum{a_kj * h_j}) # (=> Message passing?)
#
#
#
# Top 3:
# 1) create prediction based on lstm(at, e_k) -> \hat{wt}
# 2) create a_ij
# 3) Select top 3: a_ij: lstm(at, e_k, \hat{wt}_1, \hat{wt}_2, \hat{wt}_3)


## Baseline Attention Model
# 