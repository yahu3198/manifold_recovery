# Baselines @ h1=0.25, sea state 3, canonical start

B1 restarts: best FEASIBLE effort 85265.96628128215, feasible 7/12 (converged 11), classes [(0, 0, 0), (1, 0, 0)], wall 104.5s
    Zone 1 via       Solve_Succeeded              slack    0.000 effort   105079.1 sig (0, 0, 0)
    Zone 1 chord     Solve_Succeeded              slack    0.000 effort   158263.7 sig (0, 0, 0)
    Zone 1 bulge+12  Solve_Succeeded              slack    0.000 effort   105079.1 sig (0, 0, 0)
    Zone 1 bulge-12  Solve_Succeeded              slack    0.000 effort   105079.1 sig (0, 0, 0)
    Zone 2 via       Maximum_Iterations_Exceeded  slack    1.295 effort    92741.3 sig (-1, 0, 0)
    Zone 2 chord     Solve_Succeeded              slack    0.000 effort    85266.0 sig (1, 0, 0)
    Zone 2 bulge+12  Solve_Succeeded              slack    0.000 effort    85266.0 sig (1, 0, 0)
    Zone 2 bulge-12  Solve_Succeeded              slack    0.000 effort    85266.0 sig (1, 0, 0)
    Zone 3 via       Solve_Succeeded              slack    0.807 effort   134898.5 sig (2, 0, 0)
    Zone 3 chord     Solve_Succeeded              slack    0.807 effort   136538.8 sig (2, 0, 0)
    Zone 3 bulge+12  Solve_Succeeded              slack    0.807 effort    95817.3 sig (2, 0, 0)
    Zone 3 bulge-12  Solve_Succeeded              slack    0.807 effort    96517.5 sig (2, 0, 0)
B2 CEM: best R -1250.6, classes 2 [(np.int64(-1), np.int64(0), np.int64(0)), (np.int64(1), np.int64(0), np.int64(0))], wall 6.2s
B4 rejection (5s): acceptance 0.754 over 1000 samples, classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
B3 per-instance LSMO: wall 3.2s (n_train 750)