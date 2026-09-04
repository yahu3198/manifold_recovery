# Spike report (gates rev 3)
config hash: cb71255d64bb
T_h = 150.0 s, z_mode = bank, latent = 2

## G1 training: PASS
final KL 5.01 (band [0.3, 6.0]); recon 13.546 -> 4.032

## G2 conditioning: FAIL
mean lift V/V_proposal over h1 grid = 0.40 (>= 1.3); non-increasing with degradation: False (tol 0.05); Spearman(h1, V) = 0.26
h1     V     V_prop  lift   | thrust  sway  collision  terminal  (canonical start)
0.90  0.34   0.83    0.41  | 0.00   0.00   0.66      0.47
0.75  0.15   0.78    0.19  | 0.00   0.00   0.85      0.80
0.50  0.26   0.73    0.36  | 0.00   0.00   0.73      0.63
0.35  0.22   0.80    0.27  | 0.00   0.00   0.78      0.69
0.15  0.47   0.80    0.58  | 0.00   0.00   0.51      0.49
0.05  0.12   0.22    0.57  | 0.67   0.00   0.42      0.35
random starts @ h1=0.5 (V / V_prop): 0.05/0.45, 0.48/0.68, 0.31/0.69

## G3 held-out h1=0.25
(a) certified 0.38 over 3 draws (>= 0.30) -> PASS
(b) INCONCLUSIVE: fine-tune Maximum_Iterations_Exceeded, slack 1.262 (effort 145489.7, ratio 1.27 not comparable)
B1: 5/12 feasible (8 converged), classes [(0, 0, 0), (1, 0, 0)], best feasible effort 114880.4067504744 (zone Zone 1)

## G4 class coverage (zone, winding)
h1=0.9: classes among all decodes 3, among certified 2 (98 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> FAIL
h1=0.5: classes among all decodes 4, among certified 3 (78 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> PASS

## Verdict: STOP
(G1=PASS, G2=FAIL, G3a=PASS, G3b=INCONCLUSIVE, G4=FAIL)