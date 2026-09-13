# Spike report (gates rev 4)
config hash: 5eb2895806c4
T_h = 150.0 s, z_mode = bank, latent = 2

## G1 training: PASS
final KL 5.01 (band [0.3, 6.0]); recon 18.266 -> 4.221

## G2 conditioning: FAIL
starved cells (V_prop < 0.6): lift >= 1.5: False; other cells: lift >= 0.95: True; thrust dominant at h1 = 0.05: True; mean lift 1.49; Spearman(h1, V) = 0.09
h1     V     V_prop  lift   | thrust  sway  collision  terminal  (canonical start)
0.90  0.62   0.60    1.02  | 0.00   0.00   0.38      0.00
0.75  0.59   0.58    1.02  | 0.00   0.00   0.41      0.00
0.50  0.57   0.58    0.98  | 0.00   0.00   0.43      0.00
0.35  0.71   0.59    1.20  | 0.00   0.00   0.29      0.00
0.15  0.86   0.61    1.41  | 0.00   0.00   0.14      0.00
0.05  0.54   0.16    3.33  | 0.43   0.00   0.11      0.00
per-zone @ h1=0.5 (V / V_prop): Zone 1 0.75/0.48, Zone 2 0.99/0.96, Zone 3 0.04/0.20
random starts @ h1=0.5 (V / V_prop): 0.67/0.52, 0.64/0.58, 0.66/0.55

## G3 held-out h1=0.25
(a) certified 0.84 over 3 draws (>= 0.30) -> PASS
(b) fine-tuned effort 103708.0 (Zone 2) vs B1 best 93693.8 = 1.11x (<= 1.10) -> FAIL; same-zone ratio 1.11  [fine-tune Solve_Succeeded, 3 attempt(s)]
B1: 10/12 feasible (10 converged), classes [(0, 0, 0), (1, 0, 0), (2, 0, 0)], best feasible effort 93693.75699095159 (zone Zone 2)

## G4 class coverage (zone, winding)
h1=0.9: classes among all decodes 3, among certified 3 (187 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> PASS
h1=0.5: classes among all decodes 3, among certified 3 (174 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> PASS

## Verdict: STOP
(G1=PASS, G2=FAIL, G3a=PASS, G3b=FAIL, G4=PASS)