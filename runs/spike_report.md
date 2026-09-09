# Spike report (gates rev 4)
config hash: c08381ed45ed
T_h = 150.0 s, z_mode = bank, latent = 2

## G1 training: PASS
final KL 5.00 (band [0.3, 6.0]); recon 18.243 -> 4.311

## G2 conditioning: FAIL
mean lift V/V_proposal over h1 grid = 1.21 (>= 1.3); non-increasing with degradation: False (tol 0.05); Spearman(h1, V) = 0.06
h1     V     V_prop  lift   | thrust  sway  collision  terminal  (canonical start)
0.90  0.85   0.79    1.08  | 0.00   0.00   0.15      0.00
0.75  0.81   0.76    1.06  | 0.00   0.00   0.19      0.00
0.50  0.81   0.75    1.08  | 0.00   0.00   0.19      0.00
0.35  0.86   0.78    1.10  | 0.00   0.00   0.14      0.00
0.15  0.93   0.78    1.19  | 0.00   0.00   0.07      0.00
0.05  0.32   0.18    1.75  | 0.68   0.00   0.06      0.00
per-zone @ h1=0.5 (V / V_prop): Zone 1 0.82/0.67, Zone 2 1.00/1.00, Zone 3 0.67/0.60
random starts @ h1=0.5 (V / V_prop): 0.70/0.64, 0.75/0.69, 0.86/0.68

## G3 held-out h1=0.25
(a) certified 0.87 over 3 draws (>= 0.30) -> PASS
(b) fine-tuned effort 112178.9 (Zone 2) vs B1 best 84179.8 = 1.33x (<= 1.10) -> FAIL; same-zone ratio 1.33  [fine-tune Solve_Succeeded, 3 attempt(s)]
B1: 8/12 feasible (12 converged), classes [(0, 0, 0), (1, 0, 0)], best feasible effort 84179.81039081694 (zone Zone 2)

## G4 class coverage (zone, winding)
h1=0.9: classes among all decodes 3, among certified 3 (247 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> PASS
h1=0.5: classes among all decodes 3, among certified 3 (246 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> PASS

## Verdict: STOP
(G1=PASS, G2=FAIL, G3a=PASS, G3b=FAIL, G4=PASS)