# Spike report (gates rev 4)
config hash: f56f144e7a24
T_h = 150.0 s, z_mode = bank, latent = 2

## G1 training: PASS
final KL 5.01 (band [0.3, 6.0]); recon 18.711 -> 4.256

## G2 conditioning: FAIL
starved cells (V_prop < 0.6): lift >= 1.5: False; other cells: lift >= 0.95: True; thrust dominant at h1 = 0.05: True; mean lift 1.16; Spearman(h1, V) = 0.37
h1     V     V_prop  lift   | thrust  sway  collision  terminal  (canonical start)
0.90  0.46   0.44    1.04  | 0.00   0.00   0.54      0.00
0.75  0.52   0.44    1.20  | 0.00   0.00   0.48      0.00
0.50  0.50   0.44    1.13  | 0.00   0.00   0.50      0.01
0.35  0.48   0.38    1.25  | 0.00   0.00   0.51      0.01
0.15  0.49   0.44    1.11  | 0.00   0.00   0.51      0.00
0.05  0.14   0.12    1.23  | 0.63   0.00   0.49      0.03
per-zone @ h1=0.5 (V / V_prop): Zone 1 0.62/0.36, Zone 2 0.98/0.80, Zone 3 0.00/0.04
random starts @ h1=0.5 (V / V_prop): 0.59/0.37, 0.51/0.43, 0.48/0.39

## G3 held-out h1=0.25
(a) certified 0.47 over 3 draws (>= 0.30) -> PASS
(b) fine-tuned effort 124632.0 (Zone 1) vs B1 best 97079.8 = 1.28x (<= 1.10) -> FAIL; same-zone ratio 1.00  [fine-tune Solve_Succeeded, 1 attempt(s)]
B1: 7/12 feasible (7 converged), classes [(0, 0, 0), (1, 0, 0)], best feasible effort 97079.78516581716 (zone Zone 2)

## G4 class coverage (zone, winding)
h1=0.9: classes among all decodes 3, among certified 3 (129 certified / 3 draws), B1 feasible classes 2 [(0, 0, 0), (1, 0, 0)] -> PASS
h1=0.5: classes among all decodes 3, among certified 2 (153 certified / 3 draws), B1 feasible classes 2 [(0, 0, 0), (1, 0, 0)] -> PASS

## Verdict: STOP
(G1=PASS, G2=FAIL, G3a=PASS, G3b=FAIL, G4=PASS)