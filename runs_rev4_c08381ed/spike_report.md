# Spike report (gates rev 4)
config hash: 45aa48132d1b
T_h = 150.0 s, z_mode = bank, latent = 2

## G1 training: PASS
final KL 5.01 (band [0.3, 6.0]); recon 18.266 -> 4.221

## G2 conditioning: PASS
starved cells (V_prop < 0.6): lift >= 1.5: True; other cells: lift >= 0.95: True; thrust dominant at h1 = 0.05: True; mean lift 1.48; Spearman(h1, V) = 0.03
h1     V     V_prop  lift   | thrust  sway  collision  terminal  (canonical start)
0.90  0.89   0.79    1.13  | 0.00   0.00   0.11      0.00
0.75  0.82   0.76    1.08  | 0.00   0.00   0.18      0.00
0.50  0.88   0.75    1.18  | 0.00   0.00   0.12      0.00
0.35  0.94   0.78    1.21  | 0.00   0.00   0.06      0.00
0.15  0.95   0.78    1.21  | 0.00   0.00   0.05      0.00
0.05  0.57   0.18    3.09  | 0.43   0.00   0.04      0.00
per-zone @ h1=0.5 (V / V_prop): Zone 1 0.89/0.67, Zone 2 1.00/1.00, Zone 3 0.87/0.60
random starts @ h1=0.5 (V / V_prop): 0.67/0.64, 0.69/0.69, 0.86/0.68

## G3 held-out h1=0.25
(a) certified 0.95 over 3 draws (>= 0.30) -> PASS
(b) fine-tuned effort 104137.1 (Zone 1) vs B1 best 84179.8 = 1.24x (<= 1.10) -> FAIL; same-zone ratio 1.00  [fine-tune Solve_Succeeded, 1 attempt(s)]
B1: 7/12 feasible (11 converged), classes [(0, 0, 0), (1, 0, 0)], best feasible effort 84179.81039229676 (zone Zone 2)

## G4 class coverage (zone, winding)
h1=0.9: classes among all decodes 3, among certified 3 (270 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> PASS
h1=0.5: classes among all decodes 3, among certified 3 (274 certified / 3 draws), B1 feasible classes 3 [(0, 0, 0), (1, 0, 0), (2, 0, 0)] -> PASS

## Verdict: STOP
(G1=PASS, G2=PASS, G3a=PASS, G3b=FAIL, G4=PASS)