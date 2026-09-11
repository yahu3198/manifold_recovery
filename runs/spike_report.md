# Spike report (gates rev 4)
config hash: c08381ed45ed
T_h = 150.0 s, z_mode = bank, latent = 2

## G1 training: PASS
final KL 5.00 (band [0.3, 6.0]); recon 18.243 -> 4.311

## G2 conditioning: PASS
starved cells (V_prop < 0.6): lift >= 1.5: True; other cells: lift >= 0.95: True; thrust dominant at h1 = 0.05: True; mean lift 1.22; Spearman(h1, V) = 0.03
h1     V     V_prop  lift   | thrust  sway  collision  terminal  (canonical start)
0.90  0.83   0.77    1.09  | 0.00   0.00   0.17      0.00
0.75  0.81   0.77    1.05  | 0.00   0.00   0.19      0.00
0.50  0.82   0.76    1.07  | 0.00   0.00   0.18      0.00
0.35  0.84   0.77    1.09  | 0.00   0.00   0.16      0.00
0.15  0.93   0.77    1.21  | 0.00   0.00   0.07      0.00
0.05  0.34   0.19    1.78  | 0.66   0.00   0.06      0.00
per-zone @ h1=0.5 (V / V_prop): Zone 1 0.79/0.75, Zone 2 1.00/1.00, Zone 3 0.67/0.57
random starts @ h1=0.5 (V / V_prop): 0.67/0.78, 0.67/0.69, 0.65/0.69

## G3 held-out h1=0.25
(a) certified 0.89 over 10 draws (>= 0.30) -> PASS
## Verdict: GO
(G1=PASS, G2=PASS, G3a=PASS)