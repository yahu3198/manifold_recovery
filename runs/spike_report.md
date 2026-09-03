# Spike report (gates rev 2)
config hash: 636c7327ffcd
horizon_mode: detour  T_h = 144.4 s

## G1 training: PASS
final KL 5.00 (band [0.3, 6.0] = [0.3, 1.2 Cz_max]); recon 3.983 -> 0.973

## G2 conditioning: FAIL
certified fraction vs h1 on mu_H > 0 range (h1 <= 0.8): Spearman = 0.00 (> 0.5)
env-alignment vs severity: Spearman = 0.09 (diagnostic only: c = [h1] has no wind direction)
cert_frac(h1): 0.90:0.24, 0.75:0.15, 0.50:0.88, 0.35:0.85, 0.15:0.76, 0.05:0.18
failure breakdown (fraction of decodes failing each criterion):
  h1=0.90  thrust 0.06  sway 0.69  collision 0.12  terminal 0.00
  h1=0.75  thrust 0.06  sway 0.82  collision 0.11  terminal 0.00
  h1=0.50  thrust 0.06  sway 0.00  collision 0.11  terminal 0.00
  h1=0.35  thrust 0.05  sway 0.00  collision 0.14  terminal 0.00
  h1=0.15  thrust 0.20  sway 0.00  collision 0.10  terminal 0.00
  h1=0.05  thrust 0.78  sway 0.00  collision 0.17  terminal 0.00

## G3 held-out h1=0.25
(a) certified 0.90 over 3 draws (>= 0.30) -> PASS
(b) fine-tuned effort 12756.5 vs B1 best 1624.7 = 7.85x (<= 1.10) -> FAIL   [fine-tune Solve_Succeeded, slack 0.000, 2 attempt(s)]
B1: 5/7 feasible (5 converged), classes [(0, 0), (1, 0)], best feasible effort 1624.7469278038438

## G4 class coverage
h1=0.9: classes among all decodes 3, among certified 2 (72 certified over 3 draws), B1 feasible classes 2 -> PASS
h1=0.5: classes among all decodes 3, among certified 3 (264 certified over 3 draws), B1 feasible classes 3 -> PASS

## Verdict: STOP
(G1=PASS, G2=FAIL, G3a=PASS, G3b=FAIL, G4=PASS)