# Spike report
config hash: 453b3cc2da58

## G1 training: FAIL
final KL 5.00 (band [0.3, 5]); recon 12.774 -> 9.360

## G2 conditioning: FAIL
certified fraction vs h1 Spearman = -0.14 (> 0.5)
env-alignment vs severity Spearman = 0.09 (> 0.5)
cert_frac(h1): 0.90:0.07, 0.75:0.09, 0.50:0.61, 0.35:0.68, 0.15:0.69, 0.05:0.00

## G3 held-out h1=0.25: certified 0.41 (>= 0.30) -> PASS (part a)
fine-tuned cost 122923.1 vs B1 best 49080.7 (<= 1.10x) -> FAIL (part b)

G4 @ h1=0.9: model classes 1 vs B1 2 -> FAIL
G4 @ h1=0.5: model classes 1 vs B1 1 -> PASS

## Verdict: EXTEND/STOP
(G1=False, G2=False, G3a=True, G3b=False, G4=False)