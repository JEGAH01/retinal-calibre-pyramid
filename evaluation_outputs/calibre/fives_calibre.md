# FIVES_test calibre (200 images, 2048x2048, unseen)

Acc25 at GT centreline pixels, each image averaged over seeds 42/123/7, paired Wilcoxon.

| Method | Acc25 | per seed (42/123/7) | median bias |
| --- | --- | --- | --- |
| R2 | 0.224 | 0.228/0.318/0.126 | 0.573 |
| P1 | 0.125 | 0.178/0.148/0.049 | 0.368 |
| maskP1 | 0.349 | 0.367/0.291/0.389 | 0.658 |

| Comparison | Mean diff | first better on | p |
| --- | --- | --- | --- |
| R2 - P1 (PRIMARY) | +0.099 | 98% | 2e-34 |
| R2 - maskP1 | -0.125 | 10% | 2.9e-31 |
| P1 - maskP1 | -0.224 | 2% | 2.7e-34 |

## Per seed (R2 - P1, mean diff / p)

- seed 42: +0.051 / p=1.2e-21
- seed 123: +0.170 / p=1.6e-34
- seed 7: +0.077 / p=1.5e-32

## Secondary: FIVES at DRIVE field-of-view scale (x0.27, native-grid scoring)

- R2 0.416 (bias 0.780) vs P1 0.375 (bias 0.765): diff +0.041, R2 better on 85%, p=1e-24

**Primary (R2 > P1 on FIVES, p < 0.05): SUPPORTED.**
