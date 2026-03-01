# transport_selinf

Code for reproducing the experiments in *Flexible Selective Inference with Flow-based Transport Maps*, 2025. [https://arxiv.org/pdf/2506.01150](https://arxiv.org/pdf/2506.01150)


To reproduce the simulations from Section 6.2 to 6.5, run the following with varying random seeds and arguments.

```
python -m experiments.polynomial.regression.run_poly_selection --seed 0
python -m experiments.spline.run_spline --seed 0
python -m experiments.lasso.run_lassocv --seed 0
python -m experiments.pcr.run_pcr_antithetic_cv --seed 0
```

Code for the single-cell data analysis is in this [folder](experiments/single_cell).
