# transport_selinf

Code for reproducing the experiments in *Flexible Selective Inference with Flow-based Transport Maps* (2025).

The method samples a statistic conditional on a model-selection event, learns a transport map from that selective distribution to its pre-selection inference, and uses the transformed statistic for selective tests and confidence intervals.

## Installation

Run all commands from a terminal:

```bash
git clone https://github.com/liusf15/transport_selinf.git
cd transport_selinf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional: compile the Cython extension only if you need the sampling module or
the lasso experiment:

```bash
cd sampling 
python setup.py build_ext --inplace 
cd ..
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `experiments/` | Simulation runners, selection procedures, and plotting notebooks |
| `flows/` | One-dimensional spline and multivariate RealNVP transport maps |
| `sampling/` | Truncated-Gaussian and separation-of-variables samplers |
| `utils/` | Shared inference utilities |

## Simulation experiments

Run the experiment modules from the repository root. A single replication can be run with:

```bash
python -m experiments.polynomial_regression.run_poly_selection --seed 0
python -m experiments.spline.run_spline --seed 0
python -m experiments.lasso.run_lassocv --seed 0
python -m experiments.pcr.run_pcr_antithetic_cv --seed 0
```

Use `--help` on a module to view its experiment-specific options. By default, results are written below `experiments/results/2025/`; use `--rootdir` and `--date` to change that location.

For 500 simulation replications, use every integer seed from 0 through 499 inclusive while keeping the other arguments fixed. For example:

```bash
for seed in $(seq 0 499); do
  python -m experiments.spline.run_spline --seed "$seed"
done
```

Apply the same seed range to the other runners. The `--seed` argument controls the simulated-data replication; fixed design and flow-training seeds are set separately in the experiment code.


## Figures and tables

Run the following notebooks after generating their corresponding simulation
results:

- Polynomial degree selection:
  [`experiments/polynomial_regression/plot_poly.ipynb`](experiments/polynomial_regression/plot_poly.ipynb)
- Spline knot selection:
  [`experiments/spline/plot_spline.ipynb`](experiments/spline/plot_spline.ipynb)
- Cross-validated lasso:
  [`experiments/lasso/plot_lasso.ipynb`](experiments/lasso/plot_lasso.ipynb)
- Principal-component selection:
  [`experiments/pcr/plot_pcr.ipynb`](experiments/pcr/plot_pcr.ipynb)

The notebooks contain result-directory variables near their first cells. Make sure those paths match the `--rootdir` and `--date` used by the simulation runners before executing all cells.

## Single-cell analysis

The analysis for the single-cell experiment is documented in
[`experiments/single_cell/README.md`](experiments/single_cell/README.md). Extract the supplied expression matrix before running the analysis notebook:

```bash
unzip experiments/single_cell/filtered_gene_expression.zip \
  -d experiments/single_cell
```

Then run [`experiments/single_cell/pbmc_antitheticCV.ipynb`](experiments/single_cell/pbmc_antitheticCV.ipynb) to reproduce the reported QQ plot and LaTeX-formatted results table. The raw preprocessing workflow is in [`experiments/single_cell/preprocessing.Rmd`](experiments/single_cell/preprocessing.Rmd).

## License

See [LICENSE](LICENSE).
