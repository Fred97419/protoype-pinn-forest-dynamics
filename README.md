# protoype-pinn-forest-dynamics
PINN prototype developed in preparation for the Made in France postdoctoral project  including reaction-diffusion, light competition : 

```math
\frac{\partial u_i}{\partial t}
=
D_i \Delta u_i
+
r_i u_i
(1-u_i)
\left(1-\sum_j u_j\right)
\exp\left(-\sum_j \alpha_j u_j\right)
```

with inverse parameter estimation and CUDA training.

## Quick start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies 
```bash
pip install numpy matplotlib torch scipy

```


## Launch experiment

Run the main experiment with:

```bash
python inverse_problem.py --pretrain-epochs 4000 --inverse-epochs 14000
```
The script automatically uses CUDA if a compatible GPU is available, otherwise it falls back to CPU but training can be significantly slower on CPU-only machines.

## Notebook
An interactive Jupyter notebook is also available in:

`prototype.ipynb`

It provides a step-by-step version of the experiment and can be used to inspect the model, training process and results interactively.




