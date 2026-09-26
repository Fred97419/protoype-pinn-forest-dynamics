# protoype-pinn-forest-dynamics
PINN prototype developed in preparation for the Made in France postdoctoral project

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
python experiments/inverse_problem.py --pretrain-epochs 4000 --inverse-epochs 14000
```
```markdown
The script automatically uses CUDA if a compatible GPU is available, otherwise it falls back to CPU.

```



