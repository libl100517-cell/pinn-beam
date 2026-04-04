# PINN — Physics-Informed Neural Network for RC Fiber Beam Analysis

A modular Python framework for **forward** and **inverse analysis** of reinforced concrete beams using Physics-Informed Neural Networks (PINNs) with a fiber section model.

## Features

- **Mander concrete** constitutive model (unconfined, extensible to confined)
- **Bilinear steel** constitutive model with post-yield hardening
- **Fiber section** integration for arbitrary RC cross-sections
- **Four independent MLPs** for displacement, centroidal strain, moment, and axial force
- **Non-dimensionalized** governing equations for numerical stability
- **Forward analysis** — compute beam response from known parameters
- **Inverse analysis** — identify material parameters (Ec, Es, fc, fy) from observations
- **Displacement data loss** with reserved interface for **crack width** observations
- Modular, extensible architecture for research development

## Project Structure

```
pinn/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── configs/                           # Analysis configurations
│   ├── base_config.py                 # BeamConfig dataclass
│   ├── simply_supported_elastic.py    # Elastic forward config
│   ├── simply_supported_elastoplastic.py  # Elasto-plastic config
│   └── simply_supported_inverse.py    # Inverse identification config
├── materials/                         # Constitutive models
│   ├── base_material.py               # Abstract interface
│   ├── concrete_mander.py             # Mander unconfined concrete
│   └── steel_bilinear.py              # Bilinear elasto-plastic steel
├── sections/                          # Section discretisation
│   ├── fibers.py                      # Fiber and FiberCollection
│   ├── fiber_section.py               # Section resultant integration
│   └── rc_rect_section.py             # Rectangular RC section builder
├── physics/                           # Beam mechanics (non-dimensional)
│   ├── nondimensional.py              # Reference scales and conversions
│   ├── beam_equations.py              # PDE residuals (Euler-Bernoulli)
│   ├── boundary_conditions.py         # Simply supported BC residuals
│   └── losses.py                      # Weighted loss assembler
├── models/                            # Neural network architecture
│   ├── mlp.py                         # Generic MLP building block
│   ├── field_nets.py                  # Four independent field MLPs
│   ├── pinn_beam.py                   # Top-level PINN beam coordinator
│   └── inverse_parameters.py         # Trainable parameter registry
├── solvers/                           # Training and analysis drivers
│   ├── trainer.py                     # Core training loop
│   ├── forward_solver.py              # Forward analysis driver
│   └── inverse_solver.py             # Inverse analysis driver
├── plotting/                          # Visualisation utilities
│   └── plot_results.py                # Field, loss, and convergence plots
├── utils/                             # Shared utilities
│   ├── device.py                      # CUDA/CPU device selection
│   ├── seed.py                        # Reproducibility
│   ├── logger.py                      # Training metric logger
│   └── sampling.py                    # Collocation point sampling
├── examples/
│   └── run_simply_supported_beam.py   # Complete runnable example
└── tests/
    ├── test_materials.py              # Material model tests
    ├── test_sections.py               # Section integration tests
    └── test_nondimensional.py         # Scaling consistency tests
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pinn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
cd pinn
python -m examples.run_simply_supported_beam
```

This runs three analyses on a simply supported RC beam:
1. **Elastic forward** — predicts beam response with linear material behaviour
2. **Elasto-plastic forward** — predicts response with nonlinear Mander concrete and bilinear steel
3. **Inverse** — identifies the concrete elastic modulus (Ec) from synthetic displacement data

Results are saved as PNG figures in the `figures/` directory.

## Running Tests

```bash
cd pinn
python -m pytest tests/ -v
```

## Analysis Modes

### Forward Analysis

Given geometry, material parameters, section properties, loads, and boundary conditions, compute the beam response (displacement, strain, moment, axial force).

```python
from configs.simply_supported_elastic import get_config
from solvers import ForwardSolver

config = get_config()
solver = ForwardSolver(config)
results = solver.solve()
```

### Inverse Analysis

Identify unknown material parameters from observed structural response. Parameters can be fixed, partially trainable, or jointly trainable with physically meaningful bounds.

```python
from solvers import InverseSolver

trainable = {
    "Ec": {"init": 18000.0, "bounds": (10000.0, 50000.0)},
    "fc": {"init": 25.0, "bounds": (15.0, 60.0)},
}

solver = InverseSolver(
    config=config,
    trainable_params=trainable,
    observation_xi=xi_obs,
    observation_w=w_obs,
)
results = solver.solve()
print(results["identified_params"])
```

## Four-Network Architecture

The structural state is represented by **four independent MLPs**, each mapping the non-dimensional coordinate ξ ∈ [0, 1] to one field:

| Network   | Output        | Physical meaning       |
|-----------|---------------|------------------------|
| `net_w`   | w̄(ξ)         | Transverse displacement |
| `net_eps0`| ε̄₀(ξ)       | Centroidal axial strain |
| `net_M`   | M̄(ξ)         | Bending moment         |
| `net_N`   | N̄(ξ)         | Axial force            |

This design keeps each field independently modifiable and avoids the compromises of a single shared-output network.

## Non-Dimensionalization

All governing equations are formulated in **dimensionless form** for numerical stability. The reference scales are:

| Scale   | Definition             | Typical unit |
|---------|------------------------|--------------|
| L_ref   | Beam length            | mm           |
| F_ref   | Ec × A                 | N            |
| M_ref   | Ec × I / L             | N·mm         |
| w_ref   | L                      | mm           |
| eps_ref | 1                      | —            |
| kap_ref | 1 / L                  | 1/mm         |
| q_ref   | F_ref / L              | N/mm         |

The `NonDimScales` class in `physics/nondimensional.py` provides all conversions between dimensional and non-dimensional quantities.

### Dimensionless equations

- **Compatibility**: κ̄ + d²w̄/dξ² = 0
- **Transverse equilibrium**: d²M̄/dξ² + q̄ · (A_ref · L² / I_ref) = 0
- **Axial equilibrium**: dN̄/dξ = 0
- **Constitutive**: M̄_net = M̄_section, N̄_net = N̄_section

## Extensibility

The architecture supports future additions:

- **Constitutive models**: subclass `BaseMaterial` (e.g., confined Mander, Ramberg-Osgood steel)
- **Section types**: subclass `FiberSection` or create new builders (circular, T-section, prestressed)
- **Beam theories**: extend `BeamEquations` for Timoshenko or higher-order theories
- **Loading**: add point loads, multi-step loading in the config and solver
- **Data terms**: implement `data_crack_width_loss` in `losses.py` for crack observations
- **Fields**: add more MLPs in `FieldNetworks` for additional output fields

## GitHub Workflow

### Initialise Git

```bash
cd pinn
git init
git add .
git commit -m "feat: initial PINN framework for RC fiber beam analysis"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### Branch Strategy

| Branch    | Purpose                         |
|-----------|----------------------------------|
| `main`    | Stable, tested code             |
| `dev`     | Active development integration  |
| `feature/*` | Individual features           |

Example feature branches:
- `feature/confined-concrete`
- `feature/crack-width-loss`
- `feature/timoshenko-beam`
- `feature/ci-pipeline`

### Commit Message Style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Mander concrete model
feat: add bilinear steel constitutive law
fix: correct non-dimensional equilibrium residual
refactor: decouple inverse parameter registry from beam model
test: add fiber section resultant consistency test
docs: update README with non-dimensionalization details
```

## License

This project is intended for research use. Add your preferred license (MIT, Apache 2.0, etc.) before public release.
