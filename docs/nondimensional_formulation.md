# Non-dimensional Formulation

## 1. Reference Scales

| Symbol | Definition | Expression | Typical unit |
|--------|-----------|------------|-------------|
| $L_\text{ref}$ | Beam length | $L$ | mm |
| $w_\text{ref}$ | Displacement | $L$ | mm |
| $F_\text{ref}$ | Force | $E_c \cdot A$ | N |
| $M_\text{ref}$ | Moment | $E_c \cdot I / L$ | N·mm |
| $\varepsilon_\text{ref}$ | Strain | $1$ | — |
| $\kappa_\text{ref}$ | Curvature | $1/L$ | 1/mm |
| $q_\text{ref}$ | Distributed load | $F_\text{ref}/L = E_c A / L$ | N/mm |

Where $E_c$ is the concrete elastic modulus, $A$ the gross cross-section area, $I$ the gross second moment of area.

## 2. Dimensionless Variables

$$
\xi = \frac{x}{L}, \quad
\bar{w} = \frac{w}{L}, \quad
\bar{\varepsilon}_0 = \varepsilon_0, \quad
\bar{M} = \frac{M}{E_c I / L}, \quad
\bar{N} = \frac{N}{E_c A}, \quad
\bar{q} = \frac{q}{E_c A / L}, \quad
\bar{\kappa} = \kappa \cdot L
$$

## 3. Governing Equations (Dimensional)

Euler-Bernoulli beam under uniform load $q$ with fiber section:

**Compatibility:**
$$
\kappa(x) = -w''(x)
$$

**Section constitutive (fiber integration):**
$$
N = \int_A \sigma(\varepsilon_0 - y \kappa) \, dA, \qquad
M = -\int_A \sigma(\varepsilon_0 - y \kappa) \, y \, dA
$$

**Equilibrium:**
$$
M''(x) = -q, \qquad N'(x) = 0
$$

**Boundary conditions (simply supported):**
$$
w(0) = 0, \quad w(L) = 0, \quad M(0) = 0, \quad M(L) = 0
$$

**Axial constraint (no applied axial load):**
$$
N = 0 \quad \text{at boundaries}
$$

## 4. Non-dimensional Governing Equations

### 4.1 Compatibility

From $\kappa = -w''$ and $\bar{\kappa} = \kappa L$, $\bar{w} = w/L$:

$$
\frac{d^2 w}{dx^2} = \frac{w_\text{ref}}{L^2} \frac{d^2 \bar{w}}{d\xi^2} = \frac{1}{L} \frac{d^2 \bar{w}}{d\xi^2}
$$

$$
\kappa = -\frac{1}{L} \frac{d^2 \bar{w}}{d\xi^2}
\quad \Rightarrow \quad
\bar{\kappa} = -\frac{d^2 \bar{w}}{d\xi^2}
$$

In code: `kappa_bar = -d2w_bar / dxi2`

### 4.2 Section Constitutive

The fiber section is evaluated in **dimensional** space:

$$
\varepsilon_i = \varepsilon_0 - y_i \kappa
= \bar{\varepsilon}_0 \cdot \varepsilon_\text{ref} - y_i \cdot \bar{\kappa} \cdot \kappa_\text{ref}
= \bar{\varepsilon}_0 - y_i \cdot \bar{\kappa} / L
$$

$$
\sigma_i = \sigma(\varepsilon_i) \quad \text{(material model)}
$$

$$
N_\text{sec} = \sum_i \sigma_i A_i, \qquad
M_\text{sec} = \sum_i \sigma_i A_i (-y_i)
$$

Then non-dimensionalised:

$$
\bar{N}_\text{sec} = \frac{N_\text{sec}}{F_\text{ref}}, \qquad
\bar{M}_\text{sec} = \frac{M_\text{sec}}{M_\text{ref}}
$$

### 4.3 Constitutive Loss

The PINN has independent networks for $\bar{M}_\text{net}$ and $\bar{N}_\text{net}$. The constitutive loss enforces consistency with the fiber section:

$$
\mathcal{L}_\text{const\_M} = \frac{1}{n} \sum_j \left( \bar{M}_\text{net}(\xi_j) - \bar{M}_\text{sec}(\xi_j) \right)^2
$$

$$
\mathcal{L}_\text{const\_N} = \frac{1}{n} \sum_j \left( \bar{N}_\text{net}(\xi_j) - \bar{N}_\text{sec}(\xi_j) \right)^2
$$

### 4.4 Equilibrium

From $M'' = -q$:

$$
\frac{d^2 M}{dx^2} = \frac{M_\text{ref}}{L^2} \frac{d^2 \bar{M}}{d\xi^2} = -q
$$

$$
\frac{d^2 \bar{M}}{d\xi^2} = -\frac{q L^2}{M_\text{ref}}
= -\frac{q L^2}{E_c I / L}
= -\frac{q L^3}{E_c I}
$$

Using $\bar{q} = q / q_\text{ref}$:

$$
\frac{d^2 \bar{M}}{d\xi^2} = -\bar{q} \cdot C_\text{eq}
\quad \text{where} \quad
C_\text{eq} = \frac{q_\text{ref} \cdot L^2}{M_\text{ref}}
= \frac{E_c A / L \cdot L^2}{E_c I / L}
= \frac{A \cdot L^2}{I}
$$

$$
\mathcal{L}_\text{equil\_M} = \frac{1}{n} \sum_j \left( \frac{d^2 \bar{M}_\text{net}}{d\xi^2}\bigg|_{\xi_j} + \bar{q} \cdot C_\text{eq} \right)^2
$$

From $N' = 0$:

$$
\frac{d \bar{N}_\text{net}}{d\xi} = 0
$$

$$
\mathcal{L}_\text{equil\_N} = \frac{1}{n} \sum_j \left( \frac{d\bar{N}_\text{net}}{d\xi}\bigg|_{\xi_j} \right)^2
$$

### 4.5 Boundary Conditions

$$
\mathcal{L}_\text{bc} = w_\text{bc} \left[
\bar{w}(0)^2 + \bar{w}(1)^2 + \bar{M}_\text{net}(0)^2 + \bar{M}_\text{net}(1)^2
\right]
$$

### 4.6 N=0 Boundary Constraint

For a simply supported beam with no axial load, $N=0$ everywhere. This is enforced as a boundary condition: the section axial force must be zero at the supports.

Combined with $dN/d\xi = 0$ (equilibrium), this implies $N=0$ on the entire beam.

$$
\mathcal{L}_{N=0} = w_{N0} \left[
\bar{N}_\text{sec}(\xi=0)^2 + \bar{N}_\text{sec}(\xi=1)^2
\right]
$$

The gradient of this loss flows through the fiber section back to the $\varepsilon_0$ network, constraining the centroidal strain.

## 5. Total Loss

$$
\mathcal{L} = \mathcal{L}_\text{const\_M} + \mathcal{L}_\text{const\_N}
+ \mathcal{L}_\text{equil\_M} + \mathcal{L}_\text{equil\_N}
+ \mathcal{L}_\text{bc} + \mathcal{L}_{N=0}
+ \mathcal{L}_\text{data}
$$

Each term can be independently weighted via `loss_weights` in the config.

## 6. Network Architecture

Four independent MLPs, each mapping $\xi \in [0,1] \to \mathbb{R}$:

| Network | Output | Physical meaning |
|---------|--------|-----------------|
| `net_w` | $\bar{w}(\xi)$ | Non-dim displacement |
| `net_eps0` | $\bar{\varepsilon}_0(\xi)$ | Centroidal strain |
| `net_M` | $\bar{M}(\xi)$ | Non-dim bending moment |
| `net_N` | $\bar{N}(\xi)$ | Non-dim axial force |

## 7. Computation Flow

```
xi (collocation)
  │
  ├─→ net_w(xi)     → w_bar  ──→ d²w_bar/dξ² → kappa_bar ──┐
  ├─→ net_eps0(xi)  → eps0_bar ──────────────────────────────┤
  ├─→ net_M(xi)     → M_bar_net ──→ d²M_bar/dξ² → equil_M  │
  └─→ net_N(xi)     → N_bar_net ──→ dN_bar/dξ   → equil_N  │
                                                              │
                    ┌─────────────────────────────────────────┘
                    │  eps0_dim = eps0_bar * eps_ref
                    │  kappa_dim = kappa_bar * kap_ref
                    ▼
            Fiber Section Integration
            eps_i = eps0 - y_i * kappa
            sigma_i = material.stress(eps_i)
            N_sec = Σ sigma_i * A_i
            M_sec = Σ sigma_i * A_i * (-y_i)
                    │
                    ▼
            N_bar_sec = N_sec / F_ref
            M_bar_sec = M_sec / M_ref
                    │
                    ├─→ const_M = MSE(M_bar_net - M_bar_sec)
                    ├─→ const_N = MSE(N_bar_net - N_bar_sec)
                    └─→ N_zero  = N_bar_sec(boundary)² (BC)
```

## 8. Key Numerical Values (Example)

For L=3000mm, b=300mm, h=500mm, Ec=25000MPa, q=20N/mm:

| Scale | Value |
|-------|-------|
| $w_\text{ref} = L$ | 3000 mm |
| $F_\text{ref} = E_c A$ | 3.75×10⁹ N |
| $M_\text{ref} = E_c I / L$ | 2.60×10¹⁰ N·mm |
| $q_\text{ref} = E_c A / L$ | 1.25×10⁶ N/mm |
| $\bar{q}$ | 1.6×10⁻⁵ |
| $C_\text{eq} = AL²/I$ | 432 |
| $\bar{q} \cdot C_\text{eq}$ | 6.91×10⁻³ |

Expected non-dimensional field values at midspan:

| Field | Non-dim value | Note |
|-------|--------------|------|
| $\bar{w}_\text{mid}$ | ~7.3×10⁻⁵ | Very small — network must learn tiny output |
| $\bar{M}_\text{mid}$ | ~8.6×10⁻⁴ | Small |
| $\bar{\varepsilon}_0$ | ~0 | Should be near zero for symmetric loading |
| $\bar{N}$ | ~0 | Should be zero everywhere |
