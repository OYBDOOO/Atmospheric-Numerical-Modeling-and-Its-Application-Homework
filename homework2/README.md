# Homework 2: Advection Equation Solver

## Problem Description

This homework implements numerical schemes for solving the 1D linear advection equation, which is fundamental in atmospheric modeling for describing the transport of atmospheric properties.

### Governing Equation

The 1D linear advection equation:
```
∂u/∂t + c·∂u/∂x = 0
```

where:
- u(x,t) is the transported quantity (e.g., temperature, pollutant concentration)
- c is the constant advection velocity
- t is time
- x is the spatial coordinate

### Numerical Schemes Implemented

1. **FTCS (Forward in Time, Centered in Space)** - Unconditionally unstable
2. **Upwind Scheme** - First-order accurate, stable with CFL condition
3. **Lax-Wendroff Scheme** - Second-order accurate

### CFL Condition

For stability, the Courant-Friedrichs-Lewy (CFL) condition must be satisfied:
```
CFL = c·Δt/Δx ≤ 1
```

## Initial Condition

A Gaussian pulse:
```
u(x,0) = exp(-(x-x0)²/σ²)
```

## Usage

Run the script:
```bash
python advection_solver.py
```

This will:
- Solve the advection equation using different schemes
- Compare numerical solutions with the analytical solution
- Visualize the evolution of the advected pulse

## Expected Results

- Upwind scheme: Stable but diffusive
- Lax-Wendroff: Less diffusive, may show dispersion
- FTCS: Demonstrates instability (educational purpose)
