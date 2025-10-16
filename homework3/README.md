# Homework 3: Atmospheric Heat Diffusion

## Problem Description

This homework implements numerical schemes for solving the 1D heat diffusion equation, which models the vertical diffusion of heat in the atmosphere.

### Governing Equation

The 1D heat diffusion equation:
```
∂T/∂t = κ·∂²T/∂z²
```

where:
- T(z,t) is the temperature
- κ is the thermal diffusivity coefficient
- t is time
- z is the vertical coordinate (height)

### Numerical Schemes Implemented

1. **Explicit (FTCS) Scheme** - Forward in time, centered in space
   - Stability condition: κ·Δt/Δz² ≤ 0.5
   
2. **Implicit (Backward Euler) Scheme** - Unconditionally stable
   - Requires solving a tridiagonal system
   
3. **Crank-Nicolson Scheme** - Second-order accurate in time
   - Unconditionally stable

### Boundary Conditions

- Bottom boundary (z=0): Fixed temperature (Dirichlet)
- Top boundary (z=H): Zero flux (Neumann)

## Initial Condition

Temperature profile with a warm layer:
```
T(z,0) = T0 + ΔT·exp(-(z-z0)²/σ²)
```

## Usage

Run the script:
```bash
python heat_diffusion.py
```

This will:
- Solve the heat diffusion equation using different schemes
- Compare explicit and implicit methods
- Visualize temperature evolution over time

## Expected Results

- Heat diffuses from high to low temperature regions
- Implicit schemes remain stable with larger time steps
- Temperature profile smooths out over time
