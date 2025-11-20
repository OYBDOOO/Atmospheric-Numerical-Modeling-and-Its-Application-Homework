# Homework 1: Basic Numerical Methods

## Problem Description

This homework implements basic finite difference schemes for numerical differentiation.

### Topics Covered
1. Forward difference scheme
2. Backward difference scheme
3. Central difference scheme
4. Error analysis and convergence

## Problem Statement

Given a function f(x) = sin(x), compute its derivative using different finite difference schemes and compare with the analytical derivative f'(x) = cos(x).

### Finite Difference Schemes

**Forward Difference:**
```
f'(x) ≈ (f(x+h) - f(x)) / h
```

**Backward Difference:**
```
f'(x) ≈ (f(x) - f(x-h)) / h
```

**Central Difference:**
```
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
```

## Usage

Run the script:
```bash
python finite_difference.py
```

This will generate plots showing:
- Comparison of numerical derivatives with analytical solution
- Error analysis for different step sizes

## Expected Results

- Central difference should have O(h²) error
- Forward and backward differences should have O(h) error
