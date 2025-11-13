"""
Homework 3: Atmospheric Heat Diffusion

This script solves the 1D heat diffusion equation using different numerical schemes:
- Explicit (FTCS) scheme
- Implicit (Backward Euler) scheme
- Crank-Nicolson scheme

Models vertical heat diffusion in the atmosphere.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


def initial_condition(z, T0=280.0, z0=500.0, dT=10.0, sigma=200.0):
    """
    Initial temperature profile with a warm layer
    
    Parameters:
    -----------
    z : array
        Vertical coordinate (height)
    T0 : float
        Base temperature (K)
    z0 : float
        Height of warm layer (m)
    dT : float
        Temperature anomaly (K)
    sigma : float
        Width of warm layer (m)
    """
    return T0 + dT * np.exp(-(z - z0)**2 / sigma**2)


def explicit_scheme(T, kappa, dz, dt):
    """
    Explicit (FTCS) scheme for heat diffusion
    Stability condition: kappa*dt/dz^2 <= 0.5
    
    Parameters:
    -----------
    T : array
        Current temperature profile
    kappa : float
        Thermal diffusivity
    dz : float
        Spatial step size
    dt : float
        Time step size
    """
    T_new = T.copy()
    n = len(T)
    r = kappa * dt / dz**2
    
    # Interior points
    for i in range(1, n-1):
        T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
    
    # Boundary conditions
    T_new[0] = T[0]  # Fixed temperature at bottom
    # Zero flux at top: dT/dz = 0 => T[n-1] = T[n-2]
    T_new[-1] = T_new[-2]
    
    return T_new


def implicit_scheme(T, kappa, dz, dt):
    """
    Implicit (Backward Euler) scheme for heat diffusion
    Unconditionally stable - solves tridiagonal system
    
    Parameters:
    -----------
    T : array
        Current temperature profile
    kappa : float
        Thermal diffusivity
    dz : float
        Spatial step size
    dt : float
        Time step size
    """
    n = len(T)
    r = kappa * dt / dz**2
    
    # Build tridiagonal matrix in banded form
    # For solve_banded: ab[0,:] = upper diagonal, ab[1,:] = main diagonal, ab[2,:] = lower diagonal
    ab = np.zeros((3, n))
    
    # Main diagonal
    ab[1, :] = 1 + 2*r
    # Upper and lower diagonals
    ab[0, 1:] = -r  # Upper diagonal
    ab[2, :-1] = -r  # Lower diagonal
    
    # Boundary conditions
    # Bottom: fixed temperature
    ab[1, 0] = 1
    ab[0, 1] = 0
    
    # Top: zero flux (dT/dz = 0)
    ab[1, -1] = 1
    ab[2, -2] = -1
    
    # Right-hand side
    rhs = T.copy()
    rhs[0] = T[0]  # Fixed temperature at bottom
    rhs[-1] = 0    # Zero flux condition at top
    
    # Solve the system
    T_new = solve_banded((1, 1), ab, rhs)
    
    return T_new


def crank_nicolson_scheme(T, kappa, dz, dt):
    """
    Crank-Nicolson scheme for heat diffusion
    Second-order accurate in time, unconditionally stable
    
    Parameters:
    -----------
    T : array
        Current temperature profile
    kappa : float
        Thermal diffusivity
    dz : float
        Spatial step size
    dt : float
        Time step size
    """
    n = len(T)
    r = kappa * dt / (2 * dz**2)
    
    # Build tridiagonal matrix for implicit part
    ab = np.zeros((3, n))
    ab[1, :] = 1 + 2*r
    ab[0, 1:] = -r
    ab[2, :-1] = -r
    
    # Boundary conditions
    ab[1, 0] = 1
    ab[0, 1] = 0
    ab[1, -1] = 1
    ab[2, -2] = -1
    
    # Right-hand side (explicit part)
    rhs = T.copy()
    for i in range(1, n-1):
        rhs[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
    
    rhs[0] = T[0]
    rhs[-1] = 0
    
    T_new = solve_banded((1, 1), ab, rhs)
    
    return T_new


def solve_diffusion(scheme, z, T0, kappa, dz, dt, t_final):
    """
    Solve the heat diffusion equation using the specified scheme
    
    Parameters:
    -----------
    scheme : function
        Numerical scheme function
    z : array
        Vertical grid
    T0 : array
        Initial temperature profile
    kappa : float
        Thermal diffusivity
    dz : float
        Spatial step size
    dt : float
        Time step size
    t_final : float
        Final simulation time
    """
    T = T0.copy()
    t = 0
    nt = int(t_final / dt)
    
    # Store solution at different times
    snapshots = []
    snapshot_times = [0, t_final/4, t_final/2, t_final]
    
    for n in range(nt):
        T = scheme(T, kappa, dz, dt)
        t += dt
        
        # Store snapshots
        for snap_time in snapshot_times:
            if abs(t - snap_time) < dt/2:
                snapshots.append((t, T.copy()))
    
    return snapshots


def main():
    """Main function to solve and visualize heat diffusion"""
    
    # Domain parameters
    H = 1000.0  # Atmospheric height (m)
    nz = 100    # Number of vertical points
    z = np.linspace(0, H, nz)
    dz = H / (nz - 1)
    
    # Physical parameters
    kappa = 10.0  # Thermal diffusivity (m²/s)
    t_final = 3600.0  # Final time (1 hour)
    
    # Time step for explicit scheme (stability condition)
    dt_explicit = 0.4 * dz**2 / kappa
    dt_implicit = 10 * dt_explicit  # Can use larger time step
    
    print("="*60)
    print("1D Atmospheric Heat Diffusion Solver")
    print("="*60)
    print(f"Domain: [0, {H}] m with {nz} grid points")
    print(f"Spatial step: dz = {dz:.2f} m")
    print(f"Thermal diffusivity: κ = {kappa:.2f} m²/s")
    print(f"Final time: t = {t_final:.0f} s ({t_final/3600:.1f} hour)")
    print(f"\nTime steps:")
    print(f"  Explicit scheme: dt = {dt_explicit:.2f} s")
    print(f"  Implicit scheme: dt = {dt_implicit:.2f} s")
    stability_param = kappa * dt_explicit / dz**2
    print(f"\nStability parameter (explicit): κ·Δt/Δz² = {stability_param:.3f}")
    print("="*60)
    
    # Initial condition
    T0 = initial_condition(z)
    
    # Solve using different schemes
    print("\nSolving using different schemes...")
    explicit_snapshots = solve_diffusion(explicit_scheme, z, T0, kappa, dz, dt_explicit, t_final)
    implicit_snapshots = solve_diffusion(implicit_scheme, z, T0, kappa, dz, dt_implicit, t_final)
    cn_snapshots = solve_diffusion(crank_nicolson_scheme, z, T0, kappa, dz, dt_implicit, t_final)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('1D Heat Diffusion in the Atmosphere', fontsize=16)
    
    titles = ['t = 0 s', f't = {t_final/4:.0f} s', f't = {t_final/2:.0f} s', f't = {t_final:.0f} s']
    
    for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
        if idx < len(explicit_snapshots):
            t_snap, T_explicit = explicit_snapshots[idx]
            _, T_implicit = implicit_snapshots[idx]
            _, T_cn = cn_snapshots[idx]
            
            ax.plot(T_explicit, z, 'b-', linewidth=2, label='Explicit')
            ax.plot(T_implicit, z, 'r--', linewidth=2, label='Implicit')
            ax.plot(T_cn, z, 'g:', linewidth=2, label='Crank-Nicolson')
            ax.set_xlabel('Temperature (K)', fontsize=11)
            ax.set_ylabel('Height (m)', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('homework3_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'homework3_results.png'")
    plt.show()
    
    # Display temperature statistics
    print("\n" + "="*60)
    print("Temperature Statistics at Final Time")
    print("="*60)
    _, T_explicit_final = explicit_snapshots[-1]
    _, T_implicit_final = implicit_snapshots[-1]
    
    print(f"Explicit scheme:")
    print(f"  Min temperature: {T_explicit_final.min():.2f} K")
    print(f"  Max temperature: {T_explicit_final.max():.2f} K")
    print(f"  Mean temperature: {T_explicit_final.mean():.2f} K")
    print(f"\nImplicit scheme:")
    print(f"  Min temperature: {T_implicit_final.min():.2f} K")
    print(f"  Max temperature: {T_implicit_final.max():.2f} K")
    print(f"  Mean temperature: {T_implicit_final.mean():.2f} K")
    print("="*60)


if __name__ == "__main__":
    main()
