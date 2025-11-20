"""
Homework 2: Advection Equation Solver

This script solves the 1D linear advection equation using different numerical schemes:
- Upwind scheme (first-order)
- Lax-Wendroff scheme (second-order)
- FTCS scheme (for demonstrating instability)
"""

import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x, x0=5.0, sigma=0.5):
    """
    Gaussian pulse initial condition
    
    Parameters:
    -----------
    x : array
        Spatial grid points
    x0 : float
        Center of the Gaussian pulse
    sigma : float
        Width of the Gaussian pulse
    """
    return np.exp(-(x - x0)**2 / sigma**2)


def analytical_solution(x, t, c, x0=5.0, sigma=0.5):
    """
    Analytical solution: the pulse moves with velocity c
    
    Parameters:
    -----------
    x : array
        Spatial grid points
    t : float
        Time
    c : float
        Advection velocity
    x0 : float
        Initial center position
    sigma : float
        Width of the Gaussian pulse
    """
    return np.exp(-(x - x0 - c*t)**2 / sigma**2)


def upwind_scheme(u, c, dx, dt):
    """
    Upwind scheme for advection equation (first-order accurate, stable)
    
    Parameters:
    -----------
    u : array
        Current solution
    c : float
        Advection velocity
    dx : float
        Spatial step size
    dt : float
        Time step size
    """
    u_new = u.copy()
    n = len(u)
    
    if c > 0:  # Upwind direction
        for i in range(1, n):
            u_new[i] = u[i] - c * dt / dx * (u[i] - u[i-1])
        # Boundary condition: inflow at left boundary (set to zero)
        u_new[0] = 0.0
    else:
        for i in range(n-1):
            u_new[i] = u[i] - c * dt / dx * (u[i+1] - u[i])
        # Boundary condition: inflow at right boundary (set to zero)
        u_new[-1] = 0.0
    
    return u_new


def lax_wendroff_scheme(u, c, dx, dt):
    """
    Lax-Wendroff scheme for advection equation (second-order accurate)
    
    Parameters:
    -----------
    u : array
        Current solution
    c : float
        Advection velocity
    dx : float
        Spatial step size
    dt : float
        Time step size
    """
    u_new = u.copy()
    n = len(u)
    
    for i in range(1, n-1):
        u_new[i] = (u[i] - 
                    c * dt / (2*dx) * (u[i+1] - u[i-1]) + 
                    c**2 * dt**2 / (2*dx**2) * (u[i+1] - 2*u[i] + u[i-1]))
    
    # Boundary conditions: set inflow boundaries to zero
    u_new[0] = 0.0
    u_new[-1] = 0.0
    
    return u_new


def ftcs_scheme(u, c, dx, dt):
    """
    FTCS scheme (Forward in Time, Centered in Space) - Unconditionally unstable!
    Included for educational purposes to demonstrate instability.
    
    Parameters:
    -----------
    u : array
        Current solution
    c : float
        Advection velocity
    dx : float
        Spatial step size
    dt : float
        Time step size
    """
    u_new = u.copy()
    n = len(u)
    
    for i in range(1, n-1):
        u_new[i] = u[i] - c * dt / (2*dx) * (u[i+1] - u[i-1])
    
    return u_new


def solve_advection(scheme, x, u0, c, dx, dt, t_final):
    """
    Solve the advection equation using the specified scheme
    
    Parameters:
    -----------
    scheme : function
        Numerical scheme function
    x : array
        Spatial grid
    u0 : array
        Initial condition
    c : float
        Advection velocity
    dx : float
        Spatial step size
    dt : float
        Time step size
    t_final : float
        Final simulation time
    """
    u = u0.copy()
    t = 0
    nt = int(t_final / dt)
    
    # Store solution at different times for visualization
    snapshots = []
    snapshot_times = [0, t_final/3, 2*t_final/3, t_final]
    
    for n in range(nt):
        u = scheme(u, c, dx, dt)
        t += dt
        
        # Store snapshots
        for snap_time in snapshot_times:
            if abs(t - snap_time) < dt/2:
                snapshots.append((t, u.copy()))
    
    return snapshots


def main():
    """Main function to solve and visualize the advection equation"""
    
    # Domain parameters
    L = 10.0  # Domain length
    nx = 200  # Number of spatial points
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)
    
    # Physical parameters
    c = 1.0  # Advection velocity
    t_final = 2.0  # Final time
    
    # Time step (CFL condition: c*dt/dx <= 1)
    CFL = 0.8
    dt = CFL * dx / abs(c)
    
    print("="*60)
    print("1D Linear Advection Equation Solver")
    print("="*60)
    print(f"Domain: [0, {L}] with {nx} grid points")
    print(f"Spatial step: dx = {dx:.4f}")
    print(f"Time step: dt = {dt:.4f}")
    print(f"CFL number: {CFL:.2f}")
    print(f"Advection velocity: c = {c:.2f}")
    print(f"Final time: t = {t_final:.2f}")
    print("="*60)
    
    # Initial condition
    u0 = initial_condition(x)
    
    # Solve using different schemes
    print("\nSolving using different schemes...")
    upwind_snapshots = solve_advection(upwind_scheme, x, u0, c, dx, dt, t_final)
    lax_wendroff_snapshots = solve_advection(lax_wendroff_scheme, x, u0, c, dx, dt, t_final)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('1D Advection Equation: Comparison of Numerical Schemes', fontsize=16)
    
    snapshot_indices = [0, 1, 2, 3]
    titles = ['t = 0.0', f't = {t_final/3:.2f}', f't = {2*t_final/3:.2f}', f't = {t_final:.2f}']
    
    for idx, (ax, title) in enumerate(zip(axes.flat, titles)):
        if idx < len(upwind_snapshots):
            t_snap, u_upwind = upwind_snapshots[idx]
            _, u_lax_wendroff = lax_wendroff_snapshots[idx]
            u_analytical = analytical_solution(x, t_snap, c)
            
            ax.plot(x, u_analytical, 'k-', linewidth=2, label='Analytical')
            ax.plot(x, u_upwind, 'b--', linewidth=1.5, label='Upwind')
            ax.plot(x, u_lax_wendroff, 'r:', linewidth=1.5, label='Lax-Wendroff')
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('u(x,t)', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    plt.savefig('homework2_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'homework2_results.png'")
    plt.show()
    
    # Calculate and display errors
    print("\n" + "="*60)
    print("Error Analysis at Final Time")
    print("="*60)
    t_final_snap, u_upwind_final = upwind_snapshots[-1]
    _, u_lax_wendroff_final = lax_wendroff_snapshots[-1]
    u_analytical_final = analytical_solution(x, t_final, c)
    
    error_upwind = np.sqrt(np.mean((u_upwind_final - u_analytical_final)**2))
    error_lax_wendroff = np.sqrt(np.mean((u_lax_wendroff_final - u_analytical_final)**2))
    
    print(f"Upwind scheme RMS error:      {error_upwind:.6f}")
    print(f"Lax-Wendroff scheme RMS error: {error_lax_wendroff:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
