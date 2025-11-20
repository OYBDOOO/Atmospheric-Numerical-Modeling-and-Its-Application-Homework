"""
Homework 1: Basic Numerical Methods - Finite Difference Schemes

This script demonstrates different finite difference schemes for numerical differentiation
and compares them with analytical solutions.
"""

import numpy as np
import matplotlib.pyplot as plt


def test_function(x):
    """Test function: f(x) = sin(x)"""
    return np.sin(x)


def analytical_derivative(x):
    """Analytical derivative: f'(x) = cos(x)"""
    return np.cos(x)


def forward_difference(f, x, h):
    """Forward difference scheme: f'(x) ≈ (f(x+h) - f(x)) / h"""
    return (f(x + h) - f(x)) / h


def backward_difference(f, x, h):
    """Backward difference scheme: f'(x) ≈ (f(x) - f(x-h)) / h"""
    return (f(x) - f(x - h)) / h


def central_difference(f, x, h):
    """Central difference scheme: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)"""
    return (f(x + h) - f(x - h)) / (2 * h)


def main():
    """Main function to demonstrate finite difference methods"""
    
    # Define the point at which to compute the derivative
    x0 = np.pi / 4
    
    # Range of step sizes for error analysis
    h_values = np.logspace(-10, -1, 50)
    
    # Compute analytical derivative
    exact_derivative = analytical_derivative(x0)
    
    # Initialize arrays for errors
    forward_errors = np.zeros_like(h_values)
    backward_errors = np.zeros_like(h_values)
    central_errors = np.zeros_like(h_values)
    
    # Compute numerical derivatives and errors
    for i, h in enumerate(h_values):
        forward_errors[i] = abs(forward_difference(test_function, x0, h) - exact_derivative)
        backward_errors[i] = abs(backward_difference(test_function, x0, h) - exact_derivative)
        central_errors[i] = abs(central_difference(test_function, x0, h) - exact_derivative)
    
    # Plot 1: Error analysis
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.loglog(h_values, forward_errors, 'b-', label='Forward Difference', linewidth=2)
    plt.loglog(h_values, backward_errors, 'r--', label='Backward Difference', linewidth=2)
    plt.loglog(h_values, central_errors, 'g-.', label='Central Difference', linewidth=2)
    plt.loglog(h_values, h_values, 'k:', label='O(h)', linewidth=1)
    plt.loglog(h_values, h_values**2, 'k--', label='O(h²)', linewidth=1)
    plt.xlabel('Step Size (h)', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Error Analysis of Finite Difference Schemes', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Derivative comparison with a specific h
    h_demo = 0.1
    x_range = np.linspace(0, 2*np.pi, 100)
    
    analytical = analytical_derivative(x_range)
    forward_num = np.array([forward_difference(test_function, x, h_demo) for x in x_range])
    central_num = np.array([central_difference(test_function, x, h_demo) for x in x_range])
    
    plt.subplot(1, 2, 2)
    plt.plot(x_range, analytical, 'k-', label='Analytical', linewidth=2)
    plt.plot(x_range, forward_num, 'b--', label=f'Forward (h={h_demo})', linewidth=1.5)
    plt.plot(x_range, central_num, 'r:', label=f'Central (h={h_demo})', linewidth=1.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel("f'(x)", fontsize=12)
    plt.title('Numerical vs Analytical Derivative', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('homework1_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'homework1_results.png'")
    plt.show()
    
    # Print numerical results
    print("\n" + "="*60)
    print("Finite Difference Method Results")
    print("="*60)
    print(f"Test point: x = π/4 = {x0:.6f}")
    print(f"Test function: f(x) = sin(x)")
    print(f"Analytical derivative: f'(x) = cos(x) = {exact_derivative:.6f}")
    print("\nNumerical derivatives (h = 0.01):")
    h = 0.01
    print(f"  Forward difference:  {forward_difference(test_function, x0, h):.6f}")
    print(f"  Backward difference: {backward_difference(test_function, x0, h):.6f}")
    print(f"  Central difference:  {central_difference(test_function, x0, h):.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
