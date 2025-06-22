import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Parâmetros do Problema
L = 15.0  # Comprimento do domínio
T = 12.0  # Tempo total de simulação

# Condição inicial
def initial_condition(x):
    return np.exp(-20 * (x - 2)**2) + np.exp(-(x - 5)**2)

def solve_ftcs(Pe, Nx):
    """
    Resolve a equação com o método Forward-Time Centered-Space (FTCS).
    """
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    kappa = 1.0 / Pe

    # --- Análise Crítica (A "Correção") ---
    cell_Pe = Pe * dx
    print(f"--- Iniciando FTCS com Pe = {Pe} ---")
    print(f"Número de Péclet da malha (Pe*dx): {cell_Pe:.2f}")
    if cell_Pe > 2:
        print("AVISO: Pe da malha > 2. Esperam-se fortes oscilações e instabilidade!")

    # Condição de estabilidade para o passo de tempo
    dt = 0.4 * min(dx, dx**2 / (2 * kappa + 1e-9)) # Fator de segurança 0.4
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt)
    dt = T / (Nt - 1) 

    print(f"Nx = {Nx}, Nt = {Nt}, dx = {dx:.4f}, dt = {dt:.4f}")

    u = np.zeros((Nx, Nt))
    u[:, 0] = initial_condition(x)
    
    c_adv = dt / (2 * dx)
    c_diff = dt * kappa / dx**2
    
    for n in range(0, Nt - 1):
        for i in range(0, Nx):
            ip1 = (i + 1) % Nx
            im1 = (i - 1 + Nx) % Nx
            advection_term = c_adv * (u[ip1, n] - u[im1, n])
            diffusion_term = c_diff * (u[ip1, n] - 2*u[i, n] + u[im1, n])
            u[i, n+1] = u[i, n] - advection_term + diffusion_term
            
    return x, t, u


def plot_3d(x, t, u, title=None):
    # subsample only if you have HUGE arrays
    t_step = max(1, len(t) // 100)
    x_step = max(1, len(x) // 100)
    t_plot = t[::t_step]
    x_plot = x[::x_step]
    u_plot = u[::x_step, ::t_step]

    T_grid, X_grid = np.meshgrid(t_plot, x_plot)

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X_grid, T_grid, u_plot,
        cmap=cm.jet,       # ← the classic jet rainbow
        rstride=1,         # ← full resolution
        cstride=1,         # ← full resolution
        linewidth=0,       # ← no grid lines on the surface
        antialiased=True   # ← smooth shading
    )

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')    # ← just “u”
    if title:
        ax.set_title(title, pad=10)
    ax.view_init(elev=30, azim=-60)

    cb = fig.colorbar(surf, shrink=0.5, aspect=10)
    cb.set_label('u')     # optional: label your color-bar


# Caso I: Pe << 1 (Difusão Dominante). Igual o Da imagem!!

Pe1 = 1.5
Nx1 = 151 
x1, t1, u1 = solve_ftcs(Pe1, Nx1)
plot_3d(x1, t1, u1, f'Solução FTCS com Pe = {Pe1}')


plt.show()
