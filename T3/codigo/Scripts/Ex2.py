"""
Trabalho 3 - Exercício 2
Equação de Advecção-Difusão com Esquema Upwind

Este script implementa a discretização da equação unidimensional de advecção-difusão:
u_t + u_x = (1/Pe) * u_xx

Utilizando:
- Diferenças progressivas no tempo
- Diferenças centradas no espaço para derivadas de segunda ordem (Pe^-1 * u_xx)
- Estratégia Upwind de primeira ordem para o termo advectivo (u_x)

Domínio:
- Espaço: 0 ≤ x ≤ 15
- Tempo: 0 ≤ t ≤ 12
- Condição inicial: u(x,0) = exp(-20(x-2)²) + exp(-(x-5)²)
- Condições de contorno periódicas
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Parâmetros do Problema
L = 15.0  # Comprimento do domínio
T = 12.0  # Tempo total de simulação

def initial_condition(x):
    """Condição inicial: duas gaussianas"""
    return np.exp(-20 * (x - 2)**2) + np.exp(-(x - 5)**2)

def solve_upwind(Pe, Nx):
    """
    Resolve a equação de advecção-difusão usando esquema Upwind para advecção
    e diferenças centradas para difusão.
    
    Parâmetros:
    - Pe: Número de Péclet
    - Nx: Número de pontos na malha espacial
    
    Retorna:
    - x: malha espacial
    - t: malha temporal
    - u: solução numérica
    """
    
    # Malha espacial
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    kappa = 1.0 / Pe  # Coeficiente de difusão
    
    # Número de Péclet da malha
    cell_Pe = Pe * dx
    print(f"--- Iniciando Upwind com Pe = {Pe} ---")
    print(f"Número de Péclet da malha (Pe*dx): {cell_Pe:.2f}")
    
    # Passo temporal baseado em critérios de estabilidade
    # Para Upwind: CFL ≤ 1 e critério de difusão
    dt_cfl = dx  # Critério CFL para Upwind (velocidade = 1)
    dt_diff = dx**2 / (2 * kappa + 1e-9)  # Critério de difusão
    dt = 0.4 * min(dt_cfl, dt_diff)
    
    Nt = int(T / dt) + 1
    t = np.linspace(0, T, Nt)
    dt = T / (Nt - 1)  # Reajuste para tempo exato
    
    print(f"Nx = {Nx}, Nt = {Nt}, dx = {dx:.4f}, dt = {dt:.4f}")
    print(f"CFL = {dt/dx:.3f}, Número de difusão = {dt*kappa/dx**2:.3f}")
    
    # Inicialização da matriz solução
    u = np.zeros((Nx, Nt))
    u[:, 0] = initial_condition(x)
    
    # Coeficientes do esquema
    c_adv = dt / dx       # Para termo advectivo (Upwind)
    c_diff = dt * kappa / dx**2  # Para termo difusivo (centradas)
    
    # Loop temporal
    for n in range(0, Nt - 1):
        for i in range(0, Nx):
            # Índices com condições periódicas
            ip1 = (i + 1) % Nx  # i+1 (próximo)
            im1 = (i - 1 + Nx) % Nx  # i-1 (anterior)
            
            # Termo advectivo usando Upwind (primeira ordem)
            # Como a velocidade é +1 (positiva), usamos diferenças para trás
            advection_term = c_adv * (u[i, n] - u[im1, n])
            
            # Termo difusivo usando diferenças centradas (segunda ordem)
            diffusion_term = c_diff * (u[ip1, n] - 2*u[i, n] + u[im1, n])
            
            # Atualização temporal (Euler explícito)
            u[i, n+1] = u[i, n] - advection_term + diffusion_term
    
    return x, t, u

def solve_ftcs(Pe, Nx):
    """
    Resolve usando o esquema FTCS original (centradas para advecção e difusão)
    para comparação.
    """
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)
    kappa = 1.0 / Pe

    dt = 0.4 * min(dx, dx**2 / (2 * kappa + 1e-9))
    Nt = int(T / dt) + 1
    t = np.linspace(0, T, Nt)
    dt = T / (Nt - 1) 

    print(f"--- FTCS para comparação com Pe = {Pe} ---")
    print(f"Nx = {Nx}, Nt = {Nt}, dx = {dx:.4f}, dt = {dt:.4f}")

    u = np.zeros((Nx, Nt))
    u[:, 0] = initial_condition(x)
    
    c_adv = dt / (2 * dx)    # Diferenças centradas para advecção
    c_diff = dt * kappa / dx**2
    
    for n in range(0, Nt - 1):
        for i in range(0, Nx):
            ip1 = (i + 1) % Nx
            im1 = (i - 1 + Nx) % Nx
            advection_term = c_adv * (u[ip1, n] - u[im1, n])
            diffusion_term = c_diff * (u[ip1, n] - 2*u[i, n] + u[im1, n])
            u[i, n+1] = u[i, n] - advection_term + diffusion_term
            
    return x, t, u

def plot_comparison_3d(results, titles, save_path="upwind_comparison.png"):
    """Plota múltiplas soluções lado a lado em 3D"""
    num_plots = len(results)
    fig = plt.figure(figsize=(6 * num_plots, 6))
    
    for idx, ((x, t, u), title) in enumerate(zip(results, titles)):
        # Subamostragem para visualização mais rápida
        t_step = max(1, len(t) // 100)
        x_step = max(1, len(x) // 100)
        t_plot = t[::t_step]
        x_plot = x[::x_step]
        u_plot = u[::x_step, ::t_step]
        
        # Criação das malhas para plot
        T_grid, X_grid = np.meshgrid(t_plot, x_plot)
        
        # Subplot 3D
        ax = fig.add_subplot(1, num_plots, idx + 1, projection='3d')
        surf = ax.plot_surface(
            X_grid, T_grid, u_plot,
            cmap=cm.jet,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True
        )
        
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        ax.set_title(title, pad=10)
        ax.view_init(elev=30, azim=-60)
        
        # Barra de cores
        cb = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
        cb.set_label('u')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo como '{save_path}'")
    plt.show()

def plot_comparison_2d(results, titles, times_to_plot=[0, 3, 6, 9, 12]):
    """Plota comparação em 2D para tempos específicos"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for time_idx, t_target in enumerate(times_to_plot):
        ax = axes[time_idx]
        
        for (x, t, u), title in zip(results, titles):
            # Encontrar o índice de tempo mais próximo
            t_idx = np.argmin(np.abs(t - t_target))
            actual_time = t[t_idx]
            
            ax.plot(x, u[:, t_idx], label=f'{title}', linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f't = {actual_time:.1f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Remove o último subplot se não usado
    if len(times_to_plot) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_overlay_2d(results, titles, base_time=0.5, num_times=5):
    """Plota múltiplos tempos sobrepostos em um único gráfico 2D"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Gerar tempos: base_time, base_time*2, base_time*3, etc.
    times_to_plot = [base_time * (i + 1) for i in range(num_times)]
    
    # Cores para diferentes tempos
    colors = plt.cm.viridis(np.linspace(0, 1, num_times))
    
    for result_idx, ((x, t, u), title) in enumerate(zip(results, titles)):
        for time_idx, t_target in enumerate(times_to_plot):
            # Encontrar o índice de tempo mais próximo
            t_idx = np.argmin(np.abs(t - t_target))
            actual_time = t[t_idx]
            
            # Estilo de linha diferente para cada método
            linestyle = '-' if result_idx == 0 else '--'
            linewidth = 2.5 if result_idx == 0 else 2
            
            # Plot com cores diferentes para cada tempo
            ax.plot(x, u[:, t_idx], 
                   color=colors[time_idx], 
                   linestyle=linestyle,
                   linewidth=linewidth,
                   label=f'{title} - t = {actual_time:.1f}',
                   alpha=0.8)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u', fontsize=12)
    ax.set_title('Evolução Temporal Sobreposta', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def main():
    """Função principal para executar as simulações e comparações"""
    print("="*60)
    print("ANÁLISE COMPARATIVA: ESQUEMA UPWIND vs FTCS")
    print("="*60)
    
    # Parâmetros para diferentes casos de Péclet
    Pe_values = [10, 1, 0.1]  # Pe >> 1, Pe = 1, Pe << 1
    Nx = 151  # Resolução espacial
    
    # Executar simulações com Upwind
    print("\n*** SIMULAÇÕES COM ESQUEMA UPWIND ***")
    upwind_results = []
    for Pe in Pe_values:
        result = solve_upwind(Pe, Nx)
        upwind_results.append(result)
        print()
    
    # Executar simulações com FTCS para comparação
    print("\n*** SIMULAÇÕES COM ESQUEMA FTCS (para comparação) ***")
    ftcs_results = []
    for Pe in Pe_values:
        result = solve_ftcs(Pe, Nx)
        ftcs_results.append(result)
        print()
    
    # Títulos para os gráficos
    upwind_titles = [f'Upwind - Pe = {Pe}' for Pe in Pe_values]
    ftcs_titles = [f'FTCS - Pe = {Pe}' for Pe in Pe_values]
    
    # Plotar resultados Upwind
    print("\n*** VISUALIZAÇÃO DOS RESULTADOS ***")
    print("Plotando resultados com esquema Upwind...")
    plot_comparison_3d(upwind_results, upwind_titles, "upwind_results.png")
    
    # Plotar comparação entre esquemas para Pe = 10 (advecção dominante)
    print("Plotando comparação Upwind vs FTCS para Pe = 10...")
    comparison_results = [upwind_results[0], ftcs_results[0]]
    comparison_titles = ['Upwind - Pe = 10', 'FTCS - Pe = 10']
    plot_comparison_3d(comparison_results, comparison_titles, "upwind_vs_ftcs_pe10.png")
    
    # Plotar comparação 2D em tempos específicos
    print("Plotando evolução temporal em 2D...")
    plot_comparison_2d(comparison_results, comparison_titles)
    
    # Plotar evolução temporal sobreposta
    print("Plotando evolução temporal sobreposta (t=0.5, 1.0, 1.5, 2.0, 2.5)...")
    plot_overlay_2d(comparison_results, comparison_titles, base_time=0.5, num_times=5)
    
    print("\n*** ANÁLISE CONCLUÍDA ***")
    print("Arquivos gerados:")
    print("- upwind_results.png")
    print("- upwind_vs_ftcs_pe10.png")

if __name__ == "__main__":
    main() 