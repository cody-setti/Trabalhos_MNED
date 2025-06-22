"""
Trabalho – Advecção–Difusão Explícita (Upwind + Difusão Central)

Equação: 
    u_t + u_x = Pe⁻¹ · u_xx

Método numérico:
  - Euler Explícito no tempo
  - Upwind de 1ª ordem no termo advectivo (a = 1)
  - Diferenças centradas 2ª ordem no termo difusivo
  - Condições de contorno PERIÓDICAS

Gera 6 snapshots em t = 0, T/5, 2T/5, 3T/5, 4T/5, T
para quatro valores de número de Péclet.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros do problema ---
L, T    = 15.0, 12.0           # domínio espacial [0,L] e tempo final T
nx      = 301                  # pontos em x (malha mais fina, mas computacionalmente viável)
x       = np.linspace(0, L, nx)
dx      = x[1] - x[0]          # passo espacial
a       = 1.0                  # velocidade de advecção

def condicao_inicial(x):
    """u(x,0): soma de duas gaussianas em x=2 e x=5"""
    return np.exp(-20*(x-2)**2) + np.exp(-(x-5)**2)

def simulate_upwind_periodic(Pe):
    """
    Simula u_t + u_x = Pe⁻¹ u_xx usando upwind + difusão central.
    Com condições de contorno PERIÓDICAS.
    Retorna lista de 6 vetores u(x) nos tempos requisitados.
    """
    alpha = 1.0 / Pe  # coeficiente de difusão
    
    # Condições de estabilidade:
    # CFL para advecção: dt ≤ dx/|a|
    # Difusão: dt ≤ dx²/(2*alpha)
    dt_cfl = dx / abs(a)
    dt_diff = dx**2 / (2 * alpha)
    dt_max = min(dt_cfl, dt_diff)
    
    # Usar 80% do limite para segurança
    dt = 0.8 * dt_max
    nt = int(np.ceil(T / dt))
    dt = T / nt  # ajusta dt para dividir exatamente T
    
    print(f"Pe = {Pe}: dt_cfl = {dt_cfl:.6f}, dt_diff = {dt_diff:.6f}, dt = {dt:.6f}, nt = {nt}")
    
    # índices onde salvar snapshots: t = i·nt/5, i=0..5
    snap_inds = [int(i * nt / 5) for i in range(6)]
    
    # inicialização
    u = condicao_inicial(x)
    snaps = [u.copy()]
    
    # loop temporal
    for n in range(1, nt + 1):
        u_new = u.copy()
        
        # Aplicar esquema no interior e nas bordas com periodicidade
        for i in range(nx):
            # índices com periodicidade
            im1 = (i - 1) % nx  # i-1 com periodicidade
            ip1 = (i + 1) % nx  # i+1 com periodicidade
            
            # Termo advectivo (upwind para a > 0)
            # du/dx ≈ (u[i] - u[i-1])/dx
            adv_term = -a * (u[i] - u[im1]) / dx
            
            # Termo difusivo (diferenças centrais)
            # d²u/dx² ≈ (u[i+1] - 2u[i] + u[i-1])/dx²
            diff_term = alpha * (u[ip1] - 2*u[i] + u[im1]) / dx**2
            
            # Atualização temporal (Euler explícito)
            u_new[i] = u[i] + dt * (adv_term + diff_term)
        
        u = u_new
        
        # salvar snapshot
        if n in snap_inds:
            snaps.append(u.copy())
    
    return snaps

# --- Péclet e títulos para cada subplot ---
Pes     = [0.1, 1.0, 10.0, 100.0]
titulos = ['Pe ≪ 1 (Pe = 0.1)', 'Pe = 1', 'Pe na ordem de dezenas (Pe = 10)', 'Pe na ordem de centenas (Pe = 100)']
labels  = ['t = 0', 't = T/5', 't = 2T/5', 't = 3T/5', 't = 4T/5', 't = T']

# --- Plot 2x2 com todos os casos ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (Pe, titulo) in enumerate(zip(Pes, titulos)):
    print(f"\nSimulando Pe = {Pe}...")
    snaps = simulate_upwind_periodic(Pe)
    
    ax = axes[idx]
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    for i, (u_snap, label) in enumerate(zip(snaps, labels)):
        ax.plot(x, u_snap, label=label, color=colors[i], linewidth=1.5)
    
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlim(0, L)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('u', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.suptitle('Equação de Advecção-Difusão: $u_t + u_x = Pe^{-1} u_{xx}$\n' + 
             'Método Upwind + Diferenças Centrais (Condições de Contorno Periódicas)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.92)
plt.show()

# --- Análise das condições de estabilidade ---
print("\n" + "="*60)
print("ANÁLISE DAS CONDIÇÕES DE ESTABILIDADE")
print("="*60)

dx_val = dx
Pe_range = np.logspace(-2, 3, 100)  # Pe de 0.01 a 1000

dt_cfl = dx_val  # condição CFL: dt ≤ dx/a (a=1)
dt_diff = Pe_range * dx_val**2 / 2  # condição difusão: dt ≤ Pe*dx²/2

plt.figure(figsize=(10, 6))
plt.loglog(Pe_range, dt_cfl * np.ones_like(Pe_range), 'r-', linewidth=2, label='CFL: Δt ≤ Δx')
plt.loglog(Pe_range, dt_diff, 'b-', linewidth=2, label='Difusão: Δt ≤ Pe·Δx²/2')
plt.loglog(Pe_range, np.minimum(dt_cfl, dt_diff), 'g--', linewidth=2, label='min(CFL, Difusão)')

# Marcar os valores de Pe usados
for Pe in Pes:
    dt_limit = min(dt_cfl, Pe * dx_val**2 / 2)
    plt.loglog(Pe, dt_limit, 'ko', markersize=8)
    plt.annotate(f'Pe = {Pe}', (Pe, dt_limit), xytext=(10, 10), 
                textcoords='offset points', fontsize=9)

plt.xlabel('Número de Péclet (Pe)', fontsize=12)
plt.ylabel('Δt máximo permitido', fontsize=12)
plt.title('Condições de Estabilidade vs Número de Péclet', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0.01, 1000)
plt.show()

# Identificar ponto de transição
Pe_transicao = 2 / dx_val  # onde dt_cfl = dt_diff
print(f"\nPonto de transição: Pe ≈ {Pe_transicao:.2f}")
print(f"Para Pe < {Pe_transicao:.2f}: limitado pela difusão")
print(f"Para Pe > {Pe_transicao:.2f}: limitado pela advecção (CFL)") 