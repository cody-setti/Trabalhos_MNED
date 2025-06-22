"""
Trabalho – Advecção–Difusão com Diferenças Centradas

Equação: 
    u_t + u_x = Pe⁻¹ · u_xx

Método numérico (primeira parte do trabalho):
  - Euler Explícito no tempo
  - Diferenças centradas no termo advectivo
  - Diferenças centradas no termo difusivo
  - Condições de contorno PERIÓDICAS

Gera 6 snapshots em t = 0, T/5, 2T/5, 3T/5, 4T/5, T
para quatro valores de número de Péclet.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros do problema ---
L, T    = 15.0, 12.0           # domínio espacial [0,L] e tempo final T
nx      = 301                  # pontos em x
x       = np.linspace(0, L, nx)
dx      = x[1] - x[0]          # passo espacial
a       = 1.0                  # velocidade de advecção

def condicao_inicial(x):
    """u(x,0): soma de duas gaussianas em x=2 e x=5"""
    return np.exp(-20*(x-2)**2) + np.exp(-(x-5)**2)

def simulate_centered_periodic(Pe):
    """
    Simula u_t + u_x = Pe⁻¹ u_xx usando diferenças centradas.
    Com condições de contorno PERIÓDICAS.
    Retorna lista de 6 vetores u(x) nos tempos requisitados.
    """
    alpha = 1.0 / Pe  # coeficiente de difusão
    
    # Condições de estabilidade para diferenças centradas:
    # Para advecção central: dt ≤ dx²/(2*alpha) (mais restritiva que CFL padrão)
    # Para difusão: dt ≤ dx²/(2*alpha)
    # Como diferenças centradas para advecção são menos estáveis, usamos:
    dt_max = dx**2 / (2 * alpha + abs(a) * dx)
    
    # Usar fator de segurança menor para diferenças centradas
    dt = 0.5 * dt_max
    nt = int(np.ceil(T / dt))
    dt = T / nt  # ajusta dt para dividir exatamente T
    
    print(f"Pe = {Pe}: dt_max = {dt_max:.6f}, dt = {dt:.6f}, nt = {nt}")
    
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
            
            # Termo advectivo (diferenças centrais)
            # du/dx ≈ (u[i+1] - u[i-1])/(2*dx)
            adv_term = -a * (u[ip1] - u[im1]) / (2 * dx)
            
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
    try:
        snaps = simulate_centered_periodic(Pe)
        
        ax = axes[idx]
        colors = plt.cm.plasma(np.linspace(0, 1, 6))
        
        for i, (u_snap, label) in enumerate(zip(snaps, labels)):
            ax.plot(x, u_snap, label=label, color=colors[i], linewidth=1.5)
        
        ax.set_title(titulo, fontsize=12, fontweight='bold')
        ax.set_xlim(0, L)
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Instabilidade numérica\nPe = {Pe}', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        ax.set_title(f'{titulo} - INSTÁVEL', fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.suptitle('Equação de Advecção-Difusão: $u_t + u_x = Pe^{-1} u_{xx}$\n' + 
             'Método Diferenças Centrais (Condições de Contorno Periódicas)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.92)
plt.show()

print("\n" + "="*70)
print("OBSERVAÇÕES SOBRE DIFERENÇAS CENTRAIS:")
print("="*70)
print("- Para Pe altos (Pe >> 1), diferenças centrais na advecção podem ser instáveis")
print("- Oscillações numéricas podem aparecer quando Pe > 2 (critério de estabilidade)")
print("- O método upwind é mais robusto para problemas dominados por advecção") 