"""
Análise de Estabilidade - Trabalho 3
Equação de Advecção-Difusão: u_t + u_x = Pe⁻¹ u_xx

Análise das condições de estabilidade para diferentes valores de Pe
conforme solicitado no item 3 do trabalho.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
L = 15.0
dx = L / 300  # espaçamento espacial típico
a = 1.0       # velocidade de advecção

# Range de números de Péclet
Pe_values = np.logspace(-2, 3, 1000)  # Pe de 0.01 a 1000

# Condições de estabilidade
def stability_conditions(Pe, dx, a=1.0):
    """
    Calcula as diferentes condições de estabilidade
    """
    alpha = 1.0 / Pe  # coeficiente de difusão
    
    # Condição CFL para advecção (upwind)
    dt_cfl = dx / abs(a)
    
    # Condição de difusão
    dt_diff = dx**2 / (2 * alpha)
    
    # Condição combinada (mais restritiva)
    dt_combined = min(dt_cfl, dt_diff)
    
    return dt_cfl, dt_diff, dt_combined

# Calcular condições para todos os Pe
dt_cfl_vals = []
dt_diff_vals = []
dt_combined_vals = []

for Pe in Pe_values:
    dt_cfl, dt_diff, dt_combined = stability_conditions(Pe, dx, a)
    dt_cfl_vals.append(dt_cfl)
    dt_diff_vals.append(dt_diff)
    dt_combined_vals.append(dt_combined)

dt_cfl_vals = np.array(dt_cfl_vals)
dt_diff_vals = np.array(dt_diff_vals)
dt_combined_vals = np.array(dt_combined_vals)

# Ponto de transição onde as duas condições se igualam
# dt_cfl = dt_diff
# dx/a = dx²/(2*alpha) = Pe*dx²/2
# dx/a = Pe*dx²/2
# 1/a = Pe*dx/2
# Pe_transition = 2*a/dx = 2/dx (para a=1)
Pe_transition = 2 / dx

print(f"Espaçamento espacial: dx = {dx:.6f}")
print(f"Ponto de transição: Pe_transition = {Pe_transition:.2f}")

# Plotar análise de estabilidade
plt.figure(figsize=(12, 8))

plt.loglog(Pe_values, dt_cfl_vals, 'r-', linewidth=2, 
           label=f'CFL (Advecção): Δt ≤ Δx/|a| = {dx:.4f}')
plt.loglog(Pe_values, dt_diff_vals, 'b-', linewidth=2, 
           label='Difusão: Δt ≤ Pe·Δx²/2')
plt.loglog(Pe_values, dt_combined_vals, 'g--', linewidth=3, 
           label='Limitação efetiva: min(CFL, Difusão)')

# Marcar ponto de transição
plt.axvline(Pe_transition, color='k', linestyle=':', alpha=0.7, linewidth=2)
plt.text(Pe_transition*1.5, 1e-4, f'Pe = {Pe_transition:.1f}\n(transição)', 
         fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Marcar valores específicos do trabalho
Pe_trabalho = [0.1, 1.0, 10.0, 100.0]
for Pe in Pe_trabalho:
    dt_limit = min(dx/a, Pe*dx**2/2)
    plt.loglog(Pe, dt_limit, 'ko', markersize=8)
    plt.annotate(f'Pe = {Pe}', (Pe, dt_limit), 
                xytext=(10, 10), textcoords='offset points', 
                fontsize=10, ha='left')

plt.xlabel('Número de Péclet (Pe)', fontsize=14)
plt.ylabel('Δt máximo permitido', fontsize=14)
plt.title('Condições de Estabilidade vs Número de Péclet\n' + 
          f'(Δx = {dx:.4f})', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='lower left')
plt.xlim(0.01, 1000)
plt.ylim(1e-6, 1e-1)

# Adicionar regiões de dominância
plt.fill_between([0.01, Pe_transition], [plt.ylim()[0], plt.ylim()[0]], 
                [plt.ylim()[1], plt.ylim()[1]], alpha=0.2, color='blue', 
                label='')
plt.fill_between([Pe_transition, 1000], [plt.ylim()[0], plt.ylim()[0]], 
                [plt.ylim()[1], plt.ylim()[1]], alpha=0.2, color='red', 
                label='')

plt.text(0.3, 3e-3, 'Regime dominado\npela DIFUSÃO\n(Pe < {:.1f})'.format(Pe_transition), 
         fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
plt.text(50, 3e-3, 'Regime dominado\npela ADVECÇÃO\n(Pe > {:.1f})'.format(Pe_transition), 
         fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))

plt.tight_layout()
plt.show()

# Análise detalhada para os valores do trabalho
print("\n" + "="*80)
print("ANÁLISE DETALHADA PARA OS VALORES DO TRABALHO")
print("="*80)

for Pe in Pe_trabalho:
    dt_cfl, dt_diff, dt_combined = stability_conditions(Pe, dx, a)
    
    if dt_cfl < dt_diff:
        limitacao = "CFL (Advecção)"
        razao = dt_diff / dt_cfl
    else:
        limitacao = "Difusão"
        razao = dt_cfl / dt_diff
    
    print(f"\nPe = {Pe:6.1f}:")
    print(f"  • Condição CFL:     Δt ≤ {dt_cfl:.6f}")
    print(f"  • Condição Difusão: Δt ≤ {dt_diff:.6f}")
    print(f"  • Limitação:        {limitacao}")
    print(f"  • Δt efetivo:       {dt_combined:.6f}")
    print(f"  • Razão:            {razao:.2f}x mais restritiva")

print(f"\n" + "="*80)
print("IMPLICAÇÕES PARA O PLANEJAMENTO DE MÉTODOS NUMÉRICOS:")
print("="*80)
print(f"1. Para Pe < {Pe_transition:.1f} (regime difusivo):")
print("   - Limitação principal: condição de difusão")
print("   - Δt ∝ Pe (aumenta linearmente com Pe)")
print("   - Métodos implícitos são vantajosos")

print(f"\n2. Para Pe > {Pe_transition:.1f} (regime advectivo):")
print("   - Limitação principal: condição CFL")
print("   - Δt constante = Δx/|a|")
print("   - Métodos upwind são necessários para estabilidade")
print("   - Diferenças centrais na advecção causam oscilações")

print(f"\n3. Ponto de transição Pe = {Pe_transition:.1f}:")
print("   - Ambas as restrições são igualmente importantes")
print("   - Mudança de regime físico: difusão → advecção")

print("\n4. Estratégias de solução:")
print("   - Pe baixo: usar métodos implícitos (CN, backward Euler)")
print("   - Pe alto: usar upwind explícito ou métodos TVD")
print("   - Pe muito alto: considerar métodos de alta ordem espacial") 