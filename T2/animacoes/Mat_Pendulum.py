import matplotlib
matplotlib.use('Qt5Agg')  # set backend before importing pyplot

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Lib import pendulum_system, rk4_step, compute_trajectory, RefSolution, euler_explicit, rk4, euler_implicit

def main():
    # parameters
    L = 1.0           # pendulum length (m)
    θ0 = math.radians(45)
    ω0 = 0.0
    dt = 0.02         # time step (s)
    t_max = 10        # total time (s)

    # compute trajectory
    traj = compute_trajectory(θ0, ω0, t_max, dt)
    θs = traj[:,0]

    # set up figure
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-L*1.1, L*1.1)
    ax.set_ylim(-L*1.1, L*0.1)
    ax.set_aspect('equal')
    # Colors for each method
    method_colors = {
        'Euler Explícito': 'tab:blue',
        'RK4': 'tab:green',
        'Euler Implícito': 'tab:red',
        'Analítica': 'tab:orange',
    }
    # Bob size
    bob_size = 14

    # Plot objects for each method
    lines = {}
    bobs = {}
    trails = {}
    for method, color in method_colors.items():
        lines[method], = ax.plot([], [], lw=2, ls='--' if method=='Analítica' else '-', color=color, label=method)
        bobs[method], = ax.plot([], [], 'o', color=color, ms=bob_size, alpha=0.7 if method!='Analítica' else 0.5)
        trails[method], = ax.plot([], [], color=color, lw=2, alpha=0.5)

    def init():
        for method in method_colors:
            lines[method].set_data([], [])
            bobs[method].set_data([], [])
            trails[method].set_data([], [])
        return tuple(lines.values()) + tuple(bobs.values()) + tuple(trails.values())

    def update(i):
        t_vals = np.arange(0, t_max, dt)
        Ntrail = 50
        istart = max(0, i-Ntrail)

        # Compute all methods' positions
        # Euler Explicit
        _, U_euler = euler_explicit(pendulum_system, [θ0, ω0], [0, t_max], dt)
        θs_euler = U_euler[:,0]
        # RK4
        _, U_rk4 = rk4(pendulum_system, [θ0, ω0], [0, t_max], dt)
        θs_rk4 = U_rk4[:,0]
        # Euler Implicit
        _, U_impl = euler_implicit(pendulum_system, [θ0, ω0], [0, t_max], dt)
        θs_impl = U_impl[:,0]
        # Analytic
        θs_exact, _ = RefSolution(θ0, t_vals)

        method_data = {
            'Euler Explícito': θs_euler,
            'RK4': θs_rk4,
            'Euler Implícito': θs_impl,
            'Analítica': θs_exact,
        }

        artists = []
        for method, θarr in method_data.items():
            θ = θarr[i]
            x = L * math.sin(θ)
            y = -L * math.cos(θ)
            lines[method].set_data([0, x], [0, y])
            bobs[method].set_data([x], [y])
            xs_trail = L * np.sin(θarr[istart:i+1])
            ys_trail = -L * np.cos(θarr[istart:i+1])
            trails[method].set_data(xs_trail, ys_trail)
            trails[method].set_alpha(0.2 + 0.5 * (i-istart)/Ntrail)
            artists += [lines[method], bobs[method], trails[method]]
        return tuple(artists)

    frames = len(traj)
    global ani
    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True,
                        interval=dt*1000, repeat=True)
    plt.title("Simple Pendulum: Todos os Métodos")
    ax.legend(loc='upper right', fontsize=10)
    plt.show(block=True)

if __name__ == "__main__":
    main()