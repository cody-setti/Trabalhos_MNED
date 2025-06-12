import pygame
import numpy as np
import math
import sys
from Lib import pendulum_system, rk4_step, compute_trajectory

# -- Pygame animation --

def main():
    # parameters
    L = 200          # pendulum length in px
    origin = (400, 100)
    θ0 = math.radians(45)
    ω0 = 0.0
    dt = 0.02        # time step
    t_max = 20       # total simulation time

    traj = compute_trajectory(θ0, ω0, t_max, dt)

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    i = 0
    total = traj.shape[0]

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((30, 30, 30))
        θ, _ = traj[i]
        # bob position
        x = origin[0] + L * math.sin(θ)
        y = origin[1] + L * math.cos(θ)

        # draw rod
        pygame.draw.line(screen, (200,200,200), origin, (x,y), 2)
        # draw bob
        pygame.draw.circle(screen, (200,50,50), (int(x), int(y)), 16)

        pygame.display.flip()
        clock.tick(1/dt)   # match simulation dt to framerate

        i = (i+1) % total

if __name__ == "__main__":
    main()