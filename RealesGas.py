# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:23:22 2021

@author: bauer
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Particle():
    
    def __init__(self,id = 0, r = np.zeros(2), v = np.zeros(2), R = 1e-2, m = 1 , color = 'blue'):
        "r = Ortsvektor, v = Geschwindigkeitsvektor, R = Radius , m = Masse, color = Farbe"
        self.id = id
        self.r = r
        self.v = v
        self.R = R
        self.m = m 
        self.color = color
        
        
        
        
class Sim():
    
    X = 2
    Y = 2
    
    def __init__(self, dt = 10e-6, Np = 10):
        "dt = Zeitschritt, Np = Anzahl Partikel"
        
        self.dt = dt
        self.Np = Np
        
        # erstelle anzahl der Partikel
        self.particles = [Particle(id = i) for i in range(self.Np)]
        
    def particle_speeds(self):
        return[np.sqrt(np.dot(particle.v, particle.v)) for particle in self.particles]
        
    def particle_colors(self):
        return [particle.color for particle in self.particles]
    
    def collision_detection(self):
        ignore_list = list()
        for particle1 in self.particles:
            if particle1 in ignore_list:
                continue 
            x,y = particle1.r

            # collision detection for interaction from the particle with the walls 
            if (x > self.X/2 - particle1.R) or (x < - self.X/2 + particle1.R):
                particle1.v[0] *= -1
            if (y > self.Y/2 - particle1.R) or (y < - self.Y/2 + particle1.R):
                particle1.v[1] *= -1


            # collision detection for interaction from particles with each other
            for particle2 in self.particles:                
                if particle1.id == particle2.id:
                    continue
                m1, m2, r1, r2, v1, v2 = particle1.m, particle2.m, particle1.r, particle2.r, particle1.v, particle2.v
                
                if np.dot(r1-r2, r1-r2) <= (particle1.R + particle2.R)**2 + 5e-4:
                    v1_new = v1 - 2 * m1 / (m1+m2) * np.dot(v1 - v2, r1 - r2) / np.dot(r1-r2, r1-r2) * (r1-r2)
                    v2_new = v2 - 2 * m1 / (m1+m2) * np.dot(v2 - v1, r2 - r1) / np.dot(r2-r1, r2-r1) * (r2-r1)
                    particle1.v = v1_new
                    particle2.v = v2_new
                    ignore_list.append(particle2)


    def increment(self):
        self.collision_detection()
        
        for particle in self.particles:
            particle.r += self.dt * particle.v
            
    
    def particle_positions(self):
        return [particle.r for particle in self.particles]
    
    
    def E_avg(self):
        E_avg = 0 
        for particle in self.particles:
            E_avg += 0.5 * particle.m * np.dot(particle.v, particle.v)
        return E_avg/len(self.particles)
    
    def temperature(self):
        return self.E_avg() * (2/3) / kB
    
# sim Variables
Np = 150
m = 127*1.66e-27
T_init = 293.15


kB = 1.38064852e-23
v_avg = np.sqrt(3/2 * kB *2* T_init * 1/m)
n_avg = 50

sim = Sim(Np = Np) # erstelle Instanz der Simulation

for particle in sim.particles:
    particle.m = m
    particle.r = np.random.uniform([-sim.X/2 +0.1, - sim.Y/2 +0.1], [sim.X/2 - 0.1, sim.Y/2 - 0.2] )
    particle.v = v_avg* 0.92 * np.array([np.pi/4, np.pi/4])
    # particle.v = 1* np.array([np.cos(np.pi / 4), np.cos(np.pi / 4)])
    
    
# sim.particles[0].v = 2
# sim.particles[0].m = 3
# sim.particles[0].R = 8e-2
   
# plot code
fig, (ax, ax2) = plt.subplots(figsize = (5,9), nrows = 2)
ax.set_xticks([]), ax.set_yticks([])
ax.set_aspect('equal')

vs = np.arange(0,500, 25)

scatter = ax.scatter([],[])
bar  = ax2.bar(vs, [0]*len(vs), width = 0.9 * np.gradient(vs), align = 'edge', alpha = 0.8)

theo= ax2.plot(vs, 25* Np * (m /(2*np.pi * kB * sim.temperature()))**(3/2) *4 * np.pi * vs**2 * np.exp(-m * vs**2 / (2 * kB * sim.temperature())), color = 'orange')

T_text = ax.text(sim.X/2 * 0.5 , sim.Y/2 *0.92, s ="")

freqs_matrix = np.tile((np.histogram(sim.particle_speeds(), bins = vs)[0].astype(np.float64)), (n_avg, 1))


sim.particles[0].color = 'red'
def init():
    ax.set_xlim(-sim.X/2, sim.X/2)
    ax.set_ylim(-sim.Y/2, sim.Y/2)
    
    ax2.set_xlim(vs[0], vs[-1])
    ax2.set_ylim(0,Np)
    ax2.set(xlabel = "Particle Speed (m/s)", ylabel = "# of particles")
    return (scatter , *bar.patches)


def update(frame):
    sim.increment()
    
    T_text.set_text(f"{sim.temperature():.2f} K = {sim.temperature()-273.2:.2f}°C")
    
    freqs, bins = np.histogram(sim.particle_speeds(), bins = vs)
    freqs_matrix[frame%n_avg] = freqs
    freqs_mean = np.mean(freqs_matrix, axis = 0)
    freqs_max = np.max(freqs_mean)
    
    for rect, height in zip(bar.patches, freqs):
        rect.set_height(height)
    
    if np.abs(freqs_max - ax2.get_ylim()[1]) > 10:
        ax2.set_ylim(0,5 + ax2.get_ylim()[1]+ (freqs_max - ax2.get_ylim()[1]))
        fig.canvas.draw()
    scatter.set_offsets(np.array(sim.particle_positions()))
    scatter.set_color(sim.particle_colors())
    return (scatter , *bar.patches, T_text)




ani = FuncAnimation(fig, update, frames = range(24000), init_func = init, blit = True, interval = 1/30, repeat = False)
plt.show()










