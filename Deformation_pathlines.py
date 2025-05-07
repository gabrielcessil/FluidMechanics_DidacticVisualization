import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

# =====================
# VELOCITY FIELD
# =====================
def velocity_field(x, y, t):
    u = -y
    v = 1
    return u, v

# =====================
# PATHLINE INTEGRATOR
# =====================
def pathline_euler(velocity_func, r1, t1, tf, dt):
    t = np.arange(t1, tf + dt, dt)
    path = np.zeros((len(t), 2))
    path[0] = r1
    for i in range(1, len(t)):
        u, v = velocity_func(*path[i-1], t[i-1])
        path[i] = path[i-1] + np.array([u, v]) * dt
    return path, t

# =====================
# INITIAL CONFIGURATION UTILITIES
# =====================
def generate_circle(center=(0, 0), radius=1.0, num_points=30):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x0, y0 = center
    x = x0 + radius * np.cos(theta)
    y = y0 + radius * np.sin(theta)
    return np.stack((x, y), axis=1)

def generate_square(center=(0, 0), side=1.0):
    x0, y0 = center
    h = side / 2
    return [
        (x0 - h, y0 + h),
        (x0 + h, y0 + h),
        (x0 + h, y0 - h),
        (x0 - h, y0 - h),
    ]

# =====================
# MAIN VISUALIZATION FUNCTION
# =====================
def simulate_pathlines(r1_list, make_bars, t1, tf, dt, total_duration_s, filename=""):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    N = len(r1_list)

    # Generate trajectories
    paths = []
    for r1 in r1_list:
        path, times = pathline_euler(velocity_field, r1, t1, tf, dt)
        paths.append(path)

    num_frames = len(paths[0])
    interval = (total_duration_s * 1000) / num_frames

    # Figure setup
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Particle Pathlines")
    ax.grid(True)

    # Static initial shape
    _ = [ax.plot(r[0], r[1], 'o', color='firebrick', markersize=4)[0] for r in r1_list]
    for i in range(N):
        r1, r2 = r1_list[i], r1_list[(i + 1) % N]
        ax.plot([r1[0], r2[0]], [r1[1], r2[1]], '--', color='firebrick', linewidth=1.5, alpha=0.6)

    # Initial velocity field
    X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    U, V = velocity_field(X, Y, 0)
    quiver = ax.quiver(X, Y, U, V, color='rosybrown', alpha=0.5)

    # Animated objects
    points = [ax.plot([], [], 'o', color='darkslateblue', markersize=5)[0] for _ in range(N)]
    line_collections = [LineCollection([], linewidths=2) for _ in range(N)]
    for lc in line_collections:
        ax.add_collection(lc)

    bars = []
    if make_bars:
        for _ in range(N):
            bar, = ax.plot([], [], '-', color='darkslateblue', linewidth=2)
            bars.append(bar)

    # Initialization function
    def init():
        for p in points:
            p.set_data([], [])
        for lc in line_collections:
            lc.set_segments([])
        for bar in bars:
            bar.set_data([], [])
        quiver.set_UVC(U, V)
        return points + line_collections + bars + [quiver]

    # Frame update function
    def update(frame):
        artists, current_positions = [], []
        base_rgb = to_rgb("mediumslateblue")

        for i in range(N):
            path = paths[i]
            pos = path[frame]
            points[i].set_data([pos[0]], [pos[1]])
            current_positions.append(pos)
            artists.append(points[i])

            if frame > 1:
                seg = np.array([path[j:j+2] for j in range(frame-1)])
                alpha = np.linspace(0.0, 1.0, len(seg))
                colors = np.array([[*base_rgb, a] for a in alpha])
                line_collections[i].set_segments(seg)
                line_collections[i].set_color(colors)
            else:
                line_collections[i].set_segments([])
            artists.append(line_collections[i])

        if make_bars:
            for i in range(N):
                p1, p2 = current_positions[i], current_positions[(i + 1) % N]
                bars[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                artists.append(bars[i])

        all_positions = np.vstack([r1_list, current_positions])
        xmin, ymin = np.min(all_positions, axis=0) - 0.2
        xmax, ymax = np.max(all_positions, axis=0) + 0.2
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        x_vals = np.linspace(xmin, xmax, 20)
        y_vals = np.linspace(ymin, ymax, 20)
        X, Y = np.meshgrid(x_vals, y_vals)
        vel_func = np.vectorize(lambda x, y: velocity_field(x, y, times[frame]))
        U, V = vel_func(X, Y)
        quiver.set_offsets(np.c_[X.ravel(), Y.ravel()])
        quiver.set_UVC(U.ravel(), V.ravel())
        artists.append(quiver)

        return artists

    # Animation
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=interval, blit=True)
    fps = num_frames / total_duration_s
    anim.save(filename+".gif", writer=PillowWriter(fps=fps), dpi=100)
    plt.legend(["Particles"])
    plt.show()

# =====================
# Example usage
# =====================
if __name__ == "__main__":
    
    r1_list = generate_circle() 
    r2_list = generate_square()
    
    
    simulate_pathlines(
        r1_list=r1_list,
        make_bars=True,
        t1=0.0,
        tf=2.0,
        dt=0.005,
        total_duration_s=8,
        filename="circle"
    )
    
    simulate_pathlines(
        r1_list=r2_list,
        make_bars=True,
        t1=0.0,
        tf=2.0,
        dt=0.005,
        total_duration_s=8,
        filename="square"
    )
