# 2D Particle Pathline Simulation in a Velocity Field

This repository contains a Python-based simulation platform for visualizing **Lagrangian particle pathlines** in a two-dimensional velocity field. The system computes and animates particle trajectories over time, allowing for qualitative analysis of flow behavior. This tool was developed within the scope of academic studies in fluid mechanics.


<p align="center">
  <img src="examples/square.gif" alt="Example of a square" width="45%" style="margin-right: 5px;" />
  <img src="examples/circle.gif" alt="Example of a circle" width="45%" />
</p>

---

## Overview

The simulation computes the motion of particles subjected to a time-dependent velocity field, using Euler integration. The resulting pathlines are visualized dynamically, with the option to display connecting bars and a fading trail history. The initial configuration of particles can be defined as either circular or square arrangements.

---

## Key Features

- Customizable velocity field function.
- Choice between circular or square particle initialization.
- Euler-based trajectory integration.
- Fading trails to illustrate path history.
- Optional connecting bars between particles.
- Live velocity field vector visualization.
- High-resolution `.gif` export for documentation or presentation purposes.

---

## Parameters

The user can easily control the simulation by providing the following arguments:

| Parameter            | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `r1_list`            | Initial list of particle positions (`(x, y)` tuples).         |
| `make_bars`          | Boolean flag to draw connecting bars between particles.       |
| `t1`                 | Start time of the simulation.                                 |
| `tf`                 | End time of the simulation.                                   |
| `dt`                 | Time step for numerical integration.                          |
| `total_duration_s`   | Duration (in seconds) of the final `.gif` animation.          |

---

## Velocity Field

The default velocity field is defined as:

```python
def velocity_field(x, y, t):
    u = -y
    v = 1
    return u, v
```

This defines a linear rotational field with constant vertical advection. The user may replace this function to simulate different flow behaviors.

---

## Initial Configuration

Two helper functions are provided for initializing particle positions:

```python
generate_circle(center=(0, 0), radius=1.0, num_points=30)
generate_square(center=(0, 0), side=1.0)
```

---

## Output

- The resulting animation is saved as a `.gif` file named:
  ```
  pathlines_with_bars.gif
  ```
- Resolution: 1080×1080 pixels (modifiable via `dpi`).
- Frame rate automatically adjusted based on total duration.

---

## Dependencies

This project uses the following libraries:

- `numpy`
- `matplotlib`

Install them with:

```bash
pip install numpy matplotlib
```

---

## Example Usage

```python
if __name__ == "__main__":
    r1_list = generate_circle()  # or generate_square()
    simulate_pathlines(
        r1_list=r1_list,
        make_bars=True,
        t1=0.0,
        tf=2.0,
        dt=0.005,
        total_duration_s=8
    )
```
