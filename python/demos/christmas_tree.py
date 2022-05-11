import matplotlib.pyplot as plt
import numpy as np


def awful_gmsh_output(curve, offset=0):
    "Creates a .geo file with the closed curve defined by a set of x,y points"
    xv, yv = curve
    s = []
    for i, q in enumerate(zip(xv, yv)):
        s += [f"Point({i+1 + offset}) = {{{q[0]}, {q[1]}, 0.0}};\n"]
    for i in range(len(xv) - 1):
        s += [f"Line({i + 1 + offset}) = {{{i + 1 + offset}, {i + 2 + offset}}};\n"]
    s += [f"Line({len(xv) + offset}) = {{{len(xv) + offset} , {offset + 1}}};\n"]
    ll = ",".join([str(i + offset) for i in range(1, len(xv) + 1)])
    s += [f"Line Loop ({offset + 1}) = {{{ll}}};\n"]
    s += [f"Plane Surface ({offset + 1}) = {{{offset + 1}}};\n"]
    return s


def jagged_curve(npoints, t, r0, r1, xmax):
    xvals = []
    yvals = []

    dx = xmax / npoints

    xvals = [0.0]
    yvals = [0.0]

    while (xvals[-1] < xmax):
        ylast = yvals[-1]
        xlast = xvals[-1]
        xp = dx
        ds = 4 * dx
        while (ds > 3 * dx):
            x = xlast + xp
            y = r0(x) * np.arctan(t * np.sin(np.pi * x)
                                  / (1 - t * np.cos(np.pi * x))) / t + r1(x)

            ds = np.sqrt((y - ylast)**2 + (x - xlast)**2)
            print('ds=', ds / dx)
            xp *= 0.8

        xvals += [x]
        yvals += [y]

    return np.array(xvals), np.array(yvals)


# 1. Create the inner curve
npc1 = 151  # Number of points on curved sides
npc2 = 40  # Number of points on bottom
jagg = -0.95  # Jaggedness: useful range -0.1 -> -0.999
xmax = 8.0  # Size in x-direction

x, y = jagged_curve(npc1, jagg, lambda x: (0.1 + x*0.125), lambda x: 0.5*x, xmax)

# Fill in bottom
yb = np.linspace(y[-1], -y[-1], npc2)
xb = np.ones_like(yb) * x[-1]

# Flip and concatenate
xv = np.concatenate((x[1:], xb, np.flip(x[1:])))
yv = np.concatenate((y[1:], yb, np.flip(-y[1:])))

fd = open("xmas.geo", "w")
for s in awful_gmsh_output((xv, yv)):
    fd.write(s)

gmsh_offset = len(xv)

# 2. Create outer mesh
# Displacement of outer mesh to inner
dx = 0.01
dy = 0.1
x += dx
y += dy

x0 = -1.0  # Position of left-hand edge of outer mesh
yplus = y[-1] * 1.5  # Position of top and bottom relative to curve

# Number of points for each edge
ncleft = 50
nctop = 50
ncplus = 8

# Create edges (top and bottom)
xf = np.linspace(x0, x[-1], nctop)
yf = np.ones_like(xf) * yplus

# Create edge (left)
yg = np.linspace(yf[0], -yf[0], ncleft)[1:-1]
xg = np.ones_like(yg) * x0

# Create joining edges (right)
yh = np.linspace(y[-1], yplus, ncplus)[1:-1]
xh = np.ones_like(yh) * x[-1]

xv = np.concatenate((xh, np.flip(x), x, xh, np.flip(xf), xg, xf))
yv = np.concatenate((-np.flip(yh), np.flip(-y), y, yh, yf, yg, -yf))

for s in awful_gmsh_output((xv, yv), gmsh_offset):
    fd.write(s)
fd.close()

plt.plot(xv, yv, marker='o')
plt.show()
