import matplotlib.pyplot as plt
import numpy as np


def awful_gmsh_output(filename, curve, offset=0):
    "Creates a .geo file with the closed curve defined by a set of x,y points"
    xv, yv = curve
    fd = open(filename, "w")
    for i, q in enumerate(zip(xv, yv)):
        fd.write(f"Point({i+1 + offset}) = {{{q[0]}, {q[1]}, 0.0}};\n")
    for i in range(len(xv) - 1):
        fd.write(f"Line({i + 1 + offset}) = {{{i + 1 + offset}, {i + 2 + offset}}};\n")
    fd.write(f"Line({len(xv) + offset}) = {{{len(xv) + offset} , {offset + 1}}};\n")
    ll = ",".join([str(i + offset) for i in range(1, len(xv) + 1)])
    fd.write(f"Line Loop ({offset + 1}) = {{{ll}}};\n")
    fd.write(f"Plane Surface ({offset + 1}) = {{{offset + 1}}};\n")
    fd.close()


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
                                  / (1 - t * np.cos(np.pi * x))) / t + r1(x) * x

            ds = np.sqrt((y - ylast)**2 + (x - xlast)**2)
            print('ds=', ds / dx)
            xp *= 0.8

        xvals += [x]
        yvals += [y]

    return np.array(xvals), np.array(yvals)


nc = 51
x, y = jagged_curve(nc, -0.95, lambda x: 0.5, lambda x: 1.0, 4.0)

# Fill in bottom
yb = np.linspace(y[-1], -y[-1], nc)
xb = np.ones_like(yb) * x[-1]

# Flip and concatenate
xv = np.concatenate((x[1:], xb, np.flip(x[1:])))
yv = np.concatenate((y[1:], yb, np.flip(-y[1:])))

awful_gmsh_output("xmas_inner.geo", (xv, yv))

dx = 0.01
dy = 0.1
x += dx
y += dy

xf = np.linspace(-1, x[-1], nc)
yf = np.ones_like(xf) * y[-1] * 1.2

yg = np.linspace(yf[0], -yf[0], nc)[1:-1]
xg = np.ones_like(yg) * -1

yh = np.linspace(y[-1], y[-1] * 1.2, 8)[1:-1]
xh = np.ones_like(yh)*x[-1]

xv = np.concatenate((xh, np.flip(x), x, xh, np.flip(xf), xg, xf))
yv = np.concatenate((-np.flip(yh), np.flip(-y), y, yh, yf, yg, -yf))

offset = 1000
awful_gmsh_output("xmas_outer.geo", (xv, yv), offset)

plt.plot(xv, yv, marker='o')
plt.show()
