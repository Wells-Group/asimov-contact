from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

n_dofs = [4137, 13332, 17028, 22932, 33930, 51510, 89712, 170352, 245514]
newton_iterations = [9, 9, 9, 10, 10, 10, 10, 11, 11]
krylov_iterations = [154, 159, 160, 179, 188, 196, 210, 236, 237]
runtime = [6.77954976, 26.76396604, 35.45588557, 51.65037789, 83.86667789, 124.73322894, 229.36642167,
           492.29369094, 715.92167737]

plt.title("Cylinder-Cylinder contact problem (1-step approach, tetrahedron)")
fig, axs = plt.subplots(2, 1)

axs[0].plot(n_dofs, newton_iterations, marker="o", label="Newton\n solver")
axs[0].plot(n_dofs, krylov_iterations, marker="s", label="Krylov\n solver")
axs[0].set_xlabel("Degrees of freedom")
axs[0].set_ylabel("Number of iterations")
axs[0].set_xscale('log')
axs[0].grid(True, which="both")
axs[0].axis([1e3, 1e6, 0, 250])
axs[0].legend()
poly = np.polyfit(n_dofs[0:2], runtime[0:2], deg=1)

# Check that polyfit is correct:
a = (runtime[1] - runtime[0]) / (n_dofs[1] - n_dofs[0])
assert(np.isclose(poly[0], a))
b = runtime[0] - a * n_dofs[0]
assert(np.isclose(b, poly[1]))


def f(x):
    return poly[0] * x + poly[1]


axs[1].loglog(n_dofs, runtime, marker="o")
n_dofs = np.array(n_dofs)
axs[1].loglog(n_dofs, f(n_dofs), "--")
axs[1].set_xlabel("Degrees of freedom")
axs[1].set_ylabel("Runtime (Newton Solver)")

axs[1].grid(True, which="both")
axs[1].axis([1e3, 1e6, 1e0, 1e3])

plt.savefig("single_step_tet.png")

exit()

# n_dofs = np.array([13332, 17028, 22932, 33930, 51510, 89712])
# newton_iterations = np.array([[2, 2, 2, 7, 7], [2, 2, 2, 8, 7], [2, 2, 2, 8, 8],
#                              [2, 2, 2, 8, 8], [2, 2, 2, 8, 8], [2, 2, 2, 9, 10]])
# krylov_iterations = np.array([[10, 10, 10, 59, 77], [10, 10, 10, 70, 72], [9, 9, 9, 64, 89],
#                               [9, 9, 9, 63, 92], [8, 8, 8, 74, 100], [9, 9, 9, 77, 129]])

markers = ["s", "o", "*", "p", "s", "v", "p", "h"]
markersize = [6, 5, 5, 8, 5]
linestyle = ["solid", "dotted", "dashed", "dashdot", "solid"]
fig, axs = plt.subplots(2)
axs[0].set_title("Cylinder-Cylinder contact problem")

for i, step in enumerate(newton_iterations.T):
    axs[0].plot(n_dofs, step, marker=markers[i], linestyle=linestyle[i], markersize=markersize[i],
                label=f"Newton (Step {i+1})")
axs[0].grid(True, which="both")
axs[0].set_xscale('log')
axs[0].axis([1e4, 2e5, 0, 12])
box = axs[0].get_position()
axs[0].set_position([box.x0, box.y0, box.width * 0.75, box.height])
axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[0].set_xlabel("Degrees of freedom")
axs[0].set_ylabel("Number of iterations")

for i, step in enumerate(krylov_iterations.T):
    axs[1].plot(n_dofs, step, marker=markers[i], linestyle=linestyle[i], markersize=markersize[i],
                label=f"Krylov (Step {i+1})")
axs[1].grid(True, which="both")
axs[1].set_xscale('log')
axs[1].axis([1e4, 2e5, 0, 200])
box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0, box.width * 0.75, box.height])
axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axs[1].set_xlabel("Degrees of freedom")
axs[1].set_ylabel("Number of iterations")


plt.savefig(f"{krylov_iterations.shape[1]}_step.png")
