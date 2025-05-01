import matplotlib.pyplot as plt
import numpy as np

# ouput folder to save plot figure
##################################
output_folder_path = './utils/'


# parameter values to simulate with
####################################

parameters = {"muCore_muCortex_couples": [[300, 300], [300, 900], [300, 1500], [300, 3000],
                                          [1000, 1000], [1000, 3000], [1000, 5000], [1000, 10000]], # [Pa]
              "H0":[0.8, 2.25, 6] # [0.8, 1.5, 2.25, 3, 6]
             }
##
"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for m, zlow, zhigh in [('o', 0.5, 6.5)]:
    for mu_Cortex in parameters["mu_Cortex"]:
        for mu_Core in parameters["mu_Core"]:
            for H0 in parameters["H0"]:
                ax.scatter(mu_Cortex, mu_Core, H0, marker=m)

ax.set_xlabel('mu Cortex')
ax.set_ylabel('mu_Core')
ax.set_zlabel('H0')
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)
ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)
ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.5)

scatter_points = []
#for mu_Cortex in parameters["mu_Cortex"]:
#    for mu_Core in parameters["mu_Core"]:
for mu_Core, mu_Cortex in parameters["muCore_muCortex_couples"]:
    for H0 in parameters["H0"]:
        scatter_points.append([mu_Cortex, mu_Core, H0])

cmap = plt.cm.colors.LinearSegmentedColormap.from_list('blue_red', ['dodgerblue', 'red'])

scatter_points = np.array(scatter_points)
scatter = ax.scatter(scatter_points[:, 0], scatter_points[:, 1], scatter_points[:, 2], 
                     c=scatter_points[:, 2], cmap=cmap, s=50) # viridis, brg, gnuplot, gnuplot2

"""
ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10))
ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10))
ax.set_zticks(np.linspace(ax.get_zlim()[0], ax.get_zlim()[1], 10))
"""

ax.set_xlabel('mu Cortex', fontsize=12, fontweight='bold')
ax.set_ylabel('mu Core', fontsize=12, fontweight='bold')
ax.set_zlabel('H0', fontsize=12, fontweight='bold')

ax.xaxis.labelpad = 15
ax.yaxis.labelpad = 15
ax.zaxis.labelpad = 15

# Définition des graduations des axes
muCore_values = sorted(list(set(mu_Core_list[0] for mu_Core_list in parameters["muCore_muCortex_couples"])))
muCortex_values = sorted(list(set(mu_Cortex_list[1] for mu_Cortex_list in parameters["muCore_muCortex_couples"])))
ax.set_xticks(muCortex_values)
ax.set_yticks(muCore_values)
ax.set_zticks(parameters["H0"])

# Rotation des étiquettes pour une meilleure lisibilité
#ax.tick_params(axis='both', labelsize=8)
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=45)

# Ajuster la perspective et la position des axes
#ax.view_init(elev=20, azim=-20)
#ax.set_box_aspect((1, 1, 0.5))
#ax.set_proj_type('persp', focal_length=0.2)

# Déplacer les axes au fond à gauche
#ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
#ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
#ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)

# Ajouter une barre de couleur
#cbar = fig.colorbar(scatter, ax=ax, label='H0')
ax.view_init(elev=20, azim=-70) # default 30

plt.tight_layout()

plt.savefig(output_folder_path + "braingrowthFEniCS_parameters_space.svg")
#plt.show()