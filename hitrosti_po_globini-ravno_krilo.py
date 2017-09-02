import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


hitrost_vetra = 10
telon = trimesh.load_mesh("models/ravno_krilo/rakr15.stl")
telon.n = len(telon.triangles)
# hitrostni_vektorji = np.load("rezultati/ravno_krilo/hitrosti.npy")
hitrostni_vektorji = np.load("vmesni_rezultati/hitrosti.npy")
polozaj_vektorja_hitrosti = telon.triangles_center

y0 = [-12]
print("Računam intersekcije...")
rid = []
tid = []
rmi = telon.ray
for y_r in y0:
    # translacija modela
    telon.apply_translation(np.array([0, -y_r, 0]))

    tid1 = np.empty(0)
    z_r = np.linspace(-2.5, 2.5, telon.n // 20)
    ary1 = np.zeros(shape=(telon.n // 20, 3))
    ary2 = np.zeros(shape=(telon.n // 20, 3))
    ary1[:, 0] = -2
    ary2[:, 0] = 12
    ary1[:, 1] = 0
    ary2[:, 1] = 0
    ary1[:, 2] = z_r
    ary2[:, 2] = z_r
    tid11, rid12 = rmi.intersects_id(ray_origins=ary1, ray_directions=ary2)

    x_r = np.linspace(-2, 12, telon.n // 10)
    ary3 = np.zeros(shape=(telon.n // 10, 3))
    ary4 = np.zeros(shape=(telon.n // 10, 3))
    ary3[:, 0] = x_r
    ary4[:, 0] = x_r
    ary3[:, 1] = 0
    ary4[:, 1] = 0
    ary3[:, 2] = -2.5
    ary4[:, 2] = 2.5
    tid13, rid14 = rmi.intersects_id(ray_origins=ary3, ray_directions=ary4)

    tid1 = np.append(tid11, tid13)

    tid.append(np.unique(tid1))
    telon.apply_translation(np.array([0, y_r, 0]))

# graf hitrosti
for j in range(len(y0)):
    t_os = polozaj_vektorja_hitrosti[tid[j]]
    x_os = polozaj_vektorja_hitrosti[:, 0][tid[j]]
    indices = np.argsort(x_os)
    x_os = x_os[indices]
    y_os = np.linalg.norm(hitrostni_vektorji, axis=-1)[tid[j]]
    y_os = y_os[indices]
    t_os = t_os[indices]

    ind_1 = []
    ind_2 = []
    for i in range(len(x_os)):
        if t_os[i, 2] > 0:
            ind_1.append(i)
        else:
            ind_2.append(i)

    x_os_1 = x_os[ind_1]
    x_os_2 = x_os[ind_2]

    y_os_1 = y_os[ind_1]
    y_os_2 = y_os[ind_2]

    plt.plot(x_os_1, y_os_1, label=("zg. del, y = " + str(y0[j])))
    plt.plot(x_os_2, y_os_2, label=("sp. del, y = " + str(y0[j])))

plt.legend()
plt.show()

# graf pritiska, nenormiran x
cp = 1 - np.power(np.linalg.norm(hitrostni_vektorji, axis=-1) / hitrost_vetra, 2)
for j in range(len(y0)):
    t_os = polozaj_vektorja_hitrosti[tid[j]]
    x_os = polozaj_vektorja_hitrosti[:, 0][tid[j]]
    indices = np.argsort(x_os)
    x_os = x_os[indices]
    y_os = cp[tid[j]]
    y_os = y_os[indices]
    t_os = t_os[indices]

    ind_1 = []
    ind_2 = []
    for i in range(len(x_os)):
        if t_os[i, 2] > 0:
            ind_1.append(i)
        else:
            ind_2.append(i)

    x_os_1 = x_os[ind_1]
    x_os_2 = x_os[ind_2]

    y_os_1 = y_os[ind_1]
    y_os_2 = y_os[ind_2]

    plt.plot(x_os_1, y_os_1, label=("zg. del, y = " + str(y0[j])))
    plt.plot(x_os_2, y_os_2, label=("sp. del, y = " + str(y0[j])))

plt.legend()
ax = plt.gca()
ax.invert_yaxis()
plt.show()

# graf pritiska, normiran x
for j in range(len(y0)):
    t_os = polozaj_vektorja_hitrosti[tid[j]]
    x_os = polozaj_vektorja_hitrosti[:, 0][tid[j]]
    indices = np.argsort(x_os)
    x_os = x_os[indices]
    y_os = cp[tid[j]]
    y_os = y_os[indices]
    t_os = t_os[indices]

    ind_1 = []
    ind_2 = []
    for i in range(len(x_os)):
        if t_os[i, 2] > 0:
            ind_1.append(i)
        else:
            ind_2.append(i)

    x_os_1 = x_os[ind_1]
    x_os_2 = x_os[ind_2]

    normiran_x_os_1 = (x_os_1 - np.min(x_os_1)) / (np.max(x_os_1) - np.min(x_os_1))
    normiran_x_os_2 = (x_os_2 - np.min(x_os_2)) / (np.max(x_os_2) - np.min(x_os_2))

    y_os_1 = y_os[ind_1]
    y_os_2 = y_os[ind_2]

    plt.plot(normiran_x_os_1, y_os_1, label=("zg. del, y = " + str(y0[j])))
    plt.plot(normiran_x_os_2, y_os_2, label=("sp. del, y = " + str(y0[j])))

plt.legend()
ax = plt.gca()
ax.invert_yaxis()
plt.show()

# ROČNO PREVERJANJE ČE SO y_r MEJE OK IN POPRAVLJANJE NEKAJ MM SEM TER TJA (DA NE GRE TOČNO PO MEJAH)
print("Izrisujem...")
fig = plt.figure()
axes = mplot3d.Axes3D(fig)

axes.set_xlabel('x [mm]')
axes.set_ylabel('y [mm]')
axes.set_zlabel('z [mm]')

for i in range(len(y0)):
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(telon.triangles[tid[i]]))

scale = telon.triangles.flatten(-1)
axes.auto_scale_xyz(scale/2, scale/2, scale/2)
plt.show()


