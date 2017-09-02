import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from vrtincna_sled import vrtincnaSled
import racunska_orodja

mejni_kot_1 = np.pi * 2 / 4
mejni_kot_2 = np.pi / 2
telon = trimesh.load_mesh("/Users/jantomec/Documents/FS/6. semester/Zakljucna naloga/Python/models/elipticno_krilo/elkr22.stl")
# telon = trimesh.load("/Users/jantomec/Documents/FS/6. semester/Zakljucna naloga/Python/models/ravno_krilo/rakr15.stl")
telon.n = len(telon.triangles)
# fn = []
# for i in range(telon.n):
#     vektorab = telon.triangles[i, 1] - telon.triangles[i, 0]
#     vektorac = telon.triangles[i, 2] - telon.triangles[i, 0]
#     vektorn = np.cross(vektorab, vektorac)
#     fn.append(vektorn / np.linalg.norm(vektorn))
#
# fn = np.array(fn)
# telon.face_normals = fn
vrtincna_sled = vrtincnaSled(telo=telon, l=300, kot_theta=3.804*np.pi/180, mejni_kot_1=mejni_kot_1, mejni_kot_2=mejni_kot_2)

print("vrtinčna sled:", vrtincna_sled.n)

# Izris 3D modela
print("Izrisujem...")
fig = plt.figure()
axes = mplot3d.Axes3D(fig)

axes.set_xlabel('x [mm]')
axes.set_ylabel('y [mm]')
axes.set_zlabel('z [mm]')

# izris vektorskega polja
for i in range(len(telon.triangles)):
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection([telon.triangles[i]], facecolors="b", linewidths=3))
for i in range(len(vrtincna_sled.triangles)):
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection([vrtincna_sled.triangles[i]], facecolors="r", linewidths=3))

scale = telon.triangles.flatten(-1)
axes.auto_scale_xyz(0.5*scale, 0.5*scale, 0.5*scale)
print("Izrisano!")
plt.show()
