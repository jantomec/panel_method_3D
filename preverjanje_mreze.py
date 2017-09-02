import numpy as np
import trimesh
from vrtincna_sled import vrtincnaSled
import racunska_orodja
import warnings
warnings.filterwarnings(action="ignore", message="Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.")

mejni_kot_1 = np.pi * 3 / 4
mejni_kot_2 = np.pi
telon = trimesh.load("models/elipticno_krilo/elkr22.stl")
telon.n = len(telon.triangles)
fn = []
for i in range(telon.n):
    vektorab = telon.triangles[i, 1] - telon.triangles[i, 0]
    vektorac = telon.triangles[i, 2] - telon.triangles[i, 0]
    vektorn = np.cross(vektorab, vektorac)
    fn.append(vektorn / np.linalg.norm(vektorn))

fn = np.array(fn)
telon.face_normals = fn
vrtincna_sled = vrtincnaSled(telo=telon, l=30, kot_theta=3.804*np.pi/180, mejni_kot_1=mejni_kot_1, mejni_kot_2=mejni_kot_2)  # l je dolžina vrtinčne sledi - predlagano je 30 * tetiva
telon.n = len(telon.triangles)

nor = telon.face_normals

n = np.linalg.norm(nor, axis=-1)
print(sum(n))
print(len(n))

if np.abs(sum(n) - len(n)) < 10**-4:
    print("\nNormale OK!")
else:
    print("\nNormale niso OK!")

oglisca_1 = telon.triangles[:, 0]
oglisca_2 = telon.triangles[:, 1]
oglisca_3 = telon.triangles[:, 2]
lokalna_oglisca_1 = np.zeros_like(oglisca_1)
lokalna_oglisca_2 = racunska_orodja.transformacija_2(paneli=np.arange(0, telon.n, 1),
                                                     tocke=oglisca_2,
                                                     telo=telon,
                                                     dtype=np.float64)
lokalna_oglisca_3 = racunska_orodja.transformacija_2(paneli=np.arange(0, telon.n, 1),
                                                     tocke=oglisca_3,
                                                     telo=telon,
                                                     dtype=np.float64)

x2 = lokalna_oglisca_2[:, 0]
y2 = lokalna_oglisca_2[:, 1]
y3 = lokalna_oglisca_3[:, 1]

print("\nNajmanjše meje:")
print(np.min(x2))
print(np.min(y2))
print(np.min(y3))

print("\nVelikost mreže:")
print(telon.n)
print(vrtincna_sled.n)
