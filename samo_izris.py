import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

hitrost_vetra = 10

telon = trimesh.load_mesh("models/ravno_krilo/rakr15.stl")

telon.n = len(telon.triangles)

if False:
    pot = np.load("vmesni_rezultati/potenciali.npy")
    for i in range(telon.n):
        v = pot[i]
        if v <= np.mean(v):
            r = 0
            g = -np.cos(np.pi * v / (2 * np.mean(pot))) + 1
            b = np.cos(np.pi * v / (2 * np.mean(pot)))
        else:
            r = np.sin(np.pi * (v - np.mean(pot)) / (2 * (np.max(pot) - np.mean(pot))))
            g = -np.sin(np.pi * (v - np.mean(pot)) / (2 * (np.max(pot) - np.mean(pot)))) + 1
            b = 0

        if r > 1:
            r = 1
        if g > 1:
            g = 1
        if b > 1:
            b = 1

        r = np.round(r * 255)
        g = np.round(g * 255)
        b = np.round(b * 255)
        telon.visual.face_colors[i] = np.array([r, g, b, 255])

    telon.show()

if True:
    hitrostni_vektorji = np.load("vmesni_rezultati/hitrosti.npy")

    polozaj_vektorja_hitrosti = telon.triangles_center

    hitrosti = np.linalg.norm(hitrostni_vektorji, axis=-1)

    # h = hitrosti > 16
    # hitrosti[h] = hitrost_vetra

    cp = 1 - np.power(hitrosti / hitrost_vetra, 2)

    # Izris 3D modela
    print("Izrisujem...")


    # pobarvanka
    for i in range(telon.n):
        if cp[i] <= np.mean(cp):
            r = 0
            g = -np.cos(np.pi * cp[i] / (2 * np.mean(cp))) + 1
            b = np.cos(np.pi * cp[i] / (2 * np.mean(cp)))
        else:
            r = np.sin(np.pi * (cp[i] - np.mean(cp)) / (2 * (np.max(cp) - np.mean(cp))))
            g = -np.sin(np.pi * (cp[i] - np.mean(cp)) / (2 * (np.max(cp) - np.mean(cp)))) + 1
            b = 0

        if r > 1:
            r = 1
        if g > 1:
            g = 1
        if b > 1:
            b = 1

        r = np.round(r * 255)
        g = np.round(g * 255)
        b = np.round(b * 255)
        telon.visual.face_colors[i] = np.array([r, g, b, 255])

    telon.show()
    print("Izrisano!")