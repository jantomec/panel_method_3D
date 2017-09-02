import numpy as np


class vrtincnaSled:
    def __init__(self, telo, l, kot_theta, mejni_kot_1=np.pi*9/10, mejni_kot_2=np.pi/10):

        # ta ima vrtinčno sled v smeri osi x

        self.pari_robnih_panelov = telo.face_adjacency[np.where(np.logical_and(mejni_kot_1 < telo.face_adjacency_angles, telo.face_adjacency_angles < np.pi))]
        # na prvem mestu je zgornji panel:
        for i in range(len(self.pari_robnih_panelov)):
            if telo.face_normals[self.pari_robnih_panelov[i, 0]][2] < 0:
                self.pari_robnih_panelov[i] = np.flip(self.pari_robnih_panelov[i], axis=0)

        # določitev oglišč na robu (se podvajajo)
        self.robovi = []
        for i in range(len(self.pari_robnih_panelov)):
            t0 = telo.triangles[self.pari_robnih_panelov[i, 0]]
            # x0 = t0[:, 0]
            # t1 = np.array([t0[np.argsort(x0)[0]], t0[np.argsort(x0)[1]]])
            # s_ = t1[1] - t1[0]
            # if s_[1] < 0:
            #     t1 = np.flip(t1, axis=0)
            # self.robovi.append(t1)
            t1 = telo.triangles[self.pari_robnih_panelov[i, 1]]

            rob = []
            for oglišče in t0:
                enako_a = oglišče == t1[0]
                enako_b = oglišče == t1[1]
                enako_c = oglišče == t1[2]
                enako = np.array([enako_a.all(), enako_b.all(), enako_c.all()])
                if enako.any():
                    rob.append(oglišče)
            if len(rob) == 2:
                if rob[-1][1] < rob[0][1]:
                    rob = list(reversed(rob))
                self.robovi.append(rob)

        self.robovi = np.array(self.robovi)

        t = []

        for i in range(len(self.pari_robnih_panelov)):
            r = self.robovi[i, 1] - self.robovi[i, 0]
            y = np.array([0, 1, 0])
            if np.dot(r, y) > np.cos(mejni_kot_2):
                t.append(True)
            else:
                t.append(False)

        self.robovi = self.robovi[t]
        self.pari_robnih_panelov = self.pari_robnih_panelov[t]

        self.smerni_vektor = []
        self.triangles = []
        self.face_normals = []
        for i in range(len(self.pari_robnih_panelov)):
            if kot_theta == 360:
                n1 = telo.face_normals[self.pari_robnih_panelov[i][0]]
                n2 = telo.face_normals[self.pari_robnih_panelov[i][1]]
                sv1 = np.array([
                    (-n2[0] * (n1[1] ** 2 + n1[2] ** 2) + (n1[0] + n2[0]) * (n1[1] * n2[1] + n1[2] * n2[2]) - n1[0] * (n2[1] ** 2 + n2[2] ** 2)) / (np.linalg.norm(n1 - n2) * np.linalg.norm(np.cross(n1, n2))),
                    (-n1[0] ** 2 * n2[1] + n1[0] * n2[0] * (n1[1] + n2[1]) + n1[2] * n2[1] * (-n1[2] + n2[2]) - n1[1] * (n2[0] ** 2 - n1[2] * n2[2] + n2[2] ** 2)) / (np.linalg.norm(n1 - n2) * np.linalg.norm(np.cross(n1, n2))),
                    (-n1[2] * (n2[0] ** 2 - n1[1] * n2[1] + n2[1] ** 2) - n1[0] ** 2 * n2[2] + n1[1] * (-n1[1] + n2[1]) * n2[2] + n1[0] * n2[0] * (n1[2] + n2[2])) / (np.linalg.norm(n1 - n2) * np.linalg.norm(np.cross(n1, n2))),
                ])
                sv2 = -sv1
                if np.dot(sv1, n1) > 0:
                    sv = sv1
                else:
                    sv = sv2
                if np.dot(np.array([1, 0, 0]), sv) < 0:
                    sv *= -1
            else:
                sv = np.array([np.cos(kot_theta), 0, -np.sin(kot_theta)])
            self.smerni_vektor.append(sv)

            nt = np.copy(self.robovi[i][0])
            nt += sv * l
            mt = np.copy(self.robovi[i][1])
            mt += sv * l
            p1 = np.array([self.robovi[i][1], self.robovi[i][0], nt])
            p2 = np.array([self.robovi[i][1], nt, mt])

            self.triangles.append(p1)
            self.triangles.append(p2)
            n1 = np.cross(p1[1] - p1[0], p1[2] - p1[0])
            n2 = np.cross(p2[1] - p2[0], p2[2] - p2[0])
            n1 /= np.linalg.norm(n1)
            n2 /= np.linalg.norm(n2)
            if n1[2] < 0:
                print("Error")
            if n2[2] < 0:
                print("Error")

            self.face_normals.append(n1)
            self.face_normals.append(n2)

        self.triangles = np.array(self.triangles)
        self.points = self.triangles.reshape((len(self.triangles), 9))
        self.face_normals = np.array(self.face_normals)
        self.n = len(self.face_normals)


class vrtincnaSled2:
    def __init__(self, telo, l, mejni_kot_1=np.pi*9/10, mejni_kot_2=np.pi/10):

        # ta ima vrtinčno sled v smeri pravokotno na robove

        self.pari_robnih_panelov = telo.face_adjacency[np.where(np.logical_and(mejni_kot_1 < telo.face_adjacency_angles, telo.face_adjacency_angles < np.pi))]
        # na prvem mestu je zgornji panel:
        for i in range(len(self.pari_robnih_panelov)):
            if telo.face_normals[self.pari_robnih_panelov[i, 0]][2] < 0:
                self.pari_robnih_panelov[i] = np.flip(self.pari_robnih_panelov[i], axis=0)

        # določitev oglišč na robu (se podvajajo)
        self.robovi = []
        for i in range(len(self.pari_robnih_panelov)):
            t0 = telo.triangles[self.pari_robnih_panelov[i, 0]]
            # x0 = t0[:, 0]
            # t1 = np.array([t0[np.argsort(x0)[0]], t0[np.argsort(x0)[1]]])
            # s_ = t1[1] - t1[0]
            # if s_[1] < 0:
            #     t1 = np.flip(t1, axis=0)
            # self.robovi.append(t1)
            t1 = telo.triangles[self.pari_robnih_panelov[i, 1]]

            rob = []
            for oglišče in t0:
                enako_a = oglišče == t1[0]
                enako_b = oglišče == t1[1]
                enako_c = oglišče == t1[2]
                enako = np.array([enako_a.all(), enako_b.all(), enako_c.all()])
                if enako.any():
                    rob.append(oglišče)
            if len(rob) == 2:
                if rob[-1][1] < rob[0][1]:
                    rob = list(reversed(rob))
                self.robovi.append(rob)

        self.robovi = np.array(self.robovi)
        t = []

        for i in range(len(self.pari_robnih_panelov)):
            r = self.robovi[i, 1] - self.robovi[i, 0]
            y =np.array([0, 1, 0])
            if np.dot(r, y) > np.cos(mejni_kot_2):
                t.append(True)
            else:
                t.append(False)

        self.robovi = self.robovi[t]
        self.pari_robnih_panelov = self.pari_robnih_panelov[t]

        self.smerni_vektor = []
        self.triangles = []
        self.face_normals = []
        for i in range(len(self.pari_robnih_panelov)):
            n1 = telo.face_normals[self.pari_robnih_panelov[i][0]]
            n2 = telo.face_normals[self.pari_robnih_panelov[i][1]]
            sv1 = np.array([
                (-n2[0] * (n1[1] ** 2 + n1[2] ** 2) + (n1[0] + n2[0]) * (n1[1] * n2[1] + n1[2] * n2[2]) - n1[0] * (n2[1] ** 2 + n2[2] ** 2)) / (np.linalg.norm(n1 - n2) * np.linalg.norm(np.cross(n1, n2))),
                (-n1[0] ** 2 * n2[1] + n1[0] * n2[0] * (n1[1] + n2[1]) + n1[2] * n2[1] * (-n1[2] + n2[2]) - n1[1] * (n2[0] ** 2 - n1[2] * n2[2] + n2[2] ** 2)) / (np.linalg.norm(n1 - n2) * np.linalg.norm(np.cross(n1, n2))),
                (-n1[2] * (n2[0] ** 2 - n1[1] * n2[1] + n2[1] ** 2) - n1[0] ** 2 * n2[2] + n1[1] * (-n1[1] + n2[1]) * n2[2] + n1[0] * n2[0] * (n1[2] + n2[2])) / (np.linalg.norm(n1 - n2) * np.linalg.norm(np.cross(n1, n2))),
            ])
            sv2 = -sv1
            if np.dot(sv1, n1) > 0:
                sv = sv1
            else:
                sv = sv2
            self.smerni_vektor.append(sv)

            nt = np.copy(self.robovi[i][0])
            nt += sv * l
            mt = np.copy(self.robovi[i][1])
            mt += sv * l
            p1 = np.array([self.robovi[i][0], self.robovi[i][1], nt])
            p2 = np.array([self.robovi[i][1], nt, mt])

            self.triangles.append(p1)
            self.triangles.append(p2)
            n1 = np.cross(p1[1] - p1[0], p1[2] - p1[0])
            n2 = np.cross(p2[1] - p2[0], p2[2] - p2[0])
            n1 /= np.linalg.norm(n1)
            n2 /= np.linalg.norm(n2)
            if n1[2] < 0:
                n1 *= -1
            if n2[2] < 0:
                n2 *= -1

            self.face_normals.append(n1)
            self.face_normals.append(n2)

        self.triangles = np.array(self.triangles)
        self.points = self.triangles.reshape((len(self.triangles), 9))
        self.face_normals = np.array(self.face_normals)
        self.n = len(self.face_normals)
