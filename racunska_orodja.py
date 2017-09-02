import numpy as np
import pyopencl as cl


def kot_med_vektorjema(v1, v2):
    """
    Izračun kota med vektorjema, ki vrne vedno pravi kot.
    :param v1: array([vx1, vy1, vz1])
    :param v2: array([vx2, vy2, vz2])
    :return: Kot v radianih.
    """
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    smerni_v1 = v1 / a
    smerni_v2 = v2 / b
    kosinus_kota = np.dot(smerni_v1, smerni_v2)
    sinus_kota = np.linalg.norm(np.cross(smerni_v1, smerni_v2))
    return np.arctan2(sinus_kota, kosinus_kota)


def izracun_normale(oglisca):
    """
    Izračun kota normalo trikotnika, ki vrne vedno pravi kot.
    :param oglisca: array([A, B, C])
    :return: n = array([nx, ny, nz])
    """
    a = oglisca[0]
    b = oglisca[1]
    c = oglisca[2]
    ab = b - a
    ab /= np.linalg.norm(ab)
    ac = c - a
    ac /= np.linalg.norm(ac)
    return np.cross(ab, ac)


def razdalja_med_tockama(t1, t2):
    """
    Izračuna razdaljo med dvema točkama. Deluje z n-dimenzionalnimi točkami, vendar le z dvema točkama.
    :param t1: Točka 1.
    :param t2: Točka 2.
    :return: Razdalja.
    """
    return np.linalg.norm(t1 - t2)


def transformacija_g(paneli, tocke, telo, dtype):
    """
    Transformira array točk v lokalne koordinatne sisteme panelov. Torej vsako točko v vsak sistem.
    Funkcija je grafično pospešena.
    Mišljeno za transformacijo računskih točk.
    :param paneli: Array željenih indeskov panelov (arr, shape(m)).
    :param tocke: Array točk, ki jih želimo transformirati (arr, shape=(n,3)).
    :param telo: Trimesh objekt (trimesh obj).
    :param dtype: dtype.
    :return: Transformiran array točk (arr, shape=(n,m,3)).
    """
    zj = telo.face_normals[paneli]  # shape = (n,3) -> (m,3)
    zj = zj.astype(dtype=dtype)
    yj = telo.triangles[paneli, 2] - telo.triangles[paneli, 0]  # shape = (n,3) -> (m,3)
    yj = yj.astype(dtype=dtype)
    yj = np.divide(yj, np.linalg.norm(yj, axis=1).reshape(len(yj), 1))  # shape = (m,3) -> (m) -> (m,3)
    xj = np.cross(yj, zj)  # shape = (m,3)
    tocke = tocke.reshape(tocke.shape[0], 1, tocke.shape[1])
    tocke = tocke.astype(dtype=dtype)
    translatorni_vektorji = telo.triangles[paneli, 0]  # shape = (m,3)
    translatorni_vektorji = translatorni_vektorji.astype(dtype=dtype)
    translatorni_vektorji = translatorni_vektorji.reshape(
        1,
        translatorni_vektorji.shape[0],
        translatorni_vektorji.shape[1],
    )
    tocke = np.subtract(tocke, translatorni_vektorji)  # shape = (n,1,3) - (1,m,3) -> (n,m,3)
    r___ = np.stack((xj, yj, zj), axis=2)  # shape = (m,3,3)

    vektor_np = tocke.astype(np.float64)
    rotacijska_matrika_np = r___.astype(np.float64)

    res_np = np.einsum("ijk,jkl->ijl", vektor_np, rotacijska_matrika_np)

    return res_np  # shape = (n,m,3)


def transformacija_2_g(paneli, tocke, telo, dtype):
    """
    Transformira array točk v lokalne koordinatne sisteme panelov. Torej vsako točko samo v svoj sistem.
    Funkcija je grafično pospešena.
    Mišljeno za trasformacijo mej panelov.
    :param paneli: Array željenih indeskov panelov (arr, shape(m)).
    :param tocke: Array točk, ki jih želimo transformirati (arr, shape=(m,3)).
    :param telo: Trimesh objekt (trimesh obj).
    :param dtype: dtype.
    :return: Transformiran array točk (arr, shape=(m,3)).
    """
    zj = telo.face_normals[paneli]  # shape = (n,3) -> (m,3)
    zj = zj.astype(dtype=dtype)
    yj = telo.triangles[paneli, 2] - telo.triangles[paneli, 0]  # shape = (n,3) -> (m,3)
    yj = yj.astype(dtype=dtype)
    yj = np.divide(yj, np.linalg.norm(yj, axis=1).reshape(len(yj), 1))  # shape = (m,3) -> (m) -> (m,3)
    xj = np.cross(yj, zj)  # shape = (m,3)
    # tocke = tocke.reshape(tocke.shape[0], 1, tocke.shape[1])
    tocke = tocke.astype(dtype=dtype)
    translatorni_vektorji = telo.triangles[paneli, 0]  # shape = (m,3)
    translatorni_vektorji = translatorni_vektorji.astype(dtype=dtype)
    # translatorni_vektorji = translatorni_vektorji.reshape(
    #     1,
    #     translatorni_vektorji.shape[0],
    #     translatorni_vektorji.shape[1],
    # )
    tocke = np.subtract(tocke, translatorni_vektorji)  # shape = (m,3) - (m,3) -> (m,3)
    r___ = np.stack((xj, yj, zj), axis=2)  # shape = (m,3,3)
    # tocke = tocke.reshape(tocke.shape[0] * tocke.shape[1], tocke.shape[2])  # shape = (n,m,3) -> (n*m,3)

    vektor_np = tocke.astype(np.float64)
    rotacijska_matrika_np = r___.astype(np.float64)

    res_np = np.einsum("jk,jkl->jl", vektor_np, rotacijska_matrika_np)

    return res_np  # shape = (n,m,3)


def transformacija(paneli, tocke, telo, dtype):
    """
    Transformira array točk v lokalne koordinatne sisteme panelov. Torej vsako točko v vsak sistem.
    Funkcija je grafično pospešena.
    Mišljeno za transformacijo računskih točk.
    :param paneli: Array željenih indeskov panelov (arr, shape(m)).
    :param tocke: Array točk, ki jih želimo transformirati (arr, shape=(n,3)).
    :param telo: Trimesh objekt (trimesh obj).
    :param dtype: dtype.
    :return: Transformiran array točk (arr, shape=(n,m,3)).
    """
    zj = telo.face_normals[paneli]  # shape = (n,3) -> (m,3)
    zj = zj.astype(dtype=dtype)
    yj = telo.triangles[paneli, 2] - telo.triangles[paneli, 0]  # shape = (n,3) -> (m,3)
    yj = yj.astype(dtype=dtype)
    yj = np.divide(yj, np.linalg.norm(yj, axis=1).reshape(len(yj), 1))  # shape = (m,3) -> (m) -> (m,3)
    xj = np.cross(yj, zj)  # shape = (m,3)
    tocke = tocke.reshape(tocke.shape[0], 1, tocke.shape[1])
    tocke = tocke.astype(dtype=dtype)
    translatorni_vektorji = telo.triangles[paneli, 0]  # shape = (m,3)
    translatorni_vektorji = translatorni_vektorji.astype(dtype=dtype)
    translatorni_vektorji = translatorni_vektorji.reshape(
        1,
        translatorni_vektorji.shape[0],
        translatorni_vektorji.shape[1],
    )
    tocke = np.subtract(tocke, translatorni_vektorji)  # shape = (n,1,3) - (1,m,3) -> (n,m,3)
    r___ = np.stack((xj, yj, zj), axis=2)  # shape = (m,3,3)

    vektor_np = tocke.astype(np.float64)
    rotacijska_matrika_np = r___.astype(np.float64)
    m_np = np.array([telo.n]).astype(np.float64)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    rotacijska_matrika_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rotacijska_matrika_np)
    vektor_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vektor_np)
    m_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)

    prg = cl.Program(ctx, """
        __kernel void mojeIme(
            __global const double* rotacijska_matrika_buffer, 
            __global const double* vektor_buffer, 
            __global const double* m_buffer,
            __global double* res_buffer)
        {
            int i = get_global_id(2);
            int j = get_global_id(1);
            int k = get_global_id(0);
            int m = m_buffer[0];
            
            res_buffer[i+j*3+k*3*m] = vektor_buffer[j*3+k*3*m] * \
            rotacijska_matrika_buffer[i+j*3*3] + vektor_buffer[1+j*3+k*3*m] * \
            rotacijska_matrika_buffer[3+i+j*3*3] + vektor_buffer[2+j*3+k*3*m] * \
            rotacijska_matrika_buffer[6+i+j*3*3];
        }
        """).build()

    res_np = np.zeros_like(vektor_np)
    res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

    prg.mojeIme(queue, vektor_np.shape, None, rotacijska_matrika_buffer, vektor_buffer, m_buffer, res_buffer)

    cl.enqueue_copy(queue, res_np, res_buffer)

    return res_np  # shape = (n,m,3)


def transformacija_2(paneli, tocke, telo, dtype):
    """
    Transformira array točk v lokalne koordinatne sisteme panelov. Torej vsako točko samo v svoj sistem.
    Funkcija je grafično pospešena.
    Mišljeno za trasformacijo mej panelov.
    :param paneli: Array željenih indeskov panelov (arr, shape(m)).
    :param tocke: Array točk, ki jih želimo transformirati (arr, shape=(m,3)).
    :param telo: Trimesh objekt (trimesh obj).
    :param dtype: dtype.
    :return: Transformiran array točk (arr, shape=(m,3)).
    """
    zj = telo.face_normals[paneli]  # shape = (n,3) -> (m,3)
    zj = zj.astype(dtype=dtype)
    yj = telo.triangles[paneli, 2] - telo.triangles[paneli, 0]  # shape = (n,3) -> (m,3)
    yj = yj.astype(dtype=dtype)
    yj = np.divide(yj, np.linalg.norm(yj, axis=1).reshape(len(yj), 1))  # shape = (m,3) -> (m) -> (m,3)
    xj = np.cross(yj, zj)  # shape = (m,3)
    # tocke = tocke.reshape(tocke.shape[0], 1, tocke.shape[1])
    tocke = tocke.astype(dtype=dtype)
    translatorni_vektorji = telo.triangles[paneli, 0]  # shape = (m,3)
    translatorni_vektorji = translatorni_vektorji.astype(dtype=dtype)
    # translatorni_vektorji = translatorni_vektorji.reshape(
    #     1,
    #     translatorni_vektorji.shape[0],
    #     translatorni_vektorji.shape[1],
    # )
    tocke = np.subtract(tocke, translatorni_vektorji)  # shape = (m,3) - (m,3) -> (m,3)
    r___ = np.stack((xj, yj, zj), axis=2)  # shape = (m,3,3)
    # tocke = tocke.reshape(tocke.shape[0] * tocke.shape[1], tocke.shape[2])  # shape = (n,m,3) -> (n*m,3)

    vektor_np = tocke.astype(np.float64)
    rotacijska_matrika_np = r___.astype(np.float64)
    m_np = np.array([telo.n]).astype(np.float64)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    rotacijska_matrika_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rotacijska_matrika_np)
    vektor_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vektor_np)
    m_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)

    prg = cl.Program(ctx, """
        __kernel void mojeIme(
            __global const double* rotacijska_matrika_buffer, 
            __global const double* vektor_buffer, 
            __global const double* m_buffer, 
            __global double* res_buffer)
        {
            int i = get_global_id(1);
            int j = get_global_id(0);
            int m = m_buffer[0];

            res_buffer[i+j*3] = vektor_buffer[j*3] * rotacijska_matrika_buffer[i+j*3*3] + \
            vektor_buffer[1+j*3] * rotacijska_matrika_buffer[3+i+j*3*3] + vektor_buffer[2+j*3] * \
            rotacijska_matrika_buffer[6+i+j*3*3];
        }
        """).build()

    res_np = np.zeros_like(vektor_np)
    res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

    prg.mojeIme(queue, vektor_np.shape, None, rotacijska_matrika_buffer, vektor_buffer, m_buffer, res_buffer)

    cl.enqueue_copy(queue, res_np, res_buffer)

    return res_np  # shape = (n,m,3)


def rotacijska_matrika(telo, k):
    """

    :param telo:
    :param k:
    :return:
    """
    zk = telo.face_normals[k]
    yk = telo.triangles[k, 2] - telo.triangles[k, 0]
    yk /= np.linalg.norm(yk)
    xk = np.cross(yk, zk)

    a___ = np.vstack((xk, yk, zk))  # trije vektorji en nad drugim
    return np.linalg.inv(a___)


def inverzna_rotacijska_matrika(telo, k):
    """

    :param telo:
    :param k:
    :return:
    """
    zk = telo.face_normals[k]
    yk = telo.triangles[k, 2] - telo.triangles[k, 0]
    yk /= np.linalg.norm(yk)
    xk = np.cross(yk, zk)

    a___ = np.vstack((xk, yk, zk))  # trije vektorji en nad drugim
    return a___
