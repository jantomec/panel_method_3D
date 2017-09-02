# coding=utf-8

import numpy as np
import sys

from astropy.wcs.docstrings import SingularMatrix

import racunska_orodja
from programska_orodja import korak, send_ifft_notification, send_message
import trimesh
import os
from vrtincna_sled import vrtincnaSled
import pyopencl as cl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
import time
import copy
# warnings.filterwarnings(action="error", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", message="Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.")

# ############### DOLOITEV KORAKOV IZRAČUNA ################
koraki = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # vsi koraki
# koraki = [9, 10]  # brez kutta
# koraki = [2, 5, 7, 9, 10]  # za dodan/spremenjen kuttov pogoj
# koraki = [4, 5, 9, 10]  # za spremenjen kot vetra
# koraki = [11]  # samo izris

# ############### VHODNI PARAMTERI ################
hitrost_vetra = 10
kot_alfa = 5

# ############### NASTAVITVE ################
snd_msg = False
snd_ifft = False
izris_vektrosko_polje = False
f = 10  # vsak 5 vektor se izriše
vl = 3
dolzina_vrtincne_sledi = 300
kot_vrticne_sledi = 3.804 * np.pi / 180  # kot glede na x os v radianih v smeri -z osi -- če 360, potem se avtomatsko izračuna simetralni kot (manjše težave lahko povzročajo slabo definirani robovi
predznak_kutta_1 = -1
predznak_kutta_2 = -1
stopnja_kutta = -2
utezen_kutta = False

# ############### STL MODEL ################
# telon = trimesh.load_mesh("models/elipticno_krilo/elkr22.stl")
telon = trimesh.load_mesh("models/ravno_krilo/rakr15.stl")
# telon = trimesh.load_mesh("/Users/jantomec/Documents/FS/6. semester/Zakljucna naloga/Python/models/krogla.stl")

telon.n = len(telon.triangles)
veterd = hitrost_vetra * np.array([np.cos(kot_alfa * np.pi / 180), 0, np.sin(kot_alfa * np.pi / 180)])


def main(kor):

    [matrika_konstant_racunaj,  # 1
     matrika_konstant_dopolni_racunaj,  # 2
     desna_matrika_racunaj,  # 3
     izviri_racunaj,  # 4
     dvojci_racunaj,  # 5
     potenciali_matrika_konstant,  # 6
     potenciali_matrika_konstant_dopolni,  # 7
     potenciali_desna_matrika,  # 8
     izracun_potencialov_racunaj,  # 9
     izracun_hitrosti,  # 10
     izris], text = korak(št=kor)  # 11

    print("\n")
    print(text)

    mejni_kot_1 = np.pi * 3 / 4
    mejni_kot_2 = np.pi
    stopnja_utezi = -2
    dr = 0.01
    dx0 = 350
    max_buffer = 850
    vrtincna_sled = vrtincnaSled(telo=telon, l=dolzina_vrtincne_sledi, kot_theta=kot_vrticne_sledi, mejni_kot_1=mejni_kot_1, mejni_kot_2=mejni_kot_2)  # l je dolžina vrtinčne sledi - predlagano je 30 * tetiva
    platform = cl.get_platforms()[0]
    my_gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
    my_cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)

    print("\n")
    print("Število panelov na telesu:", telon.n)
    print("Število panelov na vrtinčni sledi:", vrtincna_sled.n)
    print("\n")
    print("Uporabljene naprave za računanje:")
    for i in range(len(my_cpu_devices)):
        print("    -", my_cpu_devices[i].name)
    for i in range(len(my_gpu_devices)):
        print("    -", my_gpu_devices[i].name)
    print("\n")

    if np.array([matrika_konstant_racunaj, matrika_konstant_dopolni_racunaj, desna_matrika_racunaj, izviri_racunaj, dvojci_racunaj]).any():
        matrika_konstant = np.zeros(shape=(telon.n, telon.n))
        desna_matrika = np.zeros(shape=(telon.n, telon.n))

        dirichletove_tocke = telon.triangles_center - telon.face_normals * dr
        lokalne_dirichletove_tocke = racunska_orodja.transformacija(paneli=np.arange(0, telon.n, 1),
                                                                    tocke=dirichletove_tocke,
                                                                    telo=telon,
                                                                    dtype=np.float64)
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

        if matrika_konstant_racunaj:
            print("Računam matriko konstant...")

            time0 = time.time()

            for in1 in range(telon.n // max_buffer + 1):
                for in2 in range(telon.n // max_buffer + 1):
                    if in1 == telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # čisti korner
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer  # število točk v osnovnem koordinatnem sistemu
                        nt = telon.n - in2 * max_buffer  # število trikotnikov (= število koordinatnih sistemov)

                        x_np = np.ones((n, nt))
                        y_np = np.ones((n, nt))
                        z_np = np.ones((n, nt))
                        x_np = lokalne_dirichletove_tocke[in1*max_buffer:, in2*max_buffer:, 0]
                        y_np = lokalne_dirichletove_tocke[in1*max_buffer:, in2*max_buffer:, 1]
                        z_np = lokalne_dirichletove_tocke[in1*max_buffer:, in2*max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2*max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2*max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2*max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME

                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                        __kernel void dvojci(
                            __global const double* x_b, 
                            __global const double* y_b,
                            __global const double* z_b,
                            __global const double* x2_b,
                            __global const double* y2_b,
                            __global const double* y3_b,
                            __global const double* x0_b,
                            __global const int* m_b,
                            __global const int* nt_b,
                            __global double* res_buffer)
                        {
                            int i = get_global_id(2);
                            int j = get_global_id(1);
                            int k = get_global_id(0);
                            
                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                            (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                            ( \
                                (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                    (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                    (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                ) \
                            ) - \
                            (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                            ( \
                                (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                    2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                    pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                ) \
                            );
                        }
                        """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                __kernel void dvojci(
                                                    __global const int* m_b, 
                                                    __global const int* nt_b,
                                                    __global const double* delta_b,
                                                    __global const double* f_b,
                                                    __global const double* x0_b,
                                                    __global double* res_buffer)
                                                {
                                                    int i = get_global_id(2);
                                                    int j = get_global_id(1);
                                                    int k = get_global_id(0);
    
                                                    double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                    prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                    drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                    koef_a = drugi_odvod / 2.;
                                                    koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                    koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                    f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                    f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                    koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                    (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                }
                                                """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        matrika_konstant[in1*max_buffer:, in2*max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 == telon.n // max_buffer and in2 != telon.n // max_buffer:
                        # spodnji rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1*max_buffer:, in2*max_buffer:(in2+1)*max_buffer, 0]
                        y_np = lokalne_dirichletove_tocke[in1*max_buffer:, in2*max_buffer:(in2+1)*max_buffer, 1]
                        z_np = lokalne_dirichletove_tocke[in1*max_buffer:, in2*max_buffer:(in2+1)*max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2*max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2*max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2*max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
    
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                        (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                ) - \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                        pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                __kernel void dvojci(
                                                    __global const int* m_b, 
                                                    __global const int* nt_b,
                                                    __global const double* delta_b,
                                                    __global const double* f_b,
                                                    __global const double* x0_b,
                                                    __global double* res_buffer)
                                                {
                                                    int i = get_global_id(2);
                                                    int j = get_global_id(1);
                                                    int k = get_global_id(0);
    
                                                    double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                    prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                    drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                    koef_a = drugi_odvod / 2.;
                                                    koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                    koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                    f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                    f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                    koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                    (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                }
                                                """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        matrika_konstant[in1*max_buffer:, in2*max_buffer:(in2+1)*max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 != telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # desni rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = telon.n - in2 * max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1*max_buffer:(in1+1)*max_buffer, in2*max_buffer:, 0]
                        y_np = lokalne_dirichletove_tocke[in1*max_buffer:(in1+1)*max_buffer, in2*max_buffer:, 1]
                        z_np = lokalne_dirichletove_tocke[in1*max_buffer:(in1+1)*max_buffer, in2*max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2*max_buffer:(in2+1)*max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2*max_buffer:(in2+1)*max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2*max_buffer:(in2+1)*max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
    
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                        (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                ) - \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                        pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                __kernel void dvojci(
                                                    __global const int* m_b, 
                                                    __global const int* nt_b,
                                                    __global const double* delta_b,
                                                    __global const double* f_b,
                                                    __global const double* x0_b,
                                                    __global double* res_buffer)
                                                {
                                                    int i = get_global_id(2);
                                                    int j = get_global_id(1);
                                                    int k = get_global_id(0);
    
                                                    double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                    prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                    drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                    koef_a = drugi_odvod / 2.;
                                                    koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                    koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                    f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                    f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                    koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                    (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                }
                                                """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        matrika_konstant[in1*max_buffer:(in1+1)*max_buffer, in2*max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    else:
                        # vse ostalo
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)
                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
    
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                        (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                ) - \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                        pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                __kernel void dvojci(
                                                    __global const int* m_b, 
                                                    __global const int* nt_b,
                                                    __global const double* delta_b,
                                                    __global const double* f_b,
                                                    __global const double* x0_b,
                                                    __global double* res_buffer)
                                                {
                                                    int i = get_global_id(2);
                                                    int j = get_global_id(1);
                                                    int k = get_global_id(0);
        
                                                    double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
        
                                                    prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                    drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                    4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                    1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
                                                    
                                                    koef_a = drugi_odvod / 2.;
                                                    koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                    koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                    f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                    f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
        
                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                    koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                    (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                    koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                    koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                }
                                                """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)
                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

            matrika_konstant = matrika_konstant / (4 * np.pi)
            print("\nMatrika konstant je izracunana!")
            print("Shranjujem...")
            np.save("vmesni_rezultati/matrika_konstant", matrika_konstant)
            print("Končano!")
            print("\n")

        if matrika_konstant_dopolni_racunaj:
            if vrtincna_sled.n == 0:
                print("Ni definirane vrtinčne sledi.")
                print("\n")
            print("Berem matriko konstant...")
            matrika_konstant = np.load("vmesni_rezultati/matrika_konstant.npy")

            oglisca_wake_1 = vrtincna_sled.triangles[:, 0]
            oglisca_wake_2 = vrtincna_sled.triangles[:, 1]
            oglisca_wake_3 = vrtincna_sled.triangles[:, 2]
            lokalna_oglisca_1 = np.zeros_like(oglisca_wake_1)
            lokalna_oglisca_2 = racunska_orodja.transformacija_2(paneli=np.arange(0, vrtincna_sled.n, 1),
                                                                 tocke=oglisca_wake_2,
                                                                 telo=vrtincna_sled,
                                                                 dtype=np.float64)
            lokalna_oglisca_3 = racunska_orodja.transformacija_2(paneli=np.arange(0, vrtincna_sled.n, 1),
                                                                 tocke=oglisca_wake_3,
                                                                 telo=vrtincna_sled,
                                                                 dtype=np.float64)
            lokalne_dirichletove_tocke = racunska_orodja.transformacija(paneli=np.arange(0, vrtincna_sled.n, 1),
                                                                        tocke=dirichletove_tocke,
                                                                        telo=vrtincna_sled,
                                                                        dtype=np.float64)

            time0 = time.time()

            print("Dopolnjujem matriko konstant s Kuttovim pogojem...")

            # po spremembi:
            if utezen_kutta:
                paneli_na_zgornji_strani = telon.face_normals[:, 2] >= 0
                paneli_na_spodnji_strani = telon.face_normals[:, 2] < 0

                tocke_na_sredini_robov = []
                for ki in range(vrtincna_sled.n // 2):
                    tocke_na_sredini_robov.append((vrtincna_sled.triangles[2 * ki][0] + vrtincna_sled.triangles[2 * ki][1]) / 2)
                    if vrtincna_sled.triangles[2 * ki][0][0] > dolzina_vrtincne_sledi:
                        print(vrtincna_sled.triangles[2 * ki])
                        print("Določevanje točk na robovih za kutto ne deluje!")
                        sys.exit()
                    if vrtincna_sled.triangles[2 * ki][1][0] > dolzina_vrtincne_sledi:
                        print(vrtincna_sled.triangles[2 * ki])
                        print("Določevanje točk na robovih za kutto ne deluje!")
                        sys.exit()

                utezi_zg_kutta = []
                utezi_sp_kutta = []
                for ki in range(vrtincna_sled.n // 2):
                    utezi_zg_kutta_i = np.power(np.linalg.norm(telon.triangles_center[paneli_na_zgornji_strani] - tocke_na_sredini_robov[ki], axis=-1), stopnja_kutta)  # od točke na sredini robu pa do vsakega panela na telesu
                    utezi_sp_kutta_i = np.power(np.linalg.norm(telon.triangles_center[paneli_na_spodnji_strani] - tocke_na_sredini_robov[ki], axis=-1), stopnja_kutta)
                    utezi_zg_kutta.append(utezi_zg_kutta_i)
                    utezi_sp_kutta.append(utezi_sp_kutta_i)

                utezi_zg_kutta = np.array(utezi_zg_kutta)
                utezi_sp_kutta = np.array(utezi_sp_kutta)

            # konec po spremembi

            for in1 in range(telon.n // max_buffer + 1):
                for in2 in range(vrtincna_sled.n // max_buffer + 1):
                    if in1 == telon.n // max_buffer and in2 == vrtincna_sled.n // max_buffer:
                        # čisti korner
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer  # število točk v osnovnem koordinatnem sistemu
                        nt = vrtincna_sled.n - in2 * max_buffer  # število trikotnikov (= število koordinatnih sistemov)

                        x_np = np.ones((n, nt))
                        y_np = np.ones((n, nt))
                        z_np = np.ones((n, nt))
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME

                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
    
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                        (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                ) - \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                        pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:, paneli_na_zgornji_strani] -= predznak_kutta_1 * utezene_zg_kuttove_vrednosti / (4 * np.pi)
                                matrika_konstant[in1 * max_buffer:, paneli_na_spodnji_strani] += predznak_kutta_1 * utezene_sp_kuttove_vrednosti / (4 * np.pi)

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (vrtincna_sled.n // max_buffer + 1) + in2) / ((telon.n // max_buffer + 1) * (vrtincna_sled.n // max_buffer + 1))
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 == telon.n // max_buffer and in2 != vrtincna_sled.n // max_buffer:
                        # spodnji rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                __kernel void dvojci(
                                                                    __global const double* x_b, 
                                                                    __global const double* y_b,
                                                                    __global const double* z_b,
                                                                    __global const double* x2_b,
                                                                    __global const double* y2_b,
                                                                    __global const double* y3_b,
                                                                    __global const double* x0_b,
                                                                    __global const int* m_b,
                                                                    __global const int* nt_b,
                                                                    __global double* res_buffer)
                                                                {
                                                                    int i = get_global_id(2);
                                                                    int j = get_global_id(1);
                                                                    int k = get_global_id(0);
    
                                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    ) - \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    );
                                                                }
                                                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:, paneli_na_zgornji_strani] -= predznak_kutta_1 * utezene_zg_kuttove_vrednosti / (4 * np.pi)
                                matrika_konstant[in1 * max_buffer:, paneli_na_spodnji_strani] += predznak_kutta_1 * utezene_sp_kuttove_vrednosti / (4 * np.pi)

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (vrtincna_sled.n // max_buffer + 1) + in2) / ((telon.n // max_buffer + 1) * (vrtincna_sled.n // max_buffer + 1))
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 != telon.n // max_buffer and in2 == vrtincna_sled.n // max_buffer:
                        # desni rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = vrtincna_sled.n - in2 * max_buffer  # število trikotnikov (= število koordinatnih sistemov)
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                __kernel void dvojci(
                                                                    __global const double* x_b, 
                                                                    __global const double* y_b,
                                                                    __global const double* z_b,
                                                                    __global const double* x2_b,
                                                                    __global const double* y2_b,
                                                                    __global const double* y3_b,
                                                                    __global const double* x0_b,
                                                                    __global const int* m_b,
                                                                    __global const int* nt_b,
                                                                    __global double* res_buffer)
                                                                {
                                                                    int i = get_global_id(2);
                                                                    int j = get_global_id(1);
                                                                    int k = get_global_id(0);
    
                                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    ) - \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    );
                                                                }
                                                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_zgornji_strani] -= predznak_kutta_1 * utezene_zg_kuttove_vrednosti / (4 * np.pi)
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_spodnji_strani] += predznak_kutta_1 * utezene_sp_kuttove_vrednosti / (4 * np.pi)

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (vrtincna_sled.n // max_buffer + 1) + in2) / ((telon.n // max_buffer + 1) * (vrtincna_sled.n // max_buffer + 1))
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    else:
                        # vse ostalo
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                __kernel void dvojci(
                                                                    __global const double* x_b, 
                                                                    __global const double* y_b,
                                                                    __global const double* z_b,
                                                                    __global const double* x2_b,
                                                                    __global const double* y2_b,
                                                                    __global const double* y3_b,
                                                                    __global const double* x0_b,
                                                                    __global const int* m_b,
                                                                    __global const int* nt_b,
                                                                    __global double* res_buffer)
                                                                {
                                                                    int i = get_global_id(2);
                                                                    int j = get_global_id(1);
                                                                    int k = get_global_id(0);
    
                                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    ) - \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    );
                                                                }
                                                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)
                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_1 * kuttove_vrednosti[:, i] / (4 * np.pi)  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_zgornji_strani] -= predznak_kutta_1 * utezene_zg_kuttove_vrednosti / (4 * np.pi)
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_spodnji_strani] += predznak_kutta_1 * utezene_sp_kuttove_vrednosti / (4 * np.pi)

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (vrtincna_sled.n // max_buffer + 1) + in2) / ((telon.n // max_buffer + 1) * (vrtincna_sled.n // max_buffer + 1))
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

            print("\nMatrika konstant je dopolnjena!")
            print("Shranjujem...")
            np.save("vmesni_rezultati/matrika_konstant_dopolnjeno", matrika_konstant)
            print("Končano!")
            print("\n")

        if desna_matrika_racunaj:
            print("Računam desno matriko...")
            time0 = time.time()

            for in1 in range(telon.n // max_buffer + 1):
                for in2 in range(telon.n // max_buffer + 1):
                    if in1 == telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # čisti korner
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer  # število točk v osnovnem koordinatnem sistemu
                        nt = telon.n - in2 * max_buffer  # število trikotnikov (= število koordinatnih sistemov)

                        x_np = np.ones((n, nt))
                        y_np = np.ones((n, nt))
                        z_np = np.ones((n, nt))
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME

                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                        __kernel void dvojci(
                            __global const double* x_b, 
                            __global const double* y_b,
                            __global const double* z_b,
                            __global const double* x2_b,
                            __global const double* y2_b,
                            __global const double* y3_b,
                            __global const double* x0_b,
                            __global const int* m_b,
                            __global const int* nt_b,
                            __global double* res_buffer)
                        {
                            int i = get_global_id(2);
                            int j = get_global_id(1);
                            int k = get_global_id(0);
                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            ) - \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            );
                        }
                        """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        desna_matrika[in1 * max_buffer:, in2 * max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 == telon.n // max_buffer and in2 != telon.n // max_buffer:
                        # spodnji rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                        __kernel void dvojci(
                            __global const double* x_b, 
                            __global const double* y_b,
                            __global const double* z_b,
                            __global const double* x2_b,
                            __global const double* y2_b,
                            __global const double* y3_b,
                            __global const double* x0_b,
                            __global const int* m_b,
                            __global const int* nt_b,
                            __global double* res_buffer)
                        {
                            int i = get_global_id(2);
                            int j = get_global_id(1);
                            int k = get_global_id(0);
                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            ) - \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            );
                        }
                        """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        desna_matrika[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 != telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # desni rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = telon.n - in2 * max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                        __kernel void dvojci(
                            __global const double* x_b, 
                            __global const double* y_b,
                            __global const double* z_b,
                            __global const double* x2_b,
                            __global const double* y2_b,
                            __global const double* y3_b,
                            __global const double* x0_b,
                            __global const int* m_b,
                            __global const int* nt_b,
                            __global double* res_buffer)
                        {
                            int i = get_global_id(2);
                            int j = get_global_id(1);
                            int k = get_global_id(0);
                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            ) - \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            );
                        }
                        """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        desna_matrika[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    else:
                        # vse ostalo
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_dirichletove_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                        __kernel void dvojci(
                            __global const double* x_b, 
                            __global const double* y_b,
                            __global const double* z_b,
                            __global const double* x2_b,
                            __global const double* y2_b,
                            __global const double* y3_b,
                            __global const double* x0_b,
                            __global const int* m_b,
                            __global const int* nt_b,
                            __global double* res_buffer)
                        {
                            int i = get_global_id(2);
                            int j = get_global_id(1);
                            int k = get_global_id(0);
                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            ) - \
                            log( \
                                y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                sqrt( \
                                    pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                ) \
                            );
                        }
                        """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)
                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        desna_matrika[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

            desna_matrika = desna_matrika / (4 * np.pi)
            print("\nDesna matrika je izracunana!")
            print("Shranjujem...")
            np.save("vmesni_rezultati/desna_matrika", desna_matrika)
            print("Končano!")
            print("\n")

        if izviri_racunaj:
            print("Računam izvire...")
            nx_np = telon.face_normals[:, 0].astype(np.float64)
            ny_np = telon.face_normals[:, 1].astype(np.float64)
            nz_np = telon.face_normals[:, 2].astype(np.float64)
            vx_np = np.array([veterd[0]], dtype=np.float64)
            vy_np = np.array([veterd[1]], dtype=np.float64)
            vz_np = np.array([veterd[2]], dtype=np.float64)
            nx_np = nx_np.astype(np.float64)
            ny_np = ny_np.astype(np.float64)
            nz_np = nz_np.astype(np.float64)
            vx_np = vx_np.astype(np.float64)
            vy_np = vy_np.astype(np.float64)
            vz_np = vz_np.astype(np.float64)

            ctx = cl.Context(devices=my_gpu_devices)
            queue = cl.CommandQueue(ctx)

            mf = cl.mem_flags
            nx_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nx_np)
            ny_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ny_np)
            nz_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nz_np)
            vx_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vx_np)
            vy_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vy_np)
            vz_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vz_np)

            prg = cl.Program(ctx, """
                                __kernel void dvojci(
                                    __global const double* nx_b, 
                                    __global const double* ny_b,
                                    __global const double* nz_b,
                                    __global const double* vx_b,
                                    __global const double* vy_b,
                                    __global const double* vz_b,
                                    __global double* res_buffer)
                                {
                                    int i = get_global_id(0);                                
                                    res_buffer[i] = nx_b[i] * vx_b[0] + ny_b[i] * vy_b[0] + nz_b[i] * vz_b[0];
                                    
                                }
                                """).build()

            res_np = np.zeros(telon.n, dtype=np.float64)
            res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

            prg.dvojci(queue, res_np.shape, None, nx_b, ny_b, nz_b, vx_b, vy_b, vz_b, res_buffer).wait()

            cl.enqueue_copy(queue, res_np, res_buffer).wait()
            cl.tools.clear_first_arg_caches()

            izviri = -res_np  # DODAL MINUS ZARADI USMERJENOSTI NORMAL

            print("Shranjujem...")
            np.save("vmesni_rezultati/izviri", izviri)
            print("Končano!")
            print("\n")

        if dvojci_racunaj:
            print("Berem matriko konstant...")
            matrika_konstant = np.load("vmesni_rezultati/matrika_konstant_dopolnjeno.npy")
            print("Berem desno matriko...")
            desna_matrika = np.load("vmesni_rezultati/desna_matrika.npy")
            print("Berem izvire...")
            izviri = np.load("vmesni_rezultati/izviri.npy")

            print("Izračun dvojcev...")
            desna_matrika_np = desna_matrika.astype(np.float64)
            izviri_np = izviri.astype(np.float64)

            dm = np.dot(desna_matrika_np, izviri_np)
            dvojci = np.linalg.solve(matrika_konstant, dm)
            print("Shranjujem...")
            np.save("vmesni_rezultati/dvojci", dvojci)
            print("Končano!")
            print("\n")

    if np.array([potenciali_matrika_konstant, potenciali_matrika_konstant_dopolni, potenciali_desna_matrika, izracun_potencialov_racunaj]).any():

        if potenciali_matrika_konstant:
            # matrika_konstant
            matrika_konstant = np.zeros(shape=(telon.n, telon.n))

            racunske_tocke = telon.triangles_center + telon.face_normals * dr
            lokalne_racunske_tocke = racunska_orodja.transformacija(paneli=np.arange(0, telon.n, 1),
                                                                    tocke=racunske_tocke,
                                                                    telo=telon,
                                                                    dtype=np.float64)
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

            print("Računam matriko konstant...")
            time0 = time.time()

            for in1 in range(telon.n // max_buffer + 1):
                for in2 in range(telon.n // max_buffer + 1):
                    if in1 == telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # čisti korner
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer  # število točk v osnovnem koordinatnem sistemu
                        nt = telon.n - in2 * max_buffer  # število trikotnikov (= število koordinatnih sistemov)

                        x_np = np.ones((n, nt))
                        y_np = np.ones((n, nt))
                        z_np = np.ones((n, nt))
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME

                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
    
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                        (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                ) - \
                                                (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                ( \
                                                    (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                        2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                        pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        matrika_konstant[in1 * max_buffer:, in2 * max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 == telon.n // max_buffer and in2 != telon.n // max_buffer:
                        # spodnji rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                __kernel void dvojci(
                                                                    __global const double* x_b, 
                                                                    __global const double* y_b,
                                                                    __global const double* z_b,
                                                                    __global const double* x2_b,
                                                                    __global const double* y2_b,
                                                                    __global const double* y3_b,
                                                                    __global const double* x0_b,
                                                                    __global const int* m_b,
                                                                    __global const int* nt_b,
                                                                    __global double* res_buffer)
                                                                {
                                                                    int i = get_global_id(2);
                                                                    int j = get_global_id(1);
                                                                    int k = get_global_id(0);
    
                                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    ) - \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    );
                                                                }
                                                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        matrika_konstant[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 != telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # desni rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = telon.n - in2 * max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                __kernel void dvojci(
                                                                    __global const double* x_b, 
                                                                    __global const double* y_b,
                                                                    __global const double* z_b,
                                                                    __global const double* x2_b,
                                                                    __global const double* y2_b,
                                                                    __global const double* y3_b,
                                                                    __global const double* x0_b,
                                                                    __global const int* m_b,
                                                                    __global const int* nt_b,
                                                                    __global double* res_buffer)
                                                                {
                                                                    int i = get_global_id(2);
                                                                    int j = get_global_id(1);
                                                                    int k = get_global_id(0);
    
                                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    ) - \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    );
                                                                }
                                                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    else:
                        # vse ostalo
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                __kernel void dvojci(
                                                                    __global const double* x_b, 
                                                                    __global const double* y_b,
                                                                    __global const double* z_b,
                                                                    __global const double* x2_b,
                                                                    __global const double* y2_b,
                                                                    __global const double* y3_b,
                                                                    __global const double* x0_b,
                                                                    __global const int* m_b,
                                                                    __global const int* nt_b,
                                                                    __global double* res_buffer)
                                                                {
                                                                    int i = get_global_id(2);
                                                                    int j = get_global_id(1);
                                                                    int k = get_global_id(0);
    
                                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    ) - \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    );
                                                                }
                                                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                    __kernel void dvojci(
                                                                        __global const int* m_b, 
                                                                        __global const int* nt_b,
                                                                        __global const double* delta_b,
                                                                        __global const double* f_b,
                                                                        __global const double* x0_b,
                                                                        __global double* res_buffer)
                                                                    {
                                                                        int i = get_global_id(2);
                                                                        int j = get_global_id(1);
                                                                        int k = get_global_id(0);
    
                                                                        double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                        prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                        drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                        4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                        1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                        koef_a = drugi_odvod / 2.;
                                                                        koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                        koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                        f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                        f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                        koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                        (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                        koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                        koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                    }
                                                                    """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)
                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

            print("\nMatrika konstant je izracunana!")
            print("Shranjujem...")
            np.save("vmesni_rezultati/potenciali_matrika_konstant", matrika_konstant)
            print("Končano!")
            print("\n")

        if potenciali_matrika_konstant_dopolni:
            if vrtincna_sled.n == 0:
                print("Ni definirane vrtinčne sledi.")
                print("\n")

            print("Berem matriko konstant...")
            matrika_konstant = np.load("vmesni_rezultati/potenciali_matrika_konstant.npy")

            racunske_tocke = telon.triangles_center + telon.face_normals * dr

            oglisca_wake_1 = vrtincna_sled.triangles[:, 0]
            oglisca_wake_2 = vrtincna_sled.triangles[:, 1]
            oglisca_wake_3 = vrtincna_sled.triangles[:, 2]
            lokalna_oglisca_1 = np.zeros_like(oglisca_wake_1)
            lokalna_oglisca_2 = racunska_orodja.transformacija_2(paneli=np.arange(0, vrtincna_sled.n, 1),
                                                                 tocke=oglisca_wake_2,
                                                                 telo=vrtincna_sled,
                                                                 dtype=np.float64)
            lokalna_oglisca_3 = racunska_orodja.transformacija_2(paneli=np.arange(0, vrtincna_sled.n, 1),
                                                                 tocke=oglisca_wake_3,
                                                                 telo=vrtincna_sled,
                                                                 dtype=np.float64)
            lokalne_racunske_tocke = racunska_orodja.transformacija(paneli=np.arange(0, vrtincna_sled.n, 1),
                                                                    tocke=racunske_tocke,
                                                                    telo=vrtincna_sled,
                                                                    dtype=np.float64)

            print("Dopolnjujem matriko konstant s Kuttovim pogojem...")

            # po spremembi:
            if utezen_kutta:
                paneli_na_zgornji_strani = telon.face_normals[:, 2] >= 0
                paneli_na_spodnji_strani = telon.face_normals[:, 2] < 0

                tocke_na_sredini_robov = []
                for ki in range(vrtincna_sled.n // 2):
                    tocke_na_sredini_robov.append((vrtincna_sled.triangles[2 * ki][0] + vrtincna_sled.triangles[2 * ki][1]) / 2)
                    if vrtincna_sled.triangles[2 * ki][0][0] > dolzina_vrtincne_sledi:
                        print(vrtincna_sled.triangles[2 * ki])
                        print("Določevanje točk na robovih za kutto ne deluje!")
                        sys.exit()
                    if vrtincna_sled.triangles[2 * ki][1][0] > dolzina_vrtincne_sledi:
                        print(vrtincna_sled.triangles[2 * ki])
                        print("Določevanje točk na robovih za kutto ne deluje!")
                        sys.exit()

                utezi_zg_kutta = []
                utezi_sp_kutta = []
                for ki in range(vrtincna_sled.n // 2):
                    utezi_zg_kutta_i = np.power(np.linalg.norm(telon.triangles_center[paneli_na_zgornji_strani] - tocke_na_sredini_robov[ki], axis=-1), stopnja_kutta)  # od točke na sredini robu pa do vsakega panela na telesu
                    utezi_sp_kutta_i = np.power(np.linalg.norm(telon.triangles_center[paneli_na_spodnji_strani] - tocke_na_sredini_robov[ki], axis=-1), stopnja_kutta)
                    utezi_zg_kutta.append(utezi_zg_kutta_i)
                    utezi_sp_kutta.append(utezi_sp_kutta_i)

                utezi_zg_kutta = np.array(utezi_zg_kutta)
                utezi_sp_kutta = np.array(utezi_sp_kutta)

            # konec po spremembi

            time0 = time.time()

            for in1 in range(telon.n // max_buffer + 1):
                for in2 in range(vrtincna_sled.n // max_buffer + 1):
                    if in1 == telon.n // max_buffer and in2 == vrtincna_sled.n // max_buffer:
                        # čisti korner
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer  # število točk v osnovnem koordinatnem sistemu
                        nt = vrtincna_sled.n - in2 * max_buffer  # število trikotnikov (= število koordinatnih sistemov)

                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME

                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                __kernel void dvojci(
                                                                    __global const double* x_b, 
                                                                    __global const double* y_b,
                                                                    __global const double* z_b,
                                                                    __global const double* x2_b,
                                                                    __global const double* y2_b,
                                                                    __global const double* y3_b,
                                                                    __global const double* x0_b,
                                                                    __global const int* m_b,
                                                                    __global const int* nt_b,
                                                                    __global double* res_buffer)
                                                                {
                                                                    int i = get_global_id(2);
                                                                    int j = get_global_id(1);
                                                                    int k = get_global_id(0);
    
                                                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    ) - \
                                                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                    ( \
                                                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                        sqrt( \
                                                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                        ) \
                                                                    );
                                                                }
                                                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2  # shape = (nb, nw)
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:, paneli_na_zgornji_strani] -= predznak_kutta_2 * utezene_zg_kuttove_vrednosti
                                matrika_konstant[in1 * max_buffer:, paneli_na_spodnji_strani] += predznak_kutta_2 * utezene_sp_kuttove_vrednosti

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 == telon.n // max_buffer and in2 != vrtincna_sled.n // max_buffer:
                        # spodnji rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                                    __kernel void dvojci(
                                                                                        __global const double* x_b, 
                                                                                        __global const double* y_b,
                                                                                        __global const double* z_b,
                                                                                        __global const double* x2_b,
                                                                                        __global const double* y2_b,
                                                                                        __global const double* y3_b,
                                                                                        __global const double* x0_b,
                                                                                        __global const int* m_b,
                                                                                        __global const int* nt_b,
                                                                                        __global double* res_buffer)
                                                                                    {
                                                                                        int i = get_global_id(2);
                                                                                        int j = get_global_id(1);
                                                                                        int k = get_global_id(0);
    
                                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                                        (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                                        ( \
                                                                                            (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                                            sqrt( \
                                                                                                pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                                                (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                                                (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                                            ) \
                                                                                        ) - \
                                                                                        (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                                        ( \
                                                                                            (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                                            sqrt( \
                                                                                                pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                                                2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                                                pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                                            ) \
                                                                                        );
                                                                                    }
                                                                                    """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:, paneli_na_zgornji_strani] -= predznak_kutta_2 * utezene_zg_kuttove_vrednosti
                                matrika_konstant[in1 * max_buffer:, paneli_na_spodnji_strani] += predznak_kutta_2 * utezene_sp_kuttove_vrednosti

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 != telon.n // max_buffer and in2 == vrtincna_sled.n // max_buffer:
                        # desni rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = vrtincna_sled.n - in2 * max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                                                                    __kernel void dvojci(
                                                                                        __global const double* x_b, 
                                                                                        __global const double* y_b,
                                                                                        __global const double* z_b,
                                                                                        __global const double* x2_b,
                                                                                        __global const double* y2_b,
                                                                                        __global const double* y3_b,
                                                                                        __global const double* x0_b,
                                                                                        __global const int* m_b,
                                                                                        __global const int* nt_b,
                                                                                        __global double* res_buffer)
                                                                                    {
                                                                                        int i = get_global_id(2);
                                                                                        int j = get_global_id(1);
                                                                                        int k = get_global_id(0);
    
                                                                                        res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                                                        (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                                        ( \
                                                                                            (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                                            sqrt( \
                                                                                                pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                                                (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                                                                                (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                                            ) \
                                                                                        ) - \
                                                                                        (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                                                                        ( \
                                                                                            (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                                                                            sqrt( \
                                                                                                pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                                                                                2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                                                                                pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                                                                            ) \
                                                                                        );
                                                                                    }
                                                                                    """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_zgornji_strani] -= predznak_kutta_2 * utezene_zg_kuttove_vrednosti
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_spodnji_strani] += predznak_kutta_2 * utezene_sp_kuttove_vrednosti

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    else:
                        # vse ostalo
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                __kernel void dvojci(
                                    __global const double* x_b, 
                                    __global const double* y_b,
                                    __global const double* z_b,
                                    __global const double* x2_b,
                                    __global const double* y2_b,
                                    __global const double* y3_b,
                                    __global const double* x0_b,
                                    __global const int* m_b,
                                    __global const int* nt_b,
                                    __global double* res_buffer)
                                {
                                    int i = get_global_id(2);
                                    int j = get_global_id(1);
                                    int k = get_global_id(0);
                                    res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * y2_b[j]) / x2_b[j]) * z_b[j+nt_b[0]*k] / \
                                    ( \
                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                        sqrt( \
                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                            (2 * x0_b[i+j*m_b[0]] * y_b[j+nt_b[0]*k] * y2_b[j]) / x2_b[j] + \
                                            (pow(x0_b[i+j*m_b[0]], 2) * pow(y2_b[j], 2) / pow(x2_b[j], 2)) + pow(z_b[j+nt_b[0]*k], 2) \
                                        ) \
                                    ) - \
                                    (-y_b[j+nt_b[0]*k] + (x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) * z_b[j+nt_b[0]*k] / \
                                    ( \
                                        (pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(z_b[j+nt_b[0]*k], 2)) * \
                                        sqrt( \
                                            pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k], 2) - \
                                            2 * y_b[j+nt_b[0]*k] * ((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j]) + \
                                            pow((x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j])) / x2_b[j] + y3_b[j], 2) + pow(z_b[j+nt_b[0]*k], 2) \
                                        ) \
                                    );
                                }
                                """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)
                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        kuttove_vrednosti = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2
                        kuttove_vrednosti = kuttove_vrednosti[:, ::2] + kuttove_vrednosti[:, 1::2]  # seštejem panelne pare -> shape = (nb, nw/2)

                        # pred spremembo:
                        if not utezen_kutta:
                            for i in range(len(kuttove_vrednosti[0])):
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 0]] -= predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, vrtincna_sled.pari_robnih_panelov[i + in2 * max_buffer, 1]] += predznak_kutta_2 * kuttove_vrednosti[:, i]  # SPREMEMBA

                        # konec pred spremebo

                        # po spremembi:
                        else:
                            for ki in range(len(kuttove_vrednosti[0])):

                                utezene_zg_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_zg_kutta[ki] / np.sum(utezi_zg_kutta[ki])
                                utezene_sp_kuttove_vrednosti = kuttove_vrednosti[:, ki].reshape((len(kuttove_vrednosti[:, ki]), 1)) * utezi_sp_kutta[ki] / np.sum(utezi_sp_kutta[ki])

                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_zgornji_strani] -= predznak_kutta_2 * utezene_zg_kuttove_vrednosti
                                matrika_konstant[in1 * max_buffer:(in1 + 1) * max_buffer, paneli_na_spodnji_strani] += predznak_kutta_2 * utezene_sp_kuttove_vrednosti

                        # konec po spremembi

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

            print("\nMatrika konstant je izracunana!")
            print("Shranjujem...")
            np.save("vmesni_rezultati/potenciali_matrika_konstant_dopolnjeno", matrika_konstant)
            print("Končano!")
            print("\n")

        if potenciali_desna_matrika:
            desna_matrika = np.zeros(shape=(telon.n, telon.n))

            racunske_tocke = telon.triangles_center + telon.face_normals * dr
            lokalne_racunske_tocke = racunska_orodja.transformacija(paneli=np.arange(0, telon.n, 1),
                                                                    tocke=racunske_tocke,
                                                                    telo=telon,
                                                                    dtype=np.float64)
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
            print("Računam desno matriko...")

            time0 = time.time()

            for in1 in range(telon.n // max_buffer + 1):
                for in2 in range(telon.n // max_buffer + 1):
                    if in1 == telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # čisti korner
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer  # število točk v osnovnem koordinatnem sistemu
                        nt = telon.n - in2 * max_buffer  # število trikotnikov (= število koordinatnih sistemov)

                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME

                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                ) - \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        desna_matrika[in1 * max_buffer:, in2 * max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 == telon.n // max_buffer and in2 != telon.n // max_buffer:
                        # spodnji rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = telon.n - in1 * max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                ) - \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        desna_matrika[in1 * max_buffer:, in2 * max_buffer:(in2 + 1) * max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    elif in1 != telon.n // max_buffer and in2 == telon.n // max_buffer:
                        # desni rob
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = telon.n - in2 * max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                ) - \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)

                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()

                        desna_matrika[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

                    else:
                        # vse ostalo
                        # INTEGRAL V 1D ZA NT TRIKOTNIKOV IN NxNT TOČK (direktno prenosljivo v moj program)

                        n = max_buffer
                        nt = max_buffer
                        x_np = np.ones((n, nt)).astype(np.float64)
                        y_np = np.ones((n, nt)).astype(np.float64)
                        z_np = np.ones((n, nt)).astype(np.float64)
                        x_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        z_np = lokalne_racunske_tocke[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer, 2]

                        x2_np = np.ones(nt).astype(np.float64)
                        y2_np = np.ones(nt).astype(np.float64)
                        y3_np = np.ones(nt).astype(np.float64)  # to so pa meje integracije (tudi v lokalnem koordinatnem sistemu)
                        x2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 0]
                        y2_np = lokalna_oglisca_2[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        y3_np = lokalna_oglisca_3[in2 * max_buffer:(in2 + 1) * max_buffer, 1]
                        x2_np = x2_np.astype(np.float64)
                        y2_np = y2_np.astype(np.float64)
                        y3_np = y3_np.astype(np.float64)
                        x_np = x_np.astype(np.float64)
                        y_np = y_np.astype(np.float64)
                        z_np = z_np.astype(np.float64)

                        # for i__ in range(len(x2_np)):
                        #     if x2_np[i__] == 0:
                        #         x2_np[i__] = 10 ** (-4)  # DODANO!!!!! , če je nepravilen trikotnik, ga malenkostno deformiramo
                        #     if y3_np[i__] == 0:
                        #         y3_np[i__] = 10 ** (-4)

                        x0_np = np.zeros((nt, dx0 + 4))
                        for i__ in range(nt):
                            x0_np[i__, 2:-2] = np.linspace(0, x2_np[i__], dx0, dtype=np.float64)
                            x0_np[i__, 1] = -x0_np[i__, 3]
                            x0_np[i__, 0] = -2 * x0_np[i__, 3]
                            x0_np[i__, -2] = x0_np[i__, -3] + x0_np[i__, 3]
                            x0_np[i__, -1] = x0_np[i__, -3] + 2 * x0_np[i__, 3]  # POTREBNO JE PODALJŠATI OBMOČJE NA VSAKO STRAN ZA 2 TOČKI ZARADI CENTRALNE DIFERENČNE SHEME
                        x0_np = np.array(x0_np).astype(np.float64)
                        m_np = len(x0_np[0]) * np.ones(1)
                        nt_np = nt * np.ones(1)
                        m_np = m_np.astype(np.int32)
                        nt_np = nt_np.astype(np.int32)

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        x_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_np)
                        y_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_np)
                        z_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z_np)
                        x2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x2_np)
                        y2_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y2_np)
                        y3_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y3_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)

                        prg = cl.Program(ctx, """
                                            __kernel void dvojci(
                                                __global const double* x_b, 
                                                __global const double* y_b,
                                                __global const double* z_b,
                                                __global const double* x2_b,
                                                __global const double* y2_b,
                                                __global const double* y3_b,
                                                __global const double* x0_b,
                                                __global const int* m_b,
                                                __global const int* nt_b,
                                                __global double* res_buffer)
                                            {
                                                int i = get_global_id(2);
                                                int j = get_global_id(1);
                                                int k = get_global_id(0);
                                                res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * y2_b[j] / x2_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                ) - \
                                                log( \
                                                    y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j] + \
                                                    sqrt( \
                                                        pow(x_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]], 2) + pow(y_b[j+nt_b[0]*k] - x0_b[i+j*m_b[0]] * (y2_b[j] - y3_b[j]) / x2_b[j] - y3_b[j], 2) + z_b[j+nt_b[0]*k] * z_b[j+nt_b[0]*k]
                                                    ) \
                                                );
                                            }
                                            """).build()

                        res_np = np.zeros((n, nt, m_np[0]), dtype=np.float64)
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

                        prg.dvojci(queue, res_np.shape, None, x_b, y_b, z_b, x2_b, y2_b, y3_b, x0_b, m_b, nt_b, res_buffer).wait()

                        cl.enqueue_copy(queue, res_np, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        delta_np = x0_np[:, 3]
                        delta_np = delta_np.astype(np.float64)
                        res_np = res_np.astype(np.float64)
                        m_np -= 4

                        ctx = cl.Context(devices=my_gpu_devices)
                        queue = cl.CommandQueue(ctx)

                        mf = cl.mem_flags
                        m_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m_np)
                        nt_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nt_np)
                        delta_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=delta_np)
                        f_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res_np)
                        x0_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x0_np)

                        prg = cl.Program(ctx, """
                                                                                        __kernel void dvojci(
                                                                                            __global const int* m_b, 
                                                                                            __global const int* nt_b,
                                                                                            __global const double* delta_b,
                                                                                            __global const double* f_b,
                                                                                            __global const double* x0_b,
                                                                                            __global double* res_buffer)
                                                                                        {
                                                                                            int i = get_global_id(2);
                                                                                            int j = get_global_id(1);
                                                                                            int k = get_global_id(0);
    
                                                                                            double prvi_odvod, drugi_odvod, koef_a, koef_b, koef_c;
    
                                                                                            prvi_odvod = ((1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            2. / 3. *  f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            2. / 3. *  f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / delta_b[j]);
                                                                                            drugi_odvod = ((-1. / 12. * f_b[0+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            5. / 2. * f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] + \
                                                                                            4. / 3. * f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - \
                                                                                            1. / 12. * f_b[4+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]]) / pow(delta_b[j], 2));
    
                                                                                            koef_a = drugi_odvod / 2.;
                                                                                            koef_b = prvi_odvod - drugi_odvod * x0_b[2+i+j*(m_b[0]+4)];
                                                                                            koef_c = (f_b[1+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[1+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[1+i+j*(m_b[0]+4)] + \
                                                                                            f_b[2+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[2+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[2+i+j*(m_b[0]+4)] + \
                                                                                            f_b[3+i+j*(m_b[0]+4)+k*(m_b[0]+4)*nt_b[0]] - koef_a * pow(x0_b[3+i+j*(m_b[0]+4)], 2) - koef_b * x0_b[3+i+j*(m_b[0]+4)]) / 3.;
    
                                                                                            res_buffer[i+j*m_b[0]+k*m_b[0]*nt_b[0]] = ( \
                                                                                            koef_a/3 * pow(x0_b[3+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[3+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[3+i+j*(m_b[0]+4)]) - \
                                                                                            (koef_a/3 * pow(x0_b[1+i+j*(m_b[0]+4)], 3) + \
                                                                                            koef_b/2 * pow(x0_b[1+i+j*(m_b[0]+4)], 2) + \
                                                                                            koef_c * x0_b[1+i+j*(m_b[0]+4)]);
                                                                                        }
                                                                                        """).build()

                        dintegral = np.zeros((n, nt, m_np[0]), dtype=np.float64)  # samo v realnih točkah, brez dodatnih dveh na začetku in na koncu
                        res_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, dintegral.nbytes)
                        prg.dvojci(queue, dintegral.shape, None, m_b, nt_b, delta_b, f_b, x0_b, res_buffer).wait()

                        cl.enqueue_copy(queue, dintegral, res_buffer).wait()
                        cl.tools.clear_first_arg_caches()

                        desna_matrika[in1 * max_buffer:(in1 + 1) * max_buffer, in2 * max_buffer:(in2 + 1) * max_buffer] = (np.sum(dintegral, axis=-1) - dintegral[:, :, 0] / 2 - dintegral[:, :, -1] / 2) / 2

                        time1 = time.time()
                        narejenega = (1 + in1 * (telon.n // max_buffer + 1) + in2) / (telon.n // max_buffer + 1) ** 2
                        za_narejeno_porabil = time1 - time0
                        preostalo_casa = za_narejeno_porabil / narejenega * (1 - narejenega)
                        print("\r" + str(np.round(narejenega * 100, 2)) + " %" + ", " + str(int(preostalo_casa / 60)) + " min do konca", sep='', end='', flush=True)

            print("\nDesna matrika je izracunana!")
            print("Shranjujem...")
            np.save("vmesni_rezultati/potenciali_desna_matrika", desna_matrika)
            print("Končano!")
            print("\n")

        if izracun_potencialov_racunaj:
            print("Berem izvire...")
            izviri = np.load("vmesni_rezultati/izviri.npy")
            print("Berem dvojce...")
            dvojci = np.load("vmesni_rezultati/dvojci.npy")
            print("Berem matriko konstant...")
            matrika_konstant = np.load("vmesni_rezultati/potenciali_matrika_konstant_dopolnjeno.npy")
            matrika_konstant /= 4 * np.pi
            print("Berem desno matriko...")
            desna_matrika = np.load("vmesni_rezultati/potenciali_desna_matrika.npy")
            desna_matrika /= 4 * np.pi

            print("Računam potenciale...")
            racunske_tocke = telon.triangles_center + telon.face_normals * dr

            potenciali_izvirov = np.dot(desna_matrika, izviri)
            potenciali_dvojcev = np.dot(matrika_konstant, dvojci)
            potenciali_toka = np.dot(racunske_tocke, veterd)
            potenciali = potenciali_dvojcev + potenciali_izvirov + potenciali_toka  # ORIGINAL: (+, +, +), že poskusil:
            # potenciali = potenciali_dvojcev
            # (-, +, +) # narobe;  # KROGLA: Narobe
            # (+, -, +) # narobe;  # KROGLA: Narobe
            # (-, -, +) # ne izgleda popolnoma narobe;  Samo ti dve varianti sta možni (preveril, da je potencial 0  # KROGLA:
            # (+, +, +) # ;  Samo ti dve varianti sta možni (preveril, da je potencial 0  # KROGLA:

            print("Shranjujem...")
            np.save("vmesni_rezultati/potenciali", potenciali)
            print("Končano!")
            print("\n")

    if izracun_hitrosti:
        print("Berem potenciale...")
        potenciali = np.load("vmesni_rezultati/potenciali.npy")

        print("Računam težišča...")
        hitrostne_tocke = telon.triangles_center
        lokalne_hitrostne_tocke = racunska_orodja.transformacija(paneli=np.arange(0, telon.n, 1),
                                                                 tocke=hitrostne_tocke,
                                                                 telo=telon,
                                                                 dtype=np.float64)

        print("Računam hitrosti...")
        hitrosti = []
        paneli_s_skupnim_ogliscem = []
        utezi = []
        for i in range(telon.n):
            b1 = telon.triangles[i]
            b2 = np.roll(b1, shift=1, axis=0)
            b3 = np.roll(b1, shift=2, axis=0)

            vsi_ostali_paneli = np.vstack((telon.triangles[:i], telon.triangles[i+1:]))
            indeksi = np.hstack((np.arange(0, telon.n, 1)[:i], np.arange(0, telon.n, 1)[i+1:]))

            enakost_1 = vsi_ostali_paneli == b1
            enakost_2 = vsi_ostali_paneli == b2
            enakost_3 = vsi_ostali_paneli == b3
            enakost = np.array([enakost_1, enakost_2, enakost_3])
            enakost_oglisc = np.sum(enakost, axis=-1) == 3
            enakost_oglisc = enakost_oglisc.any(axis=-1)
            enakost_oglisc = enakost_oglisc.any(axis=0)

            enakost_oglisc_s_koti = []

            v1 = telon.face_normals[indeksi[enakost_oglisc]]
            v2 = telon.face_normals[i]
            for v1_ in v1:
                alpha = racunska_orodja.kot_med_vektorjema(v1_, v2)
                enakost_oglisc_s_koti.append(alpha < np.pi / 3)

            enakost_oglisc_c = copy.copy(enakost_oglisc)
            enakost_oglisc[enakost_oglisc_c] = enakost_oglisc_s_koti

            paneli_s_skupnim_ogliscem.append(indeksi[enakost_oglisc])
            utezi.append(np.power(np.linalg.norm(hitrostne_tocke[indeksi[enakost_oglisc]] - hitrostne_tocke[i], axis=-1), stopnja_utezi))

        m11 = []
        m12 = []
        m21 = []
        m22 = []
        d1 = []
        d2 = []
        for i in range(telon.n):
            m11.append(utezi[i] * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 0] - lokalne_hitrostne_tocke[i, i, 0]) ** 2)
            m12.append(utezi[i] * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 1] - lokalne_hitrostne_tocke[i, i, 1]) * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 0] - lokalne_hitrostne_tocke[i, i, 0]))
            m21.append(utezi[i] * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 1] - lokalne_hitrostne_tocke[i, i, 1]) * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 0] - lokalne_hitrostne_tocke[i, i, 0]))
            m22.append(utezi[i] * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 1] - lokalne_hitrostne_tocke[i, i, 1]) ** 2)
            d1.append(utezi[i] * (potenciali[paneli_s_skupnim_ogliscem[i]] - potenciali[i]) * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 0] - lokalne_hitrostne_tocke[i, i, 0]))
            d2.append(utezi[i] * (potenciali[paneli_s_skupnim_ogliscem[i]] - potenciali[i]) * (lokalne_hitrostne_tocke[paneli_s_skupnim_ogliscem[i], i, 1] - lokalne_hitrostne_tocke[i, i, 1]))

        for i in range(telon.n):
            m11[i] = np.sum(m11[i], axis=-1)
            m12[i] = np.sum(m12[i], axis=-1)
            m21[i] = np.sum(m21[i], axis=-1)
            m22[i] = np.sum(m22[i], axis=-1)
            d1[i] = np.sum(d1[i], axis=-1)
            d2[i] = np.sum(d2[i], axis=-1)

        m11 = np.array(m11)
        m12 = np.array(m12)
        m21 = np.array(m21)
        m22 = np.array(m22)
        d1 = np.array(d1)
        d2 = np.array(d2)

        for i in range(telon.n):
            m = np.array([[m11[i], m12[i]],
                          [m21[i], m22[i]]])
            d = np.array([d1[i], d2[i]])
            try:
                lokalna_hitrost = np.linalg.solve(m, d)
            except SingularMatrix:
                print("Slabo definirana geometrija!")
                print("\n")
            lokalna_hitrost = np.array([lokalna_hitrost[0], lokalna_hitrost[1], 0])
            inverzna_rotacijska_matrika = racunska_orodja.inverzna_rotacijska_matrika(telo=telon, k=i)

            hitrosti.append(np.dot(lokalna_hitrost, inverzna_rotacijska_matrika))
        hitrosti = np.array(hitrosti)

        print("Zapisujem hitrosti...")
        np.save("vmesni_rezultati/hitrosti", hitrosti)
        print("Končano!")
        os.system('say "finished"')
        if snd_msg:
            send_message()
        if snd_ifft:
            send_ifft_notification()
        print("\n")

    if izris:
        hitrostni_vektorji = np.load("vmesni_rezultati/hitrosti.npy")
        # hitrostni_vektorji = np.load("rezultati/ravno_krilo/hitrosti.npy")
        polozaj_vektorja_hitrosti = telon.triangles_center

        hitrost_max = np.linalg.norm(hitrostni_vektorji, axis=-1)
        hitrost_avg = np.mean(hitrost_max)
        hitrost_min = np.min(hitrost_max)
        hit_argmax = np.argmax(hitrost_max)
        hitrost_max = np.max(hitrost_max)

        print("Hitrost vetra:", veterd)
        print("Povprečna hitrost:", hitrost_avg)
        print("Najnižja hitrost:", hitrost_min)
        print("Največja hitrost:", hitrost_max)
        # Izris 3D modela
        print("Izrisujem...")
        if izris_vektrosko_polje:
            fig = plt.figure()
            axes = mplot3d.Axes3D(fig)

            # h = np.linalg.norm(hitrostni_vektorji, axis=-1) > 16
            # hitrostni_vektorji[h] = veterd

            axes.set_xlabel('x [mm]')
            axes.set_ylabel('y [mm]')
            axes.set_zlabel('z [mm]')

            axes.quiver(polozaj_vektorja_hitrosti[:, 0][::f],
                        polozaj_vektorja_hitrosti[:, 1][::f],
                        polozaj_vektorja_hitrosti[:, 2][::f],
                        hitrostni_vektorji[:, 0][::f],
                        hitrostni_vektorji[:, 1][::f],
                        hitrostni_vektorji[:, 2][::f],
                        length=vl / hitrost_max, pivot="tail")

            scale = telon.triangles.flatten(-1)
            axes.auto_scale_xyz(scale/2, scale/2, scale/2)
            plt.show()

        # pobarvanka
        else:
            for i in range(telon.n):
                v = np.linalg.norm(hitrostni_vektorji[i])
                if v <= hitrost_avg:
                    r = 0
                    g = -np.cos(np.pi * v / (2 * hitrost_avg)) + 1
                    b = np.cos(np.pi * v / (2 * hitrost_avg))
                else:
                    r = np.sin(np.pi * (v - hitrost_avg) / (2 * (hitrost_max - hitrost_avg)))
                    g = -np.sin(np.pi * (v - hitrost_avg) / (2 * (hitrost_max - hitrost_avg))) + 1
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


for kora in koraki:
    main(kor=kora)


