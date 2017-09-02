import numpy as np
import racunska_orodja
import os


def send_message():
    cmd = """osascript<<EOF
        tell application "Messages"
            send "Končano!" to buddy "+38641343034" of service "SMS"
        end tell"""
    os.system(cmd)

def send_ifft_notification():
    cmd = """open https://maker.ifttt.com/trigger/script_done/with/key/YFEJhW4WUiXDrhlWRjaJQ"""
    os.system(cmd)


def vrhni_panel(p1, p2):
    """
    Določi, kateri izmed panelov v paru je na zgornji strani. Primeja z lego težišča.
    :param p1: Oglišča panela 1 kot so v objektu STL.
    :param p2: Oglišča panela 2 kot so v objektu STL.
    :return: 0, če p1 oz. 1, če p2, oz. False, če sta na isti višini
    """

    a1 = p1[:3]
    b1 = p1[3:6]
    c1 = p1[6:]
    a2 = p2[:3]
    b2 = p2[3:6]
    c2 = p2[6:]

    t1 = (a1 + b1 + c1) / 3
    t2 = (a2 + b2 + c2) / 3

    if t1[2] < t2[2]:
        return 1
    elif t1[2] > t2[2]:
        return 0
    else:
        # print("PANELA STA NA ISTI VIŠINI!")
        return False


def dotikanje_panelov(p1, p2):
    """
    Ta funkcija samo preveri, če se dva panela dotikata med seboj.
    :param p1: Oglišča panela 1 kot so v objektu STL.
    :param p2: Oglišča panela 2 kot so v objektu STL.
    :return: boolean
    """

    a1 = p1[:3]
    b1 = p1[3:6]
    c1 = p1[6:]
    a2 = p2[:3]
    b2 = p2[3:6]
    c2 = p2[6:]

    iste_točke = []
    numerična_napaka = 10 ** -5
    o1 = [a1, b1, c1]
    o2 = [a2, b2, c2]

    for i in range(3):

        for j in range(3):
            enakost = np.abs(o1[i] - o2[j]) < numerična_napaka
            if type(enakost[0]) != np.bool_:
                print("NEKAJ NE DELA PRI DOLOČEVANJU SKUPNIH ROBOV!")

            if enakost.all():
                iste_točke.append([i, j])
    if len(iste_točke) > 2:
        print("NEKAJ NE DELA PRI DOLOČEVANJU SKUPNIH ROBOV!")
        print(iste_točke)
        print(enakost)

    if len(iste_točke) == 2:
        return True
    else:
        return False

def skupni_rob_panelov(p1, p2):
    """
    Ta funkcija določi skupni rob dveh panelov, ki se dotikata med seboj.
    :param p1: Oglišča panela 1 kot so v objektu STL.
    :param p2: Oglišča panela 2 kot so v objektu STL.
    :return: Vektor robu, ki je definiran kot razlika krajevnih vektorjev oglišč vrhnega panela.
    """

    if not dotikanje_panelov(p1, p2):
        return False

    else:
        a1 = p1[:3]
        b1 = p1[3:6]
        c1 = p1[6:]
        a2 = p2[:3]
        b2 = p2[3:6]
        c2 = p2[6:]

        iste_točke = []
        numerična_napaka = 10 ** -5
        o1 = [a1, b1, c1]
        o2 = [a2, b2, c2]

        for i in range(3):

            for j in range(3):
                enakost = np.abs(o1[i] - o2[j]) < numerična_napaka
                if type(enakost[0]) != np.bool_:
                    print("NEKAJ NE DELA PRI DOLOČEVANJU VEKTORJA ROBOV!")

                if enakost.all():
                    iste_točke.append([i, j])
        if len(iste_točke) > 2:
            print("NEKAJ NE DELA PRI DOLOČEVANJU VEKTORJA ROBOV!")
            print(iste_točke)
            print(enakost)

        iste_točke = np.array(iste_točke)
        try:
            iste_točke = iste_točke[:, vrhni_panel(p1, p2)]
        except IndexError:
            return False

        indeks1 = np.min(iste_točke)
        indeks2 = np.max(iste_točke)

        o = [o1, o2]
        o = o[vrhni_panel(p1, p2)]

        return o[indeks2] - o[indeks1]


def robni_točki_panela(p1, p2):
    """
    Ta funkcija vrne točki, ki sta na robu.
    :param p1: Oglišča panela 1 kot so v objektu STL.
    :param p2: Oglišča panela 2 kot so v objektu STL.
    :return: [indeks1, indeks2]
    """

    if not dotikanje_panelov(p1, p2):
        return False

    else:
        a1 = p1[:3]
        b1 = p1[3:6]
        c1 = p1[6:]
        a2 = p2[:3]
        b2 = p2[3:6]
        c2 = p2[6:]

        iste_točke = []
        numerična_napaka = 10 ** -5
        o1 = [a1, b1, c1]
        o2 = [a2, b2, c2]

        for i in range(3):

            for j in range(3):
                enakost = np.abs(o1[i] - o2[j]) < numerična_napaka
                if type(enakost[0]) != np.bool_:
                    print("NEKAJ NE DELA PRI DOLOČEVANJU TOČK ROBOV!")

                if enakost.all():
                    iste_točke.append([i, j])
        if len(iste_točke) > 2:
            print("NEKAJ NE DELA PRI DOLOČEVANJU TOČK ROBOV!")
            print(iste_točke)
            print(enakost)

        iste_točke = np.array(iste_točke)
        iste_točke = iste_točke[:, vrhni_panel(p1, p2)]

        indeks1 = np.min(iste_točke)
        indeks2 = np.max(iste_točke)

        return [indeks1, indeks2]


def lokalni_koordinatni_sistem_točka(telo, p1, t):
    """
    Funkcija poskrbi za pretvorbo točke v lokalni koordinatni sistem panela.
    :param telo: Objekt, ki ga dobimo z orodjem STL.
    :param p1: Zaporedna številka panela, kot je v STL datoteki. Potrebno paziti, saj so ponekod z istim
               parametrom označena oglišča panela.
    :param t: Točka, ki je želimo transformirati (samo ena).
    :return: Točka, ki je transformirana.
    """

    if type(p1) != int:
        print("NAPAČNI VHODNI PODATKI ZA PANEL")

    # definiranje Eulerjevih kotov
    r_ai = telo.points[p1][:3]
    r_ci = telo.points[p1][6:]


    vektor_y = r_ci - r_ai
    vektor_y = vektor_y / np.linalg.norm(vektor_y)

    vektor_z = telo.units[p1]
    vektor_y = vektor_y
    vektor_x = np.cross(vektor_y, telo.units[p1])

    # določitev matrike
    m = np.array([vektor_x, vektor_y, vektor_z])
    r_zxz = np.linalg.inv(m)

    # translacija točke
    tt = t - r_ai

    # rotacija točke
    return np.dot(tt, r_zxz)


def lokalni_koordinatni_sistem_točka_vrtinčna_sled(vrtinčna_sled, p1, t):
    """
    Funkcija poskrbi za pretvorbo točke v lokalni koordinatni sistem panela vrtinčne sledi.
    :param vrtinčna_sled: Objekt, ki ga dobimo z VrtinčnaSled
    :param p1: Zaporedna številka panela, kot je v STL datoteki. Potrebno paziti, saj so ponekod z istim
               parametrom označena oglišča panela.
    :param t: Točka, ki je želimo transformirati (samo ena).
    :return: Točka, ki je transformirana.
    """

    if type(p1) != int:
        print("NAPAČNI VHODNI PODATKI ZA PANEL")

    # definiranje Eulerjevih kotov
    r_ai = vrtinčna_sled.points[p1][0]
    r_di = vrtinčna_sled.points[p1][3]

    vektor_y = r_di - r_ai
    vektor_y = vektor_y / np.linalg.norm(vektor_y)

    vektor_z = vrtinčna_sled.units[p1]
    vektor_y = vektor_y
    vektor_x = np.cross(vektor_y, vrtinčna_sled.units[p1])

    # določitev matrike
    m = np.array([vektor_x, vektor_y, vektor_z])
    r_zxz = np.linalg.inv(m)

    # translacija točke
    tt = t - r_ai

    # rotacija točke
    return np.dot(tt, r_zxz)


def presečišče_poltraka_in_ravnine(s, r0, a, b, r1):
    """
    Določi točko presečišča ravnine s premice. Če sta premica in ravnina vzporedni, vrne False. Normala ravnine
    ni pomembna, zato tudi ni pomemben vrstni red navajanja vektorjev ravnine.
    :param s: Smerni vektor premice.
    :param r0: Točka na premici.
    :param a: Prvi vektor, ki popisuje ravnino
    :param b: Drugi vektor, ki popisuje ravnino.
    :param r1: Točka na ravnini.
    :return: Če točka obstaja vrne np.array([Tx, Ty, Tz]), True, če pa ne pa vrne točko [0, 0, 0], False.
    """
    ma = np.array([[s[0], -a[0], -b[0]],
                   [s[1], -a[1], -b[1]],
                   [s[2], -a[2], -b[2]]])
    mb = r1 - r0
    try:
        t0, t1, t2 = np.linalg.solve(ma, mb)
        if t0 > 0:
            return r0 + s * t0, True
        else:
            return np.array([0, 0, 0]), False
    except np.linalg.linalg.LinAlgError:
        return np.array([0, 0, 0]), False


def ali_je_točka_znotraj_lika(p, a, b, c):
    """
    Preveri, ali je točka, ki je na isti ravnini kot lik, znotraj območja lika.
    :param p: Točka, ki nas zanima.
    :param a: Oglišče lika.
    :param b: Oglišče lika.
    :param c: Oglišče lika.
    :return: Je ali pa ni, boolean.
    """
    ab = b - a
    ac = c - a
    pa = a - p
    pb = b - p
    pc = c - p
    površina = np.linalg.norm(np.cross(ab, ac)) / 2
    alfa = np.linalg.norm(np.cross(pb, pc)) / 2
    beta = np.linalg.norm(np.cross(pc, pa)) / 2
    gama = np.linalg.norm(np.cross(pa, pb)) / 2

    if np.abs(np.dot(np.cross(pb, pa), pc)) != 0:
        print("TOČKE NISO NA ISTI RAVNINI! PREVELIKA NUMERIČNA NAPAKA!")
        print(np.abs(np.dot(np.cross(pb, pa), ab)))

    if np.abs(alfa + beta + gama - površina) < 10 ** -8:
        if alfa != 0:
            if beta != 0:
                if gama != 0:
                    return True
    else:
        return False


def ali_je_vektor_usmerjen_navznoter(ploskev, v, p, a, b, c, d):
    """
    Določi, ali je vektor usmerjen navznoter. To se potrebuje pri računanju kuttovega pogoja.
    Lik je štiristrana piramida.
    :param ploskev: Ploskev, na kateri je vektor. Definirana je z imeni oglišč v množici, npr. ["a", "p", "c"]
    :param v: Vektor.
    :param p: Vrh piramide.
    :param a: Oglišče osnovne ploskve.
    :param b: Oglišče osnovne ploskve.
    :param c: Oglišče osnovne ploskve.
    :param d: Oglišče osnovne ploskve.
    :return: Je ali pa ni, boolean.
    """
    oglišča = {"p": p, "a": a, "b": b, "c": c, "d": d}
    oglišča_ploskve = []
    for i in range(len(ploskev)):
        oglišča_ploskve.append(oglišča[ploskev[i]])
    oglišča_ploskve = np.array(oglišča_ploskve)
    t = np.sum(oglišča_ploskve, axis=0) / len(oglišča_ploskve)
    število_sekanj = 0
    vse_ploskve = [["a", "b", "p"],
                   ["a", "d", "p"],
                   ["d", "c", "p"],
                   ["b", "c", "p"],
                   ["a", "b", "c", "d"]]


    indeks_ploskve = -1
    for i in range(5):
        if set(ploskev) == set(vse_ploskve[i]):
            indeks_ploskve = i
            break
    if indeks_ploskve == -1:
        print("NEKAJ JE NAROBE PRI DOLOČEVANJU USMERJENOSTI VEKTORJEV!")
    else:
        ind = [i for i in range(5)]
        ind.pop(indeks_ploskve)
        for i in ind:
            t1 = oglišča[vse_ploskve[i][0]]
            t2 = oglišča[vse_ploskve[i][1]]
            t3 = oglišča[vse_ploskve[i][2]]
            v1 = t2 - t1
            v2 = t3 - t1
            if presečišče_poltraka_in_ravnine(s=v, r0=t, a=v1, b=v2, r1=t1)[1]:
                if ali_je_točka_znotraj_lika(p=presečišče_poltraka_in_ravnine(s=v, r0=t, a=v1, b=v2, r1=t1)[0],
                                             a=t1, b=t2, c=t3):
                    print(vse_ploskve[i][0], vse_ploskve[i][1], vse_ploskve[i][2])
                    print(t)
                    print(presečišče_poltraka_in_ravnine(s=v, r0=t, a=v1, b=v2, r1=t1)[0])
                    število_sekanj += 1

    print(število_sekanj)
    if število_sekanj % 2 == 0:
        return False
    else:
        return True


def korak(št):
    št -= 1
    koraki = np.zeros(11).astype(np.bool)
    koraki[št] = True

    text1 = "1. korak: V prvem koraku izračunam matriko vplivov dvojcev za točke na telesu za panele na telesu."
    text2 = "2. korak: V drugem koraku matriko vplivov dvojcev na telesu dopolnim z vplivi dvojcev na vrtinčni sledu.\nTo pomeni, da panelom, ki so na robu prištejem oziroma odštejem vpliv dvojca na vrtinčni sledi."
    text3 = "3. korak: V tretjem koraku izračunam desno matriko linearnega sistema enačb za določitev velikosti dvojcev.\nTa matrika predstavlja vplive porazdeljenih izvirov na telesu."
    text4 = "4. korak: V četrtem koraku izračunam velikosti izvirov, ki so definirani z velikostjo vzporednega toka."
    text5 = "5. korak: V petem koraku izračunam velikosti dvojcev z Gaussovo eliminacijo sistema enačb, ki sem ga s prejšnjimi koraki določil."
    text6 = "6. korak: V šestem koraku izračunam matriko vplivov dvojcev za točke na telesu za panele na telesu. Z njimi se določi potencial na panelih."
    text7 = "7. korak: V sedmem koraku matriko vplivov dvojcev na telesu dopolnim z vplivi dvojcev na vrtinčni sledu tako,\nda panelom, ki so na robu prištejem oziroma odštejem vpliv dvojca na vrtinčni sledi. Z njimi se določi potencial na panelih."
    text8 = "8. korak: V osmem koraku izračunam desno matriko linearnega sistema enačb za določitev velikosti dvojcev.\nTa matrika predstavlja vplive porazdeljenih izvirov na telesu. Z njimi se določi potencial na panelih."
    text9 = "9. korak: V devetem koraku izračunam velikosti potencialov s pomočjo vektorskega prodkuta velikosti izvirov in dvojcev matrik vplivov."
    text10 = "10. korak: V desetem koraku izračunam hitrosti na podlagi izračunanih potencialov. Gradient potencialov določa hitrost.\nGradient določim na podlagi sosednjih panelov in njihovih potencialov."
    text11 = "11. korak: V enajstem koraku izrišem telo in izračunane hitrosti."

    t = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11]

    return koraki, t[št]


# print(ali_je_vektor_usmerjen_navznoter(ploskev=["p", "b", "c"],
#                                        v=np.array([0.03455049,  0.54295437, -0.83905114]),
#                                        p=np.array([35.92675018,  144.97750854,   -2.90547752]),
#                                        a=np.array([7.17883224e+01,   1.36135788e+02,  -1.90942809e-02]),
#                                        b=np.array([24023.22582245,   1659.09757996,     96.06693477]),
#                                        c=np.array([24014.89944458,   1790.44958496,     39.11904472]),
#                                        d=np.array([6.50654602e+01,   2.39982910e+02,  -2.02947259e-02])))
