# =============================================================================
#  FASE 3 — Retratos fase locales y globales
#  Figuras 3, 4, 5 y 6 del informe
#
#  Figura 3 — Retratos fase locales numéricos en R² (plano G-I)
#             Tres paneles: entorno de E1 | E2 | E3
#  Figura 4 — Retratos fase globales en R²
#             Dos paneles: plano G-I | plano G-beta
#  Figura 5 — Retrato fase global en R³
#  Figura 6 — Retratos fase locales en R³
#             Tres paneles: entorno de E1 | E2 | E3
#
#  Método numérico:
#      scipy.integrate.solve_ivp con RK45
#      rtol = 1e-8,  atol = 1e-10
#      Retratos locales:  t in [0, 10] dias | 5 000 puntos
#      Retratos globales: t in [0, 60] dias | 4 000 puntos
#
#  Uso:
#      Python 3.8+  |  pip install numpy scipy matplotlib
#      python fase3_espacio_fase.py
#      (En Google Colab: descomenta la linea %matplotlib inline)
#
#  Autor: Elioncito
# =============================================================================

# %matplotlib inline   # <- descomenta si usas Google Colab

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi":     120,
})

# =============================================================================
#  1. PARAMETROS
# =============================================================================
R0    = 864.0
Ge    = 140.0
EGO   = 1.44
SI    = 0.72
sigma = 43.2
alpha = 20000.0
rho   = 0.41
k     = 432.0
d0    = 0.06
r1    = 0.84e-3
r2    = 2.4e-6
rpk   = rho + k

# =============================================================================
#  2. EQUILIBRIOS
# =============================================================================
E1 = np.array([697.22, 0.0,    0.0   ])
E2 = np.array([100.0,  11.94,  358.67])
E3 = np.array([250.0,  3.578,  47.27 ])

COL_E1 = "#C1121F"
COL_E2 = "#0B3C49"
COL_E3 = "#E07B00"

PALETTE = [
    "#0B3C49","#145C6A","#1D7382","#2A8C9B",
    "#3CA3B2","#5CB8C4","#7AC7D1","#9AD8DF",
    "#6B3074","#A94F6E","#E07B00","#C1121F",
]

# =============================================================================
#  3. SISTEMA DE EDOS
# =============================================================================
def sistema(t, y):
    G = max(y[0], 1e-10)
    I = max(y[1], 0.0)
    B = max(y[2], 0.0)
    dG = R0 + Ge - (EGO + SI * I) * G
    dI = B * sigma * G**2 / (alpha + G**2) - rpk * I
    dB = (-d0 + r1 * G - r2 * G**2) * B
    return [dG, dI, dB]

# =============================================================================
#  4. INTEGRACION
# =============================================================================
def simular(y0, t_span=(0, 60), n_pts=4000):
    """
    Integra el sistema con RK45.
    t_span: intervalo en dias
    n_pts:  puntos de evaluacion
    """
    sol = solve_ivp(
        sistema,
        t_span,
        [max(y0[0], 5.0), max(y0[1], 0.0), max(y0[2], 0.0)],
        method="RK45",
        t_eval=np.linspace(t_span[0], t_span[1], n_pts),
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )
    return sol.t, sol.y

# =============================================================================
#  5. CONDICIONES INICIALES GLOBALES
#     G in [80, 750] mg/dL | I in [0, 25] uU/mL | beta in [0, 460] u.a.
# =============================================================================
ICS_GLOBALES = [
    [ 80,  20, 380],   # cercana a E2
    [120,   8, 300],   # cercana a E2
    [150,  18, 400],   # cercana a E2
    [180,  15, 350],   # cercana a E2
    [200,   6, 150],   # zona intermedia
    [300,  12, 200],   # zona intermedia
    [350,   2,  80],   # proxima al umbral E3
    [400,   8, 120],   # zona intermedia alta
    [450,   4,  60],   # proxima al umbral E3
    [500, 0.5,  20],   # deriva hacia E1
    [600, 0.2,   5],   # deriva hacia E1
    [650, 0.5,  10],   # deriva hacia E1
    [700, 0.1,   2],   # deriva hacia E1
    [100,   3, 100],   # intermedia baja en beta
    [250,   4,  52],   # muy cercana a E3
    [ 80,   1, 200],   # glucosa baja, beta moderada
]

# =============================================================================
#  6. FIGURA 3 — RETRATOS LOCALES NUMERICOS EN R² (plano G-I)
#     t in [0, 10] dias | 5 000 puntos | perturbaciones ±5%
# =============================================================================
def figura3():
    """
    Tres paneles en plano G-I con coordenadas desplazadas.
    Condiciones iniciales: perturbaciones ±5% alrededor de cada equilibrio.
    Intervalo: t in [0, 10] dias | 5 000 puntos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Figura 3 — Retratos fase locales numéricos, plano $G$-$I$\n"
        "Coordenadas desplazadas  |  $t\\in[0,10]$ días  |  RK45",
        fontsize=11,
    )

    configs = [
        (E1, COL_E1, "$E_1$", "Nodo atractor"),
        (E2, COL_E2, "$E_2$", "Nodo atractor"),
        (E3, COL_E3, "$E_3$", "Punto silla"),
    ]
    subtitulos = [
        r"$\mathbf{(a)}$  Entorno de $E_1$",
        r"$\mathbf{(b)}$  Entorno de $E_2$",
        r"$\mathbf{(c)}$  Entorno de $E_3$",
    ]

    for ax, (Estar, col_eq, nombre, tipo), sub in \
            zip(axes, configs, subtitulos):

        ax.set_facecolor("#F5F5F5")

        # Condiciones iniciales: 12 perturbaciones ±5% y ±10%
        ics_loc = []
        for scale in [0.05, -0.05, 0.10, -0.10, 0.07, -0.07]:
            for dg in [scale, -scale * 0.5]:
                ics_loc.append([
                    Estar[0] * (1 + dg),
                    max(Estar[1] * (1 - dg * 0.8), 0.01),
                    max(Estar[2] * (1 + dg * 0.3), 0.0),
                ])

        for i, y0 in enumerate(ics_loc[:12]):
            try:
                _, Y = simular(y0, t_span=(0, 10), n_pts=5000)
            except Exception:
                continue
            col = PALETTE[i % len(PALETTE)]
            uG = Y[0] - Estar[0]
            uI = Y[1] - Estar[1]
            ax.plot(uG, uI, lw=1.5, color=col, alpha=0.85)
            m = len(uG) // 3
            if m + 4 < len(uG):
                ax.annotate("",
                            xy=(uG[m + 4], uI[m + 4]),
                            xytext=(uG[m], uI[m]),
                            arrowprops=dict(arrowstyle="-|>",
                                            lw=0.9, color=col,
                                            mutation_scale=10))

        ax.scatter(0, 0, s=160, color=col_eq,
                   edgecolor="k", lw=1.2, zorder=8)
        ax.text(-8.5, -0.15, nombre,
                fontsize=9, color=col_eq, fontweight="bold")
        ax.axhline(0, color="#AAAAAA", lw=0.5, alpha=0.5)
        ax.axvline(0, color="#AAAAAA", lw=0.5, alpha=0.5)
        ax.set_xlabel(r"$\Delta G = G - G^*$ [mg/dL]")
        ax.set_ylabel(r"$\Delta I = I - I^*$ [µU/mL]")
        titulo = (f"Retrato fase local — {nombre} ({tipo})\n"
                  "Plano G-I")
        ax.set_title(titulo, fontsize=9, pad=4)
        ax.grid(True, ls=":", alpha=0.3)
        ax.text(0.5, -0.18, sub,
                transform=ax.transAxes, ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    plt.savefig("figura3.jpeg", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: figura3.jpeg")

# =============================================================================
#  7. FIGURA 4 — RETRATOS GLOBALES EN R²
#     t in [0, 60] dias | 4 000 puntos | 16 condiciones iniciales
# =============================================================================
def figura4():
    """
    Dos paneles: plano G-I y plano G-beta.
    Intervalo: t in [0, 60] dias | 4 000 puntos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figura 4 — Retratos fase globales en $\\mathbb{R}^2$\n"
        "$t\\in[0,60]$ días  |  16 condiciones iniciales  |  RK45",
        fontsize=11,
    )

    planos = [
        (0, 1, "Retrato fase global — Plano G-I",
         "G [mg/dL]", "I [µU/mL]"),
        (0, 2, "Retrato fase global — Plano G-$\\beta$",
         "G [mg/dL]", "$\\beta$ [u.a.]"),
    ]
    subtitulos = [
        r"$\mathbf{(a)}$  Plano $G$-$I$",
        r"$\mathbf{(b)}$  Plano $G$-$\beta$",
    ]

    equils = [
        (E1, COL_E1, "$E_1$", "Nodo atractor", "^"),
        (E2, COL_E2, "$E_2$", "Nodo atractor", "o"),
        (E3, COL_E3, "$E_3$", "Punto silla",   "s"),
    ]

    for ax, (ix, iy, titulo, xl, yl), sub in \
            zip(axes, planos, subtitulos):

        ax.set_facecolor("#F5F5F5")
        ax.set_title(titulo, fontsize=9, pad=4)

        for i, y0 in enumerate(ICS_GLOBALES):
            try:
                _, Y = simular(y0, t_span=(0, 60), n_pts=4000)
            except Exception:
                continue
            col = PALETTE[i % len(PALETTE)]
            ax.plot(Y[ix], Y[iy], lw=1.5, color=col, alpha=0.80)
            m = len(Y[0]) // 3
            if m + 4 < len(Y[0]):
                ax.annotate("",
                            xy=(Y[ix][m + 4], Y[iy][m + 4]),
                            xytext=(Y[ix][m], Y[iy][m]),
                            arrowprops=dict(arrowstyle="-|>",
                                            lw=0.9, color=col,
                                            mutation_scale=10))
            ax.scatter(Y[ix][0], Y[iy][0],
                       s=22, color=col, zorder=4)

        for Eq, col_eq, nom, tipo, mk in equils:
            ax.scatter(Eq[ix], Eq[iy], s=170,
                       color=col_eq, edgecolor="k",
                       lw=1.2, zorder=9, marker=mk,
                       label=f"{nom} — {tipo}")
            ax.text(Eq[ix] + 6, Eq[iy] + 2,
                    nom, fontsize=9,
                    fontweight="bold", color=col_eq)

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(loc="upper right", fontsize=7.5,
                  frameon=True, framealpha=0.9)
        ax.grid(True, ls=":", alpha=0.3)
        ax.text(0.5, -0.14, sub,
                transform=ax.transAxes, ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    plt.savefig("figura4.jpeg", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: figura4.jpeg")

# =============================================================================
#  8. FIGURA 5 — RETRATO GLOBAL EN R³
#     t in [0, 60] dias | 4 000 puntos | 16 condiciones iniciales
# =============================================================================
def figura5():
    """
    Retrato fase global en el espacio completo (G, I, beta).
    Intervalo: t in [0, 60] dias | 4 000 puntos.
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    fig.suptitle(
        "Figura 5 — Retrato fase global en $\\mathbb{R}^3 = (G, I, \\beta)$\n"
        "$t\\in[0,60]$ días  |  16 condiciones iniciales  |  RK45",
        fontsize=11,
    )

    for i, y0 in enumerate(ICS_GLOBALES):
        try:
            _, Y = simular(y0, t_span=(0, 60), n_pts=4000)
        except Exception:
            continue
        col = PALETTE[i % len(PALETTE)]
        ax.plot(Y[0], Y[1], Y[2], lw=1.4, color=col, alpha=0.75)
        ax.scatter(Y[0][0], Y[1][0], Y[2][0],
                   s=22, color=col, zorder=4)

    equils = [
        (E1, COL_E1, "$E_1$ — Nodo atractor", "^"),
        (E2, COL_E2, "$E_2$ — Nodo atractor", "o"),
        (E3, COL_E3, "$E_3$ — Punto silla",   "s"),
    ]
    for Eq, col_eq, label, mk in equils:
        ax.scatter(Eq[0], Eq[1], Eq[2],
                   s=200, color=col_eq, edgecolor="k",
                   lw=1.2, zorder=9, marker=mk, label=label)

    ax.set_xlabel("$G$ [mg/dL]", labelpad=10)
    ax.set_ylabel("$I$ [µU/mL]", labelpad=10)
    ax.set_zlabel("$\\beta$ [u.a.]", labelpad=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=-55)
    plt.tight_layout()
    plt.savefig("figura5.jpeg", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: figura5.jpeg")

# =============================================================================
#  9. FIGURA 6 — RETRATOS LOCALES EN R³
#     t in [0, 10] dias | 5 000 puntos | perturbaciones ±8%
# =============================================================================
def figura6():
    """
    Tres paneles 3D con coordenadas desplazadas.
    Intervalo: t in [0, 10] dias | 5 000 puntos.
    """
    fig = plt.figure(figsize=(16, 5))

    configs = [
        (E1, COL_E1, "$E_1$", "Nodo atractor",  8,  1),
        (E2, COL_E2, "$E_2$", "Nodo atractor", 12,  2),
        (E3, COL_E3, "$E_3$", "Punto silla",   10,  3),
    ]
    subtitulos = [
        r"$\mathbf{(a)}$  Entorno de $E_1$",
        r"$\mathbf{(b)}$  Entorno de $E_2$",
        r"$\mathbf{(c)}$  Entorno de $E_3$",
    ]

    for (Estar, col_eq, nombre, tipo, tf, pos), sub \
            in zip(configs, subtitulos):

        ax = fig.add_subplot(1, 3, pos, projection="3d")

        # Condiciones iniciales: 8 perturbaciones
        ics_loc = []
        for dg in [0.08, -0.08, 0.12, -0.12]:
            for di in [0.06, -0.06]:
                ics_loc.append([
                    Estar[0] * (1 + dg),
                    max(Estar[1] * (1 + di), 0.01),
                    max(Estar[2] * (1 - dg * 0.4), 0.0),
                ])

        for i, y0 in enumerate(ics_loc):
            try:
                _, Y = simular(y0, t_span=(0, tf), n_pts=5000)
            except Exception:
                continue
            col = PALETTE[i % len(PALETTE)]
            uG = Y[0] - Estar[0]
            uI = Y[1] - Estar[1]
            uB = Y[2] - Estar[2]
            ax.plot(uG, uI, uB, lw=1.4, color=col, alpha=0.80)
            m = len(uG) // 3
            if m + 2 < len(uG):
                ax.quiver(uG[m], uI[m], uB[m],
                          uG[m+2]-uG[m], uI[m+2]-uI[m], uB[m+2]-uB[m],
                          color=col, lw=1.0,
                          arrow_length_ratio=0.5,
                          normalize=True,
                          length=max(abs(uG.max()), 1) * 0.15)

        ax.scatter(0, 0, 0, s=180, color=col_eq,
                   edgecolor="k", lw=1.2, zorder=8)
        ax.set_xlabel(r"$\Delta G$", labelpad=5)
        ax.set_ylabel(r"$\Delta I$", labelpad=5)
        ax.set_zlabel(r"$\Delta\beta$", labelpad=5)
        ax.set_title(
            f"Retrato local 3D — {nombre} ({tipo})\n"
            "Coordenadas desplazadas",
            fontsize=9, pad=6,
        )
        ax.view_init(elev=22, azim=-55)
        ax.text2D(0.5, -0.08, sub,
                  transform=ax.transAxes,
                  ha="center", fontsize=10)

    plt.suptitle(
        "Figura 6 — Retratos fase locales en $\\mathbb{R}^3$\n"
        "$t\\in[0,10]$ días  |  RK45",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    plt.savefig("figura6.jpeg", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: figura6.jpeg")

# =============================================================================
#  10. EJECUCION PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FASE 3 - Retratos fase locales y globales")
    print("Figuras 3, 4, 5 y 6 del informe")
    print("Modelo glucosa-insulina-celula beta")
    print("=" * 60)

    print("\n[1/4] Figura 3 — Retratos locales R² plano G-I...")
    figura3()

    print("\n[2/4] Figura 4 — Retratos globales R²...")
    figura4()

    print("\n[3/4] Figura 5 — Retrato global R³...")
    figura5()

    print("\n[4/4] Figura 6 — Retratos locales R³...")
    figura6()

    print("\nFiguras 3-6 generadas correctamente!")
    print("Archivos: figura3.jpeg, figura4.jpeg, figura5.jpeg, figura6.jpeg")
