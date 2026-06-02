# =============================================================================
#  FASE 3 — Bosquejos cualitativos locales
#  Figuras 1 y 2 del informe
#
#  Descripción:
#      Genera los bosquejos cualitativos locales alrededor de cada equilibrio
#      usando campos vectoriales (streamplot) y trayectorias del sistema
#      no lineal completo con coordenadas desplazadas.
#
#      Figura 1 — Bosquejos alrededor de los nodos atractores E1 y E2
#                 (planos G-I y G-β)
#      Figura 2 — Bosquejo alrededor del punto de silla E3
#                 (planos G-I y G-β) con variedades estable e inestable
#
#  Método numérico:
#      scipy.integrate.solve_ivp con RK45
#      rtol = 1e-8,  atol = 1e-10
#      t ∈ [0, 10] días,  5 000 puntos
#
#  Uso:
#      Python 3.8+  |  pip install numpy scipy matplotlib
#      python fase3_bosquejos.py
#      (En Google Colab: descomenta la línea %matplotlib inline)
#
#  Autor: Elioncito
# =============================================================================

# %matplotlib inline   # <- descomenta si usas Google Colab

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     120,
})

# =============================================================================
#  1. PARÁMETROS DEL MODELO (caso base)
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
rpk   = rho + k     # = 432.41

# =============================================================================
#  2. EQUILIBRIOS
# =============================================================================
E1 = np.array([697.22,  0.0,     0.0    ])
E2 = np.array([100.0,   11.94,   358.67 ])
E3 = np.array([250.0,    3.578,   47.27 ])

colores_eq  = {"E1": "#C1121F", "E2": "#0B3C49", "E3": "#E07B00"}
markers_eq  = {"E1": "^",       "E2": "o",       "E3": "s"      }
tipos_eq    = {"E1": "Nodo atractor", "E2": "Nodo atractor", "E3": "Punto de silla"}

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
#  4. CAMPO VECTORIAL EN COORDENADAS DESPLAZADAS
# =============================================================================
def campo_vectorial_local(ax, Estar, ix, iy, xlim, ylim, n=20):
    """
    Dibuja streamplot del sistema linealizado en coordenadas desplazadas
    (u_x, u_y) alrededor del equilibrio Estar.
    """
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(n):
        for j in range(n):
            state = Estar.copy().astype(float)
            state[ix] = max(Estar[ix] + X[i, j], 1e-6)
            state[iy] = max(Estar[iy] + Y[i, j], 0.0)
            f = sistema(0, state)
            U[i, j] = f[ix]
            V[i, j] = f[iy]

    mag = np.sqrt(U**2 + V**2)
    mag[mag == 0] = 1.0
    try:
        ax.streamplot(X, Y, U / mag, V / mag,
                      density=1.1, linewidth=0.8,
                      color="#BBBBBB", arrowsize=1.0,
                      arrowstyle="-|>")
    except Exception:
        step = max(1, n // 8)
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  (U / mag)[::step, ::step], (V / mag)[::step, ::step],
                  color="#BBBBBB", alpha=0.5, angles="xy",
                  width=0.003, headwidth=3.5)

# =============================================================================
#  5. INTEGRACIÓN LOCAL
# =============================================================================
def simular_local(y0, t_max=10, n_pts=5000):
    sol = solve_ivp(
        sistema,
        [0, t_max],
        [max(y0[0], 5.0), max(y0[1], 0.0), max(y0[2], 0.0)],
        method="RK45",
        t_eval=np.linspace(0, t_max, n_pts),
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )
    return sol.y

# =============================================================================
#  6. FIGURA 1 — BOSQUEJOS LOCALES ALREDEDOR DE E1 Y E2
#     Tres paneles: E1 plano G-I | E1 plano G-β | E2 plano G-I
# =============================================================================
def figura1_nodos():
    """
    Figura 1: bosquejos cualitativos locales alrededor de los
    nodos atractores E1 y E2.
    Paneles: (a) E1 plano G-I | (b) E1 plano G-β | (c) E2 plano G-I
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Figura 1 — Bosquejos cualitativos locales\n"
        "Nodos atractores $E_1$ y $E_2$  |  "
        "$t\\in[0,10]$ días  |  RK45",
        fontsize=12,
    )

    configuraciones = [
        # (equilibrio, nombre, ix, iy, xlim, ylim, título)
        (E1, "E1", 0, 1, (-8, 8), (-3.5, 3.5),
         "$P_1 \\equiv E_1$, plano $G$-$I$"),
        (E1, "E1", 0, 2, (-8, 8), (-3.5, 3.5),
         "$P_1 \\equiv E_1$, plano $G$-$\\beta$"),
        (E2, "E2", 0, 1, (-8, 8), (-3.5, 3.5),
         "$P_2 \\equiv E_2$, plano $G$-$I$"),
    ]

    labels_vars = {
        (0, 1): ("$\\Delta G = G - G^*$ [mg/dL]",
                 "$\\Delta I = I - I^*$ [$\\mu$U/mL]"),
        (0, 2): ("$\\Delta G = G - G^*$ [mg/dL]",
                 "$\\Delta\\beta = \\beta - \\beta^*$ [u.a.]"),
    }

    for ax, (Estar, nombre, ix, iy, xlim, ylim, titulo) in \
            zip(axes, configuraciones):

        ax.set_facecolor("#F8F9FA")

        # Campo vectorial
        campo_vectorial_local(ax, Estar, ix, iy, xlim, ylim, n=18)

        # Condiciones iniciales: perturbaciones en 10 direcciones
        angulos = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        for i, ang in enumerate(angulos):
            dx = xlim[1] * 0.6 * np.cos(ang)
            dy = ylim[1] * 0.6 * np.sin(ang)
            y0 = Estar.copy().astype(float)
            y0[ix] += dx
            y0[iy] += dy
            y0[0] = max(y0[0], 5.0)
            y0[1] = max(y0[1], 0.0)
            y0[2] = max(y0[2], 0.0)
            try:
                Y = simular_local(y0)
                ug = Y[ix] - Estar[ix]
                uv = Y[iy] - Estar[iy]
                col = PALETTE[i % len(PALETTE)]
                ax.plot(ug, uv, lw=1.6, color=col, alpha=0.85)
                # flecha de dirección
                m = len(ug) // 3
                if m + 5 < len(ug):
                    ax.annotate("",
                                xy=(ug[m + 5], uv[m + 5]),
                                xytext=(ug[m], uv[m]),
                                arrowprops=dict(arrowstyle="-|>",
                                                lw=1.0, color=col,
                                                mutation_scale=12))
            except Exception:
                continue

        # Equilibrio en el origen
        ax.scatter(0, 0, s=180, color=colores_eq[nombre],
                   edgecolor="k", linewidth=1.2, zorder=8,
                   marker=markers_eq[nombre],
                   label=f"$E_{nombre[1]}$ — {tipos_eq[nombre]}")
        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.axvline(0, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xlabel, ylabel = labels_vars[(ix, iy)]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titulo, fontsize=10, pad=8)
        ax.legend(loc="upper right", fontsize=8, frameon=True,
                  framealpha=0.9)
        ax.grid(True, ls=":", alpha=0.25)

    plt.tight_layout()
    plt.savefig("figura1.jpeg", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: figura1.jpeg")


# =============================================================================
#  7. FIGURA 2 — BOSQUEJO LOCAL ALREDEDOR DE E3 (PUNTO DE SILLA)
#     Dos paneles: plano G-I | plano G-β
#     Con líneas punteadas para las variedades estable e inestable
# =============================================================================
def figura2_silla():
    """
    Figura 2: bosquejo cualitativo local alrededor del punto de
    silla E3 en los planos G-I y G-β.
    Muestra las variedades estable W^s(E3) e inestable W^u(E3).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "Figura 2 — Bosquejo cualitativo local\n"
        "Punto de silla $E_3$  |  "
        "$t\\in[0,10]$ días  |  RK45",
        fontsize=12,
    )

    configuraciones = [
        (0, 1, (-8, 8), (-3.5, 3.5), "Plano $G$-$I$"),
        (0, 2, (-8, 8), (-3.5, 3.5), "Plano $G$-$\\beta$"),
    ]
    labels_vars = {
        (0, 1): ("$\\Delta G = G - G^*$ [mg/dL]",
                 "$\\Delta I = I - I^*$ [$\\mu$U/mL]"),
        (0, 2): ("$\\Delta G = G - G^*$ [mg/dL]",
                 "$\\Delta\\beta = \\beta - \\beta^*$ [u.a.]"),
    }

    for ax, (ix, iy, xlim, ylim, titulo) in zip(axes, configuraciones):

        ax.set_facecolor("#F8F9FA")

        # Campo vectorial
        campo_vectorial_local(ax, E3, ix, iy, xlim, ylim, n=18)

        # Trayectorias: condiciones iniciales en varias direcciones
        angulos = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        for i, ang in enumerate(angulos):
            dx = xlim[1] * 0.55 * np.cos(ang)
            dy = ylim[1] * 0.55 * np.sin(ang)
            y0 = E3.copy().astype(float)
            y0[ix] += dx
            y0[iy] += dy
            y0[0] = max(y0[0], 5.0)
            y0[1] = max(y0[1], 0.0)
            y0[2] = max(y0[2], 0.0)
            try:
                Y = simular_local(y0)
                ug = Y[ix] - E3[ix]
                uv = Y[iy] - E3[iy]
                col = PALETTE[i % len(PALETTE)]
                ax.plot(ug, uv, lw=1.6, color=col, alpha=0.82)
                m = len(ug) // 3
                if m + 5 < len(ug):
                    ax.annotate("",
                                xy=(ug[m + 5], uv[m + 5]),
                                xytext=(ug[m], uv[m]),
                                arrowprops=dict(arrowstyle="-|>",
                                                lw=1.0, color=col,
                                                mutation_scale=12))
            except Exception:
                continue

        # Variedad estable W^s(E3) — dirección horizontal (punteada)
        ax.axhline(0, color="#1D7382", lw=1.8, ls="--", alpha=0.85,
                   label="Variedad estable $W^s(E_3)$", zorder=5)

        # Variedad inestable W^u(E3) — dirección vertical (punteada)
        ax.axvline(0, color="#A94F6E", lw=1.8, ls="--", alpha=0.85,
                   label="Variedad inestable $W^u(E_3)$", zorder=5)

        # Equilibrio E3 en el origen
        ax.scatter(0, 0, s=180, color=colores_eq["E3"],
                   edgecolor="k", linewidth=1.2, zorder=9,
                   marker=markers_eq["E3"],
                   label="$E_3$ — Punto de silla")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xlabel, ylabel = labels_vars[(ix, iy)]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titulo, fontsize=10, pad=8)
        ax.legend(loc="upper right", fontsize=8, frameon=True,
                  framealpha=0.92)
        ax.grid(True, ls=":", alpha=0.25)

    plt.tight_layout()
    plt.savefig("figura2.jpeg", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: figura2.jpeg")


# =============================================================================
#  8. EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FASE 3 — Bosquejos cualitativos locales")
    print("Figuras 1 y 2 del informe")
    print("Modelo glucosa–insulina–célula beta")
    print("=" * 60)

    print("\n[1/2] Figura 1 — Nodos atractores E1 y E2...")
    figura1_nodos()

    print("\n[2/2] Figura 2 — Punto de silla E3...")
    figura2_silla()

    print("\n¡Figuras 1 y 2 generadas correctamente!")
    print("Archivos guardados: figura1.jpeg, figura2.jpeg")
