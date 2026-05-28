# =============================================================================
#  FASE 3 — Análisis en el espacio fase
#  Modelo glucosa–insulina–célula beta
#
#  Descripción:
#      Genera los retratos fase locales (R² y R³) y globales (R² y R³)
#      del sistema de EDOs. Incluye bosquejos cualitativos y retratos
#      numéricos alrededor de cada equilibrio.
#
#  Método numérico:
#      scipy.integrate.solve_ivp con RK45
#      rtol = 1e-8,  atol = 1e-10
#
#  Retratos locales:   t ∈ [0, 10] días,  5 000 puntos
#  Retratos globales:  t ∈ [0, 60] días,  4 000 puntos
#
#  Uso:
#      Python 3.8+  |  pip install numpy scipy matplotlib
#      python fase3_espacio_fase.py
#      (En Google Colab: descomenta la línea %matplotlib inline)
#
#  Autor: Elioncito
# =============================================================================

# %matplotlib inline   # <- descomenta si usas Google Colab

# ── Librerías ────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     110,
})

# =============================================================================
#  1. PARÁMETROS DEL MODELO (caso base)
# =============================================================================
R0    = 864.0       # mg/dL/día  — producción basal de glucosa
Ge    = 140.0       # mg/dL/día  — aporte glucémico por epinefrina
EGO   = 1.44        # 1/día      — efectividad glucosa sin insulina
SI    = 0.72        # mL/µU/día  — sensibilidad a la insulina
sigma = 43.2        # µU/mL/día  — tasa máx. de secreción de insulina
alpha = 20000.0     # (mg/dL)²   — parámetro de saturación
rho   = 0.41        # 1/día      — efecto epinefrina sobre insulina
k     = 432.0       # 1/día      — tasa de depuración de insulina
d0    = 0.06        # 1/día      — mortalidad basal células beta
r1    = 0.84e-3     # 1/(mg/dL·día) — crecimiento beta inducido por glucosa
r2    = 2.4e-6      # 1/(mg/dL)²/día — daño por glucotoxicidad

rpk = rho + k       # coeficiente total de eliminación de insulina = 432.41

# =============================================================================
#  2. EQUILIBRIOS
# =============================================================================
E1 = np.array([697.22,  0.0,     0.0    ])   # colapso pancreático
E2 = np.array([100.0,   11.94,   358.67 ])   # homeostasis
E3 = np.array([250.0,    3.578,   47.27 ])   # punto de silla (umbral)

equilibrios = {"$E_1$": E1, "$E_2$": E2, "$E_3$": E3}
tipos       = {"$E_1$": "Nodo atractor", "$E_2$": "Nodo atractor", "$E_3$": "Punto de silla"}
colores_eq  = {"$E_1$": "#C1121F", "$E_2$": "#0B3C49", "$E_3$": "#E07B00"}
markers_eq  = {"$E_1$": "^",       "$E_2$": "o",       "$E_3$": "s"      }

PALETTE = [
    "#0B3C49","#145C6A","#1D7382","#2A8C9B",
    "#3CA3B2","#5CB8C4","#7AC7D1","#9AD8DF",
    "#6B3074","#A94F6E","#E07B00","#C1121F",
]

# =============================================================================
#  3. SISTEMA DE EDOS
# =============================================================================
def sistema(t, y):
    """
    Sistema de 3 EDOs acopladas.
    y = [G, I, beta]
    """
    G = max(y[0], 1e-10)
    I = max(y[1], 0.0)
    B = max(y[2], 0.0)

    dG = R0 + Ge - (EGO + SI * I) * G
    dI = B * sigma * G**2 / (alpha + G**2) - rpk * I
    dB = (-d0 + r1 * G - r2 * G**2) * B
    return [dG, dI, dB]

# =============================================================================
#  4. FUNCIÓN DE INTEGRACIÓN
# =============================================================================
def simular(y0, t_span=(0, 60), n_pts=4000):
    """
    Integra el sistema con RK45.

    Parámetros
    ----------
    y0     : condición inicial [G0, I0, beta0]
    t_span : intervalo de integración en días (t0, tf)
    n_pts  : número de puntos de evaluación

    Retorna
    -------
    t : array de tiempos
    Y : array (3, n_pts) con las trayectorias [G, I, beta]
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
#  5. RETRATOS FASE LOCALES EN R² (coordenadas desplazadas)
# =============================================================================
def retrato_local_2D(eq_name, Estar, t_max=10, n_pts=5000):
    """
    Genera retratos locales en los planos G-I y G-beta
    usando coordenadas desplazadas (u_G, u_I, u_beta).

    Intervalo de integración: t ∈ [0, t_max] días
    Condiciones iniciales: perturbaciones ±5% alrededor de Estar,
    distribuidas en 8 direcciones del espacio de estados.
    """
    # Perturbaciones ±5% (8 condiciones iniciales)
    deltas = [0.05, -0.05]
    ics = []
    for dg in deltas:
        for di in deltas:
            for db in deltas[:1]:   # evitar combinatoria excesiva
                ics.append([
                    Estar[0] * (1 + dg),
                    max(Estar[1] * (1 + di), 0.01),
                    max(Estar[2] * (1 + db), 0.0),
                ])
    # agregar más variedad
    for scale in [0.10, -0.10, 0.15, -0.15]:
        ics.append([
            Estar[0] * (1 + scale),
            max(Estar[1] * (1 - scale * 0.5), 0.01),
            max(Estar[2] * (1 + scale * 0.3), 0.0),
        ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Retrato fase local — {eq_name}  ({tipos[eq_name]})\n"
        f"Coordenadas desplazadas  |  $t\\in[0,{t_max}]$ días  |  RK45",
        fontsize=12,
    )
    planes = [("GI", 0, 1), ("Gbeta", 0, 2)]
    labels_vars = {
        0: "$\\Delta G = G - G^*$ [mg/dL]",
        1: "$\\Delta I = I - I^*$ [$\\mu$U/mL]",
        2: "$\\Delta\\beta = \\beta - \\beta^*$ [u.a.]",
    }

    for ax, (_, ix, iy) in zip(axes, planes):
        ax.set_facecolor("#F8F9FA")
        for i, y0 in enumerate(ics):
            try:
                t_, Y = simular(y0, t_span=(0, t_max), n_pts=n_pts)
            except Exception:
                continue
            col = PALETTE[i % len(PALETTE)]
            ug = Y[ix] - Estar[ix]
            uv = Y[iy] - Estar[iy]
            ax.plot(ug, uv, lw=1.5, color=col, alpha=0.85)
            m = len(ug) // 3
            if m + 4 < len(ug):
                ax.annotate("", xy=(ug[m+4], uv[m+4]), xytext=(ug[m], uv[m]),
                            arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                            color=col, mutation_scale=11))

        ax.scatter(0, 0, s=160, color=colores_eq[eq_name],
                   edgecolor="k", linewidth=1.2, zorder=7,
                   marker=markers_eq[eq_name], label=eq_name)
        ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel(labels_vars[ix])
        ax.set_ylabel(labels_vars[iy])
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        ax.grid(True, ls=":", alpha=0.3)

    plt.tight_layout()
    fname = f"fase3_local_{eq_name.strip('$')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Guardado: {fname}")


# =============================================================================
#  6. RETRATOS FASE GLOBALES EN R²
# =============================================================================

# Condiciones iniciales globales
# G ∈ [80, 750] mg/dL | I ∈ [0, 25] µU/mL | beta ∈ [0, 460] u.a.
ICS_GLOBALES = [
    [80,   20,   380],   # cercana a E2
    [120,   8,   300],   # cercana a E2
    [150,  18,   400],   # cercana a E2
    [180,  15,   350],   # cercana a E2
    [200,   6,   150],   # zona intermedia
    [300,  12,   200],   # zona intermedia
    [350,   2,    80],   # próxima al umbral E3
    [400,   8,   120],   # zona intermedia alta
    [450,   4,    60],   # próxima al umbral E3
    [500,   0.5,  20],   # deriva hacia E1
    [600,   0.2,   5],   # deriva hacia E1
    [650,   0.5,  10],   # deriva hacia E1
    [700,   0.1,   2],   # deriva hacia E1
    [100,   3,   100],   # intermedia baja en beta
    [250,   4,    52],   # muy cercana a E3
    [80,    1,   200],   # glucosa baja, beta moderada
]


def retrato_global_2D(plano="GI"):
    """
    Genera retrato fase global en el plano especificado.

    plano : 'GI' | 'Gbeta' | 'Ibeta'
    Intervalo de integración: t ∈ [0, 60] días, 4 000 puntos.
    """
    idx = {"GI": (0, 1), "Gbeta": (0, 2), "Ibeta": (1, 2)}
    lbl = {
        0: "$G$ [mg/dL]",
        1: "$I$ [$\\mu$U/mL]",
        2: "$\\beta$ [u.a.]",
    }
    ix, iy = idx[plano]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_facecolor("#F8F9FA")
    fig.suptitle(
        f"Retrato fase global — Plano {plano.replace('beta', '-$\\\\beta$')}\n"
        f"$t\\in[0,60]$ días  |  16 condiciones iniciales  |  RK45",
        fontsize=12,
    )

    for i, y0 in enumerate(ICS_GLOBALES):
        try:
            _, Y = simular(y0, t_span=(0, 60), n_pts=4000)
        except Exception:
            continue
        col = PALETTE[i % len(PALETTE)]
        ax.plot(Y[ix], Y[iy], lw=1.6, color=col, alpha=0.82)
        m = len(Y[0]) // 3
        if m + 4 < len(Y[0]):
            ax.annotate("", xy=(Y[ix][m+4], Y[iy][m+4]),
                        xytext=(Y[ix][m], Y[iy][m]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                        color=col, mutation_scale=11))
        ax.scatter(Y[ix][0], Y[iy][0], s=28, color=col, zorder=4)

    for name, Eq in equilibrios.items():
        ax.scatter(Eq[ix], Eq[iy], s=180,
                   color=colores_eq[name], edgecolor="k",
                   linewidth=1.2, zorder=7, marker=markers_eq[name],
                   label=f"{name} — {tipos[name]}")
        ax.text(Eq[ix] + 6, Eq[iy] + 2, name,
                fontsize=10, weight="bold", color=colores_eq[name])

    ax.set_xlabel(lbl[ix])
    ax.set_ylabel(lbl[iy])
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.35)
    plt.tight_layout()
    fname = f"fase3_global_{plano}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Guardado: {fname}")


# =============================================================================
#  7. RETRATOS FASE LOCALES EN R³
# =============================================================================
def retrato_local_3D(eq_name, Estar, t_max=10, n_pts=5000):
    """
    Retrato tridimensional alrededor de un equilibrio
    usando coordenadas desplazadas (u_G, u_I, u_beta).
    Intervalo: t ∈ [0, t_max] días.
    """
    ics = []
    for dg in [0.08, -0.08, 0.12]:
        for di in [0.06, -0.06]:
            ics.append([
                Estar[0] * (1 + dg),
                max(Estar[1] * (1 + di), 0.01),
                max(Estar[2] * (1 - dg * 0.4), 0.0),
            ])

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#F8F9FA")

    for i, y0 in enumerate(ics):
        try:
            _, Y = simular(y0, t_span=(0, t_max), n_pts=n_pts)
        except Exception:
            continue
        col = PALETTE[i % len(PALETTE)]
        ug = Y[0] - Estar[0]
        ui = Y[1] - Estar[1]
        ub = Y[2] - Estar[2]
        ax.plot(ug, ui, ub, lw=1.4, color=col, alpha=0.80)

    ax.scatter(0, 0, 0, s=200, color=colores_eq[eq_name],
               edgecolor="k", linewidth=1.2, zorder=7,
               marker=markers_eq[eq_name], label=eq_name)

    ax.set_xlabel("$\\Delta G$ [mg/dL]", labelpad=8)
    ax.set_ylabel("$\\Delta I$ [$\\mu$U/mL]", labelpad=8)
    ax.set_zlabel("$\\Delta\\beta$ [u.a.]", labelpad=8)
    ax.set_title(
        f"Retrato local 3D — {eq_name}  ({tipos[eq_name]})\n"
        f"$t\\in[0,{t_max}]$ días  |  RK45",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    fname = f"fase3_local3D_{eq_name.strip('$')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Guardado: {fname}")


# =============================================================================
#  8. RETRATO FASE GLOBAL EN R³
# =============================================================================
def retrato_global_3D():
    """
    Retrato fase global en el espacio completo (G, I, beta).
    Intervalo: t ∈ [0, 60] días, 4 000 puntos.
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Retrato fase global en $\\mathbb{R}^3 = (G, I, \\beta)$\n"
        "$t\\in[0,60]$ días  |  16 condiciones iniciales  |  RK45",
        fontsize=12,
    )

    for i, y0 in enumerate(ICS_GLOBALES):
        try:
            _, Y = simular(y0, t_span=(0, 60), n_pts=4000)
        except Exception:
            continue
        col = PALETTE[i % len(PALETTE)]
        ax.plot(Y[0], Y[1], Y[2], lw=1.4, color=col, alpha=0.75)
        ax.scatter(Y[0][0], Y[1][0], Y[2][0], s=25, color=col, zorder=4)

    for name, Eq in equilibrios.items():
        ax.scatter(Eq[0], Eq[1], Eq[2], s=200,
                   color=colores_eq[name], edgecolor="k",
                   linewidth=1.2, zorder=7, marker=markers_eq[name],
                   label=f"{name} — {tipos[name]}")

    ax.set_xlabel("$G$ [mg/dL]", labelpad=10)
    ax.set_ylabel("$I$ [$\\mu$U/mL]", labelpad=10)
    ax.set_zlabel("$\\beta$ [u.a.]", labelpad=10)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("fase3_global_3D.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: fase3_global_3D.png")


# =============================================================================
#  9. EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FASE 3 — Análisis en el espacio fase")
    print("Modelo glucosa–insulina–célula beta")
    print("=" * 60)

    print("\n[1/4] Retratos locales en R²...")
    for name, Eq in equilibrios.items():
        print(f"  → {name}")
        retrato_local_2D(name, Eq)

    print("\n[2/4] Retratos globales en R²...")
    for plano in ["GI", "Gbeta"]:
        print(f"  → Plano {plano}")
        retrato_global_2D(plano)

    print("\n[3/4] Retratos locales en R³...")
    for name, Eq in equilibrios.items():
        print(f"  → {name}")
        retrato_local_3D(name, Eq)

    print("\n[4/4] Retrato global en R³...")
    retrato_global_3D()

    print("\n¡Todas las figuras generadas correctamente!")
    print("Archivos guardados en el directorio de trabajo.")
