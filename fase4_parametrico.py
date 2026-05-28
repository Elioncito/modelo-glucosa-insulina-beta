# =============================================================================
#  FASE 4 — Análisis paramétrico
#  Modelo glucosa–insulina–célula beta
#
#  Descripción:
#      Estudia cómo cambia el comportamiento del sistema al variar:
#        F4.1 — Ge (aporte glucémico por epinefrina) y rho (efecto
#               adrenérgico sobre insulina)
#        F4.2 — r1 (crecimiento de células beta) y r2 (glucotoxicidad)
#
#  Genera:
#      - Curvas de dependencia de equilibrios vs Ge  (Fig. 7)
#      - Retrato fase global G-beta Escenario C       (Fig. 8)
#      - Diagrama del discriminante en plano r1-r2    (Fig. 9)
#      - Retratos fase G-beta para Delta>0 y Delta=0  (Fig. 10)
#      - Retrato fase G-beta para Delta<0             (Fig. 11)
#
#  Método numérico:
#      scipy.integrate.solve_ivp con RK45
#      rtol = 1e-8,  atol = 1e-10
#      t ∈ [0, 60] días,  4 000 puntos de evaluación
#
#  Uso:
#      Python 3.8+  |  pip install numpy scipy matplotlib
#      python fase4_parametrico.py
#      (En Google Colab: descomenta la línea %matplotlib inline)
#
#  Autor: Elioncito
# =============================================================================

# %matplotlib inline   # <- descomenta si usas Google Colab

# ── Librerías ────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
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
#  1. PARÁMETROS FIJOS DEL MODELO
# =============================================================================
R0    = 864.0
EGO   = 1.44
SI    = 0.72
sigma = 43.2
alpha = 20000.0
k     = 432.0
d0    = 0.06
r1_base = 0.84e-3
r2_base = 2.4e-6

PALETTE = [
    "#0B3C49","#145C6A","#1D7382","#2A8C9B",
    "#3CA3B2","#5CB8C4","#7AC7D1","#9AD8DF",
    "#6B3074","#A94F6E","#E07B00","#C1121F",
]

# =============================================================================
#  2. SISTEMA DE EDOS (parámetros variables)
# =============================================================================
def sistema(t, y, Ge, rho, r1=r1_base, r2=r2_base):
    """
    Sistema de 3 EDOs con parámetros Ge, rho, r1, r2 variables.
    y = [G, I, beta]
    """
    G = max(y[0], 1e-10)
    I = max(y[1], 0.0)
    B = max(y[2], 0.0)
    rpk = rho + k

    dG = R0 + Ge - (EGO + SI * I) * G
    dI = B * sigma * G**2 / (alpha + G**2) - rpk * I
    dB = (-d0 + r1 * G - r2 * G**2) * B
    return [dG, dI, dB]

# =============================================================================
#  3. CÁLCULO DE EQUILIBRIOS
# =============================================================================
def calcular_equilibrios(Ge, rho, r1=r1_base, r2=r2_base):
    """
    Calcula los puntos de equilibrio para los parámetros dados.

    Retorna lista de tuplas (G*, I*, beta*).
    """
    rpk = rho + k
    eqs = []

    # Caso A: beta* = 0  ->  I* = 0  ->  G1* = (R0 + Ge) / EGO
    G1 = (R0 + Ge) / EGO
    eqs.append((G1, 0.0, 0.0))

    # Caso B: beta* > 0  ->  raíces de r2*G^2 - r1*G + d0 = 0
    discriminante = r1**2 - 4 * r2 * d0
    if discriminante > 0:
        G2 = (r1 - np.sqrt(discriminante)) / (2 * r2)
        G3 = (r1 + np.sqrt(discriminante)) / (2 * r2)
        for G_eq in [G2, G3]:
            I_eq = (R0 + Ge - EGO * G_eq) / (SI * G_eq)
            if I_eq > 0:
                beta_eq = rpk * I_eq * (alpha + G_eq**2) / (sigma * G_eq**2)
                if beta_eq > 0:
                    eqs.append((G_eq, I_eq, beta_eq))
    elif discriminante == 0:
        G0 = r1 / (2 * r2)
        I0 = (R0 + Ge - EGO * G0) / (SI * G0)
        if I0 > 0:
            beta0 = rpk * I0 * (alpha + G0**2) / (sigma * G0**2)
            eqs.append((G0, I0, beta0))

    return eqs, discriminante

# =============================================================================
#  4. FUNCIÓN DE INTEGRACIÓN
# =============================================================================
def simular(y0, Ge, rho, r1=r1_base, r2=r2_base,
            t_span=(0, 60), n_pts=4000):
    """
    Integra el sistema con RK45.

    Parámetros
    ----------
    y0     : condición inicial [G0, I0, beta0]
    Ge     : aporte glucémico por epinefrina (mg/dL/día)
    rho    : efecto adrenérgico sobre insulina (1/día)
    r1, r2 : parámetros de dinámica de células beta
    t_span : intervalo de integración en días (t0, tf)
    n_pts  : número de puntos de evaluación

    Retorna
    -------
    t : array de tiempos
    Y : array (3, n_pts) con trayectorias [G, I, beta]
    """
    sol = solve_ivp(
        sistema,
        t_span,
        [max(y0[0], 5.0), max(y0[1], 0.0), max(y0[2], 0.0)],
        args=(Ge, rho, r1, r2),
        method="RK45",
        t_eval=np.linspace(t_span[0], t_span[1], n_pts),
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )
    return sol.t, sol.y

# =============================================================================
#  5. F4.1 — VARIACIÓN DE Ge  (Fig. 7)
#     Dependencia de equilibrios respecto a Ge, con rho = 0.41 fijo
# =============================================================================
def fig7_dependencia_ge():
    """
    Curvas G1*(Ge), I2*(Ge), I3*(Ge), beta2*(Ge), beta3*(Ge).
    rho fijo = 0.41  |  Ge ∈ [0, 500] mg/dL/día
    """
    rho_fijo = 0.41
    Ge_vals  = np.linspace(0, 500, 300)

    G1_vals, I2_vals, I3_vals, b2_vals, b3_vals = [], [], [], [], []

    for Ge_v in Ge_vals:
        eqs, _ = calcular_equilibrios(Ge_v, rho_fijo)
        G1_vals.append(eqs[0][0])
        if len(eqs) >= 3:
            I2_vals.append(eqs[1][1])
            I3_vals.append(eqs[2][1])
            b2_vals.append(eqs[1][2])
            b3_vals.append(eqs[2][2])
        else:
            I2_vals.append(np.nan)
            I3_vals.append(np.nan)
            b2_vals.append(np.nan)
            b3_vals.append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Figura 7 — Dependencia de los equilibrios respecto a $G_e$\n"
        "($\\rho = 0.41$ fijo,  $r_1, r_2$ en valores base)",
        fontsize=12,
    )

    # Panel izquierdo: G1*(Ge)
    axes[0].plot(Ge_vals, G1_vals, color="#C1121F", lw=2)
    axes[0].axvline(140, ls="--", color="gray", lw=1, label="$G_e=140$ (base)")
    axes[0].set_xlabel("$G_e$ [mg dL$^{-1}$ d$^{-1}$]")
    axes[0].set_ylabel("$G_1^*$ [mg/dL]")
    axes[0].set_title("Glucemia de colapso $G_1^*(G_e)$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls=":", alpha=0.4)

    # Panel central: I2*(Ge) e I3*(Ge)
    axes[1].plot(Ge_vals, I2_vals, color="#0B3C49", lw=2, label="$I_2^*$ ($G^*=100$)")
    axes[1].plot(Ge_vals, I3_vals, color="#E07B00", lw=2, label="$I_3^*$ ($G^*=250$)")
    axes[1].axvline(140, ls="--", color="gray", lw=1, label="$G_e=140$ (base)")
    axes[1].set_xlabel("$G_e$ [mg dL$^{-1}$ d$^{-1}$]")
    axes[1].set_ylabel("$I^*$ [$\\mu$U/mL]")
    axes[1].set_title("Insulinemia de equilibrio $I^*(G_e)$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls=":", alpha=0.4)

    # Panel derecho: beta2*(Ge) y beta3*(Ge)
    axes[2].plot(Ge_vals, b2_vals, color="#0B3C49", lw=2, label="$\\beta_2^*$ ($G^*=100$)")
    axes[2].plot(Ge_vals, b3_vals, color="#E07B00", lw=2, label="$\\beta_3^*$ ($G^*=250$)")
    axes[2].axvline(140, ls="--", color="gray", lw=1, label="$G_e=140$ (base)")
    axes[2].set_xlabel("$G_e$ [mg dL$^{-1}$ d$^{-1}$]")
    axes[2].set_ylabel("$\\beta^*$ [u.a.]")
    axes[2].set_title("Masa $\\beta^*$ de equilibrio $\\beta^*(G_e)$")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig("fase4_Ge_parametrico.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: fase4_Ge_parametrico.png")


# =============================================================================
#  6. F4.1 — ESCENARIO C  (Fig. 8)
#     Retrato fase global G-beta con Ge=300, rho=2.00
# =============================================================================

# Condiciones iniciales para retratos globales
# G ∈ [80, 850] mg/dL | beta ∈ [0, 460] u.a.
ICS_GLOBALES = [
    [80,   20, 380], [120,  8, 300], [150, 18, 400],
    [180,  15, 350], [200,  6, 150], [300, 12, 200],
    [350,   2,  80], [400,  8, 120], [450,  4,  60],
    [500, 0.5,  20], [600, 0.2,   5],[650, 0.5,  10],
    [700, 0.1,   2], [100,  3, 100], [250,  4,  52],
    [80,   1,  200],
]


def retrato_global_Gbeta(Ge, rho, r1=r1_base, r2=r2_base, titulo=""):
    """
    Retrato fase global en el plano G-beta.
    t ∈ [0, 60] días | 4 000 puntos | RK45
    """
    eqs, _ = calcular_equilibrios(Ge, rho, r1, r2)
    colores = ["#C1121F", "#0B3C49", "#E07B00"]
    nombres = ["$E_1$", "$E_2$", "$E_3$"]
    mkrs    = ["^", "o", "s"]
    tipos_l = ["Nodo estable", "Nodo estable", "Punto silla"]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_facecolor("#F8F9FA")
    ax.set_title(titulo or
                 f"Retrato fase global G–$\\beta$\n"
                 f"$G_e={Ge}$, $\\rho={rho}$  |  $t\\in[0,60]$ días  |  RK45",
                 fontsize=11)

    for i, y0 in enumerate(ICS_GLOBALES):
        try:
            _, Y = simular(y0, Ge, rho, r1, r2, t_span=(0, 60), n_pts=4000)
        except Exception:
            continue
        col = PALETTE[i % len(PALETTE)]
        ax.plot(Y[0], Y[2], lw=1.6, color=col, alpha=0.82)
        m = len(Y[0]) // 3
        if m + 4 < len(Y[0]):
            ax.annotate("", xy=(Y[0][m+4], Y[2][m+4]),
                        xytext=(Y[0][m], Y[2][m]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                        color=col, mutation_scale=11))
        ax.scatter(Y[0][0], Y[2][0], s=25, color=col, zorder=4)

    for j, (G_eq, I_eq, b_eq) in enumerate(eqs):
        if j < len(nombres):
            ax.scatter(G_eq, b_eq, s=180, color=colores[j],
                       edgecolor="k", linewidth=1.2, zorder=7,
                       marker=mkrs[j],
                       label=f"{nombres[j]} — {tipos_l[j]}")
            ax.text(G_eq + 8, b_eq + 4, nombres[j],
                    fontsize=10, weight="bold", color=colores[j])

    ax.set_xlabel("$G$ [mg/dL]")
    ax.set_ylabel("$\\beta$ [u.a.]")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.35)
    plt.tight_layout()
    return fig, ax


def fig8_escenario_C():
    fig, ax = retrato_global_Gbeta(
        Ge=300, rho=2.00,
        titulo="Figura 8 — Retrato fase global G–$\\beta$\n"
               "$G_e=300$, $\\rho=2.00$ (alta exposición adrenal)  |  "
               "$t\\in[0,60]$ días  |  RK45"
    )
    plt.savefig("fase4_F41_escenarioC_Gbeta.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: fase4_F41_escenarioC_Gbeta.png")


# =============================================================================
#  7. F4.2 — DIAGRAMA DEL DISCRIMINANTE en plano r1-r2  (Fig. 9)
# =============================================================================
def fig9_discriminante():
    """
    Diagrama del discriminante Delta = r1² - 4*r2*d0 en el plano r1-r2.
    Muestra la curva crítica r1 = 2*sqrt(r2*d0).
    """
    r2_vals = np.linspace(0.5e-6, 6e-6, 400)
    r1_crit = 2 * np.sqrt(r2_vals * d0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#F8F9FA")

    # Región Delta > 0 (biestabilidad)
    ax.fill_between(r2_vals * 1e6, r1_crit * 1e3,
                    np.full_like(r2_vals, 2.5e-3) * 1e3,
                    alpha=0.18, color="#0B3C49", label="$\\Delta>0$: biestabilidad")

    # Región Delta < 0 (solo colapso)
    ax.fill_between(r2_vals * 1e6, np.zeros_like(r2_vals),
                    r1_crit * 1e3,
                    alpha=0.18, color="#C1121F", label="$\\Delta<0$: solo $E_1$ (colapso)")

    # Curva crítica
    ax.plot(r2_vals * 1e6, r1_crit * 1e3, color="#8B0000", lw=2.5,
            label="$\\Delta=0$: bifurcación silla-nodo\n$r_1 = 2\\sqrt{r_2 d_0}$")

    # Caso base
    ax.scatter(r2_base * 1e6, r1_base * 1e3, s=160, color="#0B3C49",
               edgecolor="k", linewidth=1.2, zorder=7, marker="o",
               label=f"Caso base ($r_1={r1_base*1e3:.2f}\\times10^{{-3}}$, "
                     f"$r_2={r2_base*1e6:.1f}\\times10^{{-6}}$)")

    # Umbral crítico r1,c
    r1c = 2 * np.sqrt(r2_base * d0)
    ax.scatter(r2_base * 1e6, r1c * 1e3, s=140, color="#E07B00",
               edgecolor="k", linewidth=1.2, zorder=7, marker="D",
               label=f"Umbral $r_{{1,c}} \\approx {r1c*1e3:.4f}\\times10^{{-3}}$")

    # Escenario C (Delta < 0)
    ax.scatter(r2_base * 1e6, 0.3e-3 * 1e3, s=140, color="#C1121F",
               edgecolor="k", linewidth=1.2, zorder=7, marker="v",
               label="Esc. C ($r_1=3.0\\times10^{-4}$, $\\Delta<0$)")

    ax.set_xlabel("$r_2 \\times 10^{-6}$ [mg$^{-2}$ dL$^2$ d$^{-1}$]")
    ax.set_ylabel("$r_1 \\times 10^{-3}$ [mg$^{-1}$ dL d$^{-1}$]")
    ax.set_title("Figura 9 — Diagrama del discriminante $\\Delta = r_1^2 - 4r_2 d_0$\n"
                 "en el plano $r_1$-$r_2$", fontsize=12)
    ax.set_xlim(0.5, 6.2)
    ax.set_ylim(0, 2.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig("fase4_discriminante_r1_r2.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: fase4_discriminante_r1_r2.png")


# =============================================================================
#  8. F4.2 — RETRATOS FASE G-beta PARA DELTA>0 Y DELTA=0  (Fig. 10)
# =============================================================================
def fig10_delta_pos_cero():
    """
    Dos retratos G-beta lado a lado: Delta>0 y Delta=0.
    t ∈ [0, 60] días | 4 000 puntos | RK45
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle("Figura 10 — Retratos fase G–$\\beta$  para $\\Delta>0$ y $\\Delta=0$\n"
                 "$t\\in[0,60]$ días  |  RK45", fontsize=12)

    escenarios = [
        {"r1": r1_base, "r2": r2_base,
         "label": "$\\Delta>0$: biestabilidad\n"
                  f"$r_1={r1_base*1e3:.2f}\\times10^{{-3}}$",
         "colores": ["#C1121F", "#0B3C49", "#E07B00"],
         "nombres": ["$E_1$", "$E_2$", "$E_3$"]},
        {"r1": 2*np.sqrt(r2_base*d0), "r2": r2_base,
         "label": "$\\Delta=0$: bifurcación silla-nodo\n"
                  f"$r_{{1,c}}\\approx{2*np.sqrt(r2_base*d0)*1e3:.4f}\\times10^{{-3}}$",
         "colores": ["#C1121F", "#6B3074"],
         "nombres": ["$E_1$", "$E_0$ (no hiperbólico)"]},
    ]

    for ax, esc in zip(axes, escenarios):
        ax.set_facecolor("#F8F9FA")
        eqs, _ = calcular_equilibrios(140, 0.41, esc["r1"], esc["r2"])
        mkrs = ["^", "D", "s"]

        for i, y0 in enumerate(ICS_GLOBALES):
            try:
                _, Y = simular(y0, 140, 0.41, esc["r1"], esc["r2"],
                               t_span=(0, 60), n_pts=4000)
            except Exception:
                continue
            col = PALETTE[i % len(PALETTE)]
            ax.plot(Y[0], Y[2], lw=1.5, color=col, alpha=0.80)
            m = len(Y[0]) // 3
            if m + 4 < len(Y[0]):
                ax.annotate("", xy=(Y[0][m+4], Y[2][m+4]),
                            xytext=(Y[0][m], Y[2][m]),
                            arrowprops=dict(arrowstyle="-|>", lw=0.9,
                                            color=col, mutation_scale=10))

        for j, (G_eq, _, b_eq) in enumerate(eqs):
            if j < len(esc["nombres"]):
                ax.scatter(G_eq, b_eq, s=180, color=esc["colores"][j],
                           edgecolor="k", linewidth=1.2, zorder=7,
                           marker=mkrs[j], label=esc["nombres"][j])
                ax.text(G_eq + 8, b_eq + 4, esc["nombres"][j],
                        fontsize=9, weight="bold", color=esc["colores"][j])

        ax.set_xlabel("$G$ [mg/dL]")
        ax.set_ylabel("$\\beta$ [u.a.]")
        ax.set_title(esc["label"], fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(True, ls=":", alpha=0.35)

    plt.tight_layout()
    plt.savefig("fase4_F42_delta_pos_cero_Gbeta.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: fase4_F42_delta_pos_cero_Gbeta.png")


# =============================================================================
#  9. F4.2 — RETRATO FASE G-beta PARA DELTA<0  (Fig. 11)
# =============================================================================
def fig11_delta_neg():
    """
    Retrato fase G-beta cuando Delta < 0.
    Solo existe E1; no hay equilibrios con beta > 0.
    t ∈ [0, 60] días | 4 000 puntos | RK45
    """
    r1_c = 3.0e-4
    r2_c = r2_base
    eqs, delta = calcular_equilibrios(140, 0.41, r1_c, r2_c)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_facecolor("#F8F9FA")
    ax.set_title(
        "Figura 11 — Retrato fase G–$\\beta$  para $\\Delta<0$\n"
        f"$r_1={r1_c:.1e}$, $r_2={r2_c:.1e}$,  "
        f"$\\Delta={delta:.2e}<0$  |  $t\\in[0,60]$ días  |  RK45",
        fontsize=11,
    )

    for i, y0 in enumerate(ICS_GLOBALES):
        try:
            _, Y = simular(y0, 140, 0.41, r1_c, r2_c,
                           t_span=(0, 60), n_pts=4000)
        except Exception:
            continue
        col = PALETTE[i % len(PALETTE)]
        ax.plot(Y[0], Y[2], lw=1.6, color=col, alpha=0.82)
        m = len(Y[0]) // 3
        if m + 4 < len(Y[0]):
            ax.annotate("", xy=(Y[0][m+4], Y[2][m+4]),
                        xytext=(Y[0][m], Y[2][m]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                        color=col, mutation_scale=11))
        ax.scatter(Y[0][0], Y[2][0], s=25, color=col, zorder=4)

    # Solo E1
    G1, _, _ = eqs[0]
    ax.scatter(G1, 0, s=200, color="#C1121F", edgecolor="k",
               linewidth=1.2, zorder=7, marker="^",
               label="$E_1$ Nodo estable (único equilibrio)")
    ax.text(G1 + 8, 4, "$E_1$", fontsize=10, weight="bold", color="#C1121F")

    ax.set_xlabel("$G$ [mg/dL]")
    ax.set_ylabel("$\\beta$ [u.a.]")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.grid(True, ls=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig("fase4_F42_delta_neg_Gbeta.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: fase4_F42_delta_neg_Gbeta.png")


# =============================================================================
#  10. EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FASE 4 — Análisis paramétrico")
    print("Modelo glucosa–insulina–célula beta")
    print("=" * 60)

    print("\n[1/5] Fig. 7 — Dependencia de equilibrios vs Ge...")
    fig7_dependencia_ge()

    print("\n[2/5] Fig. 8 — Retrato fase Escenario C (Ge=300, rho=2.00)...")
    fig8_escenario_C()

    print("\n[3/5] Fig. 9 — Diagrama del discriminante r1-r2...")
    fig9_discriminante()

    print("\n[4/5] Fig. 10 — Retratos fase Delta>0 y Delta=0...")
    fig10_delta_pos_cero()

    print("\n[5/5] Fig. 11 — Retrato fase Delta<0...")
    fig11_delta_neg()

    print("\n¡Todas las figuras generadas correctamente!")
    print("Archivos guardados en el directorio de trabajo.")
