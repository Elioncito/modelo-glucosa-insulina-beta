# =============================================================================
#  FASE 3 — Bosquejos cualitativos locales
#  Figuras 1 y 2 del informe
#
#  Figura 1 — Bosquejos alrededor de los nodos atractores E1 y E2
#             Paneles: E1 plano G-I | E1 plano G-beta | E2 plano G-I
#  Figura 2 — Bosquejo alrededor del punto de silla E3
#             Paneles: E3 plano G-I | E3 plano G-beta
#             Con variedades estable e inestable (lineas punteadas)
#
#  Uso:
#      Python 3.8+  |  pip install numpy matplotlib
#      python fase3_bosquejos.py
#
#  Autor: Elioncito
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# =============================================================================
#  FIGURA 1 — NODOS ATRACTORES E1 y E2
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

DARK  = "#173d49"
MID   = "#6aa7b6"
LIGHT = "#d9e9ed"
DOT   = "#a33a30"


def bezier_f1(ax, pts, color=DARK, lw=1.4, alpha=1, arrow=True):
    verts = [pts[0]]
    codes = [Path.MOVETO]
    for i in range(1, len(pts), 3):
        verts += pts[i:i+3]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-", lw=lw,
        color=color, alpha=alpha, zorder=3
    ))
    if arrow:
        s = path.interpolated(60).vertices
        k = len(s) // 2
        ax.annotate(
            "", xy=s[k+1], xytext=s[k-3],
            arrowprops=dict(arrowstyle="->", color=color,
                            lw=lw, mutation_scale=8)
        )


def campo_vectorial_f1(ax):
    x = np.linspace(-3, 3, 25)
    y = np.linspace(-3, 3, 25)
    X, Y = np.meshgrid(x, y)
    U = -0.52 * X + 0.10 * Y
    V = -0.18 * X - 0.80 * Y
    N = np.hypot(U, V)
    ax.quiver(X, Y, U/N, V/N,
              color=LIGHT, angles="xy", scale_units="xy",
              scale=11.5, width=0.0027,
              headwidth=3.2, headlength=4.2, alpha=0.74)


def curvas_f1(ax):
    oscuras = [
        [(-2.25,3),(-1.92,2.55),(-0.74,1.78),(-0.12,0.62)],
        [(-1.42,3),(-1.18,2.35),(-0.42,1.38),(-0.08,0.58)],
        [(1.42,3),(1.18,2.35),(0.42,1.38),(0.10,0.60)],
        [(2.22,3),(1.88,2.55),(0.74,1.78),(0.15,0.64)],
        [(-3,-1.18),(-1.72,-0.98),(-0.42,-0.42),(0.02,-0.04)],
        [(3,-1.18),(1.72,-0.98),(0.42,-0.42),(-0.02,-0.04)],
        [(-3,-3),(-2.05,-2.26),(-0.74,-1.48),(-0.16,-0.58)],
        [(-1.45,-3),(-1.02,-2.18),(-0.36,-1.22),(-0.08,-0.58)],
        [(1.43,-3),(0.98,-2.18),(0.34,-1.20),(0.08,-0.58)],
        [(3,-3),(2.05,-2.26),(0.74,-1.48),(0.16,-0.58)],
    ]
    claras = [
        [(-3,1.18),(-1.82,1.02),(-0.58,0.58),(-0.12,0.20)],
        [(3,1.18),(1.82,1.02),(0.58,0.58),(0.12,0.20)],
        [(-3,-3),(-1.58,-2.22),(-0.48,-1.18),(-0.10,-0.60)],
        [(3,-3),(1.58,-2.22),(0.48,-1.18),(0.10,-0.60)],
        [(-0.62,3),(-0.42,2.16),(-0.09,1.18),(0,0.58)],
        [(0.62,3),(0.42,2.16),(0.09,1.18),(0,0.58)],
    ]
    for c in claras:
        bezier_f1(ax, c, MID, 1.1, 0.5, False)
    for c in oscuras:
        bezier_f1(ax, c, DARK, 1.4, 0.95, True)


def panel_f1(ax, titulo, ylabel, punto, autovalores):
    campo_vectorial_f1(ax)
    curvas_f1(ax)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-3, 4, 1)); ax.set_yticks(np.arange(-3, 4, 1))
    ax.grid(True, color="#e8e8e8", lw=0.55)
    ax.axhline(0, color="#b9b9b9", lw=0.75)
    ax.axvline(0, color="#b9b9b9", lw=0.75)
    ax.set_title(titulo)
    ax.set_xlabel(r"$\Delta G = G - G^*$")
    ax.set_ylabel(ylabel)
    ax.scatter([0], [0], s=78, facecolor=DOT, edgecolor="#2b1613", zorder=10)
    ax.text(0.13, 0.15, punto, fontsize=13, weight="bold")
    ax.text(-3.22, 3.08,
            "Tipo: Nodo atractor\nAutovalores (submatriz):\n" + autovalores,
            ha="left", va="top", fontsize=7.4, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bfc4c7"))


def figura1():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.25), dpi=180)
    panel_f1(axes[0],
             r"Bosquejo local alrededor de P1 en el plano $G$-$I$",
             r"$\Delta I = I - I^*$", "P1", "-1.440, -432.410")
    panel_f1(axes[1],
             r"Bosquejo local alrededor de P1 en el plano $G$-$\beta$",
             r"$\Delta \beta = \beta - \beta^*$", "P1", "-1.440, -0.641")
    panel_f1(axes[2],
             r"Bosquejo local alrededor de P2 en el plano $G$-$I$",
             r"$\Delta I = I - I^*$", "P2", "-22.125, -420.325")
    plt.tight_layout()
    plt.savefig("figura1.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura1.jpeg guardada")


# =============================================================================
#  FIGURA 2 — PUNTO DE SILLA E3
# =============================================================================

plt.rcParams.update({
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

DARK2  = "#173d49"
MID2   = "#6fa7b4"
LIGHT2 = "#dfe8ea"
POINT2 = "#b34235"
GRAY2  = "#666666"


def bezier_f2(ax, pts, color=DARK2, lw=1.8, alpha=1.0, arrow=True):
    verts = [pts[0]]
    codes = [Path.MOVETO]
    for i in range(1, len(pts), 3):
        verts += pts[i:i+3]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-", lw=lw,
        color=color, alpha=alpha, zorder=4
    ))
    if arrow:
        s = path.interpolated(80).vertices
        k = len(s) // 2
        ax.annotate(
            "", xy=s[k+2], xytext=s[k-4],
            arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                            mutation_scale=10, shrinkA=0, shrinkB=0),
            zorder=5
        )


def campo_vectorial_f2(ax):
    x = np.linspace(-3, 3, 25)
    y = np.linspace(-3, 3, 25)
    X, Y = np.meshgrid(x, y)
    U = X; V = -Y
    N = np.hypot(U, V)
    ax.quiver(X, Y, U/(N+1e-9), V/(N+1e-9),
              color=LIGHT2, angles="xy", scale_units="xy",
              scale=12, width=0.0027,
              headwidth=3.2, headlength=4.2, alpha=0.75, zorder=1)


def curvas_silla_f2(ax):
    curvas = [
        [(-2.95,0.75),(-1.55,1.05),(-1.05,1.95),(-0.78,3.35)],
        [(-2.95,0.45),(-1.35,0.65),(-0.75,1.55),(-0.48,3.25)],
        [(-2.95,0.20),(-1.10,0.28),(-0.55,1.20),(-0.30,3.00)],
        [(0.15,3.35),(0.20,2.05),(0.36,1.05),(1.55,0.55)],
        [(0.42,3.35),(0.48,2.25),(0.70,1.20),(2.35,0.38)],
        [(0.72,3.35),(0.88,2.05),(1.25,1.15),(3.00,0.68)],
        [(-2.95,-0.25),(-1.12,-0.35),(-0.55,-1.25),(-0.30,-3.20)],
        [(-2.95,-0.55),(-1.35,-0.72),(-0.76,-1.62),(-0.48,-3.35)],
        [(-2.95,-0.85),(-1.72,-1.10),(-1.05,-2.00),(-0.78,-3.35)],
        [(0.15,-3.35),(0.20,-2.05),(0.36,-1.05),(1.55,-0.55)],
        [(0.42,-3.35),(0.48,-2.25),(0.70,-1.20),(2.35,-0.38)],
        [(0.72,-3.35),(0.88,-2.05),(1.25,-1.15),(3.00,-0.68)],
    ]
    colores = [DARK2,DARK2,MID2, DARK2,DARK2,MID2,
               MID2,DARK2,DARK2, DARK2,DARK2,MID2]
    for c, col in zip(curvas, colores):
        bezier_f2(ax, c, color=col, lw=1.75,
                  alpha=0.95 if col==DARK2 else 0.75)


def configurar_panel_f2(ax, titulo, ylabel, autovalores):
    campo_vectorial_f2(ax)
    curvas_silla_f2(ax)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-3, 4, 1)); ax.set_yticks(np.arange(-3, 4, 1))
    ax.grid(True, color="#eeeeee", lw=0.55)
    ax.axhline(0, color=DARK2, lw=1.8, ls="--", zorder=2)
    ax.axvline(0, color=GRAY2, lw=1.8, ls="--", zorder=2)
    ax.set_title(titulo, pad=12)
    ax.set_xlabel(r"$\Delta G = G - G^*$")
    ax.set_ylabel(ylabel)
    ax.scatter([0], [0], s=150, facecolor=POINT2,
               edgecolor="#2b1613", linewidth=1.2, zorder=10)
    ax.text(0.13, 0.20, "P3", fontsize=18,
            weight="bold", color="black", zorder=11)
    ax.text(-3.22, 3.18,
            "Tipo: Punto silla\n"
            "Autovalores (submatriz):\n"
            f"{autovalores}",
            ha="left", va="top", fontsize=10.5, weight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec="#bfc4c7", lw=1.0, alpha=0.97),
            zorder=20)
    legend_elements = [
        Line2D([0],[0], color=DARK2, lw=1.8, ls="--", label="Variedad estable"),
        Line2D([0],[0], color=GRAY2, lw=1.8, ls="--", label="Variedad inestable"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=10.5, frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="#d0d0d0")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1); spine.set_color("#555555")


def figura2():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=160)
    configurar_panel_f2(
        axes[0],
        r"Bosquejo local alrededor de P3 en el plano $G$-$I$",
        r"$\Delta I = I - I^*$", "-5.280, -431.146")
    configurar_panel_f2(
        axes[1],
        r"Bosquejo local alrededor de P3 en el plano $G$-$\beta$",
        r"$\Delta \beta = \beta - \beta^*$", "0.000, -4.016")
    fig.subplots_adjust(left=0.06, right=0.98,
                        top=0.86, bottom=0.20, wspace=0.18)
    fig.text(0.29, 0.08, r"(a) Plano $G$-$I$", ha="center", fontsize=25)
    fig.text(0.74, 0.08, r"(b) Plano $G$-$\beta$", ha="center", fontsize=25)
    plt.savefig("figura2.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura2.jpeg guardada")


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generando figuras 1 y 2...")
    figura1()
    figura2()
    print("Listo. Archivos: figura1.jpeg, figura2.jpeg")
