# =============================================================================
#  FASE 3 — Bosquejos cualitativos locales
#  Figuras 1 y 2 del informe
#
#  Descripción:
#      Genera los bosquejos cualitativos locales alrededor de cada equilibrio
#      usando campo vectorial esquemático (quiver) y trayectorias del sistema
#      linealizado. Incluye recuadro con tipo y autovalores en cada panel.
#
#      Figura 1 — Bosquejos alrededor de los nodos atractores E1 y E2
#                 Paneles: E1 plano G-I | E1 plano G-beta | E2 plano G-I
#      Figura 2 — Bosquejo alrededor del punto de silla E3
#                 Paneles: E3 plano G-I | E3 plano G-beta
#                 Con variedades estable e inestable (lineas punteadas)
#
#  Metodo:
#      Campo vectorial esquematico normalizado a partir del Jacobiano.
#      Trayectorias del sistema linealizado en coordenadas desplazadas.
#      t in [0, 5] dias | 500 puntos | RK45
#
#  Uso:
#      Python 3.8+  |  pip install numpy scipy matplotlib
#      python fase3_bosquejos.py
#      (En Google Colab: descomenta la linea %matplotlib inline)
#
#  Autor: Elioncito
# =============================================================================

# %matplotlib inline   # <- descomenta si usas Google Colab

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
#  1. PARAMETROS DEL MODELO (caso base)
# =============================================================================
R0    = 864.0
Ge    = 140.0
EG0   = 1.44
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
P1 = np.array([697.22, 0.0,     0.0   ])
P2 = np.array([100.0,  11.9444, 358.67])
P3 = np.array([250.0,  3.5778,  47.27 ])

equilibrios = {"P1": P1, "P2": P2, "P3": P3}
tipos = {
    "P1": "Nodo atractor",
    "P2": "Nodo atractor",
    "P3": "Punto silla",
}

# =============================================================================
#  3. ESTILO VISUAL
# =============================================================================
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     120,
})

palette_petroleo = [
    "#0B3C49", "#145C6A", "#1D7382", "#2A8C9B",
    "#3CA3B2", "#5CB8C4", "#7AC7D1", "#9AD8DF",
]
color_flechas    = "#C9D1D6"
color_equilibrio = "#8B2020"
color_caja_borde = "#8B9AA3"

# =============================================================================
#  4. JACOBIANO GENERAL
# =============================================================================
def jacobiano(G, I, beta):
    J = np.array([
        [-(EG0 + SI * I),
         -SI * G,
         0],
        [(2 * alpha * beta * sigma * G) / (alpha + G**2)**2,
         -rpk,
         (sigma * G**2) / (alpha + G**2)],
        [(r1 - 2 * r2 * G) * beta,
         0,
         -d0 + r1 * G - r2 * G**2],
    ], dtype=float)
    return J

# =============================================================================
#  5. DATOS DE PLANO
# =============================================================================
def datos_plano(indices):
    mapa = {
        (0, 1): (r"$\Delta G = G - G^*$",
                 r"$\Delta I = I - I^*$",
                 r"$G$-$I$"),
        (0, 2): (r"$\Delta G = G - G^*$",
                 r"$\Delta\beta = \beta - \beta^*$",
                 r"$G$-$\beta$"),
        (1, 2): (r"$\Delta I = I - I^*$",
                 r"$\Delta\beta = \beta - \beta^*$",
                 r"$I$-$\beta$"),
    }
    return mapa[tuple(indices)]

# =============================================================================
#  6. CAMPO VECTORIAL ESQUEMATICO
# =============================================================================
def campo_esquematico(tipo, X, Y):
    if tipo == "Nodo atractor":
        U, V = -X, -Y
    else:
        U, V = X, -Y
    norma = np.sqrt(U**2 + V**2)
    norma[norma == 0] = 1.0
    return U / norma, V / norma

# =============================================================================
#  7. TRAYECTORIAS DEL SISTEMA LINEALIZADO
#     t in [0, 5] dias | 500 puntos | RK45
# =============================================================================
def trayectorias_linealizadas(P, indices, n_tray=10,
                               t_max=5.0, n_pts=500):
    J = jacobiano(*P)
    ix, iy = indices
    radio = 2.5
    angulos = np.linspace(0, 2 * np.pi, n_tray, endpoint=False)
    trajs = []
    for ang in angulos:
        z0 = np.zeros(3)
        z0[ix] = radio * np.cos(ang)
        z0[iy] = radio * np.sin(ang)
        sol = solve_ivp(
            lambda t, z: J @ z,
            [0, t_max], z0,
            method="RK45",
            t_eval=np.linspace(0, t_max, n_pts),
            rtol=1e-8, atol=1e-10,
        )
        trajs.append((sol.y[ix], sol.y[iy]))
    return trajs

# =============================================================================
#  8. AUTOVALORES DE LA SUBMATRIZ 2x2
# =============================================================================
def autovalores_submatriz(P, indices):
    J = jacobiano(*P)
    ix, iy = indices
    J_sub = J[np.ix_([ix, iy], [ix, iy])]
    vals = np.linalg.eigvals(J_sub)
    return np.sort(vals.real)[::-1]

# =============================================================================
#  9. GRAFICAR UN PLANO LOCAL
# =============================================================================
def graficar_plano_local(nombre, P, indices, ax,
                          mostrar_variedades=False):
    ix, iy = indices
    xlabel, ylabel, nombre_plano = datos_plano(indices)
    tipo = tipos[nombre]

    # Campo vectorial
    lim = 3.3
    nx, ny = 18, 18
    xs = np.linspace(-lim, lim, nx)
    ys = np.linspace(-lim, lim, ny)
    X, Y = np.meshgrid(xs, ys)
    U, V = campo_esquematico(tipo, X, Y)
    ax.quiver(X, Y, U, V,
              color=color_flechas, alpha=0.6,
              scale=28, width=0.003,
              headwidth=4, headlength=4, headaxislength=3.5)

    # Trayectorias
    trajs = trayectorias_linealizadas(P, indices)
    for i, (u, v) in enumerate(trajs):
        col = palette_petroleo[i % len(palette_petroleo)]
        lw    = 2.0 if i < 2 else 1.2
        alpha = 0.95 if i < 2 else 0.65
        ax.plot(u, v, color=col, lw=lw, alpha=alpha)
        m = len(u) // 3
        if m + 3 < len(u):
            ax.annotate("",
                        xy=(u[m + 3], v[m + 3]),
                        xytext=(u[m], v[m]),
                        arrowprops=dict(arrowstyle="-|>",
                                        lw=0.9, color=col,
                                        mutation_scale=10))

    # Variedades para E3
    if mostrar_variedades:
        ax.axhline(0, color="#1D4E89", lw=1.8, ls="--",
                   alpha=0.85, label="Variedad estable", zorder=5)
        ax.axvline(0, color="#5C0011", lw=1.8, ls="--",
                   alpha=0.85, label="Variedad inestable", zorder=5)
        ax.legend(loc="lower right", fontsize=8,
                  frameon=True, framealpha=0.9)

    # Equilibrio en origen
    mk = "^" if nombre == "P1" else ("o" if nombre == "P2" else "s")
    ax.scatter(0, 0, s=180, color=color_equilibrio,
               edgecolor="k", linewidth=1.2, zorder=9, marker=mk)
    ax.text(0.08, 0.08, nombre,
            fontsize=11, fontweight="bold",
            color=color_equilibrio)

    # Ejes cruzados
    ax.axhline(0, color="#AAAAAA", lw=0.5, alpha=0.5)
    ax.axvline(0, color="#AAAAAA", lw=0.5, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Bosquejo local alrededor de {nombre} en el plano {nombre_plano}",
        fontsize=9, pad=5)
    ax.grid(True, ls=":", alpha=0.25, color="#CCCCCC")

    # Recuadro tipo + autovalores
    vals = autovalores_submatriz(P, indices)
    vals_str = ", ".join([f"{v:.3f}" for v in vals])
    texto = f"Tipo: {tipo}\nAutovalores (submatriz):\n{vals_str}"
    ax.text(0.02, 0.98, texto,
            transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", alpha=0.85,
                      edgecolor=color_caja_borde, linewidth=0.8))

# =============================================================================
#  10. FIGURA 1 — NODOS ATRACTORES E1 y E2
# =============================================================================
def figura1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    graficar_plano_local("P1", P1, [0, 1], axes[0])
    graficar_plano_local("P1", P1, [0, 2], axes[1])
    graficar_plano_local("P2", P2, [0, 1], axes[2])

    subtitulos = [
        r"$\mathbf{(a)}$  $P_1 \equiv E_1$, plano $G$-$I$",
        r"$\mathbf{(b)}$  $P_1 \equiv E_1$, plano $G$-$\beta$",
        r"$\mathbf{(c)}$  $P_2 \equiv E_2$, plano $G$-$I$",
    ]
    for ax, st in zip(axes, subtitulos):
        ax.text(0.5, -0.18, st,
                transform=ax.transAxes,
                ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("figura1.jpeg", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: figura1.jpeg")

# =============================================================================
#  11. FIGURA 2 — PUNTO DE SILLA E3
# =============================================================================
def figura2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    graficar_plano_local("P3", P3, [0, 1], axes[0],
                          mostrar_variedades=True)
    graficar_plano_local("P3", P3, [0, 2], axes[1],
                          mostrar_variedades=True)

    subtitulos = [
        r"$\mathbf{(a)}$  Plano $G$-$I$",
        r"$\mathbf{(b)}$  Plano $G$-$\beta$",
    ]
    for ax, st in zip(axes, subtitulos):
        ax.text(0.5, -0.18, st,
                transform=ax.transAxes,
                ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("figura2.jpeg", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Guardado: figura2.jpeg")

# =============================================================================
#  12. EJECUCION PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FASE 3 - Bosquejos cualitativos locales")
    print("Figuras 1 y 2 del informe")
    print("Modelo glucosa-insulina-celula beta")
    print("=" * 60)

    print("\n[1/2] Figura 1 - Nodos atractores E1 y E2...")
    figura1()

    print("\n[2/2] Figura 2 - Punto de silla E3...")
    figura2()

    print("\nFiguras 1 y 2 generadas correctamente!")
    print("Archivos: figura1.jpeg, figura2.jpeg")
