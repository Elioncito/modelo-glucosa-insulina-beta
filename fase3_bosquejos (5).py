import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 130,
})

# ── Parámetros ────────────────────────────────────────────────────────────────
R0=864; Ge=140; EG0=1.44; SI=0.72; sigma=43.2
alpha=20000; rho=0.41; k=432; d0=0.06; r1=0.84e-3; r2=2.4e-6
rpk = rho + k

# ── Equilibrios ───────────────────────────────────────────────────────────────
P1 = np.array([697.22, 0.0,    0.0   ])
P2 = np.array([100.0,  11.9444,358.67])
P3 = np.array([250.0,  3.5778, 47.27 ])

tipos   = {"P1":"Nodo atractor","P2":"Nodo atractor","P3":"Punto silla"}
col_eq  = {"P1":"#8B2020","P2":"#8B2020","P3":"#8B2020"}
col_caja = "#8B9AA3"

palette = ["#0B3C49","#145C6A","#1D7382","#2A8C9B",
           "#3CA3B2","#5CB8C4","#7AC7D1","#9AD8DF"]

# ── Jacobiano ─────────────────────────────────────────────────────────────────
def jac(G, I, B):
    return np.array([
        [-(EG0+SI*I),   -SI*G,  0],
        [2*alpha*B*sigma*G/(alpha+G**2)**2, -rpk, sigma*G**2/(alpha+G**2)],
        [(r1-2*r2*G)*B,  0,   -d0+r1*G-r2*G**2]
    ], float)

# ── Etiquetas de plano ────────────────────────────────────────────────────────
def datos_plano(ix, iy):
    lx = {0:r"$\Delta G = G - G^*$", 1:r"$\Delta I = I - I^*$",
          2:r"$\Delta\beta = \beta - \beta^*$"}
    return lx[ix], lx[iy]

# ── Campo vectorial esquemático (quiver) ──────────────────────────────────────
def campo_quiver(ax, tipo, lim=3.3, n=18):
    xs = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, xs)
    if tipo == "Nodo atractor":
        U, V = -X, -Y
    else:                        # Punto silla
        U, V =  X, -Y
    mag = np.sqrt(U**2+V**2); mag[mag==0]=1
    ax.quiver(X, Y, U/mag, V/mag,
              color="#C9D1D6", alpha=0.55, scale=30,
              width=0.003, headwidth=4, headlength=4,
              headaxislength=3.5, zorder=1)

# ── Trayectorias del sistema linealizado ─────────────────────────────────────
def trayectorias(P, ix, iy, n_tray=10, t_max=5.0, n_pts=500):
    J = jac(*P)
    radio = 2.5
    trajs = []
    for ang in np.linspace(0, 2*np.pi, n_tray, endpoint=False):
        z0 = np.zeros(3)
        z0[ix] = radio*np.cos(ang)
        z0[iy] = radio*np.sin(ang)
        sol = solve_ivp(lambda t,z: J@z, [0,t_max], z0,
                        method="RK45",
                        t_eval=np.linspace(0,t_max,n_pts),
                        rtol=1e-8, atol=1e-10)
        trajs.append((sol.y[ix], sol.y[iy]))
    return trajs

# ── Autovalores submatriz 2×2 ─────────────────────────────────────────────────
def autovalores_2x2(P, ix, iy):
    J = jac(*P)
    Js = J[np.ix_([ix,iy],[ix,iy])]
    vals = np.sort(np.linalg.eigvals(Js).real)[::-1]
    return vals

# ── Dibujar un panel local ────────────────────────────────────────────────────
def panel_local(ax, nombre, P, ix, iy, variedades=False):
    tipo = tipos[nombre]
    lim  = 3.3
    xl, yl = datos_plano(ix, iy)

    # campo vectorial
    campo_quiver(ax, tipo, lim)

    # trayectorias linealizadas
    for i,(u,v) in enumerate(trayectorias(P, ix, iy)):
        col = palette[i % len(palette)]
        lw  = 2.0 if i < 2 else 1.2
        alp = 0.95 if i < 2 else 0.65
        ax.plot(u, v, color=col, lw=lw, alpha=alp, zorder=3)
        m = len(u)//3
        if m+3 < len(u):
            ax.annotate("", xy=(u[m+3],v[m+3]), xytext=(u[m],v[m]),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9,
                                        color=col, mutation_scale=10),
                        zorder=4)

    # variedades para silla
    if variedades:
        ax.axhline(0, color="#1D4E89", lw=1.8, ls="--",
                   alpha=0.9, label="Variedad estable", zorder=5)
        ax.axvline(0, color="#5C0011", lw=1.8, ls="--",
                   alpha=0.9, label="Variedad inestable", zorder=5)
        ax.legend(loc="lower right", fontsize=8,
                  frameon=True, framealpha=0.92)

    # equilibrio
    mk = "^" if nombre=="P1" else ("o" if nombre=="P2" else "o")
    ax.scatter(0, 0, s=160, color=col_eq[nombre],
               edgecolor="k", lw=1.2, zorder=9, marker=mk)
    ax.text(0.10, 0.08, nombre, fontsize=11, fontweight="bold",
            color=col_eq[nombre], transform=ax.transData)

    # ejes cruzados grises
    ax.axhline(0, color="#BBBBBB", lw=0.6, alpha=0.6, zorder=2)
    ax.axvline(0, color="#BBBBBB", lw=0.6, alpha=0.6, zorder=2)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(xl, labelpad=3)
    ax.set_ylabel(yl, labelpad=3)

    nombre_plano = {(0,1):r"$G$-$I$",(0,2):r"$G$-$\beta$",(1,2):r"$I$-$\beta$"}[(ix,iy)]
    ax.set_title(f"Bosquejo local alrededor de {nombre} en el plano {nombre_plano}",
                 fontsize=9, pad=5)
    ax.grid(True, ls=":", alpha=0.22, color="#CCCCCC")

    # recuadro tipo + autovalores
    vals = autovalores_2x2(P, ix, iy)
    vals_str = ", ".join(f"{v:.3f}" for v in vals)
    texto = f"Tipo: {tipo}\nAutovalores (submatriz):\n{vals_str}"
    ax.text(0.02, 0.98, texto, transform=ax.transAxes,
            fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.88, edgecolor=col_caja, lw=0.8))

# =============================================================================
#  FIGURA 1
# =============================================================================
def figura1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_local(axes[0], "P1", P1, 0, 1)
    panel_local(axes[1], "P1", P1, 0, 2)
    panel_local(axes[2], "P2", P2, 0, 1)

    subs = [r"$\mathbf{(a)}$  $P_1 \equiv E_1$, plano $G$-$I$",
            r"$\mathbf{(b)}$  $P_1 \equiv E_1$, plano $G$-$\beta$",
            r"$\mathbf{(c)}$  $P_2 \equiv E_2$, plano $G$-$I$"]
    for ax, s in zip(axes, subs):
        ax.text(0.5, -0.20, s, transform=ax.transAxes,
                ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig("figura1.jpeg", dpi=150, bbox_inches="tight")
    plt.close()
    print("  figura1.jpeg guardada")

# =============================================================================
#  FIGURA 2
# =============================================================================
def figura2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    panel_local(axes[0], "P3", P3, 0, 1, variedades=True)
    panel_local(axes[1], "P3", P3, 0, 2, variedades=True)

    subs = [r"$\mathbf{(a)}$  Plano $G$-$I$",
            r"$\mathbf{(b)}$  Plano $G$-$\beta$"]
    for ax, s in zip(axes, subs):
        ax.text(0.5, -0.20, s, transform=ax.transAxes,
                ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig("figura2.jpeg", dpi=150, bbox_inches="tight")
    plt.close()
    print("  figura2.jpeg guardada")

# =============================================================================
#  FIGURA 3  —  retratos locales NUMÉRICOS (sistema completo, no linealizado)
# =============================================================================
def sistema(t, y):
    G = max(y[0], 1e-10); I = max(y[1], 0.); B = max(y[2], 0.)
    return [R0+Ge-(EG0+SI*I)*G,
            B*sigma*G**2/(alpha+G**2)-rpk*I,
            (-d0+r1*G-r2*G**2)*B]

def simular_local(y0, t_max=10, n_pts=5000):
    sol = solve_ivp(sistema, [0,t_max],
                    [max(y0[0],5.), max(y0[1],0.), max(y0[2],0.)],
                    method="RK45",
                    t_eval=np.linspace(0,t_max,n_pts),
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    return sol.y

def figura3():
    """
    Figura 3: tres paneles plano G-I, coordenadas desplazadas.
    Sistema no lineal completo.  t in [0,10] dias | 5 000 puntos | RK45.
    Condiciones iniciales: perturbaciones ±1 % a ±15 % alrededor de
    cada equilibrio en 12 direcciones.
    """
    configs = [
        (P1, "#C1121F", "$E_1$", "Nodo atractor"),
        (P2, "#0B3C49", "$E_2$", "Nodo atractor"),
        (P3, "#E07B00", "$E_3$", "Punto silla"),
    ]
    subs = [r"$\mathbf{(a)}$  Entorno de $E_1$",
            r"$\mathbf{(b)}$  Entorno de $E_2$",
            r"$\mathbf{(c)}$  Entorno de $E_3$"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Figura 3 — Retratos fase locales numéricos, plano $G$-$I$\n"
        "Coordenadas desplazadas  |  $t\\in[0,10]$ días  |  RK45",
        fontsize=11)

    for ax, (Estar, col_e, nombre, tipo), sub in zip(axes, configs, subs):
        ax.set_facecolor("#F5F5F5")

        # 12 condiciones iniciales con distintas escalas de perturbación
        escalas = [0.01, 0.03, 0.05, 0.08, 0.10, 0.13,
                  -0.01,-0.03,-0.05,-0.08,-0.10,-0.13]
        for i, sc in enumerate(escalas):
            y0 = [Estar[0]*(1+sc),
                  max(Estar[1]*(1-sc*0.6), 0.01),
                  max(Estar[2]*(1+sc*0.2), 0.0)]
            try:
                Y = simular_local(y0)
            except Exception:
                continue
            col = palette[i % len(palette)]
            uG = Y[0] - Estar[0]
            uI = Y[1] - Estar[1]
            ax.plot(uG, uI, color=col, lw=1.5, alpha=0.85, zorder=3)
            m = len(uG)//3
            if m+4 < len(uG):
                ax.annotate("", xy=(uG[m+4],uI[m+4]),
                            xytext=(uG[m],uI[m]),
                            arrowprops=dict(arrowstyle="-|>", lw=0.9,
                                            color=col, mutation_scale=10),
                            zorder=4)

        ax.scatter(0, 0, s=160, color=col_e,
                   edgecolor="k", lw=1.2, zorder=9)
        # etiqueta equilibrio esquina inferior izquierda
        xlim = ax.get_xlim() if ax.get_xlim() != (0,1) else (-9, 7)
        ax.set_xlim(-9, 7)
        ax.text(-8.5, ax.get_ylim()[0]*0.1 if ax.get_ylim()[0]!=0 else -0.15,
                nombre, fontsize=9, color=col_e, fontweight="bold")
        ax.axhline(0, color="#AAAAAA", lw=0.5, alpha=0.5)
        ax.axvline(0, color="#AAAAAA", lw=0.5, alpha=0.5)
        ax.set_xlabel(r"$\Delta G = G - G^*$ [mg/dL]", labelpad=3)
        ax.set_ylabel(r"$\Delta I = I - I^*$ [µU/mL]", labelpad=3)
        ax.set_title(
            f"Retrato fase local — {nombre} ({tipo})\nPlano G-I",
            fontsize=9, pad=4)
        ax.grid(True, ls=":", alpha=0.28)
        ax.text(0.5, -0.20, sub, transform=ax.transAxes,
                ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.07, 1, 0.92])
    plt.savefig("figura3.jpeg", dpi=150, bbox_inches="tight")
    plt.close()
    print("  figura3.jpeg guardada")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generando figuras 1, 2 y 3...")
    figura1()
    figura2()
    figura3()
    print("Listo.")
