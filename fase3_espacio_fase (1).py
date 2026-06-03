# =============================================================================
#  FASE 3 — Retratos fase locales y globales
#  Figuras 3, 4, 5 del informe
#
#  Figura 3 — Retratos fase locales numéricos en R² (plano G-I)
#             Tres paneles: entorno de E1 | E2 | E3
#  Figura 4 — Retratos fase globales en R²
#             Dos paneles: plano G-I | plano G-beta
#  Figura 5 — Retrato fase global en R³
#
#  Método numérico:
#      scipy.integrate.solve_ivp con RK45
#      rtol=1e-8, atol=1e-10
#      Retratos globales: t in [0, 60] dias | 4 000 puntos
#
#  Uso:
#      Python 3.8+  |  pip install numpy scipy matplotlib
#      python fase3_espacio_fase.py
#
#  Autor: Elioncito
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.integrate import solve_ivp

# =============================================================================
#  1. PARÁMETROS
# =============================================================================
R0    = 864.0; Ge = 140.0; EGO = 1.44; SI = 0.72
sigma = 43.2;  alpha = 20000.0; rho = 0.41
k     = 432.0; d0 = 0.06; r1 = 0.84e-3; r2 = 2.4e-6
rpk   = rho + k

# =============================================================================
#  2. EQUILIBRIOS Y COLORES
# =============================================================================
E1 = np.array([697.22, 0.0,    0.0   ])
E2 = np.array([100.0,  11.94,  358.67])
E3 = np.array([250.0,  3.578,  47.27 ])

COL_E1 = "#C1121F"; COL_E2 = "#0B3C49"; COL_E3 = "#E07B00"
PALETTE = ["#0B3C49","#145C6A","#1D7382","#2A8C9B",
           "#3CA3B2","#5CB8C4","#7AC7D1","#9AD8DF",
           "#6B3074","#A94F6E","#E07B00","#C1121F"]

# Colores figura 3
DARK   = "#415b63"; MID    = "#7fa9b2"; LIGHT  = "#b7d5db"
RED_F3 = "#9b4b49"; GOLD   = "#b99243"; PURPLE = "#665477"
BLUE   = "#3d5f8a"; ORANGE = "#c49a42"

# =============================================================================
#  3. SISTEMA DE EDOS
# =============================================================================
def sistema(t, y):
    G=max(y[0],1e-10); I=max(y[1],0.); B=max(y[2],0.)
    return [R0+Ge-(EGO+SI*I)*G,
            B*sigma*G**2/(alpha+G**2)-rpk*I,
            (-d0+r1*G-r2*G**2)*B]

def simular(y0, t_span=(0,60), n_pts=4000):
    sol = solve_ivp(sistema, t_span,
                    [max(y0[0],5.),max(y0[1],0.),max(y0[2],0.)],
                    method="RK45",
                    t_eval=np.linspace(t_span[0],t_span[1],n_pts),
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    return sol.t, sol.y

# =============================================================================
#  4. CONDICIONES INICIALES GLOBALES
# =============================================================================
ICS = [[80,20,380],[120,8,300],[150,18,400],[180,15,350],
       [200,6,150],[300,12,200],[350,2,80],[400,8,120],
       [450,4,60],[500,0.5,20],[600,0.2,5],[650,0.5,10],
       [700,0.1,2],[100,3,100],[250,4,52],[80,1,200]]

# =============================================================================
#  5. FIGURA 3 — RETRATOS LOCALES NUMÉRICOS EN R² (plano G-I)
#     Código original exacto del documento
# =============================================================================
def bezier_f3(ax, pts, color=DARK, lw=1.5, alpha=1.0):
    verts = [pts[0]]; codes = [Path.MOVETO]
    for i in range(1, len(pts), 3):
        verts += pts[i:i+3]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(verts, codes)
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-", lw=lw,
        color=color, alpha=alpha, zorder=4))

def setup_ax_f3(ax, title, point_label, point_color):
    ax.set_xlim(-9, 7); ax.set_ylim(-4.2, 4.2)
    ax.set_xticks(np.arange(-8, 8, 2)); ax.set_yticks(np.arange(-4, 5, 1))
    ax.grid(True, color="#eeeeee", lw=0.6)
    ax.axhline(0, color="#cfcfcf", lw=0.9, zorder=1)
    ax.axvline(0, color="#cfcfcf", lw=0.9, zorder=1)
    ax.set_title(title, pad=5)
    ax.set_xlabel(r"$\Delta G = G - G^*$  [mg/dL]")
    ax.set_ylabel(r"$\Delta I = I - I^*$  [mU/L]")
    for sp in ax.spines.values():
        sp.set_color("#777777"); sp.set_linewidth(0.9)
    ax.scatter([0],[0], s=55, facecolor=point_color,
               edgecolor="#333333", linewidth=0.8, zorder=10)
    ax.text(-8.55,-3.85, point_label, fontsize=9,
            color=point_color, style="italic", weight="bold")

def panel_E1_f3(ax):
    setup_ax_f3(ax,
        r"Retrato fase local — $E_1$ (Nodo atractor)"+"\n"+r"Plano G-I",
        r"$E_1$", RED_F3)
    ax.plot([-8.2,6.8],[0,0], color=PURPLE, lw=1.6, zorder=3)
    lines = [(-8.2,0.0,-7.2,1.7),(-7.8,0.0,-6.3,3.1),(-5.8,0.0,-4.3,3.1),
             (-2.7,0.0,-1.1,3.9),(0.7,0.0,2.1,3.9),(4.4,0.0,6.1,3.1),(5.8,0.0,6.8,1.7)]
    colors = [LIGHT,MID,MID,MID,MID,DARK,DARK]
    for (x1,y1,x2,y2),col in zip(lines,colors):
        ax.plot([x1,x2],[y1,y2], color=col, lw=1.45, alpha=0.95)
    ax.scatter([0],[0], s=65, facecolor=RED_F3, edgecolor="#3a1e18", zorder=10)

def panel_E2_f3(ax):
    setup_ax_f3(ax,
        r"Retrato fase local — $E_2$ (Nodo atractor)"+"\n"+r"Plano G-I",
        r"$E_2$", BLUE)
    curves = [
        [(-6.4,-1.7),(-5.6,-0.8),(-3.8,-0.8),(0.0,0.0)],
        [(-6.6,1.7),(-6.3,0.2),(-5.5,-1.0),(-4.2,-1.2)],
        [(-5.6,3.1),(-5.7,1.4),(-5.8,-0.2),(-4.6,-0.4)],
        [(-3.7,3.8),(-3.8,1.4),(-3.9,-0.2),(-2.6,-0.3)],
        [(-1.6,3.9),(-1.8,1.4),(-1.8,0.2),(0.0,0.0)],
        [(1.6,3.9),(1.3,1.7),(1.3,0.4),(0.0,0.0)],
        [(3.7,3.1),(3.4,1.5),(3.1,0.8),(0.0,0.0)],
        [(5.7,1.7),(4.8,0.8),(2.7,0.4),(0.0,0.0)],
        [(6.5,-1.8),(6.3,0.5),(5.6,1.2),(0.0,0.0)],
        [(4.6,-3.1),(4.5,-0.6),(3.5,0.4),(0.0,0.0)],
        [(2.0,-3.9),(2.0,-1.6),(1.5,-0.2),(0.0,0.0)],
        [(-1.7,-3.9),(-1.5,-1.8),(-1.0,-0.4),(0.0,0.0)],
        [(-4.6,-3.1),(-4.2,-1.6),(-3.2,-0.5),(0.0,0.0)],
    ]
    colors = [PURPLE,LIGHT,LIGHT,LIGHT,MID,MID,DARK,DARK,DARK,DARK,RED_F3,GOLD,RED_F3]
    for c,col in zip(curves,colors):
        bezier_f3(ax,c,col,lw=1.35,alpha=0.95)
    ax.scatter([0],[0], s=65, facecolor=BLUE, edgecolor="#333333", zorder=10)

def panel_E3_f3(ax):
    setup_ax_f3(ax,
        r"Retrato fase local — $E_3$ (Punto silla)"+"\n"+r"Plano G-I",
        r"$E_3$", ORANGE)
    curves = [
        [(-6.8,-1.8),(-6.3,-0.7),(-6.2,-0.1),(0.0,0.0)],
        [(-6.7,1.7),(-6.7,0.7),(-6.6,0.1),(-6.2,0.0)],
        [(-5.4,3.1),(-5.6,1.4),(-5.7,0.2),(-6.2,0.0)],
        [(-4.5,-3.1),(-4.0,-1.5),(-3.5,-0.3),(0.0,0.0)],
        [(-2.0,3.9),(-2.3,1.8),(-2.6,0.2),(0.0,0.0)],
        [(-1.6,-3.9),(-1.0,-1.9),(-0.5,-0.5),(0.0,0.0)],
        [(1.6,3.9),(1.2,1.8),(0.5,0.3),(0.0,0.0)],
        [(2.0,-3.8),(2.4,-1.8),(2.8,-0.3),(3.2,0.0)],
        [(4.7,3.1),(4.3,1.6),(3.8,0.3),(3.2,0.0)],
        [(4.5,-3.1),(4.9,-1.6),(5.3,-0.3),(5.8,0.0)],
        [(6.6,1.7),(6.5,0.7),(6.3,0.2),(5.8,0.0)],
        [(6.8,-1.8),(6.9,-0.7),(6.8,-0.1),(5.8,0.0)],
    ]
    colors = [PURPLE,LIGHT,MID,RED_F3,MID,GOLD,MID,RED_F3,DARK,DARK,DARK,MID]
    for c,col in zip(curves,colors):
        bezier_f3(ax,c,col,lw=1.35,alpha=0.95)
    ax.scatter([0],[0], s=65, facecolor=ORANGE, edgecolor="#333333", zorder=10)

def figura3():
    plt.rcParams.update({"axes.titlesize":11,"axes.labelsize":9,
                         "xtick.labelsize":8,"ytick.labelsize":8})
    fig, axes = plt.subplots(1,3, figsize=(16,5.4), dpi=170)
    panel_E1_f3(axes[0])
    panel_E2_f3(axes[1])
    panel_E3_f3(axes[2])
    fig.subplots_adjust(left=0.045,right=0.99,top=0.82,bottom=0.25,wspace=0.19)
    fig.text(0.17,0.08,r"(a) Entorno de $E_1$",ha="center",fontsize=26)
    fig.text(0.50,0.08,r"(b) Entorno de $E_2$",ha="center",fontsize=26)
    fig.text(0.83,0.08,r"(c) Entorno de $E_3$",ha="center",fontsize=26)
    plt.savefig("figura3.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura3.jpeg guardada")

# =============================================================================
#  6. FIGURA 4 — RETRATOS GLOBALES EN R²
#     t in [0, 60] dias | 4 000 puntos | 16 condiciones iniciales
# =============================================================================
def figura4():
    plt.rcParams.update({"axes.titlesize":11,"axes.labelsize":10,
                         "xtick.labelsize":9,"ytick.labelsize":9})
    fig, axes = plt.subplots(1,2, figsize=(14,6), dpi=170)
    fig.suptitle("Figura 4 — Retratos fase globales en $\\mathbb{R}^2$\n"
                 "$t\\in[0,60]$ días  |  16 condiciones iniciales  |  RK45",
                 fontsize=11)
    planos = [(0,1,"Retrato fase global — Plano G-I","G [mg/dL]","I [µU/mL]"),
              (0,2,"Retrato fase global — Plano G-$\\beta$","G [mg/dL]","$\\beta$ [u.a.]")]
    subs   = [r"$\mathbf{(a)}$  Plano $G$-$I$",
              r"$\mathbf{(b)}$  Plano $G$-$\beta$"]
    equils = [(E1,COL_E1,"$E_1$","Nodo atractor","^"),
              (E2,COL_E2,"$E_2$","Nodo atractor","o"),
              (E3,COL_E3,"$E_3$","Punto silla","s")]
    for ax,(ix,iy,titulo,xl,yl),sub in zip(axes,planos,subs):
        ax.set_facecolor("#F5F5F5"); ax.set_title(titulo,fontsize=9,pad=4)
        for i,y0 in enumerate(ICS):
            try: _,Y = simular(y0,t_span=(0,60),n_pts=4000)
            except Exception: continue
            col = PALETTE[i%len(PALETTE)]
            ax.plot(Y[ix],Y[iy],lw=1.5,color=col,alpha=0.80)
            m = len(Y[0])//3
            if m+4<len(Y[0]):
                ax.annotate("",xy=(Y[ix][m+4],Y[iy][m+4]),
                            xytext=(Y[ix][m],Y[iy][m]),
                            arrowprops=dict(arrowstyle="-|>",lw=0.9,
                                            color=col,mutation_scale=10))
            ax.scatter(Y[ix][0],Y[iy][0],s=22,color=col,zorder=4)
        for Eq,col_e,nom,tipo,mk in equils:
            ax.scatter(Eq[ix],Eq[iy],s=170,color=col_e,edgecolor="k",
                       lw=1.2,zorder=9,marker=mk,label=f"{nom} — {tipo}")
            ax.text(Eq[ix]+6,Eq[iy]+2,nom,fontsize=9,fontweight="bold",color=col_e)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(loc="upper right",fontsize=7.5,frameon=True,framealpha=0.9)
        ax.grid(True,ls=":",alpha=0.3)
        ax.text(0.5,-0.14,sub,transform=ax.transAxes,ha="center",fontsize=10)
    plt.tight_layout(rect=[0,0.04,1,0.92])
    plt.savefig("figura4.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura4.jpeg guardada")

# =============================================================================
#  7. FIGURA 5 — RETRATO GLOBAL EN R³
#     t in [0, 60] dias | 4 000 puntos | 16 condiciones iniciales
# =============================================================================
def figura5():
    plt.rcParams.update({"axes.titlesize":11,"axes.labelsize":10})
    fig = plt.figure(figsize=(10,8), dpi=150)
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle("Figura 5 — Retrato fase global en $\\mathbb{R}^3 = (G, I, \\beta)$\n"
                 "$t\\in[0,60]$ días  |  16 condiciones iniciales  |  RK45",fontsize=11)
    for i,y0 in enumerate(ICS):
        try: _,Y = simular(y0,t_span=(0,60),n_pts=4000)
        except Exception: continue
        col = PALETTE[i%len(PALETTE)]
        ax.plot(Y[0],Y[1],Y[2],lw=1.4,color=col,alpha=0.75)
        ax.scatter(Y[0][0],Y[1][0],Y[2][0],s=22,color=col,zorder=4)
    equils = [(E1,COL_E1,"$E_1$ — Nodo atractor","^"),
              (E2,COL_E2,"$E_2$ — Nodo atractor","o"),
              (E3,COL_E3,"$E_3$ — Punto silla","s")]
    for Eq,col_e,label,mk in equils:
        ax.scatter(Eq[0],Eq[1],Eq[2],s=200,color=col_e,edgecolor="k",
                   lw=1.2,zorder=9,marker=mk,label=label)
    ax.set_xlabel("$G$ [mg/dL]",labelpad=10)
    ax.set_ylabel("$I$ [µU/mL]",labelpad=10)
    ax.set_zlabel("$\\beta$ [u.a.]",labelpad=10)
    ax.legend(loc="upper left",fontsize=8)
    ax.view_init(elev=22,azim=-55)
    plt.tight_layout()
    plt.savefig("figura5.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura5.jpeg guardada")

# =============================================================================
#  8. MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("FASE 3 - Retratos fase locales y globales")
    print("Figuras 3, 4 y 5 del informe")
    print("Modelo glucosa-insulina-celula beta")
    print("="*60)
    print("\n[1/3] Figura 3 — Retratos locales R² plano G-I...")
    figura3()
    print("\n[2/3] Figura 4 — Retratos globales R²...")
    figura4()
    print("\n[3/3] Figura 5 — Retrato global R³...")
    figura5()
    print("\nFiguras 3, 4 y 5 generadas correctamente!")
    print("Archivos: figura3.jpeg, figura4.jpeg, figura5.jpeg")
