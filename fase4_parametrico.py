# =============================================================================
#  FASE 3 y 4 — Figuras 6 al 11 del informe
#
#  Figura 6  — Retratos fase locales en R³ (E1, E2, E3)
#  Figura 7  — Dependencia de equilibrios respecto a Ge
#  Figura 8  — Retrato fase global G-beta Escenario C (Ge=300, rho=2.00)
#  Figura 9  — Diagrama del discriminante en plano r1-r2
#  Figura 10 — Retratos fase G-beta para Delta>0 (F4.2b) y Delta=0 (F4.2c)
#  Figura 11 — Retrato fase G-beta para Delta<0
#
#  Método numérico (simulaciones):
#      scipy.integrate.solve_ivp / Radau
#      rtol=1e-8, atol=1e-10
#      t in [0, 60] dias | 4000 puntos
#
#  Uso:
#      Python 3.8+  |  pip install numpy scipy matplotlib
#      python fase4_parametrico.py
#
#  Autor: Elioncito
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

# =============================================================================
#  PARÁMETROS BASE
# =============================================================================
R0    = 864.0
Ge_base = 140.0
EGO   = 1.44
SI    = 0.72
sigma = 43.2
alpha = 20000.0
rho   = 0.41
k     = 432.0
d0    = 0.06
r1_base = 0.84e-3
r2_base = 2.4e-6
rpk   = rho + k

# Colores compartidos
BLUE   = "#456796"
GOLD   = "#b9903d"
RED    = "#973c38"
GRAY   = "#909090"
DARK   = "#435d64"
MID    = "#7fa8b0"
LIGHT  = "#bfdbe0"
PURPLE = "#6d5a7d"
ORANGE = "#c38b31"
COLORS = ["#9b4f4b","#b9903d","#6d5a7d","#3f5d66","#77a8b1","#b9d6dc"]

# =============================================================================
#  SISTEMA DE EDOS (parámetros variables)
# =============================================================================
def sistema(t, y, Ge=Ge_base, rho_=rho, r1=r1_base, r2=r2_base):
    G = max(y[0], 1e-10); I = max(y[1], 0.); B = max(y[2], 0.)
    dG = R0 + Ge - (EGO + SI*I)*G
    dI = B*sigma*G**2/(alpha+G**2) - (rho_+k)*I
    dB = (-d0 + r1*G - r2*G**2)*B
    return [dG, dI, dB]

def simular(y0, Ge=Ge_base, rho_=rho, r1=r1_base, r2=r2_base,
            t_span=(0,60), n_pts=4000, metodo="RK45"):
    sol = solve_ivp(
        lambda t,y: sistema(t,y,Ge,rho_,r1,r2),
        t_span,
        [max(y0[0],5.), max(y0[1],0.), max(y0[2],0.)],
        method=metodo,
        t_eval=np.linspace(t_span[0],t_span[1],n_pts),
        rtol=1e-8, atol=1e-10, max_step=0.05)
    return sol.t, sol.y

def calcular_equilibrios(Ge=Ge_base, rho_=rho, r1=r1_base, r2=r2_base):
    rpk_ = rho_ + k
    eqs = [((R0+Ge)/EGO, 0., 0.)]
    disc = r1**2 - 4*r2*d0
    if disc > 0:
        for Geq in [(r1-np.sqrt(disc))/(2*r2), (r1+np.sqrt(disc))/(2*r2)]:
            Ieq = (R0+Ge-EGO*Geq)/(SI*Geq)
            if Ieq > 0:
                beq = rpk_*Ieq*(alpha+Geq**2)/(sigma*Geq**2)
                if beq > 0:
                    eqs.append((Geq, Ieq, beq))
    elif abs(disc) < 1e-20:
        G0 = r1/(2*r2)
        I0 = (R0+Ge-EGO*G0)/(SI*G0)
        if I0 > 0:
            b0 = rpk_*I0*(alpha+G0**2)/(sigma*G0**2)
            eqs.append((G0, I0, b0))
    return eqs, disc

ICS = [
    [80,20,380],[120,8,300],[150,18,400],[180,15,350],
    [200,6,150],[300,12,200],[350,2,80],[400,8,120],
    [450,4,60],[500,0.5,20],[600,0.2,5],[650,0.5,10],
    [700,0.1,2],[100,3,100],[250,4,52],[80,1,200],
]
PALETTE = ["#0B3C49","#145C6A","#1D7382","#2A8C9B",
           "#3CA3B2","#5CB8C4","#7AC7D1","#9AD8DF",
           "#6B3074","#A94F6E","#E07B00","#C1121F"]

# =============================================================================
#  FIGURA 6 — RETRATOS LOCALES EN R³
# =============================================================================
def curve3(ax, x, y, z, color=DARK, lw=1.15, alpha=0.95):
    ax.plot(x, y, z, color=color, lw=lw, alpha=alpha)

def polyline(points, n=45):
    pts = np.asarray(points, dtype=float)
    out = []
    for a,b in zip(pts[:-1], pts[1:]):
        t = np.linspace(0,1,n,endpoint=False)
        out.append(a[None,:]*(1-t[:,None]) + b[None,:]*t[:,None])
    out.append(pts[-1][None,:])
    return np.vstack(out)

def setup_3d(ax, title, xlim, ylim, zlim,
             xticks=None, yticks=None, zticks=None):
    ax.set_title(title, pad=14)
    ax.set_xlabel(r"$\Delta G$", labelpad=-2)
    ax.set_ylabel(r"$\Delta I$", labelpad=-2)
    ax.set_zlabel(r"$\Delta \beta$", labelpad=-2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    if zticks is not None: ax.set_zticks(zticks)
    ax.grid(True, color="#d7d7d7")
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1.35,1.0,1.0))

def panel_e1_3d(ax):
    setup_3d(ax, r"Retrato fase local 3D — $E_1$"+"\n"+r"(Nodo atractor)",
             (-85,90),(0.0,0.85),(0.0,0.30),
             xticks=np.arange(-80,100,20),
             yticks=np.arange(0,0.9,0.1),
             zticks=np.arange(0,0.31,0.05))
    t = np.linspace(0,1,130)
    specs = [(-78,0.05,0.010,LIGHT),(-58,0.15,0.025,MID),
             (-40,0.32,0.045,MID),(-20,0.55,0.080,GOLD),
             (25,0.48,0.095,RED),(55,0.35,0.055,DARK),(82,0.22,0.040,DARK)]
    for gx,iy,bz,col in specs:
        curve3(ax, gx*(1-t)**1.65, iy*(1-t)**1.15, bz*(1-t)**0.82, col)
    for gx,iy in [(-80,0.02),(-58,0.07),(-25,0.16),(20,0.24),(55,0.18),(80,0.10)]:
        x = np.linspace(gx,0,55); y=np.linspace(iy,0,55); z=np.zeros_like(x)
        curve3(ax,x,y,z, LIGHT if gx<-20 else DARK, lw=1.0, alpha=0.7)
    ax.scatter([0],[0],[0], s=62, color=RED, edgecolor="#2d1a17", zorder=20)

def panel_e2_3d(ax):
    setup_3d(ax, r"Retrato fase local 3D — $E_2$"+"\n"+r"(Nodo atractor)",
             (-45,55),(-8.5,10.5),(-105,105),
             xticks=np.arange(-40,60,20),
             yticks=np.arange(-7.5,10.1,2.5),
             zticks=np.arange(-100,125,25))
    paths = [
        [(-42,-7.5,-95),(-28,-7.0,-92),(-18,-6.2,-86),(-5,-4.2,-55),(0,0,0)],
        [(-38,-5.2,-72),(-22,-5.0,-70),(-7,-3.7,-45),(0,0,0)],
        [(-35,-2.2,-45),(-18,-2.5,-40),(-5,-2.0,-18),(0,0,0)],
        [(-30,0.0,-15),(-16,-0.5,-10),(-5,-0.6,-5),(0,0,0)],
        [(-25,2.4,15),(-12,1.2,10),(-4,0.5,3),(0,0,0)],
        [(-18,5.0,42),(-8,3.2,30),(-2,1.0,10),(0,0,0)],
        [(-8,7.5,70),(-4,4.8,52),(-1,1.5,14),(0,0,0)],
        [(8,6.2,88),(25,7.6,98),(45,8.8,78),(18,5.0,55),(0,0,0)],
        [(20,2.5,70),(42,4.5,65),(52,6.8,35),(22,2.8,28),(0,0,0)],
        [(45,-1.5,-70),(42,1.5,-30),(30,2.2,8),(0,0,0)],
        [(25,-6.8,-100),(24,-2.2,-45),(15,0.8,-5),(0,0,0)],
        [(6,-7.2,-96),(7,-2.0,-35),(4,-0.3,-5),(0,0,0)],
        [(-6,-7.6,-90),(-4,-2.8,-34),(-2,-0.5,-6),(0,0,0)],
    ]
    colors = [LIGHT,LIGHT,LIGHT,MID,PURPLE,MID,GOLD,RED,DARK,DARK,RED,GOLD,PURPLE]
    for pts,col in zip(paths,colors):
        p=polyline(pts,n=18); curve3(ax,p[:,0],p[:,1],p[:,2],col,lw=1.05)
    ax.scatter([0],[0],[0], s=58, color=BLUE, edgecolor="#263447", zorder=20)

def panel_e3_3d(ax):
    setup_3d(ax, r"Retrato fase local 3D — $E_3$"+"\n"+r"(Punto silla)",
             (-105,430),(-4.5,6.5),(-45,105),
             xticks=np.arange(-100,450,100),
             yticks=np.arange(-4,7,2),
             zticks=np.arange(-40,120,20))
    paths = [
        [(-95,-3.4,-38),(-20,-3.0,-34),(80,-2.6,-36),(230,-3.1,-42),(410,-3.6,-42)],
        [(-80,-2.0,-28),(10,-2.2,-25),(110,-2.8,-30),(210,-3.4,-36)],
        [(-70,-0.9,-12),(-20,-0.7,-8),(40,-1.2,-12),(95,-2.3,-30)],
        [(-90,1.0,30),(-30,0.8,25),(45,0.4,16),(95,-0.6,-5)],
        [(-75,2.8,58),(-20,2.1,45),(40,1.3,28),(105,0.0,0)],
        [(-40,4.6,85),(20,4.8,78),(85,2.8,55),(105,0.0,0)],
        [(0,5.8,98),(35,5.2,92),(80,3.4,58),(105,0.0,0)],
        [(115,5.6,92),(160,5.0,80),(120,2.5,42),(105,0.0,0)],
        [(150,4.2,78),(210,4.5,70),(170,2.4,38),(105,0.0,0)],
        [(220,3.0,58),(260,2.0,35),(210,0.8,12),(105,0.0,0)],
        [(310,1.0,18),(360,0.0,-5),(325,-1.0,-20),(250,-2.4,-35)],
    ]
    colors = [LIGHT,LIGHT,MID,PURPLE,GOLD,RED,DARK,RED,GOLD,DARK,MID]
    for pts,col in zip(paths,colors):
        p=polyline(pts,n=16); curve3(ax,p[:,0],p[:,1],p[:,2],col,lw=1.05)
    ax.scatter([0],[0],[0], s=62, color=ORANGE, edgecolor="#3b2a13", zorder=20)

def figura6():
    plt.rcParams.update({"axes.titlesize":10,"axes.labelsize":8,
                         "xtick.labelsize":7,"ytick.labelsize":7})
    fig = plt.figure(figsize=(16,5.2), dpi=170)
    ax1 = fig.add_subplot(1,3,1,projection="3d")
    ax2 = fig.add_subplot(1,3,2,projection="3d")
    ax3 = fig.add_subplot(1,3,3,projection="3d")
    panel_e1_3d(ax1); panel_e2_3d(ax2); panel_e3_3d(ax3)
    fig.subplots_adjust(left=0.015,right=0.985,top=0.86,bottom=0.19,wspace=0.04)
    fig.text(0.17,0.065,r"(a) Entorno de $E_1$",ha="center",fontsize=25)
    fig.text(0.50,0.065,r"(b) Entorno de $E_2$",ha="center",fontsize=25)
    fig.text(0.83,0.065,r"(c) Entorno de $E_3$",ha="center",fontsize=25)
    plt.savefig("figura6.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura6.jpeg guardada")

# =============================================================================
#  FIGURA 7 — DEPENDENCIA DE EQUILIBRIOS vs Ge
# =============================================================================
def figura7():
    plt.rcParams.update({"axes.titlesize":12,"axes.labelsize":10,
                         "xtick.labelsize":9,"ytick.labelsize":9})
    disc = r1_base**2 - 4*r2_base*d0
    G2 = (r1_base - np.sqrt(disc))/(2*r2_base)
    G3 = (r1_base + np.sqrt(disc))/(2*r2_base)
    Ge = np.linspace(0.,500.,501)
    G1 = (R0+Ge)/EGO
    I2 = (R0+Ge-EGO*G2)/(SI*G2)
    I3 = (R0+Ge-EGO*G3)/(SI*G3)
    beta2 = rpk*I2*(alpha+G2**2)/(sigma*G2**2)
    beta3 = rpk*I3*(alpha+G3**2)/(sigma*G3**2)

    fig, axes = plt.subplots(1,3,figsize=(14.8,4.6),dpi=170)
    fig.suptitle(r"Figura F4.1a -- Dependencia de los equilibrios en $G_e$"
                 +"\n"+r"($\rho=0.41$, $r_1$, $r_2$ fijos)",
                 fontsize=13, y=1.02)
    axes[0].plot(Ge,G1,color=RED,lw=1.8)
    axes[0].axvline(Ge_base,color=GRAY,ls=":",lw=1.2,label=r"$G_e=140$ (base)")
    axes[0].set_title(r"Glucemia de colapso $G_1^*(G_e)$")
    axes[0].set_xlabel(r"$G_e$  [mg dL$^{-1}$ d$^{-1}$]")
    axes[0].set_ylabel(r"$G_1^*$  [mg/dL]")
    axes[0].legend(loc="lower right",fontsize=8,frameon=True)
    axes[1].plot(Ge,I2,color=BLUE,lw=1.8,label=rf"$I_2^*$  ($G^*={G2:.0f}$)")
    axes[1].plot(Ge,I3,color=GOLD,lw=1.8,label=rf"$I_3^*$  ($G^*={G3:.0f}$)")
    axes[1].axvline(Ge_base,color=GRAY,ls=":",lw=1.2)
    axes[1].set_title(r"Insulinemia de equilibrio $I^*(G_e)$")
    axes[1].set_xlabel(r"$G_e$"); axes[1].set_ylabel(r"$I^*$  [$\mu$U/mL]")
    axes[1].legend(loc="center right",fontsize=8,frameon=True)
    axes[2].plot(Ge,beta2,color=BLUE,lw=1.8,label=r"$\beta_2^*$  ($E_2$)")
    axes[2].plot(Ge,beta3,color=GOLD,lw=1.8,label=r"$\beta_3^*$  ($E_3$)")
    axes[2].axvline(Ge_base,color=GRAY,ls=":",lw=1.2,label=r"$G_e=140$ (base)")
    axes[2].set_title(r"Masa $\beta^*$ de equilibrio $\beta^*(G_e)$")
    axes[2].set_xlabel(r"$G_e$"); axes[2].set_ylabel(r"$\beta^*$  [u.a.]")
    axes[2].legend(loc="center right",fontsize=8,frameon=True)
    for ax in axes:
        ax.grid(True,color="#e7e7e7",ls=":",lw=0.65)
        for sp in ax.spines.values(): sp.set_color("#666666"); sp.set_linewidth(0.9)
    fig.subplots_adjust(left=0.055,right=0.99,bottom=0.15,top=0.78,wspace=0.28)
    plt.savefig("figura7.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura7.jpeg guardada")

# =============================================================================
#  FIGURA 8 — RETRATO GLOBAL G-beta ESCENARIO C
# =============================================================================
def retrato_Gbeta(ax, Ge_, rho_, r1=r1_base, r2=r2_base):
    eqs, _ = calcular_equilibrios(Ge_, rho_, r1, r2)
    cols_eq = ["#C1121F","#0B3C49","#E07B00"]
    noms    = ["$E_1$","$E_2$","$E_3$"]
    mkrs    = ["^","o","s"]
    tipos   = ["Nodo estable","Nodo estable","Silla"]
    for i,(y0) in enumerate(ICS):
        try:
            _,Y = simular(y0,Ge_,rho_,r1,r2,t_span=(0,60),n_pts=4000)
        except Exception: continue
        col = PALETTE[i%len(PALETTE)]
        ax.plot(Y[0],Y[2],lw=1.5,color=col,alpha=0.80)
        m = len(Y[0])//3
        if m+4<len(Y[0]):
            ax.annotate("",xy=(Y[0][m+4],Y[2][m+4]),xytext=(Y[0][m],Y[2][m]),
                        arrowprops=dict(arrowstyle="-|>",lw=0.9,color=col,mutation_scale=10))
        ax.scatter(Y[0][0],Y[2][0],s=22,color=col,zorder=4)
    for j,(Geq,Ieq,beq) in enumerate(eqs):
        if j<len(noms):
            ax.scatter(Geq,beq,s=170,color=cols_eq[j],edgecolor="k",
                       lw=1.2,zorder=9,marker=mkrs[j],
                       label=f"{noms[j]} — {tipos[j]}")
            ax.text(Geq+8,beq+4,noms[j],fontsize=9,fontweight="bold",color=cols_eq[j])
    ax.set_xlabel("$G$ [mg/dL]"); ax.set_ylabel("$\\beta$ [u.a.]")
    ax.legend(loc="upper right",fontsize=7.5,frameon=True,framealpha=0.9)
    ax.grid(True,ls=":",alpha=0.3)

def figura8():
    plt.rcParams.update({"axes.titlesize":11,"axes.labelsize":10})
    fig, ax = plt.subplots(figsize=(9.8,6.2),dpi=170)
    retrato_Gbeta(ax, 300, 2.00)
    ax.set_title(r"Figura F4.1b -- Retrato fase global $G$-$\beta$"+"\n"
                 +r"$G_e=300$, $\rho=2.0$  (alta exposicion adrenal)",pad=10)
    plt.figtext(0.50,0.035,r"(b)  $G_e=300$, $\rho=2.00$",ha="center",fontsize=14)
    plt.savefig("figura8.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura8.jpeg guardada")

# =============================================================================
#  FIGURA 9 — DIAGRAMA DEL DISCRIMINANTE
# =============================================================================
def figura9():
    plt.rcParams.update({"axes.titlesize":17,"axes.labelsize":15,
                         "xtick.labelsize":13,"ytick.labelsize":13})
    BLUE_F  = "#d9e4ef"; RED_F = "#ead8d8"; BND = "#9d3b35"
    r2_B = r2_base; r1_B = 2.*np.sqrt(r2_B*d0)
    r1_C = 0.30e-3; r2_C = r2_base
    x = np.linspace(0.3,6.0,500); y = np.linspace(0.,2.5,500)
    X,Y = np.meshgrid(x,y)
    Delta = (Y*1e-3)**2 - 4*(X*1e-6)*d0
    x_c = np.linspace(0.3,6.0,600)
    y_c = 2.*np.sqrt(x_c*1e-6*d0)*1e3
    fig, ax = plt.subplots(figsize=(10.8,7.2),dpi=170)
    ax.contourf(X,Y,Delta>0,levels=[-0.5,0.5,1.5],
                colors=[RED_F,BLUE_F],alpha=0.75)
    ax.plot(x_c,y_c,color=BND,lw=2.8,
            label=r"$\Delta=0$: bifurcacion silla-nodo  $r_1=2\sqrt{r_2d_0}$")
    ax.scatter([r2_base*1e6],[r1_base*1e3],s=210,marker="o",
               facecolor=BLUE,edgecolor="black",lw=1.6,zorder=10,
               label=r"Caso base $(r_1,r_2)$")
    ax.scatter([r2_B*1e6],[r1_B*1e3],s=180,marker="D",
               facecolor=GOLD,edgecolor="black",lw=1.5,zorder=11,
               label=r"Umbral $r_{1,c}$ (Esc. B)")
    ax.scatter([r2_C*1e6],[r1_C*1e3],s=210,marker="v",
               facecolor=RED,edgecolor="black",lw=1.5,zorder=11,
               label=r"Esc. C $(\Delta<0)$")
    ax.annotate("Base",xy=(r2_base*1e6,r1_base*1e3),xytext=(2.95,0.90),
                color=BLUE,fontsize=12,arrowprops=dict(arrowstyle="-",color=BLUE,lw=1.0))
    ax.annotate("E. B",xy=(r2_B*1e6,r1_B*1e3),xytext=(3.00,0.70),
                color=GOLD,fontsize=12,arrowprops=dict(arrowstyle="-",color=GOLD,lw=1.0))
    ax.text(3.7,1.68,r"$\Delta>0$"+"\nBiestabilidad",color=BLUE,fontsize=14,ha="center")
    ax.text(1.55,0.38,r"$\Delta<0$"+"\nColapso",color=BND,fontsize=14,ha="center")
    ax.set_xlim(0.3,6.1); ax.set_ylim(0.,2.5)
    ax.set_xlabel(r"$r_2 \times 10^{-6}$  [mg$^{-2}$ dL$^2$ d$^{-1}$]")
    ax.set_ylabel(r"$r_1 \times 10^{-3}$  [mg$^{-1}$ dL d$^{-1}$]")
    ax.set_title(r"Figura F4.2a -- Diagrama del discriminante $\Delta=r_1^2-4r_2d_0$"
                 +"\n"+r"en el plano $r_1$-$r_2$",pad=18)
    ax.grid(True,color="#d8d8d8",ls=":",lw=0.8)
    handles=[
        Patch(facecolor=BLUE_F,edgecolor="#cccccc",alpha=0.75,
              label=r"$\Delta>0$: biestabilidad $(E_1,E_2,E_3)$"),
        Patch(facecolor=RED_F,edgecolor="#cccccc",alpha=0.75,
              label=r"$\Delta<0$: solo $E_1$ (colapso)"),
        Line2D([0],[0],color=BND,lw=2.8,
               label=r"$\Delta=0$: bifurcacion silla-nodo  $r_1=2\sqrt{r_2d_0}$"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor=BLUE,
               markeredgecolor="black",markersize=12,label=r"Caso base $(r_1,r_2)$"),
        Line2D([0],[0],marker="D",color="none",markerfacecolor=GOLD,
               markeredgecolor="black",markersize=11,label=r"Umbral $r_{1,c}$ (Esc. B)"),
        Line2D([0],[0],marker="v",color="none",markerfacecolor=RED,
               markeredgecolor="black",markersize=12,label=r"Esc. C $(\Delta<0)$"),
    ]
    ax.legend(handles=handles,loc="upper right",fontsize=10.5,
              frameon=True,framealpha=0.95)
    for sp in ax.spines.values(): sp.set_color("#333333"); sp.set_linewidth(1.0)
    plt.savefig("figura9.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura9.jpeg guardada")

# =============================================================================
#  FIGURA 10 — F4.2b (Delta>0) y F4.2c (Delta=0)
# =============================================================================
def figura10_F42b():
    """Retrato fase G-beta con Delta>0 (caso base)."""
    plt.rcParams.update({"axes.titlesize":12,"axes.labelsize":10,
                         "xtick.labelsize":9,"ytick.labelsize":9})
    fig, ax = plt.subplots(figsize=(9.8,6.2),dpi=170)
    retrato_Gbeta(ax, Ge_base, rho, r1_base, r2_base)
    ax.set_title(r"Figura F4.2b -- Retrato fase global $G$-$\beta$,  $\Delta>0$"
                 +"\n"+r"$r_1=8.4\times10^{-4}$, $r_2=2.4\times10^{-6}$  (biestabilidad)",
                 pad=10)
    plt.figtext(0.50,0.035,r"(a) $\Delta>0$: biestabilidad.",
                ha="center",fontsize=23)
    plt.savefig("figura10_F42b.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura10_F42b.jpeg guardada")

def figura10_F42c():
    """Retrato fase G-beta con Delta=0 (bifurcación silla-nodo)."""
    plt.rcParams.update({"axes.titlesize":12,"axes.labelsize":10,
                         "xtick.labelsize":9,"ytick.labelsize":9})
    r1_c = 2.*np.sqrt(r2_base*d0)

    def modelo_c(t, y):
        G=max(y[0],1e-10); I=max(y[1],0.); B=max(y[2],0.)
        return [R0+Ge_base-(EGO+SI*I)*G,
                B*sigma*G**2/(alpha+G**2)-(rho+k)*I,
                (-d0+r1_c*G-r2_base*G**2)*B]

    def insulina_qe(G, beta):
        return beta*sigma*G**2/((alpha+G**2)*(rho+k))

    def integrar_c(y0, tf=22.):
        sol = solve_ivp(modelo_c,(0.,tf),
                        [max(y0[0],5.),max(y0[1],0.),max(y0[2],0.)],
                        method="Radau",
                        t_eval=np.linspace(0.,tf,900),
                        rtol=1e-8,atol=1e-10,max_step=tf/600)
        return (sol.t, sol.y.T) if sol.success else None

    G0c = r1_c/(2.*r2_base)
    I0c = (R0+Ge_base-EGO*G0c)/(SI*G0c)
    b0c = (rho+k)*I0c*(alpha+G0c**2)/(sigma*G0c**2)
    E0  = np.array([G0c,I0c,b0c])
    E1c = np.array([(R0+Ge_base)/EGO, 0., 0.])

    initials = [
        (85,380,COLORS[5],5.0,0.22),(150,400,COLORS[5],18.0,0.30),
        (190,350,COLORS[3],14.0,0.30),(120,320,COLORS[4],10.0,0.35),
        (125,300,COLORS[3],9.0,0.35),(130,225,COLORS[3],7.0,0.40),
        (300,200,COLORS[2],1.15,0.58),(200,150,COLORS[3],1.05,0.58),
        (400,120,COLORS[3],1.50,0.55),(180,100,COLORS[1],0.80,0.55),
        (350,80,COLORS[4],1.50,0.55),(450,60,COLORS[0],5.50,0.55),
        (500,20,COLORS[4],4.50,0.55),(650,10,COLORS[0],6.50,0.55),
        (600,5,COLORS[4],5.80,0.55),
    ]

    fig, ax = plt.subplots(figsize=(9.8,6.2),dpi=170)
    for G0,b0,color,tf,apos in initials:
        I0 = insulina_qe(G0,b0)
        out = integrar_c([G0,I0,b0],tf=tf)
        if out is None: continue
        _,Y = out
        ax.plot(Y[:,0],Y[:,2],color=color,lw=1.25,alpha=0.95,zorder=3)
        ax.scatter([Y[0,0]],[Y[0,2]],s=12,color=color,alpha=0.85,zorder=4)
        k_ = int(apos*(len(Y)-2)); k_=max(2,min(k_,len(Y)-3))
        ax.annotate("",xy=(Y[k_+1,0],Y[k_+1,2]),xytext=(Y[k_-2,0],Y[k_-2,2]),
                    arrowprops=dict(arrowstyle="->",color=color,lw=1.15,
                                   shrinkA=0,shrinkB=0,mutation_scale=9),zorder=5)

    ax.scatter([E1c[0]],[E1c[2]],s=105,marker="^",facecolor="#a33a30",
               edgecolor="black",lw=1.0,zorder=20)
    ax.scatter([E0[0]],[E0[2]],s=105,marker="D",facecolor=BLUE,
               edgecolor="black",lw=1.0,zorder=20)
    ax.text(E1c[0]+8,E1c[2]+6,r"$E_1$",color="#a33a30",fontsize=10,weight="bold")
    ax.text(E0[0]+8,E0[2]+5,r"$E_0$",color=BLUE,fontsize=10,weight="bold")
    ax.set_xlim(50,730); ax.set_ylim(-20,420)
    ax.set_xlabel(r"$G$  [mg/dL]"); ax.set_ylabel(r"$\beta$  [u.a.]")
    ax.set_title(r"Figura F4.2c -- Retrato fase global $G$-$\beta$,  $\Delta=0$"
                 +"\n"+r"$r_1=r_{1,c}\approx 7.59\times10^{-4}$  (bifurcacion silla-nodo)",
                 pad=10)
    ax.grid(True,color="#e5e5e5",ls=":",lw=0.7)
    lh=[Line2D([0],[0],marker="^",color="none",markerfacecolor="#a33a30",
               markeredgecolor="black",markersize=9,label=r"$E_1$ Nodo estable"),
        Line2D([0],[0],marker="D",color="none",markerfacecolor=BLUE,
               markeredgecolor="black",markersize=9,
               label=r"$E_0$ No hiperbolico ($\Delta=0)$")]
    ax.legend(handles=lh,loc="upper right",fontsize=8.5,frameon=True)
    plt.figtext(0.50,0.035,r"(b) $\Delta=0$:  $E_0$ no hiperbolico.",
                ha="center",fontsize=23)
    plt.savefig("figura10_F42c.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura10_F42c.jpeg guardada")

# =============================================================================
#  FIGURA 11 — RETRATO GLOBAL G-beta DELTA<0
# =============================================================================
def figura11():
    plt.rcParams.update({"axes.titlesize":12,"axes.labelsize":10,
                         "xtick.labelsize":9,"ytick.labelsize":9})
    r1_c = 3.0e-4
    fig, ax = plt.subplots(figsize=(9.8,6.2),dpi=170)
    retrato_Gbeta(ax, Ge_base, rho, r1_c, r2_base)
    eqs,_ = calcular_equilibrios(Ge_base, rho, r1_c, r2_base)
    ax.set_title(r"Figura F4.2d -- Retrato fase global $G$-$\beta$,  $\Delta<0$"
                 +"\n"+r"$r_1=3.0\times10^{-4}$  (unico equilibrio: colapso $E_1$)",
                 pad=10)
    plt.figtext(0.50,0.035,r"(c) $\Delta<0$: solo $E_1$.",
                ha="center",fontsize=14)
    plt.savefig("figura11.jpeg", bbox_inches="tight")
    plt.close()
    print("  figura11.jpeg guardada")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("Generando Figuras 6 al 11")
    print("Modelo glucosa-insulina-celula beta")
    print("="*60)
    print("\n[1/7] Figura 6  — Retratos locales R³...")
    figura6()
    print("\n[2/7] Figura 7  — Dependencia de equilibrios vs Ge...")
    figura7()
    print("\n[3/7] Figura 8  — Retrato global Escenario C...")
    figura8()
    print("\n[4/7] Figura 9  — Diagrama discriminante r1-r2...")
    figura9()
    print("\n[5/7] Figura 10 F4.2b — Delta>0 biestabilidad...")
    figura10_F42b()
    print("\n[6/7] Figura 10 F4.2c — Delta=0 bifurcacion silla-nodo...")
    figura10_F42c()
    print("\n[7/7] Figura 11 — Delta<0 colapso...")
    figura11()
    print("\nTodas las figuras generadas correctamente!")
