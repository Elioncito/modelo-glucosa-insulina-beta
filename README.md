# Análisis dinámico de un modelo glucosa–insulina–célula beta

Repositorio de código Python para el análisis numérico del sistema de EDOs
que modela la interacción entre glucosa, insulina y masa de células β pancreáticas.

## Estructura del repositorio

```
├── fase3_espacio_fase.py    # Fase 3 — Retratos fase locales y globales
├── fase4_parametrico.py     # Fase 4 — Análisis paramétrico (Ge, rho, r1, r2)
└── README.md
```

## Modelo

El sistema estudiado es:

$$\frac{dG}{dt} = R_0 + G_e - (EGO + S_I I)G$$

$$\frac{dI}{dt} = \frac{\beta \sigma G^2}{\alpha + G^2} - (\rho + k)I$$

$$\frac{d\beta}{dt} = (-d_0 + r_1 G - r_2 G^2)\beta$$

### Parámetros base

| Parámetro | Valor | Unidades |
|-----------|-------|----------|
| R₀ | 864 | mg/dL/día |
| Gₑ | 140 | mg/dL/día |
| EGO | 1.44 | 1/día |
| Sᵢ | 0.72 | mL/µU/día |
| σ | 43.2 | µU/mL/día |
| α | 20000 | (mg/dL)² |
| ρ | 0.41 | 1/día |
| k | 432 | 1/día |
| d₀ | 0.06 | 1/día |
| r₁ | 8.4×10⁻⁴ | 1/(mg/dL·día) |
| r₂ | 2.4×10⁻⁶ | 1/(mg/dL)²/día |

### Equilibrios (caso base)

| Equilibrio | G* (mg/dL) | I* (µU/mL) | β* | Tipo |
|-----------|-----------|-----------|-----|------|
| E₁ | 697.22 | 0 | 0 | Nodo atractor (colapso) |
| E₂ | 100.00 | 11.94 | 358.67 | Nodo atractor (homeostasis) |
| E₃ | 250.00 | 3.578 | 47.27 | Punto de silla (umbral) |

## Método numérico

- **Integrador:** `scipy.integrate.solve_ivp` con método RK45
- **Tolerancias:** `rtol = 1e-8`, `atol = 1e-10`
- **Retratos locales:** t ∈ [0, 10] días, 5 000 puntos
- **Retratos globales:** t ∈ [0, 60] días, 4 000 puntos

## Requisitos

```
Python >= 3.8
numpy
scipy
matplotlib
```

Instalación:

```bash
pip install numpy scipy matplotlib
```

## Uso

### Localmente

```bash
python fase3_espacio_fase.py
python fase4_parametrico.py
```

### Google Colab

1. Sube el archivo o pega su contenido en celdas
2. Descomenta la línea `# %matplotlib inline` al inicio
3. Ejecuta con `Runtime → Run all`

Cada script genera y guarda automáticamente todos los archivos PNG
en el directorio de trabajo.

## Figuras generadas

### Fase 3 (`fase3_espacio_fase.py`)

| Archivo | Descripción |
|---------|-------------|
| `fase3_local_E_1.png` | Retrato local R² alrededor de E₁ |
| `fase3_local_E_2.png` | Retrato local R² alrededor de E₂ |
| `fase3_local_E_3.png` | Retrato local R² alrededor de E₃ |
| `fase3_global_GI.png` | Retrato global R², plano G-I |
| `fase3_global_Gbeta.png` | Retrato global R², plano G-β |
| `fase3_local3D_E_1.png` | Retrato local R³ alrededor de E₁ |
| `fase3_local3D_E_2.png` | Retrato local R³ alrededor de E₂ |
| `fase3_local3D_E_3.png` | Retrato local R³ alrededor de E₃ |
| `fase3_global_3D.png` | Retrato global R³ |

### Fase 4 (`fase4_parametrico.py`)

| Archivo | Descripción |
|---------|-------------|
| `fase4_Ge_parametrico.png` | Dependencia de equilibrios vs Gₑ (Fig. 7) |
| `fase4_F41_escenarioC_Gbeta.png` | Retrato G-β Escenario C (Fig. 8) |
| `fase4_discriminante_r1_r2.png` | Diagrama discriminante Δ en plano r₁-r₂ (Fig. 9) |
| `fase4_F42_delta_pos_cero_Gbeta.png` | Retratos G-β para Δ>0 y Δ=0 (Fig. 10) |
| `fase4_F42_delta_neg_Gbeta.png` | Retrato G-β para Δ<0 (Fig. 11) |

## Autor

Elioncito
