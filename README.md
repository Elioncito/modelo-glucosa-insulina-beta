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

## Figuras del informe

### Fase 3 — Análisis en el espacio fase

| Fig. | Archivo generado | Descripción | Origen |
|------|-----------------|-------------|--------|
| 1 | `figura1.jpeg` | Bosquejos cualitativos locales alrededor de los nodos atractores E₁ y E₂ (planos G-I y G-β) | Generado con `fase3_espacio_fase.py` |
| 2 | `figura2.jpeg` | Bosquejo cualitativo local alrededor del punto de silla E₃ (planos G-I y G-β) | Generado con `fase3_espacio_fase.py` |
| 3 | `figura3.jpeg` | Retratos fase locales numéricos en el plano G-I con coordenadas desplazadas (entornos de E₁, E₂ y E₃) | Generado con `fase3_espacio_fase.py` |
| 4 | `figura4.jpeg` | Retratos fase globales en R² — planos G-I y G-β | Generado con `fase3_espacio_fase.py` |
| 5 | `figura5.jpeg` | Retrato fase global en R³ = (G, I, β) | Generado con `fase3_espacio_fase.py` |
| 6 | `figura6.jpeg` | Retratos fase locales en R³ con coordenadas desplazadas (entornos de E₁, E₂ y E₃) | Generado con `fase3_espacio_fase.py` |

### Fase 4 — Análisis paramétrico

| Fig. | Archivo generado | Descripción | Origen |
|------|-----------------|-------------|--------|
| 7 | `fase4_Ge_parametrico.jpeg` | Dependencia de los equilibrios respecto a Gₑ con ρ = 0.41 fijo (G₁*, I*, β*) | Generado con `fase4_parametrico.py` |
| 8 | `fase4_F41_escenarioC_Gbeta.jpeg` | Retrato fase global G-β para el Escenario C (Gₑ = 300, ρ = 2.00) | Generado con `fase4_parametrico.py` |
| 9 | `fase4_discriminante_r1_r2.jpeg` | Diagrama del discriminante Δ = r₁² − 4r₂d₀ en el plano r₁-r₂ | Generado con `fase4_parametrico.py` |
| 10 | `fase4_F42_delta_pos_cero_Gbeta.jpeg` | Retratos fase G-β para Δ > 0 (biestabilidad) y Δ = 0 (bifurcación silla-nodo) | Generado con `fase4_parametrico.py` |
| 11 | `fase4_F42_delta_neg_Gbeta.jpeg` | Retrato fase G-β para Δ < 0 (único equilibrio: colapso E₁) | Generado con `fase4_parametrico.py` |

## Autor

Elioncito
