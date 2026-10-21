from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "titanic.csv"
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "outputs"

FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

required = {"Survived", "Sex", "Pclass", "Age"}

if not required.issubset(df.columns):
    raise ValueError("Faltan columnas necesarias")

sns.set_theme(style="whitegrid")

def save(fig, filename):
    fig.tight_layout()
    fig.savefig(
        FIGURES / filename,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)
    print(f"OK: {filename}")


print("=" * 60)
print("FASE 2 - VISUALIZACION EXPLORATORIA")
print("=" * 60)

# GRAFICO 1: SUPERVIVENCIA GENERAL

counts = df["Survived"].value_counts().reindex(
    [0, 1], fill_value=0
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["No sobrevivio", "Sobrevivio"],
    counts.values,
    color=["#C44E52", "#55A868"]
)

ax.bar_label(bars, padding=4)
ax.set_title("Distribucion de supervivencia")
ax.set_ylabel("Numero de pasajeros")
ax.set_ylim(0, counts.max() * 1.15)

save(fig, "01_supervivencia_general.png")


# GRAFICO 2: SUPERVIVENCIA POR SEXO

sex_rates = (
    df.groupby("Sex")["Survived"]
    .mean()
    .reindex(["female", "male"])
    * 100
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["Mujeres", "Hombres"],
    sex_rates.values,
    color=["#8172B2", "#4C72B0"]
)

ax.bar_label(bars, fmt="%.2f%%", padding=4)
ax.set_ylim(0, 100)
ax.set_ylabel("Supervivencia (%)")
ax.set_title("Tasa de supervivencia por sexo")

save(fig, "02_supervivencia_sexo.png")


# GRAFICO 3: SUPERVIVENCIA POR CLASE

class_rates = (
    df.groupby("Pclass")["Survived"]
    .mean()
    .reindex([1, 2, 3])
    * 100
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["Primera", "Segunda", "Tercera"],
    class_rates.values,
    color=["#55A868", "#CCB974", "#C44E52"]
)

ax.bar_label(bars, fmt="%.2f%%", padding=4)
ax.set_ylim(0, 100)
ax.set_ylabel("Supervivencia (%)")
ax.set_title("Tasa de supervivencia por clase")

save(fig, "03_supervivencia_clase.png")


# GRAFICO 4: SEXO Y CLASE SIMULTANEAMENTE

cross = df.groupby(
    ["Pclass", "Sex"]
)["Survived"].agg(
    pasajeros="count",
    supervivientes="sum",
    tasa="mean"
)

cross["tasa"] *= 100

cross.to_csv(
    OUTPUT / "06_supervivencia_sexo_clase.csv"
)

heat = (
    df.pivot_table(
        index="Pclass",
        columns="Sex",
        values="Survived",
        aggfunc="mean"
    ) * 100
)

heat = heat.reindex(
    index=[1, 2, 3],
    columns=["female", "male"]
)

heat.columns = ["Mujeres", "Hombres"]

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    heat,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    cbar_kws={"label": "Supervivencia (%)"},
    ax=ax
)

ax.set_title("Supervivencia por sexo y clase")
ax.set_xlabel("Sexo")
ax.set_ylabel("Clase")

save(fig, "04_mapa_calor_sexo_clase.png")


# GRAFICO 5: DISTRIBUCION DE EDADES

fig, ax = plt.subplots(figsize=(9, 5))

for value, label, color in [
    (0, "No sobrevivio", "#C44E52"),
    (1, "Sobrevivio", "#55A868")
]:
    ages = df.loc[
        df["Survived"] == value, "Age"
    ].dropna()

    sns.histplot(
        ages,
        bins=25,
        stat="density",
        common_norm=False,
        alpha=0.35,
        color=color,
        label=label,
        ax=ax
    )

ax.set_title("Distribucion de edad por supervivencia")
ax.set_xlabel("Edad")
ax.set_ylabel("Densidad")
ax.legend()

save(fig, "05_distribucion_edad.png")


# GRAFICO 6: VALORES FALTANTES

missing = df.isna().sum()
missing = missing[missing > 0].sort_values()

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.barh(
    missing.index,
    missing.values,
    color="#DD8452"
)

ax.bar_label(bars, padding=4)
ax.set_xlim(0, missing.max() * 1.15)
ax.set_title("Valores faltantes por variable")
ax.set_xlabel("Numero de valores faltantes")

save(fig, "06_valores_faltantes.png")


print("\nANALISIS COMPLETADO")
print(f"Graficos generados: 6")
print("Tabla adicional: 06_supervivencia_sexo_clase.csv")
