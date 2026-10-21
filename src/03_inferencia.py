from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, norm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "titanic.csv"
OUTPUT = ROOT / "outputs"
FIGURES = ROOT / "figures"

OUTPUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
assert df["Survived"].isin([0, 1]).all()

z = norm.ppf(0.975)

def wilson(y, n):
    p = y / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / den
    half = z / den * np.sqrt(
        p*(1-p)/n + z**2/(4*n**2)
    )
    return center - half, center + half

print("=" * 65)
print("FASE 3 - INFERENCIA ESTADISTICA")
print("=" * 65)

# 1. Intervalos de confianza por sexo y clase
rows = []

for cls in [1, 2, 3]:
    for sex in ["female", "male"]:
        group = df[
            (df["Pclass"] == cls) &
            (df["Sex"] == sex)
        ]

        n = len(group)
        y = int(group["Survived"].sum())
        low, high = wilson(y, n)

        rows.append({
            "clase": cls,
            "sexo": sex,
            "n": n,
            "supervivientes": y,
            "tasa_pct": 100*y/n,
            "ic_inf_pct": 100*low,
            "ic_sup_pct": 100*high
        })

subgroups = pd.DataFrame(rows)

subgroups.round(3).to_csv(
    OUTPUT / "07_ic_sexo_clase.csv",
    index=False
)

print("\n[1] INTERVALOS DEL 95%")
print(subgroups.round(2).to_string(index=False))

# 2. Pruebas chi-cuadrado
tests = []

for variable in ["Sex", "Pclass"]:
    table = pd.crosstab(
        df[variable],
        df["Survived"]
    ).reindex(columns=[0, 1], fill_value=0)

    chi2, p, dof, expected = chi2_contingency(
        table, correction=False
    )

    if expected.min() < 5:
        raise ValueError(
            f"Frecuencias esperadas insuficientes: {variable}"
        )

    v = np.sqrt(
        chi2 / (
            table.to_numpy().sum() *
            min(table.shape[0]-1, table.shape[1]-1)
        )
    )

    tests.append({
        "variable": variable,
        "chi2": chi2,
        "gl": dof,
        "valor_p": p,
        "cramers_v": v
    })

    print(
        f"\n{variable}: "
        f"chi2={chi2:.3f}, "
        f"gl={dof}, "
        f"p={p:.3e}, "
        f"V={v:.3f}"
    )

pd.DataFrame(tests).to_csv(
    OUTPUT / "08_pruebas_chi_cuadrado.csv",
    index=False
)

# 3. Diferencia de supervivencia
female = df.loc[
    df["Sex"] == "female", "Survived"
]
male = df.loc[
    df["Sex"] == "male", "Survived"
]

pf = female.mean()
pm = male.mean()

difference = pf - pm

se = np.sqrt(
    pf*(1-pf)/len(female) +
    pm*(1-pm)/len(male)
)

print("\n[3] DIFERENCIA MUJERES - HOMBRES")
print(f"Diferencia: {100*difference:.2f} pp")
print(
    "IC95% aproximado: "
    f"[{100*(difference-z*se):.2f}, "
    f"{100*(difference+z*se):.2f}] pp"
)

# 4. Visualizacion con incertidumbre
fig, ax = plt.subplots(figsize=(9, 5.5))

groups = [
    ("female", -0.10, "#76549A", "Mujeres"),
    ("male", 0.10, "#326F9C", "Hombres")
]

for sex, offset, color, label in groups:
    t = subgroups[subgroups["sexo"] == sex]

    x = t["clase"].to_numpy() + offset
    y = t["tasa_pct"].to_numpy()

    errors = np.vstack((
        y - t["ic_inf_pct"].to_numpy(),
        t["ic_sup_pct"].to_numpy() - y
    ))

    ax.errorbar(
        x, y, yerr=errors,
        fmt="o", capsize=5,
        lw=2, markersize=7,
        color=color, label=label
    )

    for xx, yy, n in zip(x, y, t["n"]):
        ax.annotate(
            f"n={n}",
            (xx, yy),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Primera", "Segunda", "Tercera"])
ax.set_ylim(0, 108)
ax.set_xlabel("Clase")
ax.set_ylabel("Supervivencia (%)")
ax.set_title(
    "Supervivencia por sexo y clase - IC 95% Wilson"
)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    FIGURES / "07_intervalos_sexo_clase.png",
    dpi=200
)
plt.close(fig)

print("\nFASE 3 COMPLETADA")
