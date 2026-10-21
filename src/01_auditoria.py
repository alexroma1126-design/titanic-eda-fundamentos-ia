from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "titanic.csv"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PROYECTO 1 - AUDITORIA EXPLORATORIA DEL TITANIC")
print("=" * 65)

# 1. Lectura
df = pd.read_csv(DATA)

print("\n[1] DIMENSIONES")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

print("\n[2] PRIMEROS REGISTROS")
print(df.head().to_string(index=False))

print("\n[3] TIPOS DE DATOS")
print(df.dtypes.to_string())

# 2. Calidad de datos
print("\n[4] VALORES FALTANTES")
missing = pd.DataFrame({
    "faltantes": df.isna().sum(),
    "porcentaje": df.isna().mean() * 100
}).sort_values("faltantes", ascending=False)

print(missing.round(2).to_string())

print("\n[5] VALIDACIONES")
checks = {
    "duplicados_completos": int(df.duplicated().sum()),
    "duplicados_id": int(df["PassengerId"].duplicated().sum()),
    "edades_negativas": int((df["Age"] < 0).sum()),
    "tarifas_negativas": int((df["Fare"] < 0).sum()),
    "supervivencia_invalida": int(
        (~df["Survived"].isin([0, 1])).sum()
    ),
    "clase_invalida": int(
        (~df["Pclass"].isin([1, 2, 3])).sum()
    )
}

for check, value in checks.items():
    print(f"{check}: {value}")

# 3. Estadistica descriptiva
print("\n[6] ESTADISTICAS DESCRIPTIVAS")
stats = df.describe(include="all").transpose()
print(stats.to_string())

# 4. Supervivencia
print("\n[7] SUPERVIVENCIA GENERAL")
survival = df["Survived"].value_counts().sort_index()
print(survival)
print(f"Tasa: {df['Survived'].mean() * 100:.2f}%")

print("\n[8] SUPERVIVENCIA POR SEXO")
by_sex = df.groupby("Sex")["Survived"].agg(
    pasajeros="count",
    supervivientes="sum",
    tasa="mean"
)
by_sex["tasa"] *= 100
print(by_sex.round(2))

print("\n[9] SUPERVIVENCIA POR CLASE")
by_class = df.groupby("Pclass")["Survived"].agg(
    pasajeros="count",
    supervivientes="sum",
    tasa="mean"
)
by_class["tasa"] *= 100
print(by_class.round(2))

# 5. Exportar evidencias
missing.to_csv(OUTPUT / "01_valores_faltantes.csv")
stats.to_csv(OUTPUT / "02_estadisticas.csv")
by_sex.to_csv(OUTPUT / "03_supervivencia_sexo.csv")
by_class.to_csv(OUTPUT / "04_supervivencia_clase.csv")

pd.DataFrame(
    checks.items(), columns=["control", "incidencias"]
).to_csv(OUTPUT / "05_validaciones.csv", index=False)

print("\n" + "=" * 65)
print("AUDITORIA INICIAL FINALIZADA")
print("Resultados guardados en outputs/")
print("=" * 65)
