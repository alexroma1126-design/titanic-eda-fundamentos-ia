from pathlib import Path

import nbformat
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "titanic.csv"
NOTEBOOK = ROOT / "notebooks" / "EDA_Titanic_Entrega.ipynb"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def test_estructura(df):
    assert df.shape == (891, 12)
    assert {
        "PassengerId", "Survived", "Pclass",
        "Name", "Sex", "Age", "SibSp",
        "Parch", "Ticket", "Fare",
        "Cabin", "Embarked"
    } == set(df.columns)


def test_integridad(df):
    assert df["PassengerId"].is_unique
    assert df.duplicated().sum() == 0
    assert df["Survived"].isin([0, 1]).all()
    assert df["Pclass"].isin([1, 2, 3]).all()
    assert (df["Age"].dropna() >= 0).all()
    assert (df["Fare"] >= 0).all()


def test_valores_faltantes(df):
    assert df["Age"].isna().sum() == 177
    assert df["Cabin"].isna().sum() == 687
    assert df["Embarked"].isna().sum() == 2


def test_supervivencia(df):
    assert df["Survived"].sum() == 342
    assert (df["Survived"] == 0).sum() == 549
    assert df["Survived"].mean() == pytest.approx(
        342 / 891
    )


def test_tasas_por_grupo(df):
    female = df[df["Sex"] == "female"]
    male = df[df["Sex"] == "male"]

    assert len(female) == 314
    assert len(male) == 577

    assert female["Survived"].mean() == pytest.approx(
        233 / 314
    )

    assert male["Survived"].mean() == pytest.approx(
        109 / 577
    )


def test_notebook_ejecutado():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)

    code_cells = [
        c for c in nb.cells
        if c.cell_type == "code"
    ]

    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]

    assert len(nb.cells) == 19
    assert len(code_cells) == 11
    assert all(
        c.execution_count is not None
        for c in code_cells
    )
    assert len(errors) == 0
