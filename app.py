from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Prediction")
st.write("Enter the car details below to get the predicted price.")

DATA_PATH = Path(__file__).with_name("car_price_prediction.csv")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH.name} was not found. Place it in the same folder as app.py."
        )
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_engine_volume(series: pd.Series) -> pd.DataFrame:
    s = series.astype(str).fillna("")
    turbo = s.str.contains("turbo", case=False, na=False).astype(int)
    numeric = (
        s.str.replace("Turbo", "", case=False, regex=False)
        .str.replace("turbo", "", case=False, regex=False)
        .str.extract(r"([0-9]*\.?[0-9]+)")[0]
    )
    numeric = pd.to_numeric(numeric, errors="coerce")
    return pd.DataFrame({"Engine volume": numeric, "Turbo": turbo})


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if "Levy" in df.columns:
        df["Levy"] = df["Levy"].replace("-", np.nan)
        df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")

    if "Mileage" in df.columns:
        df["Mileage"] = df["Mileage"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
        df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

    if "Engine volume" in df.columns:
        engine_parts = clean_engine_volume(df["Engine volume"])
        df["Engine volume"] = engine_parts["Engine volume"]
        df["Turbo"] = engine_parts["Turbo"]
    else:
        df["Turbo"] = 0

    if "Doors" in df.columns:
        df["Doors"] = df["Doors"].astype(str)

    return df


@st.cache_resource(show_spinner=True)
def train_model():
    df = load_dataset()

    if "Price" not in df.columns:
        raise ValueError("The dataset must contain a Price column.")

    df = preprocess(df)

    y = pd.to_numeric(df["Price"], errors="coerce")
    X = df.drop(columns=["Price"])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    return model, df


try:
    with st.spinner("Loading model..."):
        model, raw_df = train_model()
except Exception as e:
    st.error(str(e))
    st.stop()

processed_df = preprocess(raw_df)


def unique_values(column: str):
    if column in raw_df.columns:
        return sorted(raw_df[column].dropna().astype(str).unique().tolist())
    return [""]


with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        levy_default = (
            float(processed_df["Levy"].median())
            if "Levy" in processed_df.columns and processed_df["Levy"].notna().any()
            else 0.0
        )
        levy = st.number_input("Levy", min_value=0.0, value=levy_default, step=1.0)

        manufacturer = st.selectbox(
            "Manufacturer", unique_values("Manufacturer")
        )

        model_name = st.text_input(
            "Model",
            value=str(raw_df["Model"].dropna().astype(str).iloc[0])
            if "Model" in raw_df.columns and raw_df["Model"].notna().any()
            else "",
        )

        prod_year_default = (
            int(processed_df["Prod. year"].median())
            if "Prod. year" in processed_df.columns and processed_df["Prod. year"].notna().any()
            else 2015
        )
        prod_year = st.number_input(
            "Prod. year", min_value=1900, max_value=2100, value=prod_year_default, step=1
        )

    with c2:
        category = st.selectbox("Category", unique_values("Category"))
        leather = st.selectbox("Leather interior", ["Yes", "No"])
        fuel_type = st.selectbox("Fuel type", unique_values("Fuel type"))

        engine_default = (
            float(processed_df["Engine volume"].median())
            if "Engine volume" in processed_df.columns and processed_df["Engine volume"].notna().any()
            else 2.0
        )
        engine_volume = st.number_input(
            "Engine volume",
            min_value=0.0,
            value=engine_default,
            step=0.1,
            format="%.1f",
        )

    with c3:
        mileage_default = (
            float(processed_df["Mileage"].median())
            if "Mileage" in processed_df.columns and processed_df["Mileage"].notna().any()
            else 100000.0
        )
        mileage = st.number_input(
            "Mileage", min_value=0.0, value=mileage_default, step=1000.0
        )

        cylinders_default = (
            float(processed_df["Cylinders"].median())
            if "Cylinders" in processed_df.columns and processed_df["Cylinders"].notna().any()
            else 4.0
        )
        cylinders = st.number_input(
            "Cylinders", min_value=1.0, value=cylinders_default, step=1.0
        )

        gear_box_type = st.selectbox("Gear box type", unique_values("Gear box type"))
        drive_wheels = st.selectbox("Drive wheels", unique_values("Drive wheels"))

    c4, c5, c6 = st.columns(3)
    with c4:
        doors = st.selectbox("Doors", unique_values("Doors"))
    with c5:
        wheel = st.selectbox("Wheel", unique_values("Wheel"))
    with c6:
        color = st.selectbox("Color", unique_values("Color"))

    airbags_default = (
        float(processed_df["Airbags"].median())
        if "Airbags" in processed_df.columns and processed_df["Airbags"].notna().any()
        else 4.0
    )
    airbags = st.number_input("Airbags", min_value=0.0, value=airbags_default, step=1.0)

    turbo = st.selectbox("Turbo", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_row = pd.DataFrame(
        [
            {
                "Levy": levy,
                "Manufacturer": manufacturer,
                "Model": model_name,
                "Prod. year": prod_year,
                "Category": category,
                "Leather interior": leather,
                "Fuel type": fuel_type,
                "Engine volume": engine_volume,
                "Mileage": mileage,
                "Cylinders": cylinders,
                "Gear box type": gear_box_type,
                "Drive wheels": drive_wheels,
                "Doors": doors,
                "Wheel": wheel,
                "Color": color,
                "Airbags": airbags,
                "Turbo": turbo,
            }
        ]
    )

    prediction = model.predict(input_row)[0]
    st.success(f"{prediction:,.2f}")
