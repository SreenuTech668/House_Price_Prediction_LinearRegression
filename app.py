import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

 
st.title("🏠 House Price Prediction using Linear Regression")

 
uploaded_file = st.file_uploader("Upload train.csv", type=["csv"])

if uploaded_file is not None:

    
    data = pd.read_csv(uploaded_file)

    
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    
    if "SalePrice" not in data.columns:
        st.error("The dataset must contain a 'SalePrice' column.")
    else:

        
        y = data["SalePrice"]

       
        X = data.drop("SalePrice", axis=1)

    
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

       
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        
        preprocessor = ColumnTransformer([
            (
                "num",
                SimpleImputer(strategy="median"),
                num_cols
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                cat_cols
            )
        ])
 
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ])

       
        model.fit(X_train, y_train)

       
        y_pred = model.predict(X_test)

        
        r2 = r2_score(y_test, y_pred)

      
        train_score = model.score(X_train, y_train)

        
        st.subheader("📊 Model Performance")

        st.success(f"R² Score: {r2 * 100:.2f}%")
        st.info(f"Train Score: {train_score * 100:.2f}%")
 
        result = pd.DataFrame({
            "Actual Price": y_test.values,
            "Predicted Price": y_pred
        })

        st.subheader("🔍 Sample Predictions")
        st.dataframe(result.head(10))
 
        st.subheader("📋 Dataset Information")
        st.write(f"Rows: {data.shape[0]}")
        st.write(f"Columns: {data.shape[1]}")

         
        st.subheader("❓ Missing Values")
        missing = data.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            st.dataframe(missing)
        else:
            st.success("No missing values found!")