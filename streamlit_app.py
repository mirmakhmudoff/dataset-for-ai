import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(layout="wide")
st.title("Video O'yin Sotuvlari Tahlili va Bashorat")

@st.cache_data
def load_data():
    df = pd.read_csv("data/vgsales.csv")
    return df

df = load_data()

# Null qiymatlar
st.header("Null qiymatlar va tozalash")
null_counts = df.isnull().sum()
st.write(null_counts)

fill_method = st.selectbox("Null qiymatlarni qanday tozalash kerak?", 
                           ("Hech nima qilmaslik", "Null qiymatlarni olib tashlash", "Null qiymatlarni ustun o'rtachasi bilan to'ldirish"))

if fill_method == "Null qiymatlarni olib tashlash":
    df_clean = df.dropna()
elif fill_method == "Null qiymatlarni ustun o'rtachasi bilan to'ldirish":
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=np.number).columns:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    df_clean.dropna(subset=['Year'], inplace=True)  # Year null qatorlarini olib tashlash
else:
    df_clean = df.copy()

# Sidebar filtrlar
st.sidebar.header("Filtrlash")
platforms = df_clean['Platform'].dropna().unique().tolist()
genres = df_clean['Genre'].dropna().unique().tolist()
years = df_clean['Year'].dropna().astype(int)
year_min, year_max = int(years.min()), int(years.max())

selected_platforms = st.sidebar.multiselect("Platformalar", platforms, default=platforms)
selected_genres = st.sidebar.multiselect("Janrlar", genres, default=genres)
selected_years = st.sidebar.slider("Yil diapazoni", year_min, year_max, (year_min, year_max))

df_filtered = df_clean[
    (df_clean['Platform'].isin(selected_platforms)) &
    (df_clean['Genre'].isin(selected_genres)) &
    (df_clean['Year'].between(*selected_years))
]

st.header(f"Filtrlash natijasi: {len(df_filtered)} satr")

if df_filtered.empty:
    st.error("Tanlangan filtrlar natijasida ma'lumot qolmadi. Iltimos, filtrlarni o'zgartiring.")
else:
    st.dataframe(df_filtered.head())

    # Statistikalar
    st.subheader("Statistik ko'rsatkichlar")
    st.write(df_filtered.describe())

    # Grafiklar
    st.subheader("Global Sotuvlar Tarqatilishi")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    sns.histplot(df_filtered['Global_Sales'], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Platformalar bo'yicha o'rtacha Global Sotuvlar")
    platform_sales = df_filtered.groupby('Platform')['Global_Sales'].mean().sort_values(ascending=False)
    st.bar_chart(platform_sales)

    st.subheader("Janrlar bo'yicha Global Sotuvlar")
    genre_sales = df_filtered.groupby('Genre')['Global_Sales'].sum()
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.pie(genre_sales, labels=genre_sales.index, autopct='%1.1f%%')
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Yillar bo'yicha Global Sotuvlar Trend")
    year_sales = df_filtered.groupby('Year')['Global_Sales'].sum()
    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.lineplot(x=year_sales.index, y=year_sales.values, ax=ax3)
    ax3.set_xlabel("Yil")
    ax3.set_ylabel("Global Sotuvlar (million dona)")
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Korrelatsiya issiqlik xaritasi")
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns
    corr = df_filtered[numeric_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)
    plt.close(fig4)

    # Regressiya modellarini qurish
    st.header("Chiziqli Regressiya")

    numeric_features = list(numeric_cols)
    if 'Global_Sales' in numeric_features:
        numeric_features.remove('Global_Sales')

    categorical_features = ['Platform', 'Genre', 'Publisher']
    categorical_features = [col for col in categorical_features if col in df_filtered.columns]

    features_selected = st.multiselect("Xususiyatlarni tanlang", numeric_features + categorical_features, default=numeric_features)

    if len(features_selected) > 0:
        X = df_filtered[features_selected].copy()
        y = df_filtered['Global_Sales'].copy()

        X = pd.get_dummies(X, columns=[col for col in categorical_features if col in features_selected], drop_first=True)
        X.fillna(0, inplace=True)
        y.fillna(0, inplace=True)

        test_size = st.slider("Test uchun ma'lumot ulushi", 0.1, 0.5, 0.2)
        if len(X) > 1 and int(len(X)*(1-test_size)) > 0 and int(len(X)*test_size) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred = lr_model.predict(X_test)

            st.subheader("Model koeffitsiyentlari")
            coefs = pd.Series(lr_model.coef_, index=X.columns)
            st.write(coefs.sort_values(ascending=False))

            st.subheader("Modelni baholash")
            st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
            st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"R²: {r2_score(y_test, y_pred):.4f}")

            st.markdown("""
            **Overfitting** — model o‘rgatish ma’lumotlariga juda moslashib, yangi ma’lumotlarga yomon ishlaydi.  
            **Underfitting** — model ma’lumotlardagi murakkabliklarni o‘rganolmaydi.
            """)
        else:
            st.warning("Ma'lumot hajmi kichik, train-test bo‘linmasini bajarib bo‘lmaydi.")

    # Logistik regressiya - janr bashorati
    st.header("Logistik Regressiya bilan Janr Bashorati")

    if len(features_selected) > 0:
        df_filtered_log = df_filtered.dropna(subset=features_selected + ['Genre'])
        if df_filtered_log.shape[0] > 10:
            label_enc = LabelEncoder()
            df_filtered_log['Genre_enc'] = label_enc.fit_transform(df_filtered_log['Genre'])

            X_log = df_filtered_log[features_selected].copy()
            X_log = pd.get_dummies(X_log, columns=[col for col in categorical_features if col in features_selected], drop_first=True)
            y_log = df_filtered_log['Genre_enc']

            test_size_log = st.slider("Logistik regressiya uchun test ulushi", 0.1, 0.5, 0.2, key='logistic')

            if len(X_log) > 1 and int(len(X_log)*(1-test_size_log)) > 0 and int(len(X_log)*test_size_log) > 0:
                X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=test_size_log, random_state=42)

                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, solver='lbfgs', n_jobs=-1))
                pipe.fit(X_train_log, y_train_log)
                y_pred_log = pipe.predict(X_test_log)

                st.write(f"Aniqlik: {accuracy_score(y_test_log, y_pred_log):.4f}")
                st.text(classification_report(y_test_log, y_pred_log, target_names=label_enc.classes_, zero_division=0))
            else:
                st.warning("Logistik regressiya uchun yetarli ma'lumot yo'q.")
        else:
            st.warning("Logistik regressiya uchun yetarli ma'lumot yo'q.")
