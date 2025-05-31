import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide")
st.title("Video O'yin Sotuvlari Tahlili va Bashorat")

@st.cache_data
def load_data():
    df = pd.read_csv("data/vgsales.csv")
    return df

df = load_data()

# Ma'lumotlarni ko'rib chiqish
st.header("Datasetni ko‘rish")
st.dataframe(df.head())

# Null qiymatlar va ularni tozalash
st.header("Null qiymatlar va ma’lumotlarni tozalash")

nulls = df.isnull().sum()
st.write("Har bir ustundagi null qiymatlar:")
st.write(nulls)

fill_option = st.selectbox("Null qiymatlarni qanday tozalash kerak?", 
                           ("Hech nima qilmaslik", "Null qiymatlarni o‘rtacha bilan to‘ldirish", "Null qiymatlarni olib tashlash"))

if fill_option == "Null qiymatlarni o‘rtacha bilan to‘ldirish":
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    df.dropna(subset=['Year'], inplace=True)  # Year uchun nulllarni olib tashlash
    st.success("Sonli ustunlardagi null qiymatlar o‘rtacha bilan to‘ldirildi, Year null qiymatlari olib tashlandi.")
elif fill_option == "Null qiymatlarni olib tashlash":
    df.dropna(inplace=True)
    st.success("Null qiymatlar mavjud bo‘lgan qatorlar olib tashlandi.")
else:
    st.info("Ma’lumotlar to‘g‘ridan-to‘g‘ri ishlatildi.")

# Filtrlash uchun variantlar
st.sidebar.header("Filtrlash")
platforms = df['Platform'].dropna().unique().tolist()
genres = df['Genre'].dropna().unique().tolist()
years = df['Year'].dropna().astype(int)
year_min, year_max = int(years.min()), int(years.max())

selected_platforms = st.sidebar.multiselect("Platformalar", platforms, default=platforms)
selected_genres = st.sidebar.multiselect("Janrlar", genres, default=genres)
selected_years = st.sidebar.slider("Yil diapazoni", year_min, year_max, (year_min, year_max))

df_filtered = df[
    (df['Platform'].isin(selected_platforms)) &
    (df['Genre'].isin(selected_genres)) &
    (df['Year'].between(*selected_years))
]

st.header("Filtrlashdan so‘ng ma’lumotlar")
st.write(f"Tanlangan ma’lumotlar soni: {len(df_filtered)}")
st.dataframe(df_filtered.head())

# Statistik ko‘rsatkichlar
st.subheader("Statistik ko‘rsatkichlar")
st.write(df_filtered.describe())

# Sotuvlar taqsimoti
st.subheader("Global Sotuvlar taqsimoti")
fig1, ax1 = plt.subplots(figsize=(10,4))
sns.histplot(df_filtered['Global_Sales'], bins=30, kde=True, ax=ax1)
st.pyplot(fig1)

# Platformalar bo‘yicha o‘rtacha sotuvlar
st.subheader("Platformalar bo‘yicha o‘rtacha Global Sotuvlar")
platform_sales = df_filtered.groupby('Platform')['Global_Sales'].mean().sort_values(ascending=False)
st.bar_chart(platform_sales)

# Janrlar bo‘yicha Global Sotuvlar pie chart
st.subheader("Janrlar bo‘yicha Global Sotuvlar")
genre_sales = df_filtered.groupby('Genre')['Global_Sales'].sum()
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.pie(genre_sales, labels=genre_sales.index, autopct='%1.1f%%')
st.pyplot(fig2)

# Yillar bo‘yicha sotuvlar trendi
st.subheader("Yillar bo‘yicha Global Sotuvlar tendensiyasi")
year_sales = df_filtered.groupby('Year')['Global_Sales'].sum()
fig3, ax3 = plt.subplots(figsize=(10,4))
sns.lineplot(x=year_sales.index, y=year_sales.values, ax=ax3)
ax3.set_xlabel("Yil")
ax3.set_ylabel("Global Sotuvlar (million dona)")
st.pyplot(fig3)

# Korrelatsiya issiqlik xaritasi
st.subheader("Numeric ustunlar orasidagi korrelatsiya")
numeric_cols = df_filtered.select_dtypes(include=np.number).columns
corr = df_filtered[numeric_cols].corr()
fig4, ax4 = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# Regressiya uchun tayyorlash
st.header("Chiziqli Regressiya Modellari")

# Xususiyatlarni avtomatik tanlash - raqamli ustunlar (target o'zgaruvchi Global_Sales)
numeric_features = list(numeric_cols)
if 'Global_Sales' in numeric_features:
    numeric_features.remove('Global_Sales')

# Kategoriyal ustunlar kodlash uchun
categorical_features = ['Platform', 'Genre', 'Publisher']
categorical_features = [col for col in categorical_features if col in df_filtered.columns]

features_selected = st.multiselect("Xususiyatlarni tanlang", numeric_features + categorical_features, default=numeric_features)

if len(features_selected) > 0:
    # Ma'lumotni tayyorlash
    X = df_filtered[features_selected].copy()
    y = df_filtered['Global_Sales'].copy()
    
    # Kategoriyal ustunlarni one-hot kodlash
    X = pd.get_dummies(X, columns=[col for col in categorical_features if col in features_selected], drop_first=True)
    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)
    
    test_size = st.slider("Test uchun ma'lumotlar ulushi", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.subheader("Model koeffitsiyentlari")
    coefs = pd.Series(model.coef_, index=X.columns)
    st.write(coefs.sort_values(ascending=False))
    
    st.subheader("Modelni baholash")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    st.write(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    st.markdown("""
    **Overfitting** — model o‘rgatish ma’lumotlariga juda moslashib, yangi ma’lumotlarga yomon ishlaydi.  
    **Underfitting** — model ma’lumotlardagi murakkabliklarni o‘rganolmaydi.
    """)

# Logistik regressiya - janrni bashorat qilish
st.header("Logistik Regressiya bilan Janr Bashorati")

if len(features_selected) > 0:
    df_filtered = df_filtered.dropna(subset=features_selected + ['Genre'])
    label_enc = LabelEncoder()
    df_filtered['Genre_enc'] = label_enc.fit_transform(df_filtered['Genre'])
    
    X_log = df_filtered[features_selected].copy()
    X_log = pd.get_dummies(X_log, columns=[col for col in categorical_features if col in features_selected], drop_first=True)
    y_log = df_filtered['Genre_enc']
    
    test_size_log = st.slider("Logistik regressiya uchun test ulushi", 0.1, 0.5, 0.2, key='logistic')
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=test_size_log, random_state=42)
    
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_log, y_train_log)
    y_pred_log = log_model.predict(X_test_log)
    
    st.write(f"Aniqlik: {accuracy_score(y_test_log, y_pred_log):.4f}")
    st.text(classification_report(y_test_log, y_pred_log, target_names=label_enc.classes_))
