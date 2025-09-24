import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("fruit_vegetable_benefits.csv")
    return df

df = load_data()

# Normalize names for case-insensitive matching
df["Name_lower"] = df["Name"].str.lower()

st.title("🍎 Fruit and Vegetable Benefit Similarity Finder")

st.markdown("""
This app helps you find **fruits or vegetables with the most similar health benefits**.

Type the name of a fruit or vegetable to get recommendations based on benefit similarity.
""")

# --- User input ---
user_input = st.text_input("Enter the name of a fruit or vegetable:").strip().lower()

if user_input:
    if user_input in df["Name_lower"].values:
        # Get original row
        item_row = df[df["Name_lower"] == user_input].iloc[0]
        item_name = item_row["Name"]
        item_benefit = item_row["Benefit"]
        item_index = item_row.name

        # --- Show benefit of selected item ---
        st.subheader(f"Selected Item: **{item_name}**")
        st.markdown(f"**Benefit:** {item_benefit}")

        # --- Vectorize benefits ---
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["Benefit"].fillna(""))

        # --- Compute similarity ---
        similarity = cosine_similarity(X)

        # --- Add similarity column ---
        similarities = similarity[item_index]
        df_result = df.copy()
        df_result["Similarity"] = similarities

        # --- Sort by similarity (excluding self) ---
        df_result = df_result.sort_values("Similarity", ascending=False)
        df_result_top = df_result[df_result["Name_lower"] != user_input].head(5)

        # --- Display Top 5 with index starting from 1 ---
        st.subheader("Top 5 Similar Items")
        df_display = df_result_top[["Name", "Benefit", "Similarity"]].reset_index(drop=True)
        df_display.index = df_display.index + 1  # Set index to start from 1
        st.dataframe(df_display)

        # --- Visualization ---
        st.subheader("Similarity Chart")

        fig, ax = plt.subplots()
        ax.barh(df_result_top["Name"], df_result_top["Similarity"], color="skyblue")
        ax.set_xlabel("Similarity")
        ax.set_title("Top 5 Similar Fruits or Vegetables")
        ax.invert_yaxis()
        st.pyplot(fig)

        # --- Optional download (hide index in CSV) ---
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Similarity Results", csv, "similarity_results.csv", "text/csv")

    else:
        st.warning("Item not found in the dataset. Please check the spelling.")
