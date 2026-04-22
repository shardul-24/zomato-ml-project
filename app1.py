import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')  # Suppress minor warnings

st.set_page_config(page_title="Restaurant Rating AI", layout="wide")
st.title("🍽️ Smart Restaurant Rating Predictor + Recommender")

# -----------------------------
# Load and train model (cached)
# -----------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("Zomato-data-.csv")

    # Clean rate column
    df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False)
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    df.dropna(subset=['rate'], inplace=True)

    # Keep a copy for recommendations & visualizations
    df_raw = df.copy()

    # Drop unnecessary columns (keep 'name' for recommendations)
    restaurant_names = None
    if 'name' in df.columns:
        restaurant_names = df['name']
        df = df.drop(columns=['name'])

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('rate', axis=1)
    y = df['rate']

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, X.columns.tolist(), X, y, df_raw, restaurant_names, feature_importance

(model, feature_columns, X_train, y_train, df_raw, restaurant_names, feature_importance) = load_and_train()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔍 Enter Restaurant Details")

votes = st.sidebar.slider("Number of Votes", 0, 1000, 100)
cost = st.sidebar.slider("Approx Cost for Two (₹)", 100, 3000, 500)
online_order = st.sidebar.selectbox("Online Order Available?", ["Yes", "No"])

# Get unique restaurant types from raw data
if 'listed_in(type)' in df_raw.columns:
    rest_types = sorted(df_raw['listed_in(type)'].dropna().unique())
else:
    rest_types = ["Buffet", "Cafes", "Dining", "Other"]
rest_type = st.sidebar.selectbox("Restaurant Type", rest_types)

# Build raw input row
input_raw = pd.DataFrame([{
    'votes': votes,
    'approx_cost(for two people)': cost,
    'online_order': online_order,
    'listed_in(type)': rest_type
}])

# Apply same one-hot encoding
input_encoded = pd.get_dummies(input_raw, drop_first=True)

# Align with training columns
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_columns]

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("✨ Predict Rating & Find Similar Restaurants"):
    prediction = model.predict(input_encoded)[0]
    
    # Display predicted rating prominently
    st.markdown(f"## ⭐ Predicted Rating: **{round(prediction, 2)} / 5**")
    
    # ---- UNIQUE FEATURE 1: Feature Importance ----
    st.subheader("📊 What influences this rating?")
    fig_imp = px.bar(feature_importance.head(10), x='importance', y='feature', orientation='h',
                     title="Top 10 Most Important Features (Global Model)")
    st.plotly_chart(fig_imp, width='stretch')  # fixed deprecation
    
    # ---- UNIQUE FEATURE 2: Similar Restaurants Recommender ----
    st.subheader("🍴 Restaurants similar to your search")
    
    # Convert to numpy array to avoid feature name warning
    query = input_encoded.values.reshape(1, -1)
    X_train_np = X_train.values  # use numpy array for NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    nn.fit(X_train_np)
    distances, indices = nn.kneighbors(query)
    
    similar_restaurants = []
    for i, idx in enumerate(indices[0]):
        if restaurant_names is not None:
            name = restaurant_names.iloc[idx]
        else:
            name = f"Restaurant {idx}"
        real_rating = y_train.iloc[idx]
        distance = distances[0][i]
        similar_restaurants.append((name, real_rating, distance))
    
    # Use consistent column name: "Similarity Score"
    sim_df = pd.DataFrame(similar_restaurants, 
                          columns=["Restaurant Name", "Actual Rating", "Similarity Score"])
    sim_df["Similarity Score"] = sim_df["Similarity Score"].round(3)
    
    st.dataframe(sim_df, width='stretch')  # fixed deprecation
    
    # ---- UNIQUE FEATURE 3: Rating Distribution ----
    st.subheader("📈 How does your predicted rating compare to real restaurants?")
    hist_data = y_train.values
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=hist_data, nbinsx=20, name="Actual Ratings", opacity=0.7))
    fig_hist.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="red",
                       annotation_text=f"Your prediction: {prediction:.2f}")
    fig_hist.update_layout(title="Distribution of Real Restaurant Ratings", xaxis_title="Rating", yaxis_title="Count")
    st.plotly_chart(fig_hist, width='stretch')  # fixed deprecation
    
    # Model performance
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    st.caption(f"📐 Model accuracy (Mean Absolute Error on training data): **{mae:.2f}** stars")

else:
    st.info("👈 Fill in the restaurant details in the sidebar and click **Predict Rating** to see the magic!")
