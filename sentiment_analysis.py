# streamlit_bert_dashboard_buttons.py

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# -------------------------------
# Load Data
# -------------------------------
st.set_page_config(page_title="EV Sentiment Dashboard", layout="wide")
st.title("EV Reviews Sentiment Dashboard (BYD & EMAS)")

DATA_PATH = "sentiment_analysis_data.csv"  # Path to your CSV
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data(DATA_PATH)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment Count", 
    "Sentiment Percentage", 
    "Example Reviews", 
    "True vs Predicted", 
    "Top Words"
])

# -------------------------------
# Tab 1: Sentiment Count per Brand
# -------------------------------
with tab1:
    st.subheader("1. Sentiment Count per Brand")
    
    brand_options_tab1 = ["All"] + list(df['type'].unique())
    brand_tab1 = st.radio("Select Brand (Tab1)", options=brand_options_tab1, index=0, horizontal=True, key="brand_tab1")
    
    sentiment_options_tab1 = ["All"] + list(df['Predicted Label'].unique())
    sentiment_tab1 = st.radio("Select Sentiment (Tab1)", options=sentiment_options_tab1, index=0, horizontal=True, key="sentiment_tab1")
    
    df_filtered1 = df.copy()
    if brand_tab1 != "All":
        df_filtered1 = df_filtered1[df_filtered1['type'] == brand_tab1]
    if sentiment_tab1 != "All":
        df_filtered1 = df_filtered1[df_filtered1['Predicted Label'] == sentiment_tab1]
    
    # Info boxes
    total_reviews = len(df_filtered1)
    positive_reviews = len(df_filtered1[df_filtered1['Predicted Label']=='positive'])
    negative_reviews = len(df_filtered1[df_filtered1['Predicted Label']=='negative'])
    neutral_reviews  = len(df_filtered1[df_filtered1['Predicted Label']=='neutral'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", total_reviews)
    col2.metric("Positive", f"{positive_reviews} ({positive_reviews/total_reviews*100:.1f}%)" if total_reviews>0 else "0")
    col3.metric("Negative", f"{negative_reviews} ({negative_reviews/total_reviews*100:.1f}%)" if total_reviews>0 else "0")
    col4.metric("Neutral", f"{neutral_reviews} ({neutral_reviews/total_reviews*100:.1f}%)" if total_reviews>0 else "0")
    
    # Bar chart
    if not df_filtered1.empty:
        sentiment_counts = df_filtered1.groupby(['type', 'Predicted Label']).size().reset_index(name='count')
        fig_bar = px.bar(sentiment_counts, x='type', y='count', color='Predicted Label', barmode='group',
                         labels={'type':'Brand', 'count':'Number of Reviews', 'Predicted Label':'Sentiment'})
        st.plotly_chart(fig_bar)
    else:
        st.info("No reviews available for selected filters.")

# -------------------------------
# Tab 2: Sentiment Percentage per Brand
# -------------------------------
with tab2:
    st.subheader("2. Sentiment Percentage per Brand")
    
    brand_options_tab2 = ["All"] + list(df['type'].unique())
    brand_tab2 = st.radio("Select Brand (Tab2)", options=brand_options_tab2, index=0, horizontal=True, key="brand_tab2")
    
    sentiment_options_tab2 = ["All"] + list(df['Predicted Label'].unique())
    sentiment_tab2 = st.radio("Select Sentiment (Tab2)", options=sentiment_options_tab2, index=0, horizontal=True, key="sentiment_tab2")
    
    df_filtered2 = df.copy()
    if brand_tab2 != "All":
        df_filtered2 = df_filtered2[df_filtered2['type'] == brand_tab2]
    if sentiment_tab2 != "All":
        df_filtered2 = df_filtered2[df_filtered2['Predicted Label'] == sentiment_tab2]
    
    # Info boxes
    col1, col2 = st.columns(2)
    col1.metric("Total Reviews", len(df_filtered2))
    for brand in df_filtered2['type'].unique():
        df_brand = df_filtered2[df_filtered2['type'] == brand]
        if not df_brand.empty:
            dominant_sentiment = df_brand['Predicted Label'].value_counts().idxmax()
            dominant_count = df_brand['Predicted Label'].value_counts().max()
            col2.metric(f"Top Sentiment ({brand})", f"{dominant_sentiment} ({dominant_count} reviews)")
    
    # Pie chart per brand
    for brand in df_filtered2['type'].unique():
        df_brand = df_filtered2[df_filtered2['type'] == brand]
        if not df_brand.empty:
            fig_pie = px.pie(df_brand, names='Predicted Label', title=f"{brand} Sentiment Distribution", hole=0.3)
            st.plotly_chart(fig_pie)

# -------------------------------
# Tab 3: Example Reviews
# -------------------------------
with tab3:
    st.subheader("3. Example Reviews")
    
    brand_options_tab3 = ["All"] + list(df['type'].unique())
    brand_tab3 = st.radio("Select Brand (Tab3)", options=brand_options_tab3, index=0, horizontal=True, key="brand_tab3")
    
    sentiment_options_tab3 = ["All"] + list(df['Predicted Label'].unique())
    sentiment_tab3 = st.radio("Select Sentiment (Tab3)", options=sentiment_options_tab3, index=0, horizontal=True, key="sentiment_tab3")
    
    num_reviews = st.slider("Number of reviews to show (Tab3)", 5, 50, 10, key="num_reviews_tab3")
    
    df_filtered3 = df.copy()
    if brand_tab3 != "All":
        df_filtered3 = df_filtered3[df_filtered3['type'] == brand_tab3]
    if sentiment_tab3 != "All":
        df_filtered3 = df_filtered3[df_filtered3['Predicted Label'] == sentiment_tab3]
    
    st.metric("Total filtered reviews", len(df_filtered3))
    st.dataframe(df_filtered3[['type','Predicted Label','review_text']].head(num_reviews))

# -------------------------------
# Tab 4: True vs Predicted
# -------------------------------
with tab4:
    st.subheader("4. True vs Predicted Sentiment")
    if 'true_label' in df.columns:
        brand_options_tab4 = ["All"] + list(df['type'].unique())
        brand_tab4 = st.radio("Select Brand (Tab4)", options=brand_options_tab4, index=0, horizontal=True, key="brand_tab4")
        
        df_filtered4 = df.copy()
        if brand_tab4 != "All":
            df_filtered4 = df_filtered4[df_filtered4['type'] == brand_tab4]
        
        correct = sum(df_filtered4['Predicted Label']==df_filtered4['true_label'])
        total = len(df_filtered4)
        accuracy = correct / total * 100 if total>0 else 0
        
        col1, col2 = st.columns(2)
        col1.metric("Total Reviews", total)
        col2.metric("Accuracy (%)", f"{accuracy:.1f}%")
        
        if not df_filtered4.empty:
            stacked_counts = df_filtered4.groupby(['type', 'true_label', 'Predicted Label']).size().reset_index(name='count')
            fig_stacked = px.bar(stacked_counts, x='type', y='count', color='Predicted Label', 
                                 facet_col='true_label', barmode='stack',
                                 labels={'type':'Brand', 'count':'Number of Reviews'},
                                 title="True vs Predicted Sentiment per Brand")
            st.plotly_chart(fig_stacked)
        else:
            st.info("No reviews available for selected filters.")
    else:
        st.info("True labels not available for stacked bar chart.")

# -------------------------------
# Tab 5: Top Words
# -------------------------------
with tab5:
    st.subheader("5. Top Words per Brand & Sentiment")
    
    brand_options_tab5 = ["All"] + list(df['type'].unique())
    brand_tab5 = st.radio("Select Brand (Tab5)", options=brand_options_tab5, index=0, horizontal=True, key="brand_tab5")
    
    sentiment_options_tab5 = ["All"] + list(df['Predicted Label'].unique())
    sentiment_tab5 = st.radio("Select Sentiment (Tab5)", options=sentiment_options_tab5, index=0, horizontal=True, key="sentiment_tab5")
    
    top_n = st.slider("Top N words to display (Tab5)", 5, 20, 10, key="top_n_tab5")
    
    df_filtered5 = df.copy()
    if brand_tab5 != "All":
        df_filtered5 = df_filtered5[df_filtered5['type'] == brand_tab5]
    if sentiment_tab5 != "All":
        df_filtered5 = df_filtered5[df_filtered5['Predicted Label'] == sentiment_tab5]
    
    # Collect words
    all_words = []
    for review in df_filtered5['review_text']:
        text = review.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = [w for w in text.split() if w not in stop_words]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    
    # Info boxes
    col1, col2 = st.columns(2)
    col1.metric("Total words in selection", len(all_words))
    col2.metric("Unique words", len(word_counts))
    
    if top_words:
        df_top_words = pd.DataFrame(top_words, columns=['Word','Count'])
        fig_word = px.bar(df_top_words, x='Count', y='Word', orientation='h', 
                          title=f"Top {top_n} words in filtered reviews",
                          labels={'Count':'Frequency','Word':'Word'})
        st.plotly_chart(fig_word)
    else:
        st.info("No words found in the filtered reviews.")


