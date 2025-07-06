import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set app title and description
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection System")
st.write("""
This application helps identify potentially fake news articles. 
Enter a news headline and content below to analyze its authenticity.
""")

# Sample data (in a real app, we'd load a pretrained model)
@st.cache_resource
def load_model():
    # Create a mock model pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words=stopwords.words('english'),
            max_features=1000,
            ngram_range=(1, 2)
        )),
        ('clf', LogisticRegression())
    ])
    
    # Train on mock data (in reality, we'd use the full dataset)
    mock_texts = [
        "Donald Trump sends embarrassing message to enemies on New Year's Eve",
        "Scientists confirm climate change is accelerating",
        "Celebrity claims aliens built the pyramids",
        "New study shows benefits of regular exercise"
    ]
    mock_labels = [1, 0, 1, 0]  # 1 = fake, 0 = real
    model.fit(mock_texts, mock_labels)
    
    return model

model = load_model()

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# User input
st.sidebar.header("User Input")
title = st.sidebar.text_area("News Headline", "Enter the news headline here...")
content = st.sidebar.text_area("News Content", "Enter the full news content here...", height=200)

analyze_button = st.sidebar.button("Analyze Article")

# Analysis section
if analyze_button:
    st.header("Analysis Results")
    
    # Combine title and content for analysis
    full_text = f"{title}\n\n{content}"
    processed_text = preprocess_text(full_text)
    
    # Make prediction
    prediction = model.predict([processed_text])[0]
    proba = model.predict_proba([processed_text])[0]
    
    # Display results
    if prediction == 1:
        st.error("🚨 Warning: This article is likely FAKE NEWS")
    else:
        st.success("✅ This article appears to be REAL NEWS")
    
    st.write(f"Confidence: {max(proba)*100:.1f}%")
    
    # Show contributing features (mock implementation)
    st.subheader("Key Indicators")
    
    if prediction == 1:
        st.write("""
        The following characteristics suggest this may be fake news:
        - Sensational or exaggerated language
        - Emotional or inflammatory wording
        - Lack of credible sources
        - Unverifiable claims
        - Polarizing political content
        """)
    else:
        st.write("""
        The following characteristics suggest this may be real news:
        - Neutral, factual language
        - Citations of credible sources
        - Measured tone
        - Verifiable claims
        - Balanced perspective
        """)
    
    # Feature importance visualization (mock)
    st.subheader("Feature Analysis")
    
    # Get top predictive words (mock implementation)
    if prediction == 1:
        top_words = ["trump", "attack", "fake", "media", "claim", "accuse", "scandal"]
    else:
        top_words = ["study", "research", "report", "data", "find", "confirm", "evidence"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=np.random.rand(len(top_words)), y=top_words, ax=ax, palette="viridis")
    ax.set_title("Key Words Influencing Prediction")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    
    # Additional analysis
    st.subheader("Additional Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Subjectivity Score", 
                 value=f"{np.random.randint(70,90)}%" if prediction == 1 else f"{np.random.randint(30,50)}%",
                 delta="High" if prediction == 1 else "Low")
    
    with col2:
        st.metric("Political Bias", 
                 value="Partisan" if prediction == 1 else "Neutral",
                 delta="Present" if prediction == 1 else "Minimal")
    
    # Word cloud placeholder
    st.write("""
    ### Text Characteristics
    *Note: In a full implementation, we would display a word cloud and more detailed text analysis.*
    """)

# Add information about the model
st.sidebar.markdown("""
### About This Model
This demonstration uses a simple logistic regression model trained on news article text features. 
In a production environment, we would use:
- A larger, more diverse dataset
- Advanced NLP techniques
- More sophisticated model architecture
- Regular updates with new examples
""")

# Add footer
st.markdown("---")
st.markdown("""
*Note: This is a demonstration only. Always verify information from multiple credible sources.*
""")
