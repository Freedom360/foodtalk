import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import tempfile
import os
import math
import base64
import io

@st.cache_resource
def setup():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

setup()

# Streamlit UI
st.title("Food Talk")
st.write("Upload your dataset to discover main discussion themes in restaurant reviews.")

uploaded_file = st.file_uploader("Upload TXT Yelp reviews", type=["txt"])
num_topics = st.slider("Number of Topics", 2, 15, 10)
passes = st.slider("Number of Passes", 5, 30, 15)

if uploaded_file:
    # ===== Load Dataset =====
    texts = uploaded_file.read().decode('utf-8').splitlines()

    # ===== Preprocessing =====
    stop_words = set(stopwords.words('english'))
    # stop_words.update(['restaurant', 'place', 'food', 'good', 'great', 'really', 
    #                    'like', 'would', 'get', 'go', 'one', 'time', 'also', 'back', 'try',
    #                      'best', 'table', 'got'])
    stop_words.update([
    # Basic restaurant terms
    'restaurant', 'place', 'food', 'meal', 'dish', 'dishes', 'menu', 'order', 'ordered',
    'eat', 'eating', 'ate', 'lunch', 'dinner', 'breakfast', 'brunch',
    
    # Service & experience
    'service', 'server', 'waiter', 'waitress', 'staff', 'table', 'seat', 'seated',
    'wait', 'waiting', 'waited', 'reservation', 'atmosphere', 'ambiance', 'experience',
    
    # Common adjectives (often not meaningful for analysis)
    'good', 'great', 'nice', 'fine', 'okay', 'ok', 'decent', 'amazing', 'awesome',
    'excellent', 'perfect', 'wonderful', 'fantastic', 'outstanding', 'best', 'worst',
    'bad', 'terrible', 'horrible', 'awful', 'disappointing',
    
    # Temporal & frequency words
    'time', 'times', 'first', 'last', 'next', 'previous', 'again', 'back', 'return',
    'visit', 'visited', 'visiting', 'came', 'come', 'coming', 'went', 'go', 'going',
    
    # Quantity & comparison
    'one', 'two', 'three', 'lot', 'lots', 'much', 'many', 'more', 'most', 'less',
    'little', 'bit', 'quite', 'very', 'really', 'pretty', 'super', 'totally',
    
    # Actions & intentions
    'try', 'tried', 'trying', 'get', 'got', 'getting', 'have', 'had', 'having',
    'would', 'could', 'should', 'will', 'definitely', 'probably', 'maybe',
    'recommend', 'recommended', 'suggest', 'worth',
    
    # Location & direction
    'here', 'there', 'around', 'near', 'close', 'far', 'local', 'area', 'neighborhood',
    
    # Pricing (unless you want to analyze price sentiment)
    'price', 'prices', 'pricing', 'cost', 'costs', 'expensive', 'cheap', 'affordable',
    'reasonable', 'pricey', 'budget', 'value', 'money', 'worth', 'pay', 'paid',
    
    # Connecting words & fillers
    'like', 'also', 'well', 'just', 'even', 'though', 'however', 'although',
    'seems', 'seemed', 'looks', 'looked', 'feels', 'felt', 'think', 'thought',
    
    # Reviews-specific terms
    'review', 'reviews', 'rating', 'star', 'stars', 'yelp', 'google', 'recommend',
    'customer', 'customers', 'people', 'everyone', 'anyone', 'somebody', 'nobody'
])
    lemmatizer = WordNetLemmatizer()

    processed_texts = []
    for text in texts:
        text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
        if len(tokens) > 5:
            processed_texts.append(tokens)

    # ===== Dictionary and Corpus =====
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # ===== Train LDA =====
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        eta='auto'
    )

    # ===== Show Topics =====
    st.subheader("Top Topics")
    for idx, topic in lda_model.print_topics(num_words=10):
        st.write(f"**Topic {idx}:** {topic}")

    # ===== Word Clouds =====
    with st.container():
        st.subheader("Word Clouds for Each Topic")

        # Dynamically calculate rows and columns for near-square layout
        cols = min(5, math.ceil(math.sqrt(num_topics)))
        rows = math.ceil(num_topics / cols)

        # Adjust figure size dynamically based on number of topics
        fig_width = cols * 4
        fig_height = rows * 3
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axes = axes.flatten() if num_topics > 1 else [axes]

        for i in range(num_topics):
            topic_words = dict(lda_model.show_topic(i, topn=30))
            wc = WordCloud(width=500, height=300, background_color='white',
                        colormap='viridis').generate_from_frequencies(topic_words)
            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f'Topic {i}', fontsize=12)
            axes[i].axis('off')

        # Hide unused axes if num_topics is not a perfect square
        for j in range(num_topics, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
            # === Download button for Word Cloud ===
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="Download Word Cloud",
            data=buf.getvalue(),
            file_name="word_cloud_topics.png",
            mime="image/png"
        )
        plt.close(fig)
    

    # ===== Interactive Visualization =====
    with st.container():
        st.subheader("Interactive Topic Visualization")
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

        # Dynamically adjust height based on number of topics (base height = 400)
        dynamic_height = min(1200, max(400, num_topics * 80))
        dynamic_width = min(1400, max(800, num_topics * 100)) 

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            pyLDAvis.save_html(vis_data, tmp_file.name)
            html_content = open(tmp_file.name, 'r', encoding='utf-8').read()
            # st.components.v1.html(html_content, height=dynamic_height, width=dynamic_width, scrolling=True)


        #         # Encode HTML to base64 for opening in new tab
        # b64 = base64.b64encode(html_content.encode()).decode()
        # href = f'<a href="data:text/html;base64,{b64}" target="_blank">🔎 Open Full Screen</a>'    
        st.components.v1.html(html_content, height=dynamic_height, width=dynamic_width, scrolling=True)
        # Provide a download button
        with open(tmp_file.name, "rb") as file:
            st.download_button(
                label="Download LDA Visualization",
                data=file,
                file_name="lda_visualization.html",
                mime="text/html"
            )
        os.unlink(tmp_file.name)

