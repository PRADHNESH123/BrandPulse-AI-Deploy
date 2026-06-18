import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#import tensorflow as tf
import joblib
import sys
import random
from datetime import datetime, timedelta
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append('C:/data/BrandPulse-AI')
from src.preprocessing import clean_tweet

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title='BrandPulse AI',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 15px;
    border-left: 4px solid #4ECDC4;
}
.positive { color: #4ECDC4; font-size: 24px; font-weight: bold; }
.negative { color: #FF6B6B; font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf    = joblib.load('C:/data/BrandPulse-AI/models/tfidf_vectorizer.pkl')
    lr_model = joblib.load('C:/data/BrandPulse-AI/models/logistic_model.pkl')
    lstm     = load_model('C:/data/BrandPulse-AI/models/lstm_best.h5')
    with open('C:/data/BrandPulse-AI/models/tokenizer.json') as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    return tfidf, lr_model, lstm, tokenizer

tfidf, lr_model, lstm_model, tokenizer = load_models()
MAX_LEN = 100

# ── Prediction function ───────────────────────────────────
def predict_sentiment(text, model_choice=None):
    cleaned = clean_tweet(text)
    if not cleaned.strip():
        return None, None, cleaned
    vec = tfidf.transform([cleaned])
    
    if model_choice == 'Naive Bayes':
        pred = nb_model.predict(vec)[0]
        conf = max(nb_model.predict_proba(vec)[0]) * 100
    else:
        pred = lr_model.predict(vec)[0]
        conf = max(lr_model.predict_proba(vec)[0]) * 100
    
    return pred, conf, cleaned

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title('📊 BrandPulse AI')
st.sidebar.markdown('*Twitter Sentiment Analysis*')
st.sidebar.divider()

model_choice = st.sidebar.selectbox(
    '🤖 Choose Model',
    ['Logistic Regression', 'LSTM (BiLSTM)']
)

st.sidebar.divider()
st.sidebar.markdown('### 📈 Model Performance')
col1, col2, col3 = st.sidebar.columns(3)
col1.metric('LR', '80%', 'Fast')
col2.metric('LSTM', '81%', 'Accurate')
col3.metric('NB', '77.36%', 'Fastest')

st.sidebar.divider()
st.sidebar.markdown('### 📁 Project Info')
st.sidebar.info('''
**Dataset:** Sentiment140
**Tweets:** 1.6M
**Classes:** Positive / Negative
**Built by:** PRADHNESH
''')

# ── Header ────────────────────────────────────────────────
st.title('  BrandPulse AI')
st.markdown('#### Real-time Twitter Sentiment Analysis — Classical NLP vs Deep Learning')
st.divider()

# ── Top metrics ───────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric('🐦 Tweets Analyzed', '1,592,958', '+1.6M')
m2.metric('🎯 Best Accuracy', '81.04%', 'LSTM')
m3.metric('⚡ Fastest Model', 'Logistic Reg', '10x faster')
m4.metric('📊 Dataset Balance', '50/50', 'Balanced')

st.divider()

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '🔍 Live Predictor',
    '📊 Sentiment Dashboard',
    '📈 Trend Analysis',
    '⚖️ Model Comparison'
])

# ════════════════════════════════════════════════════════
# TAB 1 — Live Predictor
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader(' Live Tweet Sentiment Predictor')

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            'Enter a tweet:',
            placeholder='Type your tweet here...',
            height=120
        )

        if st.button(' Analyze Sentiment', type='primary'):
            if user_input.strip():
                with st.spinner('Analyzing...'):
                    pred, conf, cleaned = predict_sentiment(
                        user_input, model_choice)

                if pred == 1:
                    st.success(f'😊 **POSITIVE** — Confidence: {conf:.1f}%')
                    st.balloons()
                elif pred == 0:
                    st.error(f'😞 **NEGATIVE** — Confidence: {conf:.1f}%')
                else:
                    st.warning('Could not analyze — tweet too short after cleaning')

                with st.expander('🔍 See preprocessing details'):
                    st.write(f'**Original:** {user_input}')
                    st.write(f'**Cleaned:** {cleaned}')
                    st.write(f'**Model used:** {model_choice}')
            else:
                st.warning('Please enter a tweet!')

    with col2:
        st.markdown('### 💡 Try these:')
        examples = [
            ("😊", "This product is absolutely amazing!"),
            ("😞", "Worst experience ever. Never again!"),
            ("😊", "Best purchase I have ever made!"),
            ("😞", "The delivery was late and damaged."),
            ("😊", "Highly recommend to everyone!")
        ]
        for emoji, ex in examples:
            if st.button(f'{emoji} {ex[:35]}...', key=ex):
                pred, conf, cleaned = predict_sentiment(ex, model_choice)
                if pred == 1:
                    st.success(f'😊 POSITIVE ({conf:.1f}%)')
                elif pred == 0:
                    st.error(f'😞 NEGATIVE ({conf:.1f}%)')

    st.divider()

    # Batch prediction
    st.subheader('📋 Batch Prediction')
    batch_input = st.text_area(
        'Enter multiple tweets (one per line):',
        height=120,
        placeholder='Tweet 1\nTweet 2\nTweet 3'
    )

    if st.button(' Analyze All Tweets'):
        if batch_input.strip():
            tweets = [t.strip() for t in batch_input.split('\n') if t.strip()]
            results = []
            progress = st.progress(0)
            for i, tweet in enumerate(tweets):
                pred, conf, cleaned = predict_sentiment(tweet, model_choice)
                results.append({
                    'Tweet': tweet[:60] + '...' if len(tweet) > 60 else tweet,
                    'Sentiment': '😊 Positive' if pred == 1 else '😞 Negative',
                    'Confidence': f'{conf:.1f}%' if conf else 'N/A'
                })
                progress.progress((i+1)/len(tweets))

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            pos = sum(1 for r in results if 'Positive' in r['Sentiment'])
            neg = len(results) - pos

            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    values=[pos, neg],
                    names=['Positive', 'Negative'],
                    color_discrete_map={
                        'Positive': '#4ECDC4',
                        'Negative': '#FF6B6B'
                    },
                    title='Batch Sentiment Distribution',
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_bar = px.bar(
                    x=['Positive', 'Negative'],
                    y=[pos, neg],
                    color=['Positive', 'Negative'],
                    color_discrete_map={
                        'Positive': '#4ECDC4',
                        'Negative': '#FF6B6B'
                    },
                    title='Sentiment Count'
                )
                st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Sentiment Dashboard
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader('📊 Sentiment Dashboard')

    # Simulated live data
    np.random.seed(42)
    n = 500
    sentiments = np.random.choice(['Positive', 'Negative'], n, p=[0.55, 0.45])
    topics = np.random.choice(
        ['Product', 'Service', 'Delivery', 'Support', 'Price'], n)
    hours = [datetime.now() - timedelta(minutes=i*5) for i in range(n)]

    df_live = pd.DataFrame({
        'time': hours,
        'sentiment': sentiments,
        'topic': topics
    })

    pos_count = (df_live['sentiment'] == 'Positive').sum()
    neg_count = (df_live['sentiment'] == 'Negative').sum()

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Tweets', n, '+500')
    col2.metric('😊 Positive', pos_count, f'{pos_count/n*100:.1f}%')
    col3.metric('😞 Negative', neg_count, f'-{neg_count/n*100:.1f}%')

    col1, col2 = st.columns(2)

    with col1:
        fig_donut = px.pie(
            values=[pos_count, neg_count],
            names=['Positive', 'Negative'],
            color_discrete_map={
                'Positive': '#4ECDC4',
                'Negative': '#FF6B6B'
            },
            hole=0.5,
            title='Overall Sentiment Distribution'
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        topic_sentiment = df_live.groupby(
            ['topic', 'sentiment']).size().reset_index(name='count')
        fig_topic = px.bar(
            topic_sentiment,
            x='topic',
            y='count',
            color='sentiment',
            color_discrete_map={
                'Positive': '#4ECDC4',
                'Negative': '#FF6B6B'
            },
            title='Sentiment by Topic',
            barmode='group'
        )
        st.plotly_chart(fig_topic, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Trend Analysis
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader('📈 Sentiment Trend — Last 24 Hours')

    hours_range = [datetime.now() - timedelta(hours=i)
                   for i in range(24, 0, -1)]
    np.random.seed(42)
    pos_trend = np.random.randint(45, 75, 24)
    neg_trend = 100 - pos_trend

    df_trend = pd.DataFrame({
        'Time': hours_range,
        'Positive': pos_trend,
        'Negative': neg_trend
    })

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_trend['Time'],
        y=df_trend['Positive'],
        name='Positive',
        fill='tozeroy',
        fillcolor='rgba(78,205,196,0.2)',
        line=dict(color='#4ECDC4', width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=df_trend['Time'],
        y=df_trend['Negative'],
        name='Negative',
        fill='tozeroy',
        fillcolor='rgba(255,107,107,0.2)',
        line=dict(color='#FF6B6B', width=2)
    ))
    fig_trend.update_layout(
        title='Sentiment Trend Over Last 24 Hours',
        xaxis_title='Time',
        yaxis_title='Tweet Count',
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric('Peak Positive Hour', f'{pos_trend.argmax()}:00', '75%')
    col2.metric('Peak Negative Hour', f'{neg_trend.argmax()}:00', '58%')
    col3.metric('Average Positive', f'{pos_trend.mean():.0f}%', 'Stable')

    # Hourly breakdown
    fig_bar = px.bar(
        df_trend,
        x='Time',
        y=['Positive', 'Negative'],
        color_discrete_map={
            'Positive': '#4ECDC4',
            'Negative': '#FF6B6B'
        },
        title='Hourly Sentiment Breakdown',
        barmode='stack'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader('⚖️ Classical NLP vs Deep Learning')

    col1, col2, col3 = st.columns(3)
    col1.metric('Logistic Regression', '~80%', 'F1: 0.80')
    col2.metric('Naive Bayes', '~78%', 'F1: 0.78')
    col3.metric('LSTM (BiLSTM)', '81.04%', 'F1: 0.81 ⭐')

    st.divider()

    comparison_data = {
        'Model': ['Logistic Regression', 'Naive Bayes', 'LSTM (BiLSTM)'],
        'Accuracy': [0.800, 0.780, 0.8104],
        'Precision': [0.800, 0.779, 0.810],
        'Recall': [0.800, 0.781, 0.810],
        'F1-Score': [0.800, 0.780, 0.810],
    }
    df_comp = pd.DataFrame(comparison_data)

    col1, col2 = st.columns(2)

    with col1:
        fig_radar = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors  = ['#4ECDC4', '#FF6B6B', '#9B59B6']

        for i, row in df_comp.iterrows():
            values = [row[m] for m in metrics]
            values.append(values[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=row['Model'],
                line_color=colors[i]
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.7, 0.85])),
            title='Model Performance Radar Chart'
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        fig_comp = px.bar(
            df_comp,
            x='Model',
            y=['Accuracy', 'F1-Score'],
            barmode='group',
            color_discrete_map={
                'Accuracy': '#4ECDC4',
                'F1-Score': '#9B59B6'
            },
            title='Accuracy vs F1-Score'
        )
        fig_comp.update_layout(yaxis_range=[0.7, 0.85])
        st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(df_comp, use_container_width=True)

    st.markdown("""
    ### 🔑 Key Takeaways
    | | Logistic Regression | Naive Bayes | LSTM |
    |---|---|---|---|
    | **Speed** | ⚡ Fast | ⚡⚡ Very Fast | 🐢 Slow |
    | **Accuracy** | ✅ Good | ⚠️ OK | ✅✅ Best |
    | **Memory** | Low | Very Low | High |
    | **Interpretable** | ✅ Yes | ✅ Yes | ❌ No |
    | **Best for** | Production | Quick deploy | Max accuracy |
    """)