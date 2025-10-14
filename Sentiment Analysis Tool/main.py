import os
import io
import re
import json
import base64
import logging
import threading
import sqlite3
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Data / ML
import numpy as np
import pandas as pd

# Visualization
import matplotlib
matplotlib.use('TkAgg')  # Ensure Tk backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages

# Sentiment
from textblob import TextBlob

# ML pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Clustering
from sklearn.cluster import KMeans

# Topic modeling
from sklearn.decomposition import LatentDirichletAllocation

# UI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Joblib for model persistence
try:
    import joblib
except Exception:
    joblib = None

# Optional VADER sentiment
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except Exception:
    SentimentIntensityAnalyzer = None
    VADER_AVAILABLE = False

# Optional language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except Exception:
    detect = None
    LANGDETECT_AVAILABLE = False

# Optional date picker
try:
    from tkcalendar import DateEntry
    TKCALENDAR_AVAILABLE = True
except Exception:
    DateEntry = None
    TKCALENDAR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    filename='sentiment_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)


class ThemeManager:
    def __init__(self):
        self.current_theme = "light"
        self.themes = {
            "light": {
                "bg": "#ffffff",
                "fg": "#2c3e50",
                "secondary_bg": "#f8f9fa",
                "accent": "#3498db",
                "success": "#27ae60",
                "danger": "#e74c3c",
                "warning": "#f39c12",
                "info": "#17a2b8",
                "card_bg": "#ffffff",
                "border": "#dee2e6",
                "text_muted": "#6c757d"
            },
            "dark": {
                "bg": "#2c3e50",
                "fg": "#ecf0f1",
                "secondary_bg": "#34495e",
                "accent": "#3498db",
                "success": "#27ae60",
                "danger": "#e74c3c",
                "warning": "#f39c12",
                "info": "#17a2b8",
                "card_bg": "#34495e",
                "border": "#5d6d7e",
                "text_muted": "#bdc3c7"
            }
        }

    def get_color(self, key: str) -> str:
        return self.themes[self.current_theme][key]

    def toggle_theme(self):
        self.current_theme = "dark" if self.current_theme == "light" else "light"

    def get_all_colors(self) -> Dict[str, str]:
        return self.themes[self.current_theme]


class DataProcessor:
    """Handles all data processing, analysis, and ML operations"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None

        # Base NLP settings
        self.stop_words = set(ENGLISH_STOP_WORDS).union({
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'use',
            'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'has', 'let', 'put', 'say', 'she', 'too', 'use', 'amp'
        })

        # Sentiment engine
        self.sentiment_engine = 'vader' if VADER_AVAILABLE else 'textblob'
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

        # Vectorizer / ML
        self.max_features = 2000
        self.ngram_range = (1, 2)
        self.model_pipeline: Optional[Pipeline] = None
        self.is_model_trained = False
        self.last_model_metrics: Dict = {}

        # Clustering vectorizer
        self.cluster_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # LDA
        self.last_lda_result: Dict = {}

    def get_active_df(self) -> Optional[pd.DataFrame]:
        return self.filtered_df if self.filtered_df is not None else self.df

    def set_sentiment_engine(self, engine: str):
        engine = engine.lower().strip()
        if engine not in {'vader', 'textblob'}:
            raise ValueError("Unsupported engine. Choose 'vader' or 'textblob'.")
        if engine == 'vader' and not VADER_AVAILABLE:
            raise RuntimeError("VADER not available. Install NLTK and download vader_lexicon.")
        self.sentiment_engine = engine

    def set_vectorizer_params(self, max_features: int, ngram_range: Tuple[int, int]):
        self.max_features = int(max_features)
        self.ngram_range = tuple(ngram_range)

    def load_data(self, file_path: str) -> bool:
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith('.xlsx'):
                try:
                    self.df = pd.read_excel(file_path, engine='openpyxl')
                except Exception:
                    self.df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.df = pd.read_json(file_path, lines=False)
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                self.df = pd.DataFrame({'review': [line.strip() for line in lines if line.strip()]})
            else:
                raise ValueError("Unsupported file type. Please load CSV, XLSX, JSON, or TXT.")

            self._preprocess_data()
            self.filtered_df = None  # reset filters
            logging.info(f"Data loaded successfully: {len(self.df)} records")
            return True

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False

    @staticmethod
    def _clean_text(text: str) -> str:
        text = str(text)
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'[^a-z0-9\s\']', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _preprocess_data(self):
        if self.df is None or self.df.empty:
            raise ValueError("No data to preprocess")

        # Find text column
        text_columns = [col for col in self.df.columns if any(keyword in col.lower()
                                                              for keyword in
                                                              ['review', 'text', 'comment', 'feedback', 'message'])]
        if not text_columns:
            text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']

        if text_columns:
            self.df['review'] = self.df[text_columns[0]].astype(str)
        else:
            raise ValueError("No text column found")

        self.df = self.df.dropna(subset=['review'])
        self.df['review'] = self.df['review'].astype(str).str.strip()
        self.df = self.df[self.df['review'] != '']
        self.df['clean_review'] = self.df['review'].apply(self._clean_text)
        self.df['word_count'] = self.df['clean_review'].apply(lambda x: len(x.split()))
        self.df['char_count'] = self.df['clean_review'].apply(len)

        # Dates
        date_columns = [col for col in self.df.columns if any(keyword in col.lower()
                                                              for keyword in ['date', 'time', 'timestamp', 'created'])]
        if date_columns:
            try:
                self.df['date'] = pd.to_datetime(self.df[date_columns[0]], errors='coerce')
                self.df = self.df.dropna(subset=['date'])
                self.df['day'] = self.df['date'].dt.date
                self.df['month'] = self.df['date'].dt.to_period('M')
                self.df['hour'] = self.df['date'].dt.hour
                self.df['weekday'] = self.df['date'].dt.day_name()
            except Exception as e:
                logging.warning(f"Failed to parse date column: {e}")

        self._apply_sentiment()

    def _apply_sentiment(self):
        if self.sentiment_engine == 'vader' and self.vader is not None:
            scores = self.df['clean_review'].apply(self.vader.polarity_scores)
            self.df['polarity'] = scores.apply(lambda s: float(s['compound']))
            self.df['subjectivity'] = self.df['review'].apply(lambda x: float(TextBlob(x).sentiment.subjectivity))
            def to_label(comp):
                if comp > 0.1:
                    return 'positive'
                elif comp < -0.1:
                    return 'negative'
                else:
                    return 'neutral'
            self.df['sentiment'] = self.df['polarity'].apply(to_label)
        else:
            self.df['polarity'] = self.df['review'].apply(lambda x: float(TextBlob(x).sentiment.polarity))
            self.df['subjectivity'] = self.df['review'].apply(lambda x: float(TextBlob(x).sentiment.subjectivity))
            def to_label(p):
                if p > 0.1:
                    return 'positive'
                elif p < -0.1:
                    return 'negative'
                else:
                    return 'neutral'
            self.df['sentiment'] = self.df['polarity'].apply(to_label)

    def detect_languages(self):
        """Add 'lang' column with ISO code using langdetect (if available)."""
        if not LANGDETECT_AVAILABLE:
            return False
        if self.df is None or 'lang' in self.df.columns:
            return True
        def detect_safe(text):
            try:
                return detect(text)
            except Exception:
                return 'unk'
        self.df['lang'] = self.df['review'].apply(detect_safe)
        return True

    def apply_filters(self, sentiments: List[str], min_words: int = 0,
                      english_only: bool = False,
                      date_from: Optional[pd.Timestamp] = None,
                      date_to: Optional[pd.Timestamp] = None):
        if self.df is None:
            return
        df = self.df.copy()
        if sentiments:
            df = df[df['sentiment'].isin(sentiments)]
        if min_words and min_words > 0:
            df = df[df['word_count'] >= int(min_words)]
        if english_only:
            if 'lang' not in df.columns:
                if self.detect_languages():
                    df = df[df['lang'] == 'en']
            else:
                df = df[df['lang'] == 'en']
        if date_from is not None and 'date' in df.columns:
            df = df[df['date'] >= date_from]
        if date_to is not None and 'date' in df.columns:
            df = df[df['date'] <= date_to]
        self.filtered_df = df

    def reset_filters(self):
        self.filtered_df = None

    def remove_duplicates(self):
        if self.df is None:
            return 0
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['clean_review']).reset_index(drop=True)
        after = len(self.df)
        self.filtered_df = None
        return before - after

    def get_advanced_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        if df is None:
            df = self.get_active_df()
        if df is None or df.empty:
            return {}
        stats = {
            'total_records': int(len(df)),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_polarity': float(df['polarity'].mean()),
            'avg_subjectivity': float(df['subjectivity'].mean()) if 'subjectivity' in df else None,
            'avg_word_count': float(df['word_count'].mean()),
            'polarity_std': float(df['polarity'].std()),
        }
        try:
            stats['most_positive_review'] = str(df.loc[df['polarity'].idxmax(), 'review'])[:200] + "..."
        except Exception:
            stats['most_positive_review'] = "N/A"
        try:
            stats['most_negative_review'] = str(df.loc[df['polarity'].idxmin(), 'review'])[:200] + "..."
        except Exception:
            stats['most_negative_review'] = "N/A"

        if 'date' in df.columns and not df['date'].isna().all():
            try:
                stats.update({
                    'date_range': f"{df['date'].min()} to {df['date'].max()}",
                    'reviews_per_day': float(df.groupby('day').size().mean()) if 'day' in df.columns else None,
                    'peak_hour': int(df['hour'].mode().iloc[0]) if 'hour' in df.columns and not df['hour'].isna().all() else 'N/A'
                })
            except Exception:
                pass
        return stats

    def perform_clustering(self, n_clusters: int = 3) -> Dict:
        try:
            df = self.get_active_df()
            if df is None or df.empty:
                return {'success': False, 'error': 'No data available for clustering.'}

            X = self.cluster_vectorizer.fit_transform(df['clean_review'])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)

            df = df.copy()
            df['cluster'] = clusters

            feature_names = self.cluster_vectorizer.get_feature_names_out()
            cluster_terms = {}
            centers = kmeans.cluster_centers_
            for i in range(n_clusters):
                top_indices = centers[i].argsort()[-10:][::-1]
                cluster_terms[f'Cluster {i}'] = [feature_names[idx] for idx in top_indices]

            active_index = self.get_active_df().index
            self.df.loc[active_index, 'cluster'] = clusters

            return {
                'cluster_distribution': pd.Series(clusters).value_counts().to_dict(),
                'cluster_terms': cluster_terms,
                'success': True
            }

        except Exception as e:
            logging.error(f"Clustering error: {e}")
            return {'success': False, 'error': str(e)}

    def train_sentiment_model(self) -> Dict:
        try:
            df = self.get_active_df()
            if df is None or len(df) < 50:
                return {'success': False, 'error': 'Insufficient data for model training (need >= 50).'}

            X = df['clean_review'].values
            y = df['sentiment'].values

            self.model_pipeline = Pipeline(steps=[
                ('tfidf', TfidfVectorizer(max_features=self.max_features,
                                          stop_words='english',
                                          ngram_range=self.ngram_range)),
                ('clf', LogisticRegression(max_iter=1000))
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
            )

            self.model_pipeline.fit(X_train, y_train)
            predictions = self.model_pipeline.predict(X_test)

            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, predictions, labels=['negative', 'neutral', 'positive']).tolist()

            self.is_model_trained = True
            self.last_model_metrics = {
                'report': report,
                'confusion_matrix': cm,
                'labels': ['negative', 'neutral', 'positive']
            }

            return {
                'success': True,
                'accuracy': float(report.get('accuracy', 0.0)),
                'detailed_report': report
            }

        except Exception as e:
            logging.error(f"Model training error: {e}")
            return {'success': False, 'error': str(e)}

    def save_model(self, file_path: str) -> bool:
        if not self.is_model_trained or self.model_pipeline is None:
            raise RuntimeError("No trained model to save.")
        if joblib is None:
            raise RuntimeError("joblib is not installed.")
        try:
            payload = {
                'pipeline': self.model_pipeline,
                'vectorizer_params': {'max_features': self.max_features, 'ngram_range': self.ngram_range},
                'engine': self.sentiment_engine
            }
            joblib.dump(payload, file_path)
            return True
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return False

    def load_model(self, file_path: str) -> bool:
        if joblib is None:
            raise RuntimeError("joblib is not installed.")
        try:
            payload = joblib.load(file_path)
            self.model_pipeline = payload.get('pipeline')
            vp = payload.get('vectorizer_params', {})
            self.max_features = vp.get('max_features', self.max_features)
            self.ngram_range = tuple(vp.get('ngram_range', self.ngram_range))
            self.sentiment_engine = payload.get('engine', self.sentiment_engine)
            self.is_model_trained = self.model_pipeline is not None
            return self.is_model_trained
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

    def predict_text(self, text: str) -> Dict:
        if not text or not text.strip():
            return {'success': False, 'error': 'Empty input text'}

        cleaned = self._clean_text(text)

        if self.is_model_trained and self.model_pipeline is not None:
            try:
                pred = self.model_pipeline.predict([cleaned])[0]
                try:
                    proba = getattr(self.model_pipeline[-1], "predict_proba", None)
                    conf = float(np.max(proba(self.model_pipeline[0].transform([cleaned])))) if proba else None
                except Exception:
                    conf = None
                return {'success': True, 'engine': 'model', 'sentiment': pred, 'polarity': None, 'confidence': conf}
            except Exception as e:
                logging.warning(f"Model prediction failed, falling back: {e}")

        if self.sentiment_engine == 'vader' and self.vader is not None:
            s = self.vader.polarity_scores(cleaned)
            comp = float(s['compound'])
            if comp > 0.1:
                label = 'positive'
            elif comp < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            conf = float(max(s['pos'], s['neg'], s['neu']))
            return {'success': True, 'engine': 'vader', 'sentiment': label, 'polarity': comp, 'confidence': conf}
        else:
            pol = float(TextBlob(text).sentiment.polarity)
            if pol > 0.1:
                label = 'positive'
            elif pol < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            return {'success': True, 'engine': 'textblob', 'sentiment': label, 'polarity': pol, 'confidence': None}

    def get_keyword_analysis(self, df: Optional[pd.DataFrame] = None, top_n: int = 10) -> Dict:
        try:
            if df is None:
                df = self.get_active_df()
            if df is None or df.empty:
                return {}

            keyword_analysis = {}
            for sentiment in ['positive', 'negative', 'neutral']:
                text = ' '.join(df[df['sentiment'] == sentiment]['clean_review'])
                words = re.findall(r'\b[a-z]{3,}\b', text.lower())
                filtered = [w for w in words if w not in self.stop_words]
                keyword_analysis[sentiment] = Counter(filtered).most_common(top_n)

            return keyword_analysis

        except Exception as e:
            logging.error(f"Keyword analysis error: {e}")
            return {}

    def get_ngram_analysis(self, ngram_range=(2, 2), top_n=10) -> Dict:
        df = self.get_active_df()
        if df is None or df.empty:
            return {}
        results = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            docs = df[df['sentiment'] == sentiment]['clean_review']
            if docs.empty:
                results[sentiment] = []
                continue
            vect = CountVectorizer(stop_words='english', ngram_range=ngram_range, min_df=1)
            X = vect.fit_transform(docs)
            freqs = np.asarray(X.sum(axis=0)).ravel()
            terms = np.array(vect.get_feature_names_out())
            idxs = freqs.argsort()[::-1][:top_n]
            results[sentiment] = list(zip(terms[idxs], freqs[idxs].astype(int)))
        return results

    def run_lda(self, n_topics=8, max_features=5000, max_df=0.95, min_df=2, random_state=42) -> Dict:
        df = self.get_active_df()
        if df is None or df.empty:
            return {'success': False, 'error': 'No data available for LDA.'}

        try:
            vect = CountVectorizer(stop_words='english',
                                   max_features=max_features,
                                   max_df=max_df,
                                   min_df=min_df)
            X = vect.fit_transform(df['clean_review'])
            if X.shape[0] < 10 or X.shape[1] < n_topics:
                return {'success': False, 'error': 'Not enough data/features for the requested number of topics.'}

            lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method='batch')
            lda.fit(X)

            feature_names = np.array(vect.get_feature_names_out())
            top_words = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                words = feature_names[top_indices]
                weights = topic[top_indices]
                top_words.append({'topic': topic_idx, 'words': words.tolist(), 'weights': weights.tolist()})

            doc_topics = lda.transform(X)  # shape: (n_docs, n_topics)
            dominant_topic = doc_topics.argmax(axis=1).tolist()

            result = {
                'success': True,
                'n_topics': n_topics,
                'top_words': top_words,
                'dominant_topic': dominant_topic
            }
            self.last_lda_result = result
            return result
        except Exception as e:
            logging.error(f"LDA error: {e}")
            return {'success': False, 'error': str(e)}


class ChartGenerator:
    def __init__(self, theme_manager: ThemeManager):
        self.theme = theme_manager
        self.setup_matplotlib_theme()

    def setup_matplotlib_theme(self):
        colors = self.theme.get_all_colors()
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': colors['bg'],
            'axes.facecolor': colors['card_bg'],
            'axes.edgecolor': colors['border'],
            'axes.labelcolor': colors['fg'],
            'text.color': colors['fg'],
            'xtick.color': colors['fg'],
            'ytick.color': colors['fg'],
            'grid.color': colors['border'],
            'figure.edgecolor': colors['border']
        })

    def create_sentiment_distribution(self, df: pd.DataFrame, chart_type: str = 'bar') -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_order = ['positive', 'negative', 'neutral']
        sentiment_counts = df['sentiment'].value_counts().reindex(sentiment_order).fillna(0).astype(int)

        colors = {
            'positive': self.theme.get_color('success'),
            'negative': self.theme.get_color('danger'),
            'neutral': self.theme.get_color('info')
        }
        color_list = [colors[s] for s in sentiment_order]

        if chart_type == 'bar':
            bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=color_list)
            ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Sentiment', fontsize=12)
            ax.set_ylabel('Number of Reviews', fontsize=12)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.pie(sentiment_counts.values, labels=sentiment_counts.index,
                   autopct='%1.1f%%', colors=color_list, startangle=90)
            ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    def create_polarity_analysis(self, df: pd.DataFrame) -> plt.Figure:
        fig = plt.figure(figsize=(15, 10))

        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(df['polarity'], bins=30, color=self.theme.get_color('accent'), alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color=self.theme.get_color('danger'), linestyle='--', linewidth=2)
        ax1.set_title('Polarity Distribution', fontweight='bold')
        ax1.set_xlabel('Polarity Score')
        ax1.set_ylabel('Frequency')

        ax2 = plt.subplot(2, 3, 2)
        sns.boxplot(data=df, x='sentiment', y='polarity', ax=ax2,
                    order=['negative', 'neutral', 'positive'],
                    palette=[self.theme.get_color('danger'), self.theme.get_color('info'), self.theme.get_color('success')])
        ax2.set_title('Polarity by Sentiment Category', fontweight='bold')
        ax2.set_xlabel('Sentiment')

        ax3 = plt.subplot(2, 3, 3)
        colors_map = {'positive': self.theme.get_color('success'),
                      'negative': self.theme.get_color('danger'),
                      'neutral': self.theme.get_color('info')}
        for sentiment in df['sentiment'].unique():
            data = df[df['sentiment'] == sentiment]
            ax3.scatter(data['word_count'], data['polarity'],
                        label=sentiment, color=colors_map.get(sentiment, '#888'), alpha=0.6)
        ax3.set_title('Word Count vs Polarity', fontweight='bold')
        ax3.set_xlabel('Word Count')
        ax3.set_ylabel('Polarity Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        if 'day' in df.columns:
            ax4 = plt.subplot(2, 3, 4)
            daily_polarity = df.groupby('day')['polarity'].mean()
            ax4.plot(daily_polarity.index, daily_polarity.values,
                     color=self.theme.get_color('accent'), linewidth=2, marker='o')
            ax4.set_title('Average Polarity Over Time', fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Average Polarity')
            plt.xticks(rotation=45)

        ax5 = plt.subplot(2, 3, 5)
        subj = df.get('subjectivity', pd.Series([0]*len(df)))
        scatter = ax5.scatter(subj, df['polarity'], c=df['word_count'], cmap='viridis', alpha=0.6)
        ax5.set_title('Subjectivity vs Polarity', fontweight='bold')
        ax5.set_xlabel('Subjectivity')
        ax5.set_ylabel('Polarity')
        plt.colorbar(scatter, ax=ax5, label='Word Count')

        ax6 = plt.subplot(2, 3, 6)
        word_count_range = pd.cut(df['word_count'], bins=5,
                                  labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        sentiment_by_length = df.groupby([word_count_range, 'sentiment']).size().unstack(fill_value=0)
        sentiment_by_length.plot(kind='bar', ax=ax6, color=[colors_map.get(col, '#888') for col in sentiment_by_length.columns])
        ax6.set_title('Sentiment by Review Length', fontweight='bold')
        ax6.set_xlabel('Word Count Range')
        ax6.set_ylabel('Number of Reviews')
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def create_temporal_analysis(self, df: pd.DataFrame) -> plt.Figure:
        if 'date' not in df.columns or df['date'].isna().all():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No date information available',
                    ha='center', va='center', fontsize=16, transform=ax.transAxes)
            return fig

        fig = plt.figure(figsize=(15, 12))

        ax1 = plt.subplot(3, 2, 1)
        if 'day' in df.columns:
            daily_sentiment = df.groupby(['day', 'sentiment']).size().unstack(fill_value=0)
            daily_sentiment.plot(kind='area', ax=ax1, alpha=0.7,
                                 color=[self.theme.get_color('success'),
                                        self.theme.get_color('danger'),
                                        self.theme.get_color('info')])
            ax1.set_title('Daily Sentiment Trends', fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Reviews')

        if 'hour' in df.columns:
            ax2 = plt.subplot(3, 2, 2)
            hourly = df.groupby('hour')['sentiment'].value_counts().unstack(fill_value=0)
            hourly.plot(kind='bar', ax=ax2,
                        color=[self.theme.get_color('success'),
                               self.theme.get_color('danger'),
                               self.theme.get_color('info')])
            ax2.set_title('Hourly Sentiment Patterns', fontweight='bold')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Number of Reviews')
            plt.xticks(rotation=0)

        if 'weekday' in df.columns:
            ax3 = plt.subplot(3, 2, 3)
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly = df.groupby('weekday')['sentiment'].value_counts().unstack(fill_value=0)
            weekly = weekly.reindex(weekday_order)
            weekly.plot(kind='bar', ax=ax3,
                        color=[self.theme.get_color('success'),
                               self.theme.get_color('danger'),
                               self.theme.get_color('info')])
            ax3.set_title('Weekly Sentiment Patterns', fontweight='bold')
            ax3.set_xlabel('Day of Week')
            ax3.set_ylabel('Number of Reviews')
            plt.xticks(rotation=45)

        if 'month' in df.columns:
            ax4 = plt.subplot(3, 2, 4)
            monthly = df.groupby('month')['polarity'].mean()
            ax4.plot(monthly.index.astype(str), monthly.values,
                     color=self.theme.get_color('accent'), linewidth=3, marker='o', markersize=8)
            ax4.set_title('Monthly Polarity Trends', fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Average Polarity')
            plt.xticks(rotation=45)

        if 'day' in df.columns:
            ax5 = plt.subplot(3, 2, 5)
            daily_volume = df.groupby('day').size()
            ax5.fill_between(daily_volume.index, daily_volume.values,
                             color=self.theme.get_color('accent'), alpha=0.7)
            ax5.set_title('Review Volume Over Time', fontweight='bold')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Number of Reviews')

        ax6 = plt.subplot(3, 2, 6)
        df_sorted = df.sort_values('date')
        cumavg = df_sorted['polarity'].expanding().mean()
        ax6.plot(df_sorted['date'], cumavg, color=self.theme.get_color('warning'), linewidth=2)
        ax6.axhline(y=0, color=self.theme.get_color('danger'), linestyle='--', alpha=0.7)
        ax6.set_title('Cumulative Average Polarity', fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Cumulative Average Polarity')

        plt.tight_layout()
        return fig

    def create_wordcloud(self, df: pd.DataFrame, sentiment: str = 'all') -> plt.Figure:
        sentiments = ['positive', 'negative', 'neutral'] if sentiment == 'all' else [sentiment]
        fig, axes = plt.subplots(1, len(sentiments), figsize=(6 * len(sentiments), 6))
        if len(sentiments) == 1:
            axes = [axes]

        for i, sent in enumerate(sentiments):
            text = ' '.join(df[df['sentiment'] == sent]['clean_review'])
            if text.strip():
                wc = WordCloud(width=800, height=600, background_color='white',
                               colormap='viridis', max_words=200, collocations=False).generate(text)
                axes[i].imshow(wc, interpolation='bilinear')
                axes[i].set_title(f'{sent.title()} Sentiment Words', fontweight='bold', fontsize=14)
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No {sent} reviews found',
                             ha='center', va='center', fontsize=12, transform=axes[i].transAxes)
                axes[i].axis('off')

        plt.tight_layout()
        return fig

    def create_confusion_matrix(self, cm: np.ndarray, labels: List[str]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix', fontweight='bold')
        plt.tight_layout()
        return fig

    def create_top_ngrams_bar(self, ngram_results: Dict, title_prefix="Top n-grams") -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sentiments = ['positive', 'negative', 'neutral']
        colors = [self.theme.get_color('success'), self.theme.get_color('danger'), self.theme.get_color('info')]
        for i, sent in enumerate(sentiments):
            pairs = ngram_results.get(sent, [])
            terms = [t for t, _ in pairs]
            counts = [c for _, c in pairs]
            axes[i].barh(terms[::-1], counts[::-1], color=colors[i])
            axes[i].set_title(f"{title_prefix} - {sent.title()}")
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel('n-gram')
        plt.tight_layout()
        return fig

    def create_lda_topics_figure(self, top_words: List[Dict]) -> plt.Figure:
        cols = 2
        rows = int(np.ceil(len(top_words) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, max(6, 3 * rows)))
        axes = np.array(axes).reshape(-1)
        for i, topic in enumerate(top_words):
            ax = axes[i]
            words = topic['words']
            weights = topic['weights']
            ax.barh(words[::-1], weights[::-1], color=self.theme.get_color('accent'))
            ax.set_title(f"Topic {topic['topic']}")
            ax.set_xlabel("Weight")
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        return fig


class DatabaseManager:
    def __init__(self):
        self.db_path = 'sentiment_analysis.db'
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                upload_date TEXT,
                total_records INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                avg_polarity REAL,
                analysis_summary TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_analysis(self, filename: str, stats: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO analyses 
            (filename, upload_date, total_records, positive_count, negative_count, 
             neutral_count, avg_polarity, analysis_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            datetime.now().isoformat(),
            stats.get('total_records', 0),
            stats.get('sentiment_distribution', {}).get('positive', 0),
            stats.get('sentiment_distribution', {}).get('negative', 0),
            stats.get('sentiment_distribution', {}).get('neutral', 0),
            stats.get('avg_polarity', 0),
            json.dumps(stats)
        ))
        conn.commit()
        conn.close()

    def get_analysis_history(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM analyses ORDER BY upload_date DESC')
        results = cursor.fetchall()
        conn.close()

        history = []
        for row in results:
            history.append({
                'id': row[0],
                'filename': row[1],
                'upload_date': row[2],
                'total_records': row[3],
                'positive_count': row[4],
                'negative_count': row[5],
                'neutral_count': row[6],
                'avg_polarity': row[7],
                'summary': json.loads(row[8]) if row[8] else {}
            })
        return history


class SentimentAnalysisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.theme_manager = ThemeManager()
        self.data_processor = DataProcessor()
        self.chart_generator = ChartGenerator(self.theme_manager)
        self.db_manager = DatabaseManager()
        self.current_figure = None

        # Filter state
        self.filter_vars = {
            'positive': tk.BooleanVar(value=True),
            'negative': tk.BooleanVar(value=True),
            'neutral': tk.BooleanVar(value=True),
        }
        self.min_words_var = tk.IntVar(value=0)
        self.english_only_var = tk.BooleanVar(value=False)
        self.start_date_var = tk.StringVar(value="")
        self.end_date_var = tk.StringVar(value="")

        # LDA controls
        self.lda_topics_var = tk.IntVar(value=8)
        self.lda_max_features_var = tk.IntVar(value=5000)
        self.lda_min_df_var = tk.IntVar(value=2)
        self.lda_max_df_var = tk.DoubleVar(value=0.95)

        # Sentiment engine UI var
        self.engine_var = tk.StringVar(value='VADER' if VADER_AVAILABLE else 'TextBlob')

        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        self.root.title("Advanced Sentiment Analysis Tool - Professional Edition")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 800)

        # Main container: removed 'minsize' to avoid Tk option error
        self.main_container = ttk.PanedWindow(self.root, orient='horizontal')
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Sidebar
        self.create_sidebar()

        # Main content
        self.create_main_content()

        # Initialize sash position instead of using minsize option
        self.root.after(50, lambda: self.main_container.sashpos(0, 360))

        # Status bar
        self.create_status_bar()

        # Menu bar
        self.create_menu()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Report", command=self.export_report)
        file_menu.add_command(label="Export Charts", command=self.export_charts)
        file_menu.add_command(label="Export PDF Bundle", command=self.export_plots_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Basic Analysis", command=self.show_basic_analysis)
        analysis_menu.add_command(label="Advanced Analysis", command=self.show_advanced_analysis)
        analysis_menu.add_command(label="Temporal Analysis", command=self.show_temporal_analysis)
        analysis_menu.add_command(label="Text Clustering", command=self.perform_clustering)
        analysis_menu.add_command(label="Topic Modeling (LDA)", command=self.show_topic_modeling)
        analysis_menu.add_command(label="Train Model", command=self.train_model)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Theme", command=self.toggle_theme)
        view_menu.add_command(label="Analysis History", command=self.show_history)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Vectorizer Settings", command=self.open_vectorizer_settings_dialog)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        # help_menu.add_command(label="About", command=self.show_about)

    def create_sidebar(self):
        sidebar_frame = ttk.Frame(self.main_container, width=340)
        self.main_container.add(sidebar_frame, weight=1)

        ttk.Label(sidebar_frame, text="Sentiment Analysis", font=('Arial', 16, 'bold')).pack(pady=(10, 10))

        # Engine selector
        engine_frame = ttk.LabelFrame(sidebar_frame, text="Sentiment Engine", padding=10)
        engine_frame.pack(fill='x', padx=10, pady=5)
        engine_combo = ttk.Combobox(engine_frame, textvariable=self.engine_var,
                                    values=["VADER", "TextBlob"], state="readonly")
        engine_combo.pack(fill='x')
        ttk.Button(engine_frame, text="Apply Engine", command=self.apply_engine).pack(pady=5, fill='x')

        # Data management
        data_frame = ttk.LabelFrame(sidebar_frame, text="Data Management", padding=10)
        data_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(data_frame, text="Load Data File", command=self.load_data).pack(fill='x', pady=2)
        ttk.Button(data_frame, text="Data Preview", command=self.show_data_preview).pack(fill='x', pady=2)
        ttk.Button(data_frame, text="Remove Duplicates", command=self.remove_duplicates).pack(fill='x', pady=2)

        # Filters
        filter_frame = ttk.LabelFrame(sidebar_frame, text="Filters", padding=10)
        filter_frame.pack(fill='x', padx=10, pady=5)
        ttk.Checkbutton(filter_frame, text="Positive", variable=self.filter_vars['positive']).pack(anchor='w')
        ttk.Checkbutton(filter_frame, text="Negative", variable=self.filter_vars['negative']).pack(anchor='w')
        ttk.Checkbutton(filter_frame, text="Neutral", variable=self.filter_vars['neutral']).pack(anchor='w')

        min_frame = ttk.Frame(filter_frame)
        min_frame.pack(fill='x', pady=3)
        ttk.Label(min_frame, text="Min word count:").pack(side='left')
        ttk.Spinbox(min_frame, from_=0, to=1000, textvariable=self.min_words_var, width=7).pack(side='left', padx=5)

        ttk.Checkbutton(filter_frame, text="English only (requires langdetect)", variable=self.english_only_var).pack(anchor='w', pady=(3, 6))

        # Date range picker (optional tkcalendar)
        date_frame = ttk.LabelFrame(filter_frame, text="Date Range", padding=6)
        date_frame.pack(fill='x')
        row = ttk.Frame(date_frame)
        row.pack(fill='x', pady=2)
        ttk.Label(row, text="From:").pack(side='left')
        if TKCALENDAR_AVAILABLE:
            self.from_date_widget = DateEntry(row, textvariable=self.start_date_var, width=12, date_pattern='y-mm-dd')
        else:
            self.from_date_widget = ttk.Entry(row, textvariable=self.start_date_var, width=14)
            self.from_date_widget.insert(0, "")
        self.from_date_widget.pack(side='left', padx=5)

        row2 = ttk.Frame(date_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="To:").pack(side='left')
        if TKCALENDAR_AVAILABLE:
            self.to_date_widget = DateEntry(row2, textvariable=self.end_date_var, width=12, date_pattern='y-mm-dd')
        else:
            self.to_date_widget = ttk.Entry(row2, textvariable=self.end_date_var, width=14)
            self.to_date_widget.insert(0, "")
        self.to_date_widget.pack(side='left', padx=22)

        btns = ttk.Frame(filter_frame)
        btns.pack(fill='x', pady=(8, 0))
        ttk.Button(btns, text="Apply Filters", command=self.apply_filters).pack(side='left', expand=True, fill='x', padx=(0, 5))
        ttk.Button(btns, text="Reset", command=self.reset_filters).pack(side='left', expand=True, fill='x', padx=(5, 0))

        # Analysis options
        analysis_frame = ttk.LabelFrame(sidebar_frame, text="Analysis Options", padding=10)
        analysis_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(analysis_frame, text="Basic Analysis", command=self.show_basic_analysis).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Advanced Analysis", command=self.show_advanced_analysis).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Temporal Analysis", command=self.show_temporal_analysis).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Text Clustering", command=self.perform_clustering).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Topic Modeling (LDA)", command=self.show_topic_modeling).pack(fill='x', pady=2)
        ttk.Button(analysis_frame, text="Keyword Analysis", command=self.show_keyword_analysis).pack(fill='x', pady=2)

        # Visualizations
        viz_frame = ttk.LabelFrame(sidebar_frame, text="Visualizations", padding=10)
        viz_frame.pack(fill='x', padx=10, pady=5)
        self.chart_var = tk.StringVar(value="Sentiment Distribution")
        chart_options = ["Sentiment Distribution", "Polarity Analysis", "Word Cloud", "Temporal Trends"]
        ttk.Label(viz_frame, text="Chart Type:").pack(anchor='w')
        self.chart_combo = ttk.Combobox(viz_frame, textvariable=self.chart_var, values=chart_options, state="readonly")
        self.chart_combo.pack(fill='x', pady=2)
        self.chart_combo.bind('<<ComboboxSelected>>', self.on_chart_select)
        ttk.Button(viz_frame, text="Generate Chart", command=self.generate_chart).pack(fill='x', pady=2)
        ttk.Button(viz_frame, text="Save Chart", command=self.save_chart).pack(fill='x', pady=2)

        # ML
        model_frame = ttk.LabelFrame(sidebar_frame, text="Machine Learning", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(model_frame, text="Train Model", command=self.train_model).pack(fill='x', pady=2)
        ttk.Button(model_frame, text="Model Performance", command=self.show_model_performance).pack(fill='x', pady=2)
        ttk.Button(model_frame, text="Save Model", command=self.save_model).pack(fill='x', pady=2)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(fill='x', pady=2)

        # Quick Predict
        qp_frame = ttk.LabelFrame(sidebar_frame, text="Quick Predict", padding=10)
        qp_frame.pack(fill='x', padx=10, pady=5)
        self.qp_text = tk.Text(qp_frame, height=5, wrap='word')
        self.qp_text.pack(fill='x')
        ttk.Button(qp_frame, text="Predict", command=self.quick_predict).pack(fill='x', pady=5)
        self.qp_result = ttk.Label(qp_frame, text="Result: -")
        self.qp_result.pack(anchor='w')

        # Export
        export_frame = ttk.LabelFrame(sidebar_frame, text="Export Options", padding=10)
        export_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(export_frame, text="Export Report", command=self.export_report).pack(fill='x', pady=2)
        ttk.Button(export_frame, text="Export Data", command=self.export_data).pack(fill='x', pady=2)

        # Appearance
        theme_frame = ttk.LabelFrame(sidebar_frame, text="Appearance", padding=10)
        theme_frame.pack(fill='x', padx=10, pady=5)
        ttk.Button(theme_frame, text="Toggle Dark/Light Mode", command=self.toggle_theme).pack(fill='x', pady=2)

        # Data info
        self.info_frame = ttk.LabelFrame(sidebar_frame, text="Data Info", padding=10)
        self.info_frame.pack(fill='x', padx=10, pady=5)
        self.info_text = tk.Text(self.info_frame, height=6, wrap='word', state='disabled')
        scrollbar = ttk.Scrollbar(self.info_frame, orient='vertical', command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def create_main_content(self):
        content_frame = ttk.Frame(self.main_container)
        self.main_container.add(content_frame, weight=4)

        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tabs
        self.welcome_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.welcome_tab, text="Welcome")
        self.create_welcome_content()

        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Analysis Results")

        self.charts_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.charts_tab, text="Visualizations")

        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data View")

        self.topics_tab = ttk.Frame(self.notebook)  # LDA tab
        self.notebook.add(self.topics_tab, text="Topics (LDA)")

        self.reports_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reports_tab, text="Reports")

    def create_welcome_content(self):
        welcome_container = ttk.Frame(self.welcome_tab)
        welcome_container.pack(fill='both', expand=True, padx=20, pady=20)
        ttk.Label(welcome_container, text="Advanced Sentiment Analysis Tool", font=('Arial', 24, 'bold')).pack(pady=(50, 20))
        ttk.Label(welcome_container, text="Professional-grade sentiment analysis with machine learning capabilities",
                  font=('Arial', 12)).pack(pady=(0, 30))
        features_frame = ttk.Frame(welcome_container)
        features_frame.pack(fill='x', pady=20)
        features = [
            ("📊 Multi-format Data Support", "CSV, Excel, JSON, TXT files"),
            ("🤖 Machine Learning Models", "Train and save custom classifiers"),
            ("📈 Advanced Visualizations", "Charts and word clouds"),
            ("⏰ Temporal Analysis", "Time-based sentiment trends"),
            ("🎯 Text Clustering", "Hidden themes and topics"),
            ("🧩 Topic Modeling (LDA)", "Discover latent topics"),
            ("🔡 N-gram Analysis", "Unigrams, bigrams, trigrams"),
            ("🧰 Filters & Dedupe", "Filter by sentiment, word count, date; remove duplicates"),
            ("🌐 Language Filter", "Detect and keep English only (optional)"),
            ("🌓 Theme Support", "Dark and light mode interface")
        ]
        for i, (title, desc) in enumerate(features):
            f = ttk.Frame(features_frame)
            f.grid(row=i // 2, column=i % 2, padx=20, pady=6, sticky='w')
            ttk.Label(f, text=title, font=('Arial', 11, 'bold')).pack(anchor='w')
            ttk.Label(f, text=desc, font=('Arial', 9)).pack(anchor='w')
        ttk.Button(welcome_container, text="Load Data to Get Started", command=self.load_data, style='Accent.TButton').pack(pady=30)

    def create_status_bar(self):
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side='left', padx=5, pady=2)
        self.progress = ttk.Progressbar(self.status_bar, mode='indeterminate')
        self.progress.pack(side='right', padx=5, pady=2, fill='x', expand=True)

    def apply_theme(self):
        colors = self.theme_manager.get_all_colors()
        style = ttk.Style()
        if self.theme_manager.current_theme == "dark":
            style.theme_use('clam')
        else:
            style.theme_use('default')
        style.configure('TFrame', background=colors['bg'])
        style.configure('TLabel', background=colors['bg'], foreground=colors['fg'])
        style.configure('TButton', background=colors['secondary_bg'], foreground=colors['fg'])
        style.configure('Accent.TButton', background=colors['accent'], foreground='white')
        style.configure('TNotebook', background=colors['bg'])
        style.configure('TNotebook.Tab', background=colors['secondary_bg'], foreground=colors['fg'])
        self.root.configure(bg=colors['bg'])
        self.chart_generator.setup_matplotlib_theme()

    def update_status(self, message: str, show_progress: bool = False):
        self.status_label.config(text=message)
        if show_progress:
            self.progress.start()
        else:
            self.progress.stop()
        self.root.update_idletasks()

    def apply_engine(self):
        try:
            selected = self.engine_var.get().strip().lower()
            engine = 'vader' if selected == 'vader' else 'textblob'
            self.data_processor.set_sentiment_engine(engine)
            if self.data_processor.df is not None:
                self.data_processor._apply_sentiment()
                self.reset_filters()
                self.update_info_panel()
                messagebox.showinfo("Success", f"Sentiment engine set to {selected.title()} and re-applied.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("All Supported", "*.csv;*.xlsx;*.json;*.txt"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx"),
                ("JSON Files", "*.json"),
                ("Text Files", "*.txt")
            ]
        )
        if not file_path:
            return

        self.update_status("Loading data...", True)

        def load_thread():
            try:
                success = self.data_processor.load_data(file_path)
                if success:
                    self.root.after(0, self.on_data_loaded, file_path)
                else:
                    self.root.after(0, self.on_data_load_error, "Failed to load data")
            except Exception as e:
                self.root.after(0, self.on_data_load_error, str(e))

        threading.Thread(target=load_thread, daemon=True).start()

    def on_data_loaded(self, file_path: str):
        self.update_status("Data loaded successfully")
        self.update_info_panel()
        stats = self.data_processor.get_advanced_statistics()
        self.db_manager.save_analysis(os.path.basename(file_path), stats)
        self.notebook.select(self.analysis_tab)
        self.show_basic_analysis()
        messagebox.showinfo("Success", f"Data loaded successfully!\n{stats.get('total_records', 0)} records processed.")

    def on_data_load_error(self, error_msg: str):
        self.update_status("Ready")
        messagebox.showerror("Error", f"Failed to load data:\n{error_msg}")

    def update_info_panel(self):
        stats = self.data_processor.get_advanced_statistics()
        info_text = []
        info_text.append(f"Records: {stats.get('total_records', 0):,}")
        dist = stats.get('sentiment_distribution', {})
        info_text.append(f"Positive: {dist.get('positive', 0)}")
        info_text.append(f"Negative: {dist.get('negative', 0)}")
        info_text.append(f"Neutral: {dist.get('neutral', 0)}")
        info_text.append(f"Avg Polarity: {stats.get('avg_polarity', 0):.3f}")
        if stats.get('avg_subjectivity') is not None:
            info_text.append(f"Avg Subjectivity: {stats.get('avg_subjectivity', 0):.3f}")

        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "\n".join(info_text))
        self.info_text.config(state='disabled')

    def show_basic_analysis(self):
        df = self.data_processor.get_active_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        for widget in self.analysis_tab.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.analysis_tab)
        scrollbar = ttk.Scrollbar(self.analysis_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        stats = self.data_processor.get_advanced_statistics(df)
        ttk.Label(scrollable_frame, text="Basic Analysis Results", font=('Arial', 16, 'bold')).pack(pady=(10, 20))

        overview_frame = ttk.LabelFrame(scrollable_frame, text="Overview", padding=10)
        overview_frame.pack(fill='x', padx=20, pady=10)
        overview_text = f"""
Total Reviews: {stats.get('total_records', 0):,}
Average Polarity: {stats.get('avg_polarity', 0):.3f}
Average Subjectivity: {stats.get('avg_subjectivity', 0) if stats.get('avg_subjectivity') is not None else 'N/A'}
Average Word Count: {stats.get('avg_word_count', 0):.1f}
Polarity Standard Deviation: {stats.get('polarity_std', 0):.3f}
""".strip()
        if 'date_range' in stats:
            overview_text += f"\nDate Range: {stats['date_range']}"
            if stats.get('reviews_per_day') is not None:
                overview_text += f"\nAverage Reviews per Day: {stats.get('reviews_per_day', 0):.1f}"

        ttk.Label(overview_frame, text=overview_text, font=('Courier', 10)).pack(anchor='w')

        sentiment_frame = ttk.LabelFrame(scrollable_frame, text="Sentiment Distribution", padding=10)
        sentiment_frame.pack(fill='x', padx=20, pady=10)
        sentiment_dist = stats.get('sentiment_distribution', {})
        total = sum(sentiment_dist.values()) or 1
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total * 100)
            ttk.Label(sentiment_frame,
                      text=f"{sentiment.title()}: {count:,} ({percentage:.1f}%)",
                      font=('Arial', 11)).pack(anchor='w', pady=2)

        if 'most_positive_review' in stats:
            samples_frame = ttk.LabelFrame(scrollable_frame, text="Sample Reviews", padding=10)
            samples_frame.pack(fill='x', padx=20, pady=10)
            ttk.Label(samples_frame, text="Most Positive:", font=('Arial', 10, 'bold')).pack(anchor='w')
            ttk.Label(samples_frame, text=stats['most_positive_review'], wraplength=1000, font=('Arial', 9)).pack(anchor='w', pady=(0, 10))
            ttk.Label(samples_frame, text="Most Negative:", font=('Arial', 10, 'bold')).pack(anchor='w')
            ttk.Label(samples_frame, text=stats['most_negative_review'], wraplength=1000, font=('Arial', 9)).pack(anchor='w')

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def show_advanced_analysis(self):
        if self.data_processor.get_active_df() is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        self.update_status("Performing advanced analysis...", True)

        def analyze_thread():
            try:
                cluster_results = self.data_processor.perform_clustering()
                keyword_results = self.data_processor.get_keyword_analysis()
                bigrams = self.data_processor.get_ngram_analysis(ngram_range=(2, 2))
                trigrams = self.data_processor.get_ngram_analysis(ngram_range=(3, 3))
                self.root.after(0, self.display_advanced_results, cluster_results, keyword_results, bigrams, trigrams)
            except Exception as e:
                self.root.after(0, self.on_analysis_error, str(e))

        threading.Thread(target=analyze_thread, daemon=True).start()

    def display_advanced_results(self, cluster_results: Dict, keyword_results: Dict, bigrams: Dict, trigrams: Dict):
        self.update_status("Advanced analysis complete")

        for widget in self.analysis_tab.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.analysis_tab)
        scrollbar = ttk.Scrollbar(self.analysis_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(scrollable_frame, text="Advanced Analysis Results", font=('Arial', 16, 'bold')).pack(pady=(10, 20))

        if cluster_results.get('success'):
            cluster_frame = ttk.LabelFrame(scrollable_frame, text="Text Clustering Analysis", padding=10)
            cluster_frame.pack(fill='x', padx=20, pady=10)
            ttk.Label(cluster_frame, text="Cluster Distribution:", font=('Arial', 12, 'bold')).pack(anchor='w')
            for cluster, count in cluster_results['cluster_distribution'].items():
                ttk.Label(cluster_frame, text=f"Cluster {cluster}: {count} reviews", font=('Arial', 10)).pack(anchor='w', padx=20)
            ttk.Label(cluster_frame, text="\nTop Terms by Cluster:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(10, 0))
            for cluster, terms in cluster_results['cluster_terms'].items():
                ttk.Label(cluster_frame, text=f"{cluster}: {', '.join(terms[:10])}", font=('Arial', 10), wraplength=1200).pack(anchor='w', padx=20)

        if keyword_results:
            keyword_frame = ttk.LabelFrame(scrollable_frame, text="Keyword Analysis (Unigrams)", padding=10)
            keyword_frame.pack(fill='x', padx=20, pady=10)
            for sentiment, keywords in keyword_results.items():
                ttk.Label(keyword_frame, text=f"{sentiment.title()} Keywords:", font=('Arial', 12, 'bold')).pack(anchor='w', pady=(10, 5))
                keyword_text = ", ".join([f"{word} ({count})" for word, count in keywords[:10]])
                ttk.Label(keyword_frame, text=keyword_text, font=('Arial', 10), wraplength=1200).pack(anchor='w', padx=20)

        try:
            fig_bi = self.chart_generator.create_top_ngrams_bar(bigrams, title_prefix="Top Bigrams")
            self.embed_matplotlib_figure(scrollable_frame, fig_bi)
            fig_tri = self.chart_generator.create_top_ngrams_bar(trigrams, title_prefix="Top Trigrams")
            self.embed_matplotlib_figure(scrollable_frame, fig_tri)
        except Exception as e:
            logging.warning(f"Failed to render n-gram charts: {e}")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def embed_matplotlib_figure(self, parent, fig):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()

    def show_temporal_analysis(self):
        if self.data_processor.get_active_df() is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        if 'date' not in self.data_processor.get_active_df().columns:
            messagebox.showinfo("Info", "No date information available for temporal analysis.")
            return
        self.notebook.select(self.charts_tab)
        self.chart_var.set("Temporal Trends")
        self.generate_chart()

    def perform_clustering(self):
        if self.data_processor.get_active_df() is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        cluster_dialog = tk.Toplevel(self.root)
        cluster_dialog.title("Text Clustering")
        cluster_dialog.geometry("300x150")
        cluster_dialog.transient(self.root)
        cluster_dialog.grab_set()

        ttk.Label(cluster_dialog, text="Number of clusters:").pack(pady=10)
        cluster_var = tk.IntVar(value=3)
        ttk.Spinbox(cluster_dialog, from_=2, to=12, textvariable=cluster_var).pack(pady=5)

        def perform_clustering_action():
            cluster_dialog.destroy()
            self.update_status("Performing clustering...", True)

            def cluster_thread():
                try:
                    results = self.data_processor.perform_clustering(cluster_var.get())
                    self.root.after(0, self.show_clustering_results, results)
                except Exception as e:
                    self.root.after(0, self.on_analysis_error, str(e))

            threading.Thread(target=cluster_thread, daemon=True).start()

        ttk.Button(cluster_dialog, text="Perform Clustering", command=perform_clustering_action).pack(pady=10)

    def show_clustering_results(self, results: Dict):
        self.update_status("Clustering complete")

        if not results.get('success'):
            messagebox.showerror("Error", f"Clustering failed: {results.get('error', 'Unknown error')}")
            return

        results_window = tk.Toplevel(self.root)
        results_window.title("Clustering Results")
        results_window.geometry("900x650")

        text_frame = ttk.Frame(results_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap='word', font=('Arial', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        results_text = "CLUSTERING ANALYSIS RESULTS\n" + "=" * 50 + "\n\n"
        results_text += "Cluster Distribution:\n"
        for cluster, count in results['cluster_distribution'].items():
            results_text += f"  Cluster {cluster}: {count} reviews\n"
        results_text += "\nTop Terms by Cluster:\n"
        for cluster, terms in results['cluster_terms'].items():
            results_text += f"\n{cluster}:\n"
            results_text += f"  {', '.join(terms)}\n"

        text_widget.insert(tk.END, results_text)
        text_widget.config(state='disabled')
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        messagebox.showinfo("Success", "Clustering analysis completed successfully!")

    def train_model(self):
        if self.data_processor.get_active_df() is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        self.update_status("Training model...", True)

        def train_thread():
            try:
                results = self.data_processor.train_sentiment_model()
                self.root.after(0, self.show_model_results, results)
            except Exception as e:
                self.root.after(0, self.on_analysis_error, str(e))

        threading.Thread(target=train_thread, daemon=True).start()

    def show_model_results(self, results: Dict):
        self.update_status("Model training complete")

        if not results.get('success'):
            messagebox.showerror("Error", f"Model training failed: {results.get('error', 'Unknown error')}")
            return

        results_window = tk.Toplevel(self.root)
        results_window.title("Model Training Results")
        results_window.geometry("900x600")

        ttk.Label(results_window, text=f"Model Accuracy: {results['accuracy']:.3f}", font=('Arial', 14, 'bold')).pack(pady=10)

        report_frame = ttk.LabelFrame(results_window, text="Classification Report", padding=10)
        report_frame.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget = tk.Text(report_frame, wrap='word', font=('Courier', 9))
        scrollbar = ttk.Scrollbar(report_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        detailed = results['detailed_report']
        report_text = f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
        for label, metrics in detailed.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                report_text += f"{label:>12} {metrics.get('precision',0):>10.2f} {metrics.get('recall',0):>10.2f} {metrics.get('f1-score',0):>10.2f} {metrics.get('support',0):>10.0f}\n"
        report_text += "\n"
        for label in ['macro avg', 'weighted avg']:
            if label in detailed:
                m = detailed[label]
                report_text += f"{label:>12} {m.get('precision',0):>10.2f} {m.get('recall',0):>10.2f} {m.get('f1-score',0):>10.2f} {m.get('support',0):>10.0f}\n"
        text_widget.insert(tk.END, report_text)
        text_widget.config(state='disabled')
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        try:
            cm = np.array(self.data_processor.last_model_metrics.get('confusion_matrix'))
            labels = self.data_processor.last_model_metrics.get('labels', ['negative', 'neutral', 'positive'])
            fig_cm = self.chart_generator.create_confusion_matrix(cm, labels)
            self.embed_matplotlib_figure(results_window, fig_cm)
        except Exception as e:
            logging.warning(f"Failed to render confusion matrix: {e}")

        messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {results['accuracy']:.3f}")

    def show_model_performance(self):
        if not self.data_processor.is_model_trained:
            messagebox.showwarning("Warning", "Please train a model first.")
            return
        report = self.data_processor.last_model_metrics.get('report', {})
        acc = report.get('accuracy', 0.0)
        messagebox.showinfo("Model Performance", f"Model is trained.\nAccuracy: {acc:.3f}\nSee 'Train Model' window for full report.")

    def save_model(self):
        if not self.data_processor.is_model_trained:
            messagebox.showwarning("Warning", "Please train a model first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                                 filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            ok = self.data_processor.save_model(file_path)
            if ok:
                messagebox.showinfo("Success", f"Model saved to {file_path}")
            else:
                messagebox.showerror("Error", "Failed to save model.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            ok = self.data_processor.load_model(file_path)
            if ok:
                messagebox.showinfo("Success", f"Model loaded from {file_path}")
            else:
                messagebox.showerror("Error", "Failed to load model.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_keyword_analysis(self):
        if self.data_processor.get_active_df() is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        self.update_status("Analyzing keywords...", True)

        def keyword_thread():
            try:
                results = self.data_processor.get_keyword_analysis()
                self.root.after(0, self.show_keyword_results, results)
            except Exception as e:
                self.root.after(0, self.on_analysis_error, str(e))

        threading.Thread(target=keyword_thread, daemon=True).start()

    def show_keyword_results(self, results: Dict):
        self.update_status("Keyword analysis complete")

        if not results:
            messagebox.showerror("Error", "Failed to perform keyword analysis.")
            return

        results_window = tk.Toplevel(self.root)
        results_window.title("Keyword Analysis Results")
        results_window.geometry("800x600")

        notebook = ttk.Notebook(results_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        for sentiment, keywords in results.items():
            tab_frame = ttk.Frame(notebook)
            notebook.add(tab_frame, text=sentiment.title())

            columns = ('Rank', 'Keyword', 'Count')
            tree = ttk.Treeview(tab_frame, columns=columns, show='headings')
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150)
            for i, (keyword, count) in enumerate(keywords, 1):
                tree.insert('', 'end', values=(i, keyword, count))
            tree.pack(fill='both', expand=True, padx=10, pady=10)

        messagebox.showinfo("Success", "Keyword analysis completed successfully!")

    def on_chart_select(self, event=None):
        self.generate_chart()

    def get_chart_df(self):
        df = self.data_processor.get_active_df()
        if df is None or df.empty:
            return None
        return df

    def generate_chart(self):
        df = self.get_chart_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        chart_type = self.chart_var.get()
        for widget in self.charts_tab.winfo_children():
            widget.destroy()

        self.update_status(f"Generating {chart_type}...", True)

        try:
            if chart_type == "Sentiment Distribution":
                fig = self.chart_generator.create_sentiment_distribution(df, 'bar')
            elif chart_type == "Polarity Analysis":
                fig = self.chart_generator.create_polarity_analysis(df)
            elif chart_type == "Word Cloud":
                fig = self.chart_generator.create_wordcloud(df)
            elif chart_type == "Temporal Trends":
                fig = self.chart_generator.create_temporal_analysis(df)
            else:
                fig = self.chart_generator.create_sentiment_distribution(df, 'bar')

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            canvas = FigureCanvasTkAgg(fig, self.charts_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
            toolbar_frame = ttk.Frame(self.charts_tab)
            toolbar_frame.pack(fill='x', padx=10, pady=5)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

            self.current_figure = fig
            self.update_status(f"{chart_type} generated successfully")

        except Exception as e:
            self.update_status("Ready")
            messagebox.showerror("Error", f"Failed to generate chart: {str(e)}")

    def save_chart(self):
        if not hasattr(self, 'current_figure') or self.current_figure is None:
            messagebox.showwarning("Warning", "No chart to save. Generate a chart first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("JPEG files", "*.jpg")]
        )
        if file_path:
            try:
                self.current_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Chart saved to {file_path}")
                logging.info(f"Chart saved: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save chart: {str(e)}")

    def export_charts(self):
        df = self.get_chart_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        directory = filedialog.askdirectory(title="Select Folder to Save Charts")
        if not directory:
            return
        try:
            figs = []
            figs.append(('sentiment_distribution.png', self.chart_generator.create_sentiment_distribution(df, 'bar')))
            figs.append(('polarity_analysis.png', self.chart_generator.create_polarity_analysis(df)))
            figs.append(('wordcloud.png', self.chart_generator.create_wordcloud(df)))
            if 'date' in df.columns and not df['date'].isna().all():
                figs.append(('temporal_trends.png', self.chart_generator.create_temporal_analysis(df)))
            for name, fig in figs:
                path = os.path.join(directory, name)
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            messagebox.showinfo("Success", f"Charts exported to {directory}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export charts: {e}")

    def export_plots_pdf(self):
        df = self.get_chart_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save PDF Bundle"
        )
        if not file_path:
            return
        try:
            with PdfPages(file_path) as pdf:
                figs = [
                    self.chart_generator.create_sentiment_distribution(df, 'bar'),
                    self.chart_generator.create_polarity_analysis(df),
                    self.chart_generator.create_wordcloud(df)
                ]
                if 'date' in df.columns and not df['date'].isna().all():
                    figs.append(self.chart_generator.create_temporal_analysis(df))
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            messagebox.showinfo("Success", f"PDF bundle saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF bundle: {e}")

    def show_data_preview(self):
        df = self.data_processor.get_active_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        for widget in self.data_tab.winfo_children():
            widget.destroy()

        columns = list(df.columns)
        tree = ttk.Treeview(self.data_tab, columns=columns, show='headings')

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, minwidth=100)

        v_scrollbar = ttk.Scrollbar(self.data_tab, orient='vertical', command=tree.yview)
        h_scrollbar = ttk.Scrollbar(self.data_tab, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        for _, row in df.head(1000).iterrows():
            values = []
            for val in row:
                s = str(val)
                values.append((s[:80] + "...") if len(s) > 80 else s)
            tree.insert('', 'end', values=values)

        tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        self.data_tab.grid_rowconfigure(0, weight=1)
        self.data_tab.grid_columnconfigure(0, weight=1)

        self.notebook.select(self.data_tab)
        info_label = ttk.Label(self.data_tab, text=f"Showing first 1000 rows of {len(df)} total records")
        info_label.grid(row=2, column=0, columnspan=2, pady=5)

    def export_report(self):
        df = self.data_processor.get_active_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("Text files", "*.txt"), ("JSON files", "*.json")]
        )
        if not file_path:
            return

        self.update_status("Generating report...", True)

        def export_thread():
            try:
                stats = self.data_processor.get_advanced_statistics(df)
                keywords = self.data_processor.get_keyword_analysis(df)

                if file_path.endswith('.html'):
                    self.generate_html_report(file_path, stats, keywords, df)
                elif file_path.endswith('.json'):
                    self.generate_json_report(file_path, stats, keywords)
                else:
                    self.generate_text_report(file_path, stats, keywords)
                self.root.after(0, self.on_export_success, file_path)
            except Exception as e:
                self.root.after(0, self.on_export_error, str(e))

        threading.Thread(target=export_thread, daemon=True).start()

    def fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def generate_html_report(self, file_path: str, stats: Dict, keywords: Dict, df: pd.DataFrame):
        dist_fig = self.chart_generator.create_sentiment_distribution(df)
        dist_b64 = self.fig_to_base64(dist_fig)
        plt.close(dist_fig)

        wc_fig = self.chart_generator.create_wordcloud(df)
        wc_b64 = self.fig_to_base64(wc_fig)
        plt.close(wc_fig)

        temporal_b64 = ''
        if 'date' in df.columns and not df['date'].isna().all():
            temp_fig = self.chart_generator.create_temporal_analysis(df)
            temporal_b64 = self.fig_to_base64(temp_fig)
            plt.close(temp_fig)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Sentiment Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .neutral {{ color: #3498db; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 Sentiment Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📈 Overview Statistics</h2>
        <div class="metric"><strong>Total Reviews:</strong> {stats.get('total_records', 0):,}</div>
        <div class="metric"><strong>Average Polarity:</strong> {stats.get('avg_polarity', 0):.3f}</div>
        <div class="metric"><strong>Average Word Count:</strong> {stats.get('avg_word_count', 0):.1f}</div>
    </div>

    <div class="section">
        <h2>🎯 Sentiment Distribution</h2>
        <img src="data:image/png;base64,{dist_b64}" alt="Sentiment Distribution"/>
    </div>

    <div class="section">
        <h2>☁️ Word Cloud</h2>
        <img src="data:image/png;base64,{wc_b64}" alt="Word Cloud"/>
    </div>
"""
        if temporal_b64:
            html_content += f"""
    <div class="section">
        <h2>⏰ Temporal Trends</h2>
        <img src="data:image/png;base64,{temporal_b64}" alt="Temporal Trends"/>
    </div>
"""

        if keywords:
            html_content += """
    <div class="section">
        <h2>🔑 Top Keywords by Sentiment</h2>
"""
            for sentiment, word_counts in keywords.items():
                html_content += f"""
        <h3 class="{sentiment}">{sentiment.title()} Keywords</h3>
        <table>
            <tr><th>Keyword</th><th>Frequency</th></tr>
"""
                for word, count in word_counts[:10]:
                    html_content += f"<tr><td>{word}</td><td>{count}</td></tr>"
                html_content += "</table>"
            html_content += "</div>"

        html_content += """
</div>
</body>
</html>
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def generate_json_report(self, file_path: str, stats: Dict, keywords: Dict):
        report_data = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'tool': 'Advanced Sentiment Analysis Tool'
            },
            'statistics': stats,
            'keyword_analysis': keywords
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    def generate_text_report(self, file_path: str, stats: Dict, keywords: Dict):
        report_text = f"""
ADVANCED SENTIMENT ANALYSIS REPORT
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW STATISTICS
{'-' * 20}
Total Reviews: {stats.get('total_records', 0):,}
Average Polarity: {stats.get('avg_polarity', 0):.3f}
Average Subjectivity: {stats.get('avg_subjectivity', 0) if stats.get('avg_subjectivity') is not None else 'N/A'}
Average Word Count: {stats.get('avg_word_count', 0):.1f}
Polarity Standard Deviation: {stats.get('polarity_std', 0):.3f}

SENTIMENT DISTRIBUTION
{'-' * 20}
"""
        sentiment_dist = stats.get('sentiment_distribution', {})
        total = sum(sentiment_dist.values()) or 1
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total * 100)
            report_text += f"{sentiment.title()}: {count:,} ({percentage:.1f}%)\n"

        if keywords:
            report_text += f"\nKEYWORD ANALYSIS\n{'-' * 20}\n"
            for sentiment, word_counts in keywords.items():
                report_text += f"\n{sentiment.title()} Keywords:\n"
                for word, count in word_counts[:10]:
                    report_text += f"  {word}: {count}\n"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

    def export_data(self):
        df = self.data_processor.get_active_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("JSON files", "*.json")]
        )
        if not file_path:
            return

        try:
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            elif file_path.endswith('.json'):
                df.to_json(file_path, orient='records', indent=2)
            messagebox.showinfo("Success", f"Data exported to {file_path}")
            logging.info(f"Data exported: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def toggle_theme(self):
        self.theme_manager.toggle_theme()
        self.apply_theme()
        theme_name = "Dark" if self.theme_manager.current_theme == "dark" else "Light"
        self.update_status(f"Switched to {theme_name} theme")
        if hasattr(self, 'current_figure') and self.current_figure is not None:
            self.generate_chart()

    def show_history(self):
        history = self.db_manager.get_analysis_history()
        if not history:
            messagebox.showinfo("Info", "No analysis history found.")
            return

        history_window = tk.Toplevel(self.root)
        history_window.title("Analysis History")
        history_window.geometry("800x600")

        columns = ('Date', 'Filename', 'Records', 'Positive', 'Negative', 'Neutral', 'Avg Polarity')
        tree = ttk.Treeview(history_window, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=110)
        for record in history:
            tree.insert('', 'end', values=(
                record['upload_date'][:19],
                record['filename'],
                record['total_records'],
                record['positive_count'],
                record['negative_count'],
                record['neutral_count'],
                f"{record['avg_polarity']:.3f}"
            ))
        scrollbar = ttk.Scrollbar(history_window, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)

    def show_topic_modeling(self):
        if self.data_processor.get_active_df() is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        self.notebook.select(self.topics_tab)
        self.build_topics_tab_ui()

    def build_topics_tab_ui(self):
        for w in self.topics_tab.winfo_children():
            w.destroy()
        ctl = ttk.Frame(self.topics_tab)
        ctl.pack(fill='x', padx=10, pady=10)
        ttk.Label(ctl, text="Number of topics:").pack(side='left', padx=5)
        ttk.Spinbox(ctl, from_=2, to=20, textvariable=self.lda_topics_var, width=5).pack(side='left')
        ttk.Label(ctl, text="Max features:").pack(side='left', padx=10)
        ttk.Spinbox(ctl, from_=500, to=20000, increment=500, textvariable=self.lda_max_features_var, width=6).pack(side='left')
        ttk.Label(ctl, text="min_df:").pack(side='left', padx=10)
        ttk.Spinbox(ctl, from_=1, to=20, textvariable=self.lda_min_df_var, width=5).pack(side='left')
        ttk.Label(ctl, text="max_df:").pack(side='left', padx=10)
        ttk.Entry(ctl, textvariable=self.lda_max_df_var, width=6).pack(side='left')
        ttk.Button(ctl, text="Run LDA", command=self.run_lda).pack(side='left', padx=15)

        self.topics_output = ttk.Frame(self.topics_tab)
        self.topics_output.pack(fill='both', expand=True, padx=10, pady=10)

    def run_lda(self):
        df = self.data_processor.get_active_df()
        if df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        self.update_status("Running LDA topic modeling...", True)

        n_topics = int(self.lda_topics_var.get())
        max_features = int(self.lda_max_features_var.get())
        min_df = int(self.lda_min_df_var.get())
        try:
            max_df = float(self.lda_max_df_var.get())
        except Exception:
            max_df = 0.95

        def lda_thread():
            try:
                result = self.data_processor.run_lda(
                    n_topics=n_topics,
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df
                )
                self.root.after(0, self.display_lda_results, result)
            except Exception as e:
                self.root.after(0, self.on_analysis_error, str(e))

        threading.Thread(target=lda_thread, daemon=True).start()

    def display_lda_results(self, result: Dict):
        self.update_status("LDA complete")
        for w in self.topics_output.winfo_children():
            w.destroy()

        if not result.get('success'):
            messagebox.showerror("LDA Error", result.get('error', 'Unknown error'))
            return

        # Text summary
        text_frame = ttk.LabelFrame(self.topics_output, text="Top words per topic", padding=10)
        text_frame.pack(fill='x', padx=5, pady=5)
        text_box = tk.Text(text_frame, wrap='word', height=12, font=('Courier', 10))
        for topic in result['top_words']:
            words = ", ".join(topic['words'])
            text_box.insert(tk.END, f"Topic {topic['topic']:02d}: {words}\n")
        text_box.config(state='disabled')
        text_box.pack(fill='x')

        # Figure
        fig = self.chart_generator.create_lda_topics_figure(result['top_words'])
        self.embed_matplotlib_figure(self.topics_output, fig)

        # Dominant topic counts
        dom_counts = Counter(result['dominant_topic'])
        dom_frame = ttk.LabelFrame(self.topics_output, text="Dominant topic distribution", padding=10)
        dom_frame.pack(fill='x', padx=5, pady=5)
        dom_text = ", ".join([f"T{t}: {c}" for t, c in sorted(dom_counts.items())])
        ttk.Label(dom_frame, text=dom_text).pack(anchor='w')

    def on_analysis_error(self, error_msg: str):
        self.update_status("Analysis failed")
        messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_msg}")
        logging.error(f"Analysis error: {error_msg}")

    def on_export_success(self, file_path: str):
        self.update_status("Export completed")
        messagebox.showinfo("Success", f"Report exported to {file_path}")
        logging.info(f"Report exported: {file_path}")

    def on_export_error(self, error_msg: str):
        self.update_status("Export failed")
        messagebox.showerror("Export Error", f"Export failed:\n{error_msg}")
        logging.error(f"Export error: {error_msg}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        try:
            plt.close('all')
            logging.info("Application closed")
        except Exception:
            pass
        self.root.destroy()

    def apply_filters(self):
        sentiments = [k for k, v in self.filter_vars.items() if v.get()]
        min_words = self.min_words_var.get()
        english_only = self.english_only_var.get()

        def parse_date(s):
            s = s.strip()
            if not s:
                return None
            try:
                return pd.to_datetime(s, errors='coerce')
            except Exception:
                return None

        date_from = parse_date(self.start_date_var.get())
        date_to = parse_date(self.end_date_var.get())

        self.data_processor.apply_filters(sentiments, min_words, english_only, date_from, date_to)
        self.update_info_panel()

        current_tab = self.notebook.select()
        if current_tab == str(self.charts_tab):
            self.generate_chart()
        elif current_tab == str(self.data_tab):
            self.show_data_preview()

    def reset_filters(self):
        self.filter_vars['positive'].set(True)
        self.filter_vars['negative'].set(True)
        self.filter_vars['neutral'].set(True)
        self.min_words_var.set(0)
        self.english_only_var.set(False)
        self.start_date_var.set("")
        self.end_date_var.set("")
        self.data_processor.reset_filters()
        self.update_info_panel()

    def remove_duplicates(self):
        if self.data_processor.df is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        removed = self.data_processor.remove_duplicates()
        self.update_info_panel()
        messagebox.showinfo("Deduplication", f"Removed {removed} duplicate review(s).")

    def quick_predict(self):
        txt = self.qp_text.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showwarning("Warning", "Enter some text to predict.")
            return
        res = self.data_processor.predict_text(txt)
        if not res.get('success'):
            self.qp_result.config(text=f"Result: Error - {res.get('error')}")
            return
        emoji = {'positive': '😊', 'neutral': '😐', 'negative': '☹️'}.get(res['sentiment'], '')
        conf = res.get('confidence')
        conf_str = f" (conf: {conf:.2f})" if conf is not None else ""
        self.qp_result.config(text=f"Result: {res['sentiment'].title()} {emoji} via {res['engine']}{conf_str}")

    def open_vectorizer_settings_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Vectorizer Settings")
        dlg.geometry("300x200")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="Max Features:").pack(pady=(10, 0))
        mf_var = tk.IntVar(value=self.data_processor.max_features)
        ttk.Spinbox(dlg, from_=500, to=20000, increment=100, textvariable=mf_var).pack(pady=5)

        ttk.Label(dlg, text="n-gram range (min,max):").pack(pady=(10, 0))
        frm = ttk.Frame(dlg)
        frm.pack(pady=5)
        ng_min = tk.IntVar(value=self.data_processor.ngram_range[0])
        ng_max = tk.IntVar(value=self.data_processor.ngram_range[1])
        ttk.Spinbox(frm, from_=1, to=3, textvariable=ng_min, width=5).pack(side='left', padx=5)
        ttk.Spinbox(frm, from_=1, to=3, textvariable=ng_max, width=5).pack(side='left', padx=5)

        def apply():
            mn, mx = int(ng_min.get()), int(ng_max.get())
            if mn > mx:
                messagebox.showwarning("Invalid", "Min n-gram cannot exceed max.")
                return
            self.data_processor.set_vectorizer_params(int(mf_var.get()), (mn, mx))
            dlg.destroy()
            messagebox.showinfo("Settings", "Vectorizer settings updated. Retrain model to apply.")

        ttk.Button(dlg, text="Apply", command=apply).pack(pady=10)


def main():
    try:
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'textblob', 'wordcloud', 'sklearn', 'tkinter'
        ]
        missing_packages = []
        for package in required_packages:
            try:
                if package == 'tkinter':
                    import tkinter  # noqa
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"Missing required packages: {', '.join(missing_packages)}")
            print("Please install them using: pip install " + ' '.join(p for p in missing_packages if p != 'tkinter'))
            print("Optional: pip install langdetect tkcalendar nltk")
            return

        app = SentimentAnalysisApp()
        app.run()

    except Exception as e:
        logging.error(f"Application startup error: {e}")
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()