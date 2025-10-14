#!/usr/bin/env python3
"""
Comprehensive Machine Learning Pipeline for Heart Disease UCI Dataset
Author: Claude AI
Date: September 2025
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib

# Plot styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HeartDiseaseMLPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.data = None
        self.X = None
        self.y = None

        self.X_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_index = None
        self.test_index = None

        self.scaler = StandardScaler()
        self.pca = None
        self.models = {}
        self.best_model = None
        self.selected_features = None
        self.final_pipeline = None

    def load_data(self):
        """Load the Heart Disease UCI dataset"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]
            self.data = pd.read_csv(url, names=columns, na_values='?')
            print("✅ Dataset loaded successfully from UCI repository")
        except Exception as e:
            print(f"❌ Could not load from UCI ({e}). Creating sample dataset for demonstration...")
            np.random.seed(self.random_state)
            n_samples = 303
            self.data = pd.DataFrame({
                'age': np.random.normal(54, 9, n_samples).astype(int),
                'sex': np.random.choice([0, 1], n_samples),
                'cp': np.random.choice([0, 1, 2, 3], n_samples),
                'trestbps': np.random.normal(131, 17, n_samples).astype(int),
                'chol': np.random.normal(246, 51, n_samples).astype(int),
                'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
                'restecg': np.random.choice([0, 1, 2], n_samples),
                'thalach': np.random.normal(149, 22, n_samples).astype(int),
                'exang': np.random.choice([0, 1], n_samples),
                'oldpeak': np.random.exponential(1, n_samples),
                'slope': np.random.choice([0, 1, 2], n_samples),
                'ca': np.random.choice([0, 1, 2, 3], n_samples),
                'thal': np.random.choice([1, 2, 3], n_samples),  # normalized encoding
                'target': np.random.choice([0, 1], n_samples)
            })
        print(f"Dataset shape: {self.data.shape}")
        return self.data

    def data_preprocessing(self):
        """Comprehensive data preprocessing and cleaning"""
        print("\n🔧 Starting Data Preprocessing...")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values:\n{self.data.isnull().sum()}")

        # Handle missing values (numerics -> median)
        if self.data.isnull().sum().sum() > 0:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)

        # Convert target to binary (0 = no disease, 1 = disease)
        self.data['target'] = (self.data['target'] > 0).astype(int)

        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']

        # Scale all features (fit on full data for this demo; in prod use only train)
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        print("✅ Data preprocessing completed")
        return self.X_scaled, self.y

    def exploratory_data_analysis(self):
        """Perform basic EDA"""
        print("\n📊 Performing Exploratory Data Analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Target distribution
        counts = self.y.value_counts().reindex([0, 1], fill_value=0)
        axes[0, 0].pie(counts, labels=['No Disease', 'Disease'], autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Heart Disease Distribution')

        # 2. Age distribution by target
        for target in [0, 1]:
            subset = self.data[self.data['target'] == target]
            axes[0, 1].hist(subset['age'], alpha=0.7, label=f'Target {target}', bins=20)
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Age Distribution by Target')
        axes[0, 1].legend()

        # 3. Correlation heatmap
        correlation_matrix = self.data.corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=axes[1, 0], fmt='.2f', square=True)
        axes[1, 0].set_title('Feature Correlation Heatmap')

        # 4. Boxplot for key features
        key_features = ['age', 'trestbps', 'chol', 'thalach']
        self.data[key_features].boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('Boxplot of Key Numerical Features')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        print("\nDataset Statistics:")
        print(self.data.describe())
        return correlation_matrix

    def apply_pca(self, n_components=None):
        """Apply Principal Component Analysis"""
        print("\n🎯 Applying PCA for Dimensionality Reduction...")
        pca_temp = PCA()
        pca_temp.fit(self.X_scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)

        if n_components is None:
            n_components = np.argmax(cumsum >= 0.95) + 1

        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(self.X_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_) + 1), self.pca.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Explained Variance per Component')

        axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        print(f"✅ PCA completed. {n_components} components explain {cumsum[n_components - 1]:.2%} of variance")
        return pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_components)])

    def feature_selection(self):
        """Apply multiple feature selection techniques and combine results"""
        print("\n🎯 Performing Feature Selection...")
        selected = {}

        # 1) Random Forest importance
        rf = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        rf.fit(self.X_scaled, self.y)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        selected['rf_top5'] = feature_importance.head(5)['feature'].tolist()

        # 2) RFE
        rfe = RFE(RandomForestClassifier(random_state=self.random_state), n_features_to_select=5)
        rfe.fit(self.X_scaled, self.y)
        selected['rfe'] = self.X.columns[rfe.support_].tolist()

        # 3) Chi-square (requires non-negative)
        X_non_negative = self.X_scaled - self.X_scaled.min() + 1
        chi2_selector = SelectKBest(chi2, k=5)
        chi2_selector.fit(X_non_negative, self.y)
        selected['chi2'] = self.X.columns[chi2_selector.get_support()].tolist()

        # Plot importance
        plt.figure(figsize=(12, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Combine features from all methods
        final_features = sorted(set().union(*selected.values()))
        self.selected_features = final_features

        print("✅ Feature Selection Results:")
        for method, feats in selected.items():
            print(f"{method}: {feats}")
        print(f"Final combined features ({len(final_features)}): {final_features}")

        return self.X_scaled[final_features], selected

    def train_supervised_models(self, X_selected, y):
        """Train multiple supervised learning models"""
        print("\n🤖 Training Supervised Learning Models...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        # Keep indices to refit a final pipeline on raw data later
        self.train_index = self.X_train.index
        self.test_index = self.X_test.index

        models_config = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=200),
            'SVM': SVC(random_state=self.random_state, probability=True)
        }

        results = {}
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            print(f"✅ {name} - Accuracy: {accuracy:.3f}, F1-Score: {f1:.3f}")

        self.models = results
        return results

    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n📈 Evaluating Models...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.models.keys())
        x = np.arange(len(model_names))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [self.models[m][metric] for m in model_names]
            axes[0, 0].bar(x + i * width, values, width, label=metric.capitalize())
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        for name, results in self.models.items():
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
                auc_score = results['auc_score']
                axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['f1_score'])
        best_predictions = self.models[best_model_name]['predictions']
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')

        if hasattr(self.models[best_model_name]['model'], 'feature_importances_'):
            importance = self.models[best_model_name]['model'].feature_importances_
            feature_names = self.X_train.columns
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance',
                                                                                                    ascending=True)
            axes[1, 1].barh(imp_df['feature'], imp_df['importance'])
            axes[1, 1].set_title(f'Feature Importance - {best_model_name}')
            axes[1, 1].set_xlabel('Importance')

        plt.tight_layout()
        plt.show()

        print("\n📊 Detailed Model Results:")
        results_df = pd.DataFrame({
            model: {
                'Accuracy': f"{res['accuracy']:.3f}",
                'Precision': f"{res['precision']:.3f}",
                'Recall': f"{res['recall']:.3f}",
                'F1-Score': f"{res['f1_score']:.3f}",
                'AUC Score': f"{res['auc_score']:.3f}" if res['auc_score'] else "N/A"
            }
            for model, res in self.models.items()
        }).T
        print(results_df)

        self.best_model = self.models[best_model_name]['model']
        print(f"\n🏆 Best Model: {best_model_name}")
        return results_df

    def unsupervised_learning(self):
        """Apply clustering algorithms"""
        print("\n🔍 Applying Unsupervised Learning...")
        inertias = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].plot(K_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method for Optimal K')
        axes[0].grid(True, alpha=0.3)

        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        clusters_kmeans = kmeans.fit_predict(self.X_scaled)

        hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
        clusters_hierarchical = hierarchical.fit_predict(self.X_scaled)

        pca_viz = PCA(n_components=2)
        X_pca_viz = pca_viz.fit_transform(self.X_scaled)

        scatter = axes[1].scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=clusters_kmeans, cmap='viridis', alpha=0.7)
        axes[1].set_title('K-Means Clustering Results')
        axes[1].set_xlabel('First Principal Component')
        axes[1].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[1])

        scatter2 = axes[2].scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=clusters_hierarchical, cmap='plasma', alpha=0.7)
        axes[2].set_title('Hierarchical Clustering Results')
        axes[2].set_xlabel('First Principal Component')
        axes[2].set_ylabel('Second Principal Component')
        plt.colorbar(scatter2, ax=axes[2])

        plt.tight_layout()
        plt.show()

        from sklearn.metrics import adjusted_rand_score, silhouette_score
        ari_kmeans = adjusted_rand_score(self.y, clusters_kmeans)
        ari_hierarchical = adjusted_rand_score(self.y, clusters_hierarchical)
        silhouette_kmeans = silhouette_score(self.X_scaled, clusters_kmeans)
        silhouette_hierarchical = silhouette_score(self.X_scaled, clusters_hierarchical)

        print(f"✅ K-Means Clustering:")
        print(f"   Adjusted Rand Index: {ari_kmeans:.3f}")
        print(f"   Silhouette Score: {silhouette_kmeans:.3f}")
        print(f"✅ Hierarchical Clustering:")
        print(f"   Adjusted Rand Index: {ari_hierarchical:.3f}")
        print(f"   Silhouette Score: {silhouette_hierarchical:.3f}")

        return clusters_kmeans, clusters_hierarchical

    def hyperparameter_tuning(self):
        """Optimize model hyperparameters"""
        print("\n⚙️ Hyperparameter Tuning...")
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l2']
            }
        }
        models_for_tuning = {
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }

        tuned_models = {}
        for name, model in models_for_tuning.items():
            print(f"Tuning {name}...")
            random_search = RandomizedSearchCV(
                model, param_grids[name], n_iter=20, cv=5, scoring='f1',
                random_state=self.random_state, n_jobs=-1
            )
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            y_pred_tuned = best_model.predict(self.X_test)
            tuned_models[name] = {
                'model': best_model,
                'best_params': random_search.best_params_,
                'cv_score': random_search.best_score_,
                'test_f1': f1_score(self.y_test, y_pred_tuned),
                'test_accuracy': accuracy_score(self.y_test, y_pred_tuned)
            }
            print(
                f"✅ {name} - Best CV F1: {random_search.best_score_:.3f}, Test F1: {tuned_models[name]['test_f1']:.3f}")
            print(f"   Best parameters: {random_search.best_params_}")

        best_tuned_name = max(tuned_models.keys(), key=lambda x: tuned_models[x]['test_f1'])
        self.best_model = tuned_models[best_tuned_name]['model']
        print(f"\n🏆 Best Tuned Model: {best_tuned_name}")
        print(f"   Test F1-Score: {tuned_models[best_tuned_name]['test_f1']:.3f}")
        return tuned_models, self.best_model

    def save_model(self, filename='final_model.pkl'):
        """
        Save a fully-fitted pipeline:
          - Column selection -> StandardScaler -> Best model
        """
        print(f"\n💾 Saving model to {filename}...")
        if self.best_model is None:
            raise RuntimeError("No best model to save. Run training/evaluation first.")

        # Use selected features, or fall back to all
        features_to_use = self.selected_features if self.selected_features else list(self.X.columns)

        # Create a fresh pipeline and fit on the raw training data for consistency
        preprocessor = ColumnTransformer(
            [('scale', StandardScaler(), features_to_use)],
            remainder='drop'
        )
        model_clone = clone(self.best_model)
        final_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', model_clone)
        ])

        # Fit on raw train split using the same indices captured earlier
        X_train_raw = self.X.loc[self.train_index, features_to_use]
        y_train_raw = self.y.loc[self.train_index]
        final_pipeline.fit(X_train_raw, y_train_raw)
        self.final_pipeline = final_pipeline

        # Evaluate on raw test split
        X_test_raw = self.X.loc[self.test_index, features_to_use]
        y_test_raw = self.y.loc[self.test_index]
        y_pred = final_pipeline.predict(X_test_raw)
        acc = accuracy_score(y_test_raw, y_pred)
        f1 = f1_score(y_test_raw, y_pred)

        # Save the fitted pipeline
        joblib.dump(final_pipeline, filename)

        # Save metadata
        metadata = {
            'all_feature_names': list(self.X.columns),
            'feature_names': features_to_use,
            'target_names': ['No Disease', 'Disease'],
            'model_type': type(self.best_model).__name__,
            'performance_metrics': {
                'accuracy': acc,
                'f1_score': f1
            }
        }
        joblib.dump(metadata, filename.replace('.pkl', '_metadata.pkl'))

        print(f"✅ Model and metadata saved successfully!")
        print(f"   Test Accuracy: {acc:.3f}, Test F1: {f1:.3f}")
        return filename

    def run_complete_pipeline(self):
        """Run the entire ML pipeline"""
        print("🚀 Starting Comprehensive Heart Disease ML Pipeline")
        print("=" * 60)

        self.load_data()
        X_processed, y = self.data_preprocessing()
        _ = self.exploratory_data_analysis()
        _ = self.apply_pca()
        X_selected, _ = self.feature_selection()
        _ = self.train_supervised_models(X_selected, y)
        _ = self.evaluate_models()
        _ = self.unsupervised_learning()
        _, _ = self.hyperparameter_tuning()
        model_filename = self.save_model()

        print("\n🎉 Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📁 Model saved as: {model_filename}")
        print(f"🏆 Best model: {type(self.best_model).__name__}")

        return {
            'data': self.data,
            'processed_features': X_selected,
            'model_results': self.models,
            'best_model': self.best_model,
        }


if __name__ == "__main__":
    pipeline = HeartDiseaseMLPipeline()
    results = pipeline.run_complete_pipeline()
    print("\n📋 Pipeline Summary:")
    print(f"Dataset shape: {pipeline.data.shape}")
    print(f"Number of models trained: {len(results['model_results'])}")
    print(f"Best model type: {type(results['best_model']).__name__}")
    print("✅ All objectives completed successfully!")