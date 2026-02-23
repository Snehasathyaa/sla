"""
ML Training Script - Smart SLA Prediction System
Trains: SLA Violation Classifier + Resolution Days Regressor
Generates: All evaluation graphs
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance
import joblib

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 120,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#0f172a'
})

COLORS = {
    'primary': '#3b82f6',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'purple': '#8b5cf6',
    'cyan': '#06b6d4',
    'bg': '#0f172a',
    'surface': '#1e293b',
    'text': '#e2e8f0',
    'muted': '#64748b'
}
PALETTE = [COLORS['primary'], COLORS['danger'], COLORS['success'],
           COLORS['warning'], COLORS['purple'], COLORS['cyan']]

os.makedirs('static/graphs', exist_ok=True)
os.makedirs('models', exist_ok=True)


def styled_fig(figsize=(12, 7)):
    fig = plt.figure(figsize=figsize, facecolor=COLORS['bg'])
    return fig


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(COLORS['surface'])
    ax.tick_params(colors=COLORS['text'])
    ax.xaxis.label.set_color(COLORS['text'])
    ax.yaxis.label.set_color(COLORS['text'])
    ax.title.set_color(COLORS['text'])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['muted'])
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess():
    df = pd.read_csv('dataset/grievance_dataset.csv')
    print(f"Loaded {len(df)} records  |  SLA violation rate: {df['sla_violated'].mean():.2%}")

    le_cat = LabelEncoder()
    le_dep = LabelEncoder()
    le_pri = LabelEncoder()
    le_dis = LabelEncoder()

    df['category_enc'] = le_cat.fit_transform(df['category'])
    df['department_enc'] = le_dep.fit_transform(df['department'])
    df['priority_enc'] = le_pri.fit_transform(df['priority'])
    df['district_enc'] = le_dis.fit_transform(df['district'])

    encoders = {
        'category': le_cat, 'department': le_dep,
        'priority': le_pri, 'district': le_dis
    }

    FEATURES = [
        'category_enc', 'department_enc', 'priority_enc', 'district_enc',
        'urban', 'day_of_week', 'officer_experience', 'workload_at_submission',
        'complaint_length', 'resubmission', 'sla_threshold_days'
    ]
    FEATURE_NAMES = [
        'Category', 'Department', 'Priority', 'District',
        'Urban/Rural', 'Day of Week', 'Officer Experience', 'Workload',
        'Complaint Length', 'Resubmission', 'SLA Threshold'
    ]

    X = df[FEATURES].values
    y_cls = df['sla_violated'].values
    y_reg = df['resolution_days'].values

    return df, X, y_cls, y_reg, FEATURES, FEATURE_NAMES, encoders


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAPH 1 — EDA: SLA Violation by Category & Priority
# ══════════════════════════════════════════════════════════════════════════════

def plot_eda(df):
    fig = styled_fig(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # 2a. Violation rate by category
    ax1 = fig.add_subplot(gs[0])
    cat_viol = df.groupby('category')['sla_violated'].mean().sort_values(ascending=True)
    bars = ax1.barh(cat_viol.index, cat_viol.values * 100,
                    color=[COLORS['danger'] if v > 0.5 else COLORS['warning']
                           if v > 0.35 else COLORS['success'] for v in cat_viol.values])
    ax1.set_xlim(0, 100)
    for bar, v in zip(bars, cat_viol.values):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{v*100:.1f}%', va='center', color=COLORS['text'], fontsize=9)
    style_ax(ax1, 'SLA Violation Rate by Category', 'Violation Rate (%)', '')

    # 2b. Violations by priority
    ax2 = fig.add_subplot(gs[1])
    pri_order = ['Low', 'Medium', 'High', 'Critical']
    pri_data = df[df['priority'].isin(pri_order)]
    pri_viol = pri_data.groupby('priority')['sla_violated'].mean().reindex(pri_order)
    clrs = [COLORS['success'], COLORS['warning'], COLORS['danger'], '#dc2626']
    bars2 = ax2.bar(pri_viol.index, pri_viol.values * 100, color=clrs, width=0.55)
    for bar, v in zip(bars2, pri_viol.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{v*100:.1f}%', ha='center', color=COLORS['text'], fontsize=9)
    style_ax(ax2, 'SLA Violation Rate by Priority', 'Priority Level', 'Violation Rate (%)')

    # 2c. Resolution days distribution
    ax3 = fig.add_subplot(gs[2])
    violated = df[df['sla_violated'] == 1]['resolution_days']
    on_time = df[df['sla_violated'] == 0]['resolution_days']
    ax3.hist(on_time, bins=30, alpha=0.75, color=COLORS['success'], label='On-Time', density=True)
    ax3.hist(violated, bins=30, alpha=0.75, color=COLORS['danger'], label='SLA Violated', density=True)
    ax3.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    style_ax(ax3, 'Resolution Days Distribution', 'Resolution Days', 'Density')

    fig.suptitle('Exploratory Data Analysis — Kerala Grievance Dataset',
                 color=COLORS['text'], fontsize=15, fontweight='bold', y=1.02)
    plt.savefig('static/graphs/eda_analysis.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ EDA graph saved")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_models(X, y_cls, y_reg, FEATURE_NAMES):
    X_train, X_test, y_cls_train, y_cls_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)

    # ── Classifier ─────────────────────────────────────────────────────────
    print("\nTraining classifiers...")
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                    random_state=42, class_weight='balanced')
    gb_clf = GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                         learning_rate=0.1, random_state=42)
    lr_clf = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')

    rf_clf.fit(X_train, y_cls_train)
    gb_clf.fit(X_train, y_cls_train)
    lr_clf.fit(X_train, y_cls_train)

    # ── Regressor ──────────────────────────────────────────────────────────
    print("Training regressor...")
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    rf_reg.fit(X_train, y_reg_train)

    # Save best classifier (RF)
    joblib.dump(rf_clf, 'models/sla_classifier.pkl')
    joblib.dump(rf_reg, 'models/resolution_regressor.pkl')
    print("✔ Models saved")

    return (rf_clf, gb_clf, lr_clf, rf_reg,
            X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test)


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRAPH 2 — Learning Curves (Accuracy & Loss proxy)
# ══════════════════════════════════════════════════════════════════════════════

def plot_learning_curves(rf_clf, X, y_cls):
    print("Generating learning curves...")
    train_sizes, train_scores, val_scores = learning_curve(
        rf_clf, X, y_cls, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 12), n_jobs=-1)

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    # Compute "loss" as 1 - accuracy
    train_loss = 1 - train_mean
    val_loss   = 1 - val_mean

    fig = styled_fig(figsize=(14, 5))
    ax1, ax2 = fig.subplots(1, 2)

    # Accuracy
    ax1.plot(train_sizes, train_mean, color=COLORS['primary'], lw=2.5, marker='o', label='Train Accuracy')
    ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color=COLORS['primary'])
    ax1.plot(train_sizes, val_mean, color=COLORS['success'], lw=2.5, marker='s', label='Validation Accuracy')
    ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color=COLORS['success'])
    ax1.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    ax1.set_ylim(0.5, 1.02)
    style_ax(ax1, 'Model Accuracy — Learning Curve', 'Training Samples', 'Accuracy')

    # Loss
    ax2.plot(train_sizes, train_loss, color=COLORS['warning'], lw=2.5, marker='o', label='Train Loss')
    ax2.fill_between(train_sizes, train_loss - train_std, train_loss + train_std,
                     alpha=0.2, color=COLORS['warning'])
    ax2.plot(train_sizes, val_loss, color=COLORS['danger'], lw=2.5, marker='s', label='Validation Loss')
    ax2.fill_between(train_sizes, val_loss - val_std, val_loss + val_std,
                     alpha=0.2, color=COLORS['danger'])
    ax2.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    style_ax(ax2, 'Model Loss — Learning Curve', 'Training Samples', 'Loss (1 - Accuracy)')

    fig.suptitle('Random Forest — Learning Curves', color=COLORS['text'],
                 fontsize=14, fontweight='bold')
    plt.savefig('static/graphs/learning_curves.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ Learning curves saved")


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRAPH 3 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(models_dict, X_test, y_test):
    metrics = {}
    for name, model in models_dict.items():
        preds = model.predict(X_test)
        metrics[name] = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-Score': f1_score(y_test, preds)
        }

    df_m = pd.DataFrame(metrics).T
    fig = styled_fig(figsize=(12, 6))
    ax = fig.add_subplot(111)

    x = np.arange(len(df_m.columns))
    width = 0.25
    bars_colors = [COLORS['primary'], COLORS['success'], COLORS['warning']]

    for i, (model_name, row) in enumerate(df_m.iterrows()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, row.values, width, label=model_name,
                      color=bars_colors[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    color=COLORS['text'], fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_m.columns, color=COLORS['text'])
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    style_ax(ax, 'Model Comparison — Classification Metrics', '', 'Score')
    plt.savefig('static/graphs/model_comparison.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ Model comparison saved")

    return df_m


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAPH 4 — Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(rf_clf, X_test, y_cls_test):
    y_pred = rf_clf.predict(X_test)
    cm = confusion_matrix(y_cls_test, y_pred)

    fig = styled_fig(figsize=(8, 6))
    ax = fig.add_subplot(111)

    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.color_palette("Blues", as_cmap=True),
                linewidths=2, linecolor=COLORS['bg'],
                annot_kws={'size': 16, 'weight': 'bold', 'color': 'white'},
                cbar_kws={'shrink': 0.8})

    ax.set_xticklabels(['On-Time', 'SLA Violated'], color=COLORS['text'])
    ax.set_yticklabels(['On-Time', 'SLA Violated'], color=COLORS['text'], rotation=0)
    style_ax(ax, 'Confusion Matrix — Random Forest', 'Predicted', 'Actual')
    ax.set_facecolor(COLORS['surface'])

    # Accuracy overlay
    acc = accuracy_score(y_cls_test, y_pred)
    fig.text(0.5, 0.02, f'Overall Accuracy: {acc:.2%}', ha='center',
             color=COLORS['success'], fontsize=12, fontweight='bold')

    plt.savefig('static/graphs/confusion_matrix.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ Confusion matrix saved")


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRAPH 5 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(models_dict, X_test, y_test):
    fig = styled_fig(figsize=(9, 7))
    ax = fig.add_subplot(111)

    colors_list = [COLORS['primary'], COLORS['success'], COLORS['warning']]
    for (name, model), color in zip(models_dict.items(), colors_list):
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', color=COLORS['muted'], lw=1.5, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color=COLORS['muted'])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'], loc='lower right')
    style_ax(ax, 'ROC Curves — All Models', 'False Positive Rate', 'True Positive Rate')

    plt.savefig('static/graphs/roc_curves.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ ROC curves saved")


# ══════════════════════════════════════════════════════════════════════════════
# 8. GRAPH 6 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(rf_clf, FEATURE_NAMES):
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)
    sorted_names = [FEATURE_NAMES[i] for i in indices]
    sorted_vals = importances[indices]

    clrs = [COLORS['danger'] if v > 0.15 else COLORS['warning']
            if v > 0.08 else COLORS['primary'] for v in sorted_vals]

    fig = styled_fig(figsize=(10, 7))
    ax = fig.add_subplot(111)
    bars = ax.barh(sorted_names, sorted_vals * 100, color=clrs)
    for bar, v in zip(bars, sorted_vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{v*100:.1f}%', va='center', color=COLORS['text'], fontsize=9)
    style_ax(ax, 'Feature Importance — Random Forest Classifier',
             'Importance (%)', 'Feature')

    plt.savefig('static/graphs/feature_importance.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ Feature importance saved")


# ══════════════════════════════════════════════════════════════════════════════
# 9. GRAPH 7 — Cross-Validation Scores
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_scores(rf_clf, gb_clf, lr_clf, X, y_cls):
    print("Running cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, model in [('Random Forest', rf_clf),
                         ('Gradient Boosting', gb_clf),
                         ('Logistic Regression', lr_clf)]:
        scores = cross_val_score(model, X, y_cls, cv=skf, scoring='f1')
        results[name] = scores

    fig = styled_fig(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Box plot
    ax1 = fig.add_subplot(gs[0])
    bp = ax1.boxplot(list(results.values()), patch_artist=True,
                     medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], PALETTE[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xticklabels(list(results.keys()), rotation=15, color=COLORS['text'])
    style_ax(ax1, '5-Fold CV — F1-Score Distribution', 'Model', 'F1-Score')

    # Fold-by-fold line
    ax2 = fig.add_subplot(gs[1])
    folds = np.arange(1, 6)
    for (name, scores), color in zip(results.items(), PALETTE[:3]):
        ax2.plot(folds, scores, marker='o', color=color, lw=2, label=name)
        ax2.fill_between(folds, scores - 0.01, scores + 0.01, alpha=0.1, color=color)
    ax2.set_xticks(folds)
    ax2.set_xticklabels([f'Fold {i}' for i in folds], color=COLORS['text'])
    ax2.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    style_ax(ax2, '5-Fold CV — F1-Score per Fold', 'Fold', 'F1-Score')

    fig.suptitle('Cross-Validation Analysis', color=COLORS['text'],
                 fontsize=14, fontweight='bold')
    plt.savefig('static/graphs/cv_scores.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ CV scores saved")


# ══════════════════════════════════════════════════════════════════════════════
# 10. GRAPH 8 — Regression Analysis (Resolution Days)
# ══════════════════════════════════════════════════════════════════════════════

def plot_regression(rf_reg, X_test, y_reg_test):
    y_pred = rf_reg.predict(X_test)
    mae  = mean_absolute_error(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2   = r2_score(y_reg_test, y_pred)

    print(f"\nRegression Metrics — MAE: {mae:.2f}d | RMSE: {rmse:.2f}d | R²: {r2:.3f}")

    fig = styled_fig(figsize=(14, 5))
    ax1, ax2, ax3 = fig.subplots(1, 3)

    # Actual vs Predicted
    ax1.scatter(y_reg_test, y_pred, alpha=0.4, color=COLORS['primary'],
                s=15, edgecolors='none')
    lims = [min(y_reg_test.min(), y_pred.min()), max(y_reg_test.max(), y_pred.max())]
    ax1.plot(lims, lims, '--', color=COLORS['warning'], lw=2, label='Perfect Prediction')
    ax1.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    ax1.text(0.05, 0.90, f'R² = {r2:.3f}', transform=ax1.transAxes,
             color=COLORS['success'], fontsize=11, fontweight='bold')
    style_ax(ax1, 'Actual vs Predicted — Resolution Days',
             'Actual Days', 'Predicted Days')

    # Residuals
    residuals = y_reg_test - y_pred
    ax2.hist(residuals, bins=40, color=COLORS['purple'], alpha=0.75, edgecolor='none')
    ax2.axvline(0, color=COLORS['warning'], lw=2, linestyle='--')
    style_ax(ax2, 'Residuals Distribution', 'Residual (days)', 'Count')

    # Metrics bar
    metric_names = ['MAE', 'RMSE', 'R² × 10']
    metric_vals = [mae, rmse, r2 * 10]
    bars = ax3.bar(metric_names, metric_vals,
                   color=[COLORS['primary'], COLORS['warning'], COLORS['success']])
    for bar, v in zip(bars, [mae, rmse, r2]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{v:.3f}', ha='center', color=COLORS['text'], fontsize=10)
    style_ax(ax3, 'Regressor Performance Metrics', '', 'Value')

    fig.suptitle('Resolution Days Regressor — Random Forest',
                 color=COLORS['text'], fontsize=14, fontweight='bold')
    plt.savefig('static/graphs/regression_analysis.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ Regression analysis saved")


# ══════════════════════════════════════════════════════════════════════════════
# 11. GRAPH 9 — Department & District Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(df):
    pivot = df.pivot_table(values='sla_violated', index='category',
                           columns='priority', aggfunc='mean') * 100
    pivot = pivot.reindex(columns=['Low', 'Medium', 'High', 'Critical'])

    fig = styled_fig(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(pivot, annot=True, fmt='.1f', ax=ax,
                cmap='YlOrRd', linewidths=1.5, linecolor=COLORS['bg'],
                annot_kws={'size': 11, 'weight': 'bold'},
                cbar_kws={'label': 'SLA Violation Rate (%)', 'shrink': 0.8})
    ax.set_facecolor(COLORS['surface'])
    style_ax(ax, 'SLA Violation Rate (%) by Category & Priority', 'Priority', 'Category')
    ax.tick_params(colors=COLORS['text'])

    plt.savefig('static/graphs/heatmap.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ Heatmap saved")


# ══════════════════════════════════════════════════════════════════════════════
# 12. GRAPH 10 — Monthly Trend
# ══════════════════════════════════════════════════════════════════════════════

def plot_trends(df):
    df2 = df.copy()
    df2['submission_date'] = pd.to_datetime(df2['submission_date'])
    df2['month'] = df2['submission_date'].dt.to_period('M')

    monthly = df2.groupby('month').agg(
        total=('sla_violated', 'count'),
        violations=('sla_violated', 'sum'),
        escalations=('escalated', 'sum')
    ).reset_index()
    monthly['month_str'] = monthly['month'].astype(str)
    monthly['violation_rate'] = monthly['violations'] / monthly['total'] * 100

    fig = styled_fig(figsize=(14, 5))
    ax1, ax2 = fig.subplots(1, 2)

    # Total complaints + violations
    ax1.fill_between(range(len(monthly)), monthly['total'],
                     alpha=0.3, color=COLORS['primary'])
    ax1.plot(range(len(monthly)), monthly['total'],
             color=COLORS['primary'], lw=2, label='Total Complaints')
    ax1.fill_between(range(len(monthly)), monthly['violations'],
                     alpha=0.4, color=COLORS['danger'])
    ax1.plot(range(len(monthly)), monthly['violations'],
             color=COLORS['danger'], lw=2, label='SLA Violations')
    ax1.plot(range(len(monthly)), monthly['escalations'],
             color=COLORS['warning'], lw=2, linestyle='--', label='Escalations')

    tick_step = max(1, len(monthly) // 8)
    ax1.set_xticks(range(0, len(monthly), tick_step))
    ax1.set_xticklabels(monthly['month_str'].iloc[::tick_step], rotation=45,
                        color=COLORS['text'], fontsize=8)
    ax1.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    style_ax(ax1, 'Monthly Complaints, Violations & Escalations',
             'Month', 'Count')

    # Violation rate trend
    ax2.plot(range(len(monthly)), monthly['violation_rate'],
             color=COLORS['purple'], lw=2.5, marker='o', markersize=4)
    ax2.fill_between(range(len(monthly)), monthly['violation_rate'],
                     alpha=0.2, color=COLORS['purple'])
    ax2.axhline(monthly['violation_rate'].mean(), color=COLORS['warning'],
                lw=1.5, linestyle='--', label=f'Avg {monthly["violation_rate"].mean():.1f}%')
    ax2.set_xticks(range(0, len(monthly), tick_step))
    ax2.set_xticklabels(monthly['month_str'].iloc[::tick_step], rotation=45,
                        color=COLORS['text'], fontsize=8)
    ax2.legend(facecolor=COLORS['surface'], labelcolor=COLORS['text'])
    style_ax(ax2, 'Monthly SLA Violation Rate (%)', 'Month', 'Violation Rate (%)')

    fig.suptitle('Temporal Trends in Grievance Management',
                 color=COLORS['text'], fontsize=14, fontweight='bold')
    plt.savefig('static/graphs/trends.png', facecolor=COLORS['bg'])
    plt.close()
    print("✔ Trends graph saved")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SMART SLA PREDICTION SYSTEM — MODEL TRAINING")
    print("=" * 60)

    df, X, y_cls, y_reg, FEATURES, FEATURE_NAMES, encoders = load_and_preprocess()
    joblib.dump(encoders, 'models/encoders.pkl')

    # EDA
    plot_eda(df)
    plot_heatmap(df)
    plot_trends(df)

    # Train
    (rf_clf, gb_clf, lr_clf, rf_reg,
     X_train, X_test, y_cls_train, y_cls_test,
     y_reg_train, y_reg_test) = train_models(X, y_cls, y_reg, FEATURE_NAMES)

    # Evaluation Graphs
    plot_learning_curves(rf_clf, X, y_cls)
    plot_confusion_matrix(rf_clf, X_test, y_cls_test)

    models_dict = {
        'Random Forest': rf_clf,
        'Gradient Boosting': gb_clf,
        'Logistic Regression': lr_clf
    }
    plot_roc_curves(models_dict, X_test, y_cls_test)
    df_metrics = plot_model_comparison(models_dict, X_test, y_cls_test)
    plot_feature_importance(rf_clf, FEATURE_NAMES)
    plot_cv_scores(rf_clf, gb_clf, lr_clf, X, y_cls)
    plot_regression(rf_reg, X_test, y_reg_test)

    # Final report
    print("\n" + "=" * 60)
    print("  FINAL CLASSIFICATION REPORT — Random Forest")
    print("=" * 60)
    y_pred_final = rf_clf.predict(X_test)
    print(classification_report(y_cls_test, y_pred_final,
                                target_names=['On-Time', 'SLA Violated']))
    print("\nModel Comparison Summary:")
    print(df_metrics.round(4))
    print("\n✅ All models and graphs saved successfully!")
    print(f"   Graphs: static/graphs/ ({len(os.listdir('static/graphs'))} files)")
    print("   Models: models/sla_classifier.pkl, models/resolution_regressor.pkl")


if __name__ == '__main__':
    main()
