import math
import pandas as pd
import numpy as np
import json
from flask import Blueprint, jsonify
from ..auth import jwt_required, get_jwt_identity
from ..models import Dataset
import os
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

_TEXT_COL_NAMES = {"text", "content", "document", "sentence", "summary", "abstract", "article"}

def _is_categorical(series, df_len):
    """True if an object-dtype column looks like a label/category, not free text."""
    if series.name.lower() in _TEXT_COL_NAMES:
        return False
    if df_len > 0 and series.nunique() / df_len > 0.5:
        return False
    return True


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        # NaN and infinity are not JSON: an all-blank column has no statistics.
        return float(obj) if math.isfinite(obj) else None
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


eda_bp = Blueprint("eda", __name__)


@eda_bp.route("/datasets/<dataset_id>/eda", methods=["GET"])
@jwt_required()
def get_eda_analysis(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Load the dataset
        file_path = dataset.file_path
        if not os.path.exists(file_path):
            return jsonify({"error": "Dataset file not found"}), 404

        # Read the dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Perform EDA analysis
        eda_results = perform_eda_analysis(df, dataset.task_type)

        # Convert numpy types to Python native types for JSON serialization
        eda_results = convert_numpy_types(eda_results)

        return jsonify(eda_results), 200

    except Exception as e:
        import traceback

        print(f"EDA Analysis Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@eda_bp.route("/datasets/<dataset_id>/eda/visualizations", methods=["GET"])
@jwt_required()
def get_eda_visualizations(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Load the dataset
        file_path = dataset.file_path
        if not os.path.exists(file_path):
            return jsonify({"error": "Dataset file not found"}), 404

        # Read the dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Generate visualizations
        visualizations = generate_visualizations(df, dataset.task_type)

        # Convert numpy types to Python native types for JSON serialization
        visualizations = convert_numpy_types(visualizations)

        return jsonify(visualizations), 200

    except Exception as e:
        import traceback

        print(f"EDA Visualizations Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def perform_eda_analysis(df, task_type):
    """Perform comprehensive EDA analysis on the dataset"""

    try:
        analysis = {}

        # Try each analysis function individually
        try:
            analysis["dataset_overview"] = get_dataset_overview(df)
        except Exception as e:
            print(f"Error in get_dataset_overview: {str(e)}")
            analysis["dataset_overview"] = {
                "shape": {"rows": len(df), "columns": len(df.columns)}
            }

        try:
            analysis["column_analysis"] = get_column_analysis(df)
        except Exception as e:
            print(f"Error in get_column_analysis: {str(e)}")
            analysis["column_analysis"] = {}

        try:
            analysis["data_quality"] = get_data_quality_analysis(df)
        except Exception as e:
            print(f"Error in get_data_quality_analysis: {str(e)}")
            analysis["data_quality"] = {"issues": [], "overall_quality_score": 0}

        try:
            analysis["statistical_summary"] = get_statistical_summary(df)
        except Exception as e:
            print(f"Error in get_statistical_summary: {str(e)}")
            analysis["statistical_summary"] = {
                "numeric_summary": {},
                "categorical_summary": {},
            }

        try:
            analysis["task_specific_analysis"] = get_task_specific_analysis(
                df, task_type
            )
        except Exception as e:
            print(f"Error in get_task_specific_analysis: {str(e)}")
            analysis["task_specific_analysis"] = {}

        try:
            analysis["insights"] = generate_insights(df, task_type)
        except Exception as e:
            print(f"Error in generate_insights: {str(e)}")
            analysis["insights"] = []

        return analysis
    except Exception as e:
        import traceback

        print(f"Error in perform_eda_analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Return a minimal analysis to prevent complete failure
        return {
            "dataset_overview": {
                "shape": {"rows": len(df), "columns": len(df.columns)}
            },
            "column_analysis": {},
            "data_quality": {"issues": [], "overall_quality_score": 0},
            "statistical_summary": {"numeric_summary": {}, "categorical_summary": {}},
            "task_specific_analysis": {},
            "insights": [
                {
                    "type": "error",
                    "title": "Analysis Error",
                    "message": str(e),
                    "priority": "high",
                }
            ],
        }


def get_dataset_overview(df):
    """Get basic dataset overview"""
    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "memory_usage": df.memory_usage(deep=True).sum(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "column_names": df.columns.tolist(),
    }


def get_column_analysis(df):
    """Analyze each column in detail"""
    column_analysis = {}

    for col in df.columns:
        col_data = df[col]
        analysis = {
            "dtype": str(col_data.dtype),
            "null_count": col_data.isnull().sum(),
            "null_percentage": (col_data.isnull().sum() / len(df)) * 100,
            "unique_count": col_data.nunique(),
            "unique_percentage": (col_data.nunique() / len(df)) * 100,
        }

        # Add data type specific analysis
        if col_data.dtype in ["int64", "float64"]:
            analysis.update(
                {
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std(),
                    "skewness": col_data.skew(),
                    "kurtosis": col_data.kurtosis(),
                }
            )
        elif col_data.dtype == "object":
            # Text analysis
            text_lengths = col_data.astype(str).str.len()
            try:
                most_common = col_data.value_counts().head(10).to_dict()
            except Exception as e:
                print(f"Error in value_counts for column {col}: {e}")
                most_common = {}

            analysis.update(
                {
                    "avg_text_length": text_lengths.mean(),
                    "min_text_length": text_lengths.min(),
                    "max_text_length": text_lengths.max(),
                    "most_common_values": most_common,
                }
            )

        column_analysis[col] = analysis

    return column_analysis


def get_data_quality_analysis(df):
    """Analyze data quality issues"""
    quality_issues = []

    # Check for missing values
    missing_data = df.isnull().sum()
    high_missing = missing_data[missing_data > len(df) * 0.1]
    if len(high_missing) > 0:
        quality_issues.append(
            {
                "type": "high_missing_values",
                "message": f"Columns with >10% missing values: {list(high_missing.index)}",
                "severity": "warning",
            }
        )

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        quality_issues.append(
            {
                "type": "duplicate_rows",
                "message": f"Found {duplicates} duplicate rows",
                "severity": "info",
            }
        )

    # Check for constant columns
    constant_cols = df.columns[df.nunique() <= 1].tolist()
    if constant_cols:
        quality_issues.append(
            {
                "type": "constant_columns",
                "message": f"Constant columns found: {constant_cols}",
                "severity": "warning",
            }
        )

    # Check for highly correlated numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append(
                        {
                            "col1": corr_matrix.columns[i],
                            "col2": corr_matrix.columns[j],
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )

        if high_corr_pairs:
            quality_issues.append(
                {
                    "type": "high_correlation",
                    "message": f"Highly correlated columns found: {len(high_corr_pairs)} pairs",
                    "severity": "info",
                    "details": high_corr_pairs,
                }
            )

    return {
        "issues": quality_issues,
        "overall_quality_score": calculate_quality_score(df, quality_issues),
    }


def calculate_quality_score(df, issues):
    """Calculate overall data quality score"""
    score = 100

    # Deduct points for issues
    for issue in issues:
        if issue["severity"] == "warning":
            score -= 15
        elif issue["severity"] == "info":
            score -= 5

    # Deduct points for missing values
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    score -= missing_percentage * 0.5

    return max(0, min(100, score))


def get_statistical_summary(df):
    """Get statistical summary of the dataset"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    summary = {"numeric_summary": {}, "categorical_summary": {}}

    # Numeric summary
    if len(numeric_cols) > 0:
        try:
            numeric_desc = df[numeric_cols].describe()
            summary["numeric_summary"] = numeric_desc.to_dict()
        except Exception as e:
            print(f"Error in numeric summary: {e}")
            summary["numeric_summary"] = {}

    # Categorical summary — skip free-text columns
    for col in categorical_cols:
        if not _is_categorical(df[col], len(df)):
            continue
        summary["categorical_summary"][col] = {
            "count": df[col].count(),
            "unique": df[col].nunique(),
            "top": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
            "freq": df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
        }

    return summary


def get_task_specific_analysis(df, task_type):
    """Perform task-specific analysis"""
    analysis = {}

    if task_type in ["classification", "document_classification"]:
        analysis = get_classification_analysis(df)
    elif task_type in ["ner", "named_entity_recognition"]:
        analysis = get_ner_analysis(df)
    elif task_type == "summarization":
        analysis = get_summarization_analysis(df)
    elif task_type in ["topic_modeling", "clustering"]:
        analysis = get_topic_analysis(df)
    elif task_type == "judgment_prediction":
        analysis = get_judgment_analysis(df)

    return analysis


def get_classification_analysis(df):
    """Analyze classification dataset"""
    analysis = {}

    # Find label column
    label_col = None
    for col in df.columns:
        if col.lower() in ["label", "labels", "class", "category", "target"]:
            label_col = col
            break

    if label_col:
        try:
            label_distribution = df[label_col].value_counts().to_dict()
            analysis["label_distribution"] = label_distribution
            analysis["class_balance"] = {
                "balanced": len(set(label_distribution.values())) == 1,
                "imbalance_ratio": max(label_distribution.values())
                / min(label_distribution.values())
                if label_distribution
                else 1,
            }
        except Exception as e:
            print(f"Error in label distribution analysis: {e}")
            analysis["label_distribution"] = {}
            analysis["class_balance"] = {"balanced": True, "imbalance_ratio": 1}

        # Find text column
        text_col = None
        for col in df.columns:
            if col.lower() in ["text", "content", "document", "sentence"]:
                text_col = col
                break

        if text_col:
            # Text length analysis by class
            df["text_length"] = df[text_col].astype(str).str.len()
            try:
                length_by_class = (
                    df.groupby(label_col)["text_length"]
                    .agg(["mean", "std", "min", "max"])
                    .to_dict()
                )
                analysis["text_length_by_class"] = length_by_class
            except Exception as e:
                print(f"Error in text length analysis: {e}")
                analysis["text_length_by_class"] = {}

            # Most distinctive terms per class: the words that appear in a
            # class far more often than elsewhere, so a user can see what each
            # label actually captures ("interactive keywords per class").
            try:
                analysis["keywords_by_class"] = _keywords_by_class(
                    df, text_col, label_col
                )
            except Exception as e:
                print(f"Error in keyword analysis: {e}")
                analysis["keywords_by_class"] = {}

    return analysis


def _keywords_by_class(df, text_col, label_col, top_n=10):
    """Return {class: [word, ...]} of terms over-represented in each class.

    Uses a per-class document frequency ratio against the class's overall share,
    which needs no extra dependencies and is stable on small legal datasets.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 1), max_features=2000, binary=True
    )
    texts = df[text_col].astype(str).fillna("")
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return {}

    terms = vectorizer.get_feature_names_out()
    labels = df[label_col].astype(str)
    total_docs = len(labels)
    if total_docs == 0:
        return {}

    result = {}
    for label in labels.unique():
        mask = (labels == label).to_numpy()
        class_docs = int(mask.sum())
        if class_docs == 0:
            continue
        present = matrix[mask].sum(axis=0).A1  # docs in class containing term
        overall = matrix.sum(axis=0).A1  # docs overall containing term
        with np.errstate(divide="ignore", invalid="ignore"):
            class_rate = present / class_docs
            overall_rate = overall / total_docs
            lift = np.where(overall_rate > 0, class_rate / overall_rate, 0)
        top_idx = np.argsort(lift)[::-1]
        keywords = [
            terms[i]
            for i in top_idx
            if present[i] >= max(2, int(0.1 * class_docs)) and lift[i] > 1.0
        ][:top_n]
        result[str(label)] = keywords
    return result


def get_ner_analysis(df):
    """Analyze NER dataset"""
    analysis = {}

    # Find text column
    text_col = None
    for col in df.columns:
        if col.lower() in ["text", "content", "document", "sentence"]:
            text_col = col
            break

    if text_col:
        # Text length analysis
        text_lengths = df[text_col].astype(str).str.len()
        analysis["text_length_stats"] = {
            "mean": text_lengths.mean(),
            "std": text_lengths.std(),
            "min": text_lengths.min(),
            "max": text_lengths.max(),
        }

        # Word count analysis
        word_counts = df[text_col].astype(str).str.split().str.len()
        analysis["word_count_stats"] = {
            "mean": word_counts.mean(),
            "std": word_counts.std(),
            "min": word_counts.min(),
            "max": word_counts.max(),
        }

    return analysis


def get_summarization_analysis(df):
    """Analyze summarization dataset"""
    analysis = {}

    # Find text and summary columns
    text_col = None
    summary_col = None

    for col in df.columns:
        if col.lower() in ["text", "content", "document", "article"]:
            text_col = col
        elif col.lower() in ["summary", "abstract", "summary_text"]:
            summary_col = col

    if text_col and summary_col:
        # Length analysis
        text_lengths = df[text_col].astype(str).str.len()
        summary_lengths = df[summary_col].astype(str).str.len()

        analysis["length_analysis"] = {
            "text_length": {
                "mean": text_lengths.mean(),
                "std": text_lengths.std(),
                "min": text_lengths.min(),
                "max": text_lengths.max(),
            },
            "summary_length": {
                "mean": summary_lengths.mean(),
                "std": summary_lengths.std(),
                "min": summary_lengths.min(),
                "max": summary_lengths.max(),
            },
            "compression_ratio": (summary_lengths / text_lengths).mean(),
        }

    return analysis


def get_topic_analysis(df):
    """Analyze topic modeling/clustering dataset"""
    analysis = {}

    # Find text column
    text_col = None
    for col in df.columns:
        if col.lower() in ["text", "content", "document", "article"]:
            text_col = col
            break

    if text_col:
        # Text length analysis
        text_lengths = df[text_col].astype(str).str.len()
        analysis["text_length_stats"] = {
            "mean": text_lengths.mean(),
            "std": text_lengths.std(),
            "min": text_lengths.min(),
            "max": text_lengths.max(),
        }

        # Word count analysis
        word_counts = df[text_col].astype(str).str.split().str.len()
        analysis["word_count_stats"] = {
            "mean": word_counts.mean(),
            "std": word_counts.std(),
            "min": word_counts.min(),
            "max": word_counts.max(),
        }

    return analysis


def get_judgment_analysis(df):
    """Analyze judgment prediction dataset"""
    analysis = {}

    # Find judgment column
    judgment_col = None
    for col in df.columns:
        if col.lower() in ["judgment", "outcome", "decision", "verdict"]:
            judgment_col = col
            break

    if judgment_col:
        try:
            judgment_distribution = df[judgment_col].value_counts().to_dict()
            analysis["judgment_distribution"] = judgment_distribution
            analysis["class_balance"] = {
                "balanced": len(set(judgment_distribution.values())) == 1,
                "imbalance_ratio": max(judgment_distribution.values())
                / min(judgment_distribution.values())
                if judgment_distribution
                else 1,
            }
        except Exception as e:
            print(f"Error in judgment distribution analysis: {e}")
            analysis["judgment_distribution"] = {}
            analysis["class_balance"] = {"balanced": True, "imbalance_ratio": 1}

    return analysis


def generate_insights(df, task_type):
    """Generate insights and recommendations"""
    insights = []

    # Data quality insights
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percentage > 5:
        insights.append(
            {
                "type": "data_quality",
                "title": "Missing Data Detected",
                "message": f"{missing_percentage:.1f}% of data is missing. Consider data imputation strategies.",
                "priority": "high",
            }
        )

    # Dataset size insights
    if len(df) < 100:
        insights.append(
            {
                "type": "dataset_size",
                "title": "Small Dataset",
                "message": "Dataset has fewer than 100 rows. Consider collecting more data for better model performance.",
                "priority": "medium",
            }
        )
    elif len(df) > 10000:
        insights.append(
            {
                "type": "dataset_size",
                "title": "Large Dataset",
                "message": "Large dataset detected. Consider using sampling or batch processing for faster training.",
                "priority": "low",
            }
        )

    # Task-specific insights
    if task_type in ["classification", "document_classification"]:
        label_col = None
        for col in df.columns:
            if col.lower() in ["label", "labels", "class", "category", "target"]:
                label_col = col
                break

        if label_col:
            label_distribution = df[label_col].value_counts()
            if len(label_distribution) > 10:
                insights.append(
                    {
                        "type": "classification",
                        "title": "Many Classes",
                        "message": f"Dataset has {len(label_distribution)} classes. Consider if all classes are necessary.",
                        "priority": "medium",
                    }
                )

            values = label_distribution.values
            imbalance_ratio = values.max() / values.min()
            if imbalance_ratio > 5:
                insights.append(
                    {
                        "type": "classification",
                        "title": "Class Imbalance",
                        "message": f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f}). Consider balancing techniques.",
                        "priority": "high",
                    }
                )

    return insights


def generate_visualizations(df, task_type):
    """Generate various visualizations for the dataset"""
    visualizations = {}

    # Basic visualizations
    visualizations["missing_data"] = create_missing_data_plot(df)
    visualizations["data_types"] = create_data_types_plot(df)

    # Numeric visualizations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        visualizations["numeric_distributions"] = create_numeric_distributions(
            df, numeric_cols
        )
        if len(numeric_cols) > 1:
            visualizations["correlation_heatmap"] = create_correlation_heatmap(
                df, numeric_cols
            )

    # NOTE: the generic "Top 10 values" categorical chart was removed — it was
    # meaningless for text-heavy legal datasets and added noise. The label
    # distribution and distinctive-terms-per-class views cover that need.

    # Task-specific visualizations
    if task_type in ["classification", "document_classification"]:
        visualizations.update(create_classification_visualizations(df))
    elif task_type in ["ner", "named_entity_recognition"]:
        visualizations.update(create_ner_visualizations(df))
    elif task_type == "summarization":
        visualizations.update(create_summarization_visualizations(df))

    return visualizations


def create_missing_data_plot(df):
    """Create missing data visualization"""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if len(missing_data) == 0:
        return None

    fig = go.Figure(
        data=[
            go.Bar(x=missing_data.index, y=missing_data.values, name="Missing Values")
        ]
    )

    fig.update_layout(
        title="Missing Data by Column",
        xaxis_title="Columns",
        yaxis_title="Missing Values Count",
        height=400,
    )

    return json.dumps(fig, cls=PlotlyJSONEncoder)


def create_data_types_plot(df):
    """Create data types distribution plot"""
    dtype_counts = df.dtypes.value_counts()

    fig = go.Figure(
        data=[go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values)]
    )

    fig.update_layout(title="Data Types Distribution", height=400)

    return json.dumps(fig, cls=PlotlyJSONEncoder)


def create_numeric_distributions(df, numeric_cols):
    """Create distributions for numeric columns"""
    if len(numeric_cols) == 0:
        return None

    # Limit to first 6 columns for performance
    cols_to_plot = numeric_cols[:6]

    fig = go.Figure()

    for col in cols_to_plot:
        fig.add_trace(go.Histogram(x=df[col].dropna(), name=col, opacity=0.7))

    fig.update_layout(
        title="Numeric Columns Distributions",
        xaxis_title="Value",
        yaxis_title="Frequency",
        barmode="overlay",
        height=400,
    )

    return json.dumps(fig, cls=PlotlyJSONEncoder)


def create_correlation_heatmap(df, numeric_cols):
    """Create correlation heatmap for numeric columns"""
    if len(numeric_cols) < 2:
        return None

    corr_matrix = df[numeric_cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
        )
    )

    fig.update_layout(title="Correlation Heatmap", height=400)

    return json.dumps(fig, cls=PlotlyJSONEncoder)


def create_classification_visualizations(df):
    """Create classification-specific visualizations"""
    visualizations = {}

    # Find label column
    label_col = None
    for col in df.columns:
        if col.lower() in ["label", "labels", "class", "category", "target"]:
            label_col = col
            break

    if label_col:
        # Class distribution
        label_distribution = df[label_col].value_counts()

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=label_distribution.index, values=label_distribution.values
                )
            ]
        )

        fig.update_layout(title="Class Distribution", height=400)

        visualizations["class_distribution"] = json.dumps(fig, cls=PlotlyJSONEncoder)

        # Find text column for length analysis
        text_col = None
        for col in df.columns:
            if col.lower() in ["text", "content", "document", "sentence"]:
                text_col = col
                break

        if text_col:
            # Text length by class. Box plots encode the middle 50% as the box
            # and the fences (1.5 x IQR) as the whiskers; text outside them is
            # unusually long or short, so label it explicitly instead of leaving
            # the fences undiscoverable.
            df["text_length"] = df[text_col].astype(str).str.len()

            fig = go.Figure()

            for label in df[label_col].unique():
                label_data = df[df[label_col] == label]["text_length"]
                fig.add_trace(
                    go.Box(
                        y=label_data,
                        name=str(label),
                        boxpoints="outliers",
                        hoveron="points+boxes",
                    )
                )

            fig.update_layout(
                title=(
                    "Text Length Distribution by Class "
                    "(box = middle 50% of texts; whiskers = 1.5×IQR; "
                    "dots = unusually long/short texts)"
                ),
                xaxis_title="Class",
                yaxis_title="Text length (characters)",
                height=400,
            )

            visualizations["text_length_by_class"] = json.dumps(
                fig, cls=PlotlyJSONEncoder
            )

    return visualizations


def create_ner_visualizations(df):
    """Create NER-specific visualizations"""
    visualizations = {}

    # Find text column
    text_col = None
    for col in df.columns:
        if col.lower() in ["text", "content", "document", "sentence"]:
            text_col = col
            break

    if text_col:
        # Text length distribution
        text_lengths = df[text_col].astype(str).str.len()

        fig = go.Figure(data=[go.Histogram(x=text_lengths, nbinsx=30)])

        fig.update_layout(
            title="Text Length Distribution",
            xaxis_title="Text Length (characters)",
            yaxis_title="Frequency",
            height=400,
        )

        visualizations["text_length_distribution"] = json.dumps(
            fig, cls=PlotlyJSONEncoder
        )

        # Word count distribution
        word_counts = df[text_col].astype(str).str.split().str.len()

        fig = go.Figure(data=[go.Histogram(x=word_counts, nbinsx=30)])

        fig.update_layout(
            title="Word Count Distribution",
            xaxis_title="Word Count",
            yaxis_title="Frequency",
            height=400,
        )

        visualizations["word_count_distribution"] = json.dumps(
            fig, cls=PlotlyJSONEncoder
        )

    return visualizations


def create_summarization_visualizations(df):
    """Create summarization-specific visualizations"""
    visualizations = {}

    # Find text and summary columns
    text_col = None
    summary_col = None

    for col in df.columns:
        if col.lower() in ["text", "content", "document", "article"]:
            text_col = col
        elif col.lower() in ["summary", "abstract", "summary_text"]:
            summary_col = col

    if text_col and summary_col:
        # Length comparison
        text_lengths = df[text_col].astype(str).str.len()
        summary_lengths = df[summary_col].astype(str).str.len()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=text_lengths,
                y=summary_lengths,
                mode="markers",
                name="Text vs Summary Length",
            )
        )

        fig.update_layout(
            title="Text Length vs Summary Length",
            xaxis_title="Text Length (characters)",
            yaxis_title="Summary Length (characters)",
            height=400,
        )

        visualizations["length_comparison"] = json.dumps(fig, cls=PlotlyJSONEncoder)

        # Compression ratio distribution
        compression_ratios = summary_lengths / text_lengths

        fig = go.Figure(data=[go.Histogram(x=compression_ratios, nbinsx=30)])

        fig.update_layout(
            title="Compression Ratio Distribution",
            xaxis_title="Compression Ratio (Summary/Text)",
            yaxis_title="Frequency",
            height=400,
        )

        visualizations["compression_ratio"] = json.dumps(fig, cls=PlotlyJSONEncoder)

    return visualizations
