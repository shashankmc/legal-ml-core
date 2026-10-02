from flask import Blueprint, request, jsonify, current_app
from ..auth import jwt_required, get_jwt_identity
from ..db import db
from ..models import Model, Dataset, ModelComparison, TrainingJob
import json
import plotly.graph_objs as go
import plotly.utils
import numpy as np
from sklearn.manifold import TSNE

visualization_bp = Blueprint("visualization", __name__)


@visualization_bp.route("/model-performance/<model_id>", methods=["GET"])
@jwt_required()
def get_model_performance(model_id):
    try:
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.status != "completed" or not model.metrics:
            return jsonify({"error": "Model metrics not available"}), 400

        metrics = model.metrics

        # Create performance visualization
        performance_data = {
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
        }

        # Create bar chart for metrics
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(performance_data.keys()),
                    y=list(performance_data.values()),
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                )
            ]
        )

        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
        )

        performance_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return (
            jsonify(
                {
                    "performance_data": performance_data,
                    "performance_chart": performance_chart,
                    "detailed_metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/confusion-matrix/<model_id>", methods=["GET"])
@jwt_required()
def get_confusion_matrix(model_id):
    try:
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.status != "completed":
            return jsonify({"error": "Model not ready"}), 400

        # Get confusion matrix data from Redis (stored during training)
        cm_key = f"confusion_matrix_{model_id}"
        cm_data = current_app.redis_client.get(cm_key)

        if not cm_data:
            return jsonify({"error": "Confusion matrix data not available"}), 404

        cm_info = json.loads(cm_data)
        cm = np.array(cm_info["matrix"])
        labels = cm_info["labels"]

        # Create confusion matrix heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
            )
        )

        fig.update_layout(
            title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual"
        )

        confusion_matrix_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return (
            jsonify(
                {
                    "confusion_matrix": cm.tolist(),
                    "labels": labels,
                    "confusion_matrix_chart": confusion_matrix_chart,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/feature-importance/<model_id>", methods=["GET"])
@jwt_required()
def get_feature_importance(model_id):
    try:
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.status != "completed":
            return jsonify({"error": "Model not ready"}), 400

        # Get feature importance data from Redis
        fi_key = f"feature_importance_{model_id}"
        fi_data = current_app.redis_client.get(fi_key)

        if not fi_data:
            return jsonify({"error": "Feature importance data not available"}), 404

        fi_info = json.loads(fi_data)
        features = fi_info["features"]
        importance = fi_info["importance"]

        # Sort by importance
        sorted_data = sorted(
            zip(features, importance), key=lambda x: x[1], reverse=True
        )
        top_features = sorted_data[:20]  # Top 20 features

        feature_names = [item[0] for item in top_features]
        feature_importance = [item[1] for item in top_features]

        # Create horizontal bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    y=feature_names,
                    x=feature_importance,
                    orientation="h",
                    marker_color="lightblue",
                )
            ]
        )

        fig.update_layout(
            title="Top 20 Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
        )

        feature_importance_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return (
            jsonify(
                {
                    "feature_importance": {
                        "features": feature_names,
                        "importance": feature_importance,
                    },
                    "feature_importance_chart": feature_importance_chart,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/clustering-visualization/<model_id>", methods=["GET"])
@jwt_required()
def get_clustering_visualization(model_id):
    try:
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.status != "completed":
            return jsonify({"error": "Model not ready"}), 400

        # Get clustering data from Redis
        cluster_key = f"clustering_data_{model_id}"
        cluster_data = current_app.redis_client.get(cluster_key)

        if not cluster_data:
            return jsonify({"error": "Clustering data not available"}), 404

        cluster_info = json.loads(cluster_data)
        embeddings = np.array(cluster_info["embeddings"])
        labels = cluster_info["labels"]

        # Reduce dimensionality for visualization
        if embeddings.shape[1] > 2:
            # Use t-SNE for dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings

        # Create scatter plot
        unique_labels = list(set(labels))
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        fig = go.Figure()

        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            fig.add_trace(
                go.Scatter(
                    x=embeddings_2d[mask, 0],
                    y=embeddings_2d[mask, 1],
                    mode="markers",
                    name=f"Cluster {label}",
                    marker=dict(color=colors[i % len(colors)], size=8),
                )
            )

        fig.update_layout(
            title="Clustering Visualization (t-SNE)",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
        )

        clustering_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return (
            jsonify(
                {
                    "clustering_data": {
                        "embeddings_2d": embeddings_2d.tolist(),
                        "labels": labels,
                    },
                    "clustering_chart": clustering_chart,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/topic-modeling/<model_id>", methods=["GET"])
@jwt_required()
def get_topic_modeling_visualization(model_id):
    try:
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.status != "completed":
            return jsonify({"error": "Model not ready"}), 400

        # Get topic modeling data from Redis
        topic_key = f"topic_data_{model_id}"
        topic_data = current_app.redis_client.get(topic_key)

        if not topic_data:
            return jsonify({"error": "Topic modeling data not available"}), 404

        topic_info = json.loads(topic_data)
        topics = topic_info["topics"]
        topic_weights = topic_info["topic_weights"]

        # Create topic visualization
        fig = go.Figure()

        for i, topic in enumerate(topics):
            fig.add_trace(
                go.Bar(
                    x=[f"Topic {i}"],
                    y=[topic_weights[i]],
                    name=f"Topic {i}",
                    text=[", ".join(topic[:5])],  # Top 5 words
                    textposition="auto",
                )
            )

        fig.update_layout(
            title="Topic Distribution",
            xaxis_title="Topics",
            yaxis_title="Weight",
            barmode="group",
        )

        topic_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return (
            jsonify(
                {
                    "topics": topics,
                    "topic_weights": topic_weights,
                    "topic_chart": topic_chart,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/model-comparison", methods=["POST"])
@jwt_required()
def create_model_comparison():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()

        if "name" not in data or "model_ids" not in data:
            return jsonify({"error": "name and model_ids are required"}), 400

        model_ids = data["model_ids"]
        if not isinstance(model_ids, list) or len(model_ids) < 2:
            return jsonify({"error": "At least 2 model IDs are required"}), 400

        # Verify all models belong to user and are completed
        models = []
        for model_id in model_ids:
            model = Model.query.filter_by(id=model_id, user_id=user_id).first()
            if not model:
                return jsonify({"error": f"Model {model_id} not found"}), 404
            if model.status != "completed":
                return jsonify({"error": f"Model {model_id} is not completed"}), 400
            models.append(model)

        # Create comparison metrics
        comparison_metrics = []
        for model in models:
            metrics = model.metrics or {}
            comparison_metrics.append(
                {
                    "model_id": model.id,
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                }
            )

        # Create comparison chart
        model_names = [m["model_name"] for m in comparison_metrics]
        metrics_names = ["accuracy", "precision", "recall", "f1_score"]

        fig = go.Figure()

        for metric in metrics_names:
            values = [m[metric] for m in comparison_metrics]
            fig.add_trace(
                go.Bar(name=metric.replace("_", " ").title(), x=model_names, y=values)
            )

        fig.update_layout(
            title="Model Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode="group",
        )

        comparison_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Save comparison
        comparison = ModelComparison(
            user_id=user_id,
            name=data["name"],
            model_ids=model_ids,
            comparison_metrics={
                "metrics": comparison_metrics,
                "chart": comparison_chart,
            },
        )

        db.session.add(comparison)
        db.session.commit()

        return (
            jsonify(
                {
                    "message": "Model comparison created successfully",
                    "comparison": comparison.to_dict(),
                    "comparison_chart": comparison_chart,
                }
            ),
            201,
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/comparisons", methods=["GET"])
@jwt_required()
def get_model_comparisons():
    try:
        user_id = get_jwt_identity()
        comparisons = (
            ModelComparison.query.filter_by(user_id=user_id)
            .order_by(ModelComparison.created_at.desc())
            .all()
        )

        return (
            jsonify(
                {"comparisons": [comparison.to_dict() for comparison in comparisons]}
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/comparisons/<comparison_id>", methods=["GET"])
@jwt_required()
def get_model_comparison(comparison_id):
    try:
        user_id = get_jwt_identity()
        comparison = ModelComparison.query.filter_by(
            id=comparison_id, user_id=user_id
        ).first()

        if not comparison:
            return jsonify({"error": "Comparison not found"}), 404

        return jsonify({"comparison": comparison.to_dict()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@visualization_bp.route("/parameter-comparison", methods=["GET"])
@jwt_required()
def get_parameter_comparison():
    """Experiment-monitoring endpoint: hyperparameters vs. a chosen final metric,
    across all of the user's completed training jobs of one model type — lets the
    frontend plot e.g. learning_rate vs. accuracy to see how a parameter affected
    performance across the user's own real runs. Purely retrospective (reads
    existing Model.parameters/TrainingJob.metrics) — no new schema needed.
    """
    try:
        user_id = get_jwt_identity()
        model_type = request.args.get("model_type")
        metric = request.args.get("metric", "accuracy")

        if not model_type:
            return jsonify({"error": "model_type is required"}), 400

        models = Model.query.filter_by(
            user_id=user_id, model_type=model_type, status="completed"
        ).all()

        points = []
        for model in models:
            job = (
                TrainingJob.query.filter_by(model_id=model.id, status="completed")
                .order_by(TrainingJob.completed_at.desc())
                .first()
            )
            if not job:
                continue
            metric_value = (job.metrics or {}).get(metric)
            if metric_value is None:
                metric_value = (model.metrics or {}).get(metric)
            if metric_value is None:
                continue
            points.append(
                {
                    "model_id": model.id,
                    "training_job_id": job.id,
                    "parameters": model.parameters or {},
                    "metric_value": metric_value,
                }
            )

        return jsonify({"model_type": model_type, "metric": metric, "points": points}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
