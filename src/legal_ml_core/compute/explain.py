import lime
import lime.lime_text
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def generate_lime_explanation(model_data, text, num_features=10):
    """
    Generate LIME explanation for text classification
    Returns feature importance for the given text
    """
    try:
        preprocessor = model_data["preprocessor"]
        vectorizer = model_data["vectorizer"]
        model = model_data["model"]
        label_encoder = model_data.get("label_encoder")

        # Create LIME explainer for text
        explainer = lime.lime_text.LimeTextExplainer(
            class_names=label_encoder.classes_.tolist() if label_encoder else None
        )

        # Define prediction function
        def predict_fn(texts):
            cleaned = [preprocessor.clean_text(t) for t in texts]
            X = vectorizer.transform(cleaned)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            else:
                # For models without predict_proba, return one-hot encoded predictions
                preds = model.predict(X)
                n_classes = len(label_encoder.classes_) if label_encoder else 2
                proba = np.zeros((len(preds), n_classes))
                for i, pred in enumerate(preds):
                    proba[i, pred] = 1.0
                return proba

        # Get prediction first so we know which label to ask LIME to explain —
        # explain_instance() only computes weights for the labels it's told
        # about (default: just label 1), so passing the actual predicted
        # class here is required, not optional. Fixed during extraction:
        # the original code omitted `labels=`, so `exp.as_list(label=pred_idx)`
        # below raised KeyError whenever the prediction wasn't class 1.
        cleaned_text = preprocessor.clean_text(text)
        X = vectorizer.transform([cleaned_text])
        prediction = model.predict(X)[0]
        pred_idx = int(prediction)

        if label_encoder:
            predicted_class = label_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = str(prediction)

        # Generate explanation
        exp = explainer.explain_instance(
            text, predict_fn, labels=[pred_idx], num_features=num_features, num_samples=1000
        )

        # Extract explanation data
        explanation = {
            "predicted_class": predicted_class,
            "features": [],
            "type": "lime_text",
        }

        # Get feature weights for predicted class
        for feature, weight in exp.as_list(label=pred_idx):
            explanation["features"].append(
                {"feature": feature, "weight": float(weight)}
            )

        return explanation

    except Exception as e:
        print(f"Error generating LIME explanation: {str(e)}")
        return {"error": str(e)}


def generate_shap_explanation(model_data, texts, max_samples=100):
    """
    Generate SHAP explanation for text classification
    Works best with tree-based models
    """
    try:
        preprocessor = model_data["preprocessor"]
        vectorizer = model_data["vectorizer"]
        model = model_data["model"]
        label_encoder = model_data.get("label_encoder")

        # Check if model is tree-based (better SHAP support)
        is_tree_based = isinstance(
            model, (DecisionTreeClassifier, RandomForestClassifier)
        )

        # Prepare texts
        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [preprocessor.clean_text(t) for t in texts]
        X = vectorizer.transform(cleaned_texts)

        # For tree-based models, use TreeExplainer
        if is_tree_based:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            # Format explanation for first text
            explanation = {
                "type": "shap_tree",
                "feature_importance": [],
                "base_value": float(explainer.expected_value[0])
                if isinstance(explainer.expected_value, np.ndarray)
                else float(explainer.expected_value),
            }

            # Get SHAP values for first prediction
            if len(shap_values.shape) == 3:  # Multi-class
                shap_vals = shap_values[0][0]  # First text, first class
            else:
                shap_vals = shap_values[0]

            # Get top features
            top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
            for idx in top_indices:
                if shap_vals[idx] != 0:  # Only non-zero contributions
                    explanation["feature_importance"].append(
                        {
                            "feature": feature_names[idx],
                            "shap_value": float(shap_vals[idx]),
                        }
                    )
        else:
            # For linear models, use LinearExplainer
            if isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)

                feature_names = vectorizer.get_feature_names_out()

                explanation = {
                    "type": "shap_linear",
                    "feature_importance": [],
                    "base_value": float(explainer.expected_value),
                }

                # Get SHAP values for first prediction
                shap_vals = (
                    shap_values[0] if len(shap_values.shape) == 2 else shap_values[0][0]
                )

                # Get top features
                top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
                for idx in top_indices:
                    if shap_vals[idx] != 0:
                        explanation["feature_importance"].append(
                            {
                                "feature": feature_names[idx],
                                "shap_value": float(shap_vals[idx]),
                            }
                        )
            else:
                # For other models, use KernelExplainer (slower but works for all models)
                # Use a background dataset
                background_size = min(max_samples, X.shape[0])
                background = shap.sample(X, background_size)

                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)
                    if hasattr(model, "predict_proba")
                    else model.predict(x),
                    background,
                )

                shap_values = explainer.shap_values(X[0:1])

                explanation = {
                    "type": "shap_kernel",
                    "feature_importance": [],
                    "base_value": float(explainer.expected_value[0])
                    if isinstance(explainer.expected_value, np.ndarray)
                    else float(explainer.expected_value),
                }

                feature_names = vectorizer.get_feature_names_out()
                shap_vals = (
                    shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                )

                top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
                for idx in top_indices:
                    if shap_vals[idx] != 0:
                        explanation["feature_importance"].append(
                            {
                                "feature": feature_names[idx],
                                "shap_value": float(shap_vals[idx]),
                            }
                        )

        # Add prediction
        prediction = model.predict(X[0:1])[0]
        if label_encoder:
            explanation["predicted_class"] = label_encoder.inverse_transform(
                [prediction]
            )[0]
        else:
            explanation["predicted_class"] = str(prediction)

        return explanation

    except Exception as e:
        print(f"Error generating SHAP explanation: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}
