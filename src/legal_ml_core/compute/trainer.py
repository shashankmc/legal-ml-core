import json

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import TrainerCallback

from .preprocessing import TextPreprocessor


class TrainingCancelled(Exception):
    """Raised when a Stop request (Redis `job_cancel_<id>`) is observed."""


class TrainingMetricCallback(TrainerCallback):
    """Feeds HuggingFace Trainer's on_log events into a MetricLogger, giving the
    LegalBERT/RoBERTa path genuine per-step training curves (loss, eval_accuracy,
    etc.) — unlike the classical sklearn algorithms below, which fit synchronously
    with no epoch loop and only get coarse phase markers (logged by the caller in
    legal_ml_core.compute.tasks, not here).
    """

    def __init__(self, metric_logger):
        self.metric_logger = metric_logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self.metric_logger:
            return
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.metric_logger.log(metric_name=key, value=value, epoch=state.epoch)


class TrainingCancelCallback(TrainerCallback):
    """Ends HF training at the first step boundary after a Stop request, i.e.
    well within one logging step (logging_steps=10). Polled every step: one
    Redis EXISTS per optimizer step is negligible next to a forward/backward
    pass."""

    def __init__(self, cancel_check):
        self.cancel_check = cancel_check
        self.cancelled = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.cancel_check():
            self.cancelled = True
            control.should_training_stop = True
        return control


class MLModelTrainer:
    def __init__(self, redis_client, metric_logger=None, cancel_check=None):
        self.redis_client = redis_client
        self.metric_logger = metric_logger
        # Zero-arg callable returning True once Stop was requested.
        self.cancel_check = cancel_check
        self.preprocessor = None
        self.vectorizer = None
        self.model = None
        self.label_encoder = None

    def preprocess_data(self, df, task_type, preprocessing_options):
        self.preprocessor = TextPreprocessor(preprocessing_options)

        # Get text column (assume first column is text)
        text_column = df.columns[0]
        texts = df[text_column].apply(self.preprocessor.clean_text)

        # Handle different task types
        if task_type in [
            "classification",
            "document_classification",
            "judgment_prediction",
        ]:
            # Assume second column is label
            if len(df.columns) < 2:
                raise ValueError("Classification datasets must have at least 2 columns")

            label_column = df.columns[1]
            labels = df[label_column]

            # Encode labels
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)

            return texts, encoded_labels, labels.unique().tolist()

        elif task_type in ["ner", "named_entity_recognition"]:
            # For NER, we'll do entity extraction
            return texts, None, None

        elif task_type in ["summarization"]:
            # Assume second column is summary
            if len(df.columns) < 2:
                raise ValueError("Summarization datasets must have at least 2 columns")

            summary_column = df.columns[1]
            summaries = df[summary_column]

            return texts, summaries, None

        elif task_type in ["topic_modeling", "clustering"]:
            return texts, None, None

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def create_vectorizer(self, method, max_features, ngram_range):
        if method == "tfidf":
            return TfidfVectorizer(
                max_features=max_features,
                ngram_range=tuple(ngram_range),
                stop_words="english",
            )
        elif method == "count":
            return CountVectorizer(
                max_features=max_features,
                ngram_range=tuple(ngram_range),
                stop_words="english",
            )
        else:
            raise ValueError(f"Unsupported vectorization method: {method}")

    def train_classification_model(self, texts, labels, model_type, parameters):
        # Create vectorizer
        vectorizer_params = parameters.get("vectorizer", {})
        self.vectorizer = self.create_vectorizer(
            vectorizer_params.get("method", "tfidf"),
            vectorizer_params.get("max_features", 5000),
            vectorizer_params.get("ngram_range", [1, 2]),
        )

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # "none" is the catalogue's name for sklearn's default of equal weights.
        class_weight = parameters.get("class_weight")
        class_weight = None if class_weight == "none" else class_weight

        # Create model based on type
        if model_type == "svm":
            # SVM scores are not probabilities: SVC without probability=True has
            # no predict_proba at all. Enable it so downstream calibration (and
            # the confidence display) has something to work with.
            self.model = SVC(
                kernel=parameters.get("kernel", "rbf"),
                C=parameters.get("C", 1.0),
                gamma=parameters.get("gamma", "scale"),
                degree=parameters.get("degree", 3),
                probability=True,
                class_weight=class_weight,
                random_state=42,
            )
        elif model_type == "decision_tree":
            self.model = DecisionTreeClassifier(
                max_depth=parameters.get("max_depth"),
                min_samples_split=parameters.get("min_samples_split", 2),
                min_samples_leaf=parameters.get("min_samples_leaf", 1),
                criterion=parameters.get("criterion", "gini"),
                class_weight=class_weight,
                random_state=42,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=parameters.get("n_estimators", 100),
                max_depth=parameters.get("max_depth"),
                min_samples_split=parameters.get("min_samples_split", 2),
                min_samples_leaf=parameters.get("min_samples_leaf", 1),
                class_weight=class_weight,
                random_state=42,
            )
        elif model_type == "naive_bayes":
            self.model = MultinomialNB(
                alpha=parameters.get("alpha", 1.0),
                fit_prior=parameters.get("fit_prior", True),
            )
        elif model_type == "logistic_regression":
            penalty = parameters.get("penalty", "l2")
            solver = parameters.get("solver", "lbfgs")
            # Without a ratio, scikit-learn 1.8 and later fit elasticnet as plain l2.
            mix = {"l1_ratio": parameters.get("l1_ratio", 0.5)} if penalty == "elasticnet" else {}
            self.model = LogisticRegression(
                C=parameters.get("C", 1.0),
                # "none" is the catalogue's name for scikit-learn's None.
                penalty=None if penalty == "none" else penalty,
                solver=solver,
                class_weight=class_weight,
                random_state=42,
                **mix,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model
        self.model.fit(X_train, y_train)

        # Calibrate probabilities on a held-out split so displayed confidence is
        # closer to a true probability. Not all classifiers expose probabilities
        # in a comparable way (naive_bayes is already roughly calibrated), and
        # calibration can fail on tiny/degenerate datasets — so it is best-effort
        # and never blocks training.
        #
        # A balanced model skips it: sigmoid calibration refits scores to the
        # label frequencies of the unweighted data, which puts back the very bias
        # balancing removed and hands rare labels back to the common ones. Its
        # probabilities are therefore not calibrated, and the metrics say so.
        calibrated = False
        # Equal-size CalibratedClassifierCV folds need at least 2 samples per class.
        n_classes = len(set(y_train))
        if class_weight is None and hasattr(self.model, "predict_proba") and n_classes > 1:
            try:
                self.model = CalibratedClassifierCV(
                    self.model, method="sigmoid", cv=3
                )
                self.model.fit(X_train, y_train)
                calibrated = True
            except Exception:
                # Fall back to the uncalibrated estimator.
                self.model = self.model.estimator if hasattr(self.model, "estimator") else self.model
                calibrated = False

        # Evaluate model
        y_pred = self.model.predict(X_test)

        label_names = self.label_encoder.classes_.tolist() if self.label_encoder else None
        # A class of two can land wholly on the training side of the split.
        every_class = list(range(len(label_names))) if label_names else None
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=every_class).tolist(),
            "confusion_matrix_labels": label_names,
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0,
                labels=every_class, target_names=label_names
            ),
            "probabilities_calibrated": calibrated,
        }

        # Also store in Redis for backward compatibility
        cm = confusion_matrix(y_test, y_pred)
        self.redis_client.setex(
            f'confusion_matrix_{parameters["model_id"]}',
            86400,  # 24 hours
            json.dumps(
                {"matrix": cm.tolist(), "labels": self.label_encoder.classes_.tolist()}
            ),
        )

        return metrics

    def train_clustering_model(self, texts, model_type, parameters):
        # Create vectorizer
        vectorizer_params = parameters.get("vectorizer", {})
        self.vectorizer = self.create_vectorizer(
            vectorizer_params.get("method", "tfidf"),
            vectorizer_params.get("max_features", 5000),
            vectorizer_params.get("ngram_range", [1, 2]),
        )

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)

        points = X.toarray()

        # Create model based on type
        if model_type == "kmeans":
            self.model = KMeans(
                n_clusters=parameters.get("n_clusters", 8),
                init=parameters.get("init", "k-means++"),
                max_iter=parameters.get("max_iter", 300),
                random_state=parameters.get("random_state", 42),
            )
        elif model_type == "hierarchical":
            distance = parameters.get("affinity", "euclidean")
            # A document with no usable words has no angle, and metric="cosine"
            # refuses the whole run. Precomputed distances keep it, far from the rest.
            if distance == "cosine":
                points = cosine_distances(points)
                distance = "precomputed"
            self.model = AgglomerativeClustering(
                n_clusters=parameters.get("n_clusters", 8),
                linkage=parameters.get("linkage", "ward"),
                # scikit-learn renamed AgglomerativeClustering's `affinity` kwarg
                # to `metric` (removed in newer releases) — the external parameter
                # name in AVAILABLE_MODELS/the API stays "affinity" for callers.
                metric=distance,
            )
        elif model_type == "dbscan":
            self.model = DBSCAN(
                eps=parameters.get("eps", 0.5),
                min_samples=parameters.get("min_samples", 5),
                metric=parameters.get("metric", "euclidean"),
            )
        else:
            raise ValueError(f"Unsupported clustering model type: {model_type}")

        # Train model
        labels = self.model.fit_predict(points)

        # Store clustering data for visualization
        embeddings = X.toarray()
        self.redis_client.setex(
            f'clustering_data_{parameters["model_id"]}',
            86400,  # 24 hours
            json.dumps({"embeddings": embeddings.tolist(), "labels": labels.tolist()}),
        )

        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Calculate silhouette score
        silhouette = 0.0
        if n_clusters > 1 and len(set(labels)) > 1:
            try:
                silhouette = silhouette_score(X.toarray(), labels)
            except Exception:
                silhouette = 0.0

        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette_score": silhouette,
        }

        return metrics

    def train_topic_modeling(self, texts, model_type, parameters):
        # Create vectorizer
        vectorizer_params = parameters.get("vectorizer", {})
        self.vectorizer = CountVectorizer(
            max_features=vectorizer_params.get("max_features", 1000),
            ngram_range=tuple(vectorizer_params.get("ngram_range", [1, 2])),
            stop_words="english",
        )

        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)

        # Create model based on type
        if model_type == "lda":
            self.model = LatentDirichletAllocation(
                n_components=parameters.get("n_components", 10),
                max_iter=parameters.get("max_iter", 100),
                learning_decay=parameters.get("learning_decay", 0.7),
                random_state=parameters.get("random_state", 42),
            )
        elif model_type == "nmf":
            # scikit-learn split NMF's single `alpha` kwarg into `alpha_W`/
            # `alpha_H` (removed in newer releases) — apply the same strength
            # to both, matching the old single-`alpha` behavior. The external
            # parameter name in AVAILABLE_MODELS/the API stays "alpha".
            nmf_alpha = parameters.get("alpha", 0.0)
            self.model = NMF(
                n_components=parameters.get("n_components", 10),
                max_iter=parameters.get("max_iter", 1000),
                alpha_W=nmf_alpha,
                alpha_H=nmf_alpha,
                l1_ratio=parameters.get("l1_ratio", 0.0),
                random_state=parameters.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unsupported topic modeling type: {model_type}")

        # Train model
        self.model.fit(X)
        if not self.model.components_.any():
            raise ValueError(
                "The regularisation (alpha) removed every topic on this Dataset. Choose a smaller alpha."
            )

        # Get topics
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)

        # Get topic weights for documents
        topic_weights = self.model.transform(X)

        # Store topic data
        self.redis_client.setex(
            f'topic_data_{parameters["model_id"]}',
            86400,  # 24 hours
            json.dumps(
                {"topics": topics, "topic_weights": topic_weights.mean(axis=0).tolist()}
            ),
        )

        metrics = {
            "n_topics": parameters.get("n_components", 10),
            "perplexity": getattr(self.model, "perplexity", lambda x: 0)(X)
            if hasattr(self.model, "perplexity")
            else 0,
            # Persisted (not just Redis) so the UI can suggest human-readable
            # names for topics long after the 24h Redis cache expires.
            "topic_keywords": topics,
        }

        return metrics

    def train_transformer_model(self, texts, labels, model_type, parameters):
        import torch
        from torch.utils.data import Dataset as TorchDataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        defaults = {
            "legalbert": "nlpaueb/legal-bert-base-uncased",
            "roberta": "roberta-base",
        }
        model_name = parameters.get("model_name", defaults[model_type])
        max_length = int(parameters.get("max_length", 256))
        learning_rate = float(parameters.get("learning_rate", 2e-5))
        batch_size = int(parameters.get("batch_size", 16))
        epochs = int(parameters.get("epochs", 3))

        texts_list = list(texts)
        X_train, X_test, y_train, y_test = train_test_split(
            texts_list, list(labels), test_size=0.2, random_state=42, stratify=labels
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        class TextDataset(TorchDataset):
            def __init__(self, texts, labels):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True, max_length=max_length
                )
                self.labels = labels

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        n_labels = len(set(labels))
        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)

        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_labels
        )

        output_dir = parameters.get(
            "output_dir", f"/tmp/transformer_{parameters['model_id']}"
        )
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            use_cpu=not torch.cuda.is_available(),
        )

        def compute_metrics(eval_pred):
            logits, eval_labels = eval_pred
            preds = logits.argmax(-1)
            return {
                "accuracy": accuracy_score(eval_labels, preds),
                "precision": precision_score(eval_labels, preds, average="weighted", zero_division=0),
                "recall": recall_score(eval_labels, preds, average="weighted", zero_division=0),
                "f1_score": f1_score(eval_labels, preds, average="weighted", zero_division=0),
            }

        callbacks = [TrainingMetricCallback(self.metric_logger)] if self.metric_logger else []
        cancel_callback = None
        if self.cancel_check:
            cancel_callback = TrainingCancelCallback(self.cancel_check)
            callbacks.append(cancel_callback)

        hf_trainer = Trainer(
            model=hf_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        hf_trainer.train()
        # Stopped: skip evaluate/predict and keep no model (D4/OV-E4).
        if (cancel_callback and cancel_callback.cancelled) or (
            self.cancel_check and self.cancel_check()
        ):
            raise TrainingCancelled()
        eval_results = hf_trainer.evaluate()

        predictions = hf_trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(-1)

        label_names = self.label_encoder.classes_.tolist() if self.label_encoder else None
        # A class of two can land wholly on the training side of the split.
        every_class = list(range(len(label_names))) if label_names else None
        metrics = {
            "accuracy": float(eval_results.get("eval_accuracy", accuracy_score(y_test, y_pred))),
            "precision": float(eval_results.get("eval_precision", precision_score(y_test, y_pred, average="weighted", zero_division=0))),
            "recall": float(eval_results.get("eval_recall", recall_score(y_test, y_pred, average="weighted", zero_division=0))),
            "f1_score": float(eval_results.get("eval_f1_score", f1_score(y_test, y_pred, average="weighted", zero_division=0))),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=every_class).tolist(),
            "confusion_matrix_labels": label_names,
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0,
                labels=every_class, target_names=label_names
            ),
        }

        self.model = hf_model
        self.tokenizer = tokenizer
        return metrics

    def save_model(self, model_path):
        model_data = {
            "preprocessor": self.preprocessor,
            "vectorizer": self.vectorizer,
            "model": self.model,
            "label_encoder": self.label_encoder,
            "tokenizer": getattr(self, "tokenizer", None),
        }
        joblib.dump(model_data, model_path)
