import uuid
from datetime import datetime

from .db import MLBase, db


class MLProject(MLBase):
    """Optional package-owned project grouping (increment 4, `ml_projects`).

    Hosts that have their own project table do not use this — they plug a
    ``PROJECT_VALIDATOR`` into the app config instead (see db_utils). This table
    exists so a self-contained ML-only host gets project grouping without
    bringing its own model. It is only consulted for reference validation when
    ``PROJECT_SCOPE="package_table"``.
    """

    __tablename__ = "ml_projects"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(36), nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Dataset(MLBase):
    __tablename__ = "datasets"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(36), nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(500), nullable=False)
    row_count = db.Column(db.Integer, nullable=False)
    columns = db.Column(db.JSON)
    task_type = db.Column(
        db.String(50), nullable=False
    )  # classification, ner, summarization, etc.
    project_id = db.Column(db.String(36))
    dataset_type = db.Column(
        db.String(50), default="unlabeled"
    )  # 'unlabeled', 'training', 'test'
    annotation_progress = db.Column(db.Integer, default=0)  # 0-100
    text_column = db.Column(
        db.String(100)
    )  # Name of column containing text to annotate
    label_column = db.Column(db.String(100))  # Name of column to store labels
    labels = db.Column(
        db.JSON
    )  # Custom labels for annotation (e.g., ['Positive', 'Negative', 'Neutral'])
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    models = db.relationship(
        "Model", backref="dataset", lazy=True, cascade="all, delete-orphan"
    )
    annotation_tasks = db.relationship(
        "AnnotationTask", backref="dataset", lazy=True, cascade="all, delete-orphan"
    )

    def to_dict(self):
        annotation_count = len(self.annotation_tasks) if self.annotation_tasks else 0
        annotated_count = (
            sum(1 for task in self.annotation_tasks if task.status == "annotated")
            if self.annotation_tasks
            else 0
        )

        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "row_count": self.row_count,
            "columns": self.columns,
            "task_type": self.task_type,
            "project_id": self.project_id,
            "dataset_type": self.dataset_type,
            "annotation_progress": int(
                (annotated_count / annotation_count * 100)
                if annotation_count > 0
                else 0
            ),
            "annotated_count": annotated_count,
            "total_count": annotation_count,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Model(MLBase):
    __tablename__ = "models"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(36), nullable=False, index=True)
    dataset_id = db.Column(db.String(36), db.ForeignKey("datasets.id"), nullable=False)
    project_id = db.Column(db.String(36), nullable=True)
    name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(
        db.String(100), nullable=False
    )  # svm, decision_tree, legalbert, etc.
    parameters = db.Column(db.JSON)
    status = db.Column(
        db.String(50), default="pending"
    )  # pending, training, completed, failed
    file_path = db.Column(db.String(500))
    metrics = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    training_jobs = db.relationship(
        "TrainingJob", backref="model", lazy=True, cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
            "name": self.name,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "status": self.status,
            "file_path": self.file_path,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class TrainingJob(MLBase):
    __tablename__ = "training_jobs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(36), nullable=False, index=True)
    model_id = db.Column(db.String(36), db.ForeignKey("models.id"), nullable=False)
    status = db.Column(
        db.String(50), default="pending"
    )  # pending, running, completed, failed, cancelled
    progress = db.Column(db.Integer, default=0)
    metrics = db.Column(db.JSON)
    error_message = db.Column(db.Text)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    metric_points = db.relationship(
        "TrainingMetricPoint",
        backref="training_job",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="TrainingMetricPoint.step",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "model_id": self.model_id,
            "status": self.status,
            "progress": self.progress,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "created_at": self.created_at.isoformat(),
        }


class ModelComparison(MLBase):
    __tablename__ = "model_comparisons"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(36), nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    model_ids = db.Column(db.JSON)  # Array of model IDs
    comparison_metrics = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "model_ids": self.model_ids,
            "comparison_metrics": self.comparison_metrics,
            "created_at": self.created_at.isoformat(),
        }


class AnnotationTask(MLBase):
    __tablename__ = "annotation_tasks"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = db.Column(db.String(36), db.ForeignKey("datasets.id"), nullable=False)
    item_index = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    annotation_label = db.Column(db.String(1000))  # JSON string for multi-label/NER
    status = db.Column(
        db.String(50), default="pending"
    )  # 'pending', 'annotated', 'skipped'
    label_source = db.Column(db.String(20))  # 'human', 'llm_confirmed', 'llm_bulk'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    suggestions = db.relationship(
        "LLMSuggestion", backref="task", lazy=True, cascade="all, delete-orphan"
    )

    # Unique constraint on dataset_id and item_index
    __table_args__ = (
        db.UniqueConstraint("dataset_id", "item_index", name="uq_dataset_item"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "item_index": self.item_index,
            "text": self.text,
            "annotation_label": self.annotation_label,
            "status": self.status,
            "label_source": self.label_source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class LLMCredential(MLBase):
    """A user's own LLM key (BYOK). ``encrypted_key`` is Fernet ciphertext."""

    __tablename__ = "llm_credentials"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(255), nullable=False, unique=True)
    encrypted_key = db.Column(db.Text, nullable=False)
    base_url = db.Column(db.String(500), nullable=False)
    model = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class LLMAcknowledgement(MLBase):
    """Records that an owner accepted sending a scope's text to a destination.

    ``scope_id`` is the dataset's project, or the dataset itself when it has none.
    """

    __tablename__ = "llm_acknowledgements"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(255), nullable=False)
    scope_id = db.Column(db.String(255), nullable=False)
    destination = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint(
            "owner_id", "scope_id", "destination", name="uq_llm_acknowledgement"
        ),
    )

    def to_dict(self):
        return {
            "destination": self.destination,
            "created_at": self.created_at.isoformat(),
        }


class LLMPrompt(MLBase):
    """One version of a dataset's labelling prompt. Versions are never edited."""

    __tablename__ = "llm_prompts"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = db.Column(
        db.String(36), db.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    version = db.Column(db.Integer, nullable=False)
    template = db.Column(db.Text, nullable=False)
    definitions = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("dataset_id", "version", name="uq_llm_prompt_version"),
    )

    def to_dict(self):
        return {
            "version": self.version,
            "template": self.template,
            "definitions": self.definitions or {},
            "created_at": self.created_at.isoformat(),
        }


class LLMSuggestion(MLBase):
    """An LLM's proposed label. It is not a label until a human accepts it."""

    __tablename__ = "llm_suggestions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = db.Column(
        db.String(36),
        db.ForeignKey("annotation_tasks.id", ondelete="CASCADE"),
        nullable=False,
    )
    model = db.Column(db.String(255), nullable=False)
    prompt_version = db.Column(db.Integer, nullable=False)
    label = db.Column(db.String(1000), nullable=False)
    rationale = db.Column(db.Text)
    status = db.Column(
        db.String(20), default="pending", nullable=False
    )  # 'pending', 'accepted', 'corrected', 'rejected', 'bulk_accepted'
    bulk_agreement = db.Column(db.JSON)  # the agreement check that unlocked bulk accept
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        db.UniqueConstraint(
            "task_id", "model", "prompt_version", name="uq_llm_suggestion"
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "label": self.label,
            "rationale": self.rationale,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class LLMUsage(MLBase):
    """Shared-key tokens per owner per month, for the default usage meter."""

    __tablename__ = "llm_usage"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = db.Column(db.String(255), nullable=False)
    period = db.Column(db.String(7), nullable=False)  # 'YYYY-MM'
    tokens = db.Column(db.Integer, default=0, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("owner_id", "period", name="uq_llm_usage_period"),
    )


class TrainingMetricPoint(MLBase):
    """A single logged metric value during a training run.

    For the HuggingFace `Trainer` path (LegalBERT/RoBERTa) this is a genuine
    per-logging-step point (loss, accuracy, ...). For the classical sklearn
    algorithms, which fit synchronously with no epoch loop, only coarse phase
    markers are logged (metric_name like "training_started"/"training_done") —
    see legal_ml_core.compute.tasks.MetricLogger.
    """

    __tablename__ = "training_metric_points"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    training_job_id = db.Column(
        db.String(36), db.ForeignKey("training_jobs.id"), nullable=False
    )
    step = db.Column(db.Integer, nullable=False)
    epoch = db.Column(db.Float, nullable=True)
    metric_name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.Index(
            "ix_training_metric_points_job_metric_step",
            "training_job_id",
            "metric_name",
            "step",
        ),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "training_job_id": self.training_job_id,
            "step": self.step,
            "epoch": self.epoch,
            "metric_name": self.metric_name,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
        }
