"""Dataset rows becoming annotation work, and saved work becoming labels."""
import logging

import pandas as pd
from sqlalchemy.exc import IntegrityError

from .db import db
from .models import AnnotationTask, Dataset

logger = logging.getLogger(__name__)

# The column holding each task type's answer, as required at upload. Task types
# left out (topic modeling, clustering) have no labels, so no annotation work.
LABEL_COLUMNS = {
    "classification": "label",
    "document_classification": "label",
    "ner": "entities",
    "named_entity_recognition": "entities",
    "summarization": "summary",
    "judgment_prediction": "judgment",
}

# Task types whose trainer learns from the label column. NER training ignores
# its entities column, so saved NER spans have nothing to feed yet.
TRAINED_ON_LABELS = {"classification", "document_classification", "judgment_prediction"}

# Task types offered on the annotation screen. Summarization is left out: no
# trainer reads the summary column, so a summary written there would go nowhere.
QUEUED_TASK_TYPES = TRAINED_ON_LABELS | {"ner", "named_entity_recognition"}

# Key on the text column's entry in ``Dataset.columns``: how many rows of the file
# need a label. The host reads it too, so both count what the queue would offer.
ROWS_TO_LABEL = "rows_to_label"


def _find_column(df, name):
    """The column of ``df`` called ``name``, ignoring case, else ``None``."""
    for column in df.columns:
        if str(column).lower() == name.lower():
            return column
    return None


def _is_blank(values):
    return values.isna() | values.astype(str).str.strip().eq("")


def _read_rows(file_path):
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    if file_path.endswith(".json"):
        return pd.read_json(file_path)
    return pd.read_csv(file_path)


def _rows_needing_label(df, dataset):
    """Which rows need a label: those with text and a blank label.

    ``None`` when the dataset has no labels to give, or no text to read.
    """
    label_name = dataset.label_column or LABEL_COLUMNS.get(dataset.task_type)
    text_column = _find_column(df, dataset.text_column or "text")
    if not label_name or text_column is None:
        return None

    label_column = _find_column(df, label_name)
    needs_label = ~_is_blank(df[text_column])
    if label_column is not None:
        needs_label &= _is_blank(df[label_column])
    return needs_label


def record_rows_to_label(dataset, df):
    """Note in ``dataset.columns`` how many rows of ``df`` need a label; caller commits."""
    needs_label = _rows_needing_label(df, dataset)
    if needs_label is None:
        return

    text_name = (dataset.text_column or "text").lower()
    dataset.columns = [
        {**column, ROWS_TO_LABEL: int(needs_label.sum())}
        if str(column["name"]).lower() == text_name
        else column
        for column in dataset.columns or []
    ]


def _recorded_rows_to_label(dataset):
    for column in dataset.columns or []:
        if ROWS_TO_LABEL in column:
            return column[ROWS_TO_LABEL]
    return None


def create_annotation_tasks(dataset):
    """Add one pending task per unlabelled row of ``dataset``; caller commits."""
    if dataset.task_type not in QUEUED_TASK_TYPES:
        return

    try:
        df = _read_rows(dataset.file_path)
    except Exception as e:
        logger.warning("dataset %s: cannot read rows to annotate: %s", dataset.id, e)
        return

    needs_label = _rows_needing_label(df, dataset)
    if needs_label is None:
        return
    record_rows_to_label(dataset, df)

    text_column = _find_column(df, dataset.text_column or "text")
    for position in needs_label.to_numpy().nonzero()[0]:
        db.session.add(
            AnnotationTask(
                dataset_id=dataset.id,
                item_index=int(position),
                text=str(df[text_column].iloc[position]),
                status="pending",
            )
        )


def ensure_project_tasks(project_id, owner_id):
    """Create the missing tasks of ``owner_id``'s datasets on ``project_id``.

    Runs when the queue is read rather than when a dataset is linked: linking is
    the host's route, in several places, and this also picks up datasets linked
    before tasks existed. A dataset that already has tasks is left alone, so
    saved and skipped work is never touched, and so is one with no rows to label,
    so its file is not read again.
    """
    datasets = Dataset.query.filter_by(project_id=project_id, owner_id=str(owner_id)).all()
    for dataset in datasets:
        if dataset.annotation_tasks or _recorded_rows_to_label(dataset) == 0:
            continue
        create_annotation_tasks(dataset)
        try:
            db.session.commit()
        except IntegrityError:
            # A concurrent read created them first (uq_dataset_item).
            db.session.rollback()


def apply_saved_annotations(df, dataset):
    """``df`` as its text and label columns, in the order the trainer reads them.

    Saved annotations become labels and still-unlabelled rows are dropped.
    """
    if dataset is None or dataset.task_type not in TRAINED_ON_LABELS:
        return df

    label_name = dataset.label_column or LABEL_COLUMNS[dataset.task_type]
    label_column = _find_column(df, label_name)
    if label_column is None:
        label_column = label_name
        df = df.assign(**{label_name: None})

    df = df.astype({label_column: object})
    for task in dataset.annotation_tasks:
        if task.status == "annotated":
            df.loc[task.item_index, label_column] = task.annotation_label

    text_column = _find_column(df, dataset.text_column or "text")
    if text_column is None:
        text_column = df.columns[0]
    return df.loc[~_is_blank(df[label_column]), [text_column, label_column]].reset_index(drop=True)
