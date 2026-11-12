"""One path for turning a file into a Dataset.

The upload route and any host that brings data in another way (BAIGLE's
dataset library, for example) call `create_dataset`, so every dataset passes
the same checks and gets the same row and column information.
"""

from .annotation_queue import record_rows_to_label
from .db import db
from .models import Dataset


class DatasetInvalid(ValueError):
    """The file cannot become a dataset. The message is safe to show a student."""


def create_dataset(
    owner_id,
    file_path,
    name,
    description,
    task_type,
    annotation_mode=False,
    labels=None,
):
    """Validate the file at `file_path` and store it as a Dataset owned by `owner_id`.

    Raises `DatasetInvalid` with a student-readable message for every refusal.
    The caller owns the file: it is neither moved nor deleted here.
    """
    from .routes.datasets import VALID_TASK_TYPES, validate_dataset_file

    name = (name or "").strip()
    if not name:
        raise DatasetInvalid("Dataset name is required")
    if task_type not in VALID_TASK_TYPES:
        raise DatasetInvalid(f'Invalid task type. Valid types: {", ".join(VALID_TASK_TYPES)}')

    df, error = validate_dataset_file(file_path, task_type, annotation_mode=annotation_mode)
    if error:
        raise DatasetInvalid(error)

    columns_info = [
        {
            "name": col,
            "type": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }
        for col in df.columns
    ]
    dataset = Dataset(
        owner_id=owner_id,
        name=name,
        description=description,
        file_path=file_path,
        row_count=len(df),
        columns=columns_info,
        task_type=task_type,
        labels=labels or None,
    )
    record_rows_to_label(dataset, df)
    db.session.add(dataset)
    db.session.commit()
    return dataset
