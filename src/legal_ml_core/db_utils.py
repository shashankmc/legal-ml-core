from sqlalchemy import Column, String, Table, text


def ensure_external_fk_stub_tables(db):
    """Registers minimal stand-in tables for `users`/`projects` — owned by
    the host app, not this package — if they aren't already present.

    `Dataset.user_id`, `Model.user_id`/`project_id`, `TrainingJob.user_id`,
    and `ModelComparison.user_id` declare foreign keys into these tables.
    SQLAlchemy resolves FK targets against the shared metadata when it
    configures mappers (on first query), which needs *a* `users`/`projects`
    `Table` object to exist there — it doesn't need the host's real ORM
    classes, just a table with the referenced column.

    In the Flask API process this is a no-op: the host app's `models.py` has
    already defined the real `User`/`Project` classes (with all their
    columns/relationships) on this same metadata by the time
    `init_ml_feature` runs, so `db.metadata.tables` already has real
    entries. In the worker process — which never imports the host app's own
    model definitions — this is what makes FK resolution succeed at all.
    """
    if "users" not in db.metadata.tables:
        Table("users", db.metadata, Column("id", String(36), primary_key=True))
    if "projects" not in db.metadata.tables:
        Table(
            "projects",
            db.metadata,
            Column("id", String(36), primary_key=True),
            Column("user_id", String(36)),
        )


def project_belongs_to_user(session, project_id, user_id):
    """True if a `projects` row with this id/user_id exists.

    The `Project` model itself belongs to the host application, not this
    package (projects are generic app-level grouping, not ML-specific), so we
    can't import its ORM class here. The `projects` table name/columns are
    already part of the stable contract implied by `Dataset.project_id` and
    `Model.project_id`'s foreign keys, so a direct Core query against them is
    the least-coupled way to validate a project reference from within the
    package.
    """
    row = session.execute(
        text("SELECT 1 FROM projects WHERE id = :pid AND user_id = :uid"),
        {"pid": project_id, "uid": user_id},
    ).first()
    return row is not None
