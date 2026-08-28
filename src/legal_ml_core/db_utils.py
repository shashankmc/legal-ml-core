from sqlalchemy import text


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
