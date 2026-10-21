# Package-side Alembic environment is intentionally minimal: the host app owns
# the database connection and runs migrations. This directory exists so the
# package can ship its own table DDL via version_locations.
