-- Model Registry: a stage per registered version, free notes, and the matching MLflow
-- model version when the source run was tracked in MLflow.
ALTER TABLE models ADD COLUMN stage TEXT NOT NULL DEFAULT 'none'
    CHECK (stage IN ('none', 'staging', 'production', 'archived'));
ALTER TABLE models ADD COLUMN notes TEXT NOT NULL DEFAULT '';
ALTER TABLE models ADD COLUMN mlflow_version TEXT;
