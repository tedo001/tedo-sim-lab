-- 0001: the lab's index. One database per workspace (project). Paths are stored
-- relative to the workspace when they are inside it, so a project folder can be
-- moved or opened on another machine.

CREATE TABLE experiments (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    task        TEXT NOT NULL,
    spec_yaml   TEXT NOT NULL,
    spec_hash   TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',          -- JSON list
    created_at  TEXT NOT NULL
);
CREATE INDEX experiments_hash ON experiments(spec_hash);

CREATE TABLE runs (
    id             TEXT PRIMARY KEY,
    experiment_id  TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    parent_run_id  TEXT REFERENCES runs(id) ON DELETE SET NULL,   -- set for reproductions
    status         TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed',
                       'early_stopped', 'cancelled', 'failed', 'interrupted')),
    runner         TEXT NOT NULL,
    run_dir        TEXT NOT NULL,
    device         TEXT,
    git_commit     TEXT,
    mlflow_run_id  TEXT,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    ended_at       TEXT,
    duration_s     REAL,
    error          TEXT
);
CREATE INDEX runs_experiment ON runs(experiment_id);
CREATE INDEX runs_status ON runs(status);

CREATE TABLE params (
    run_id  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,                             -- JSON
    PRIMARY KEY (run_id, key)
);

CREATE TABLE metrics (
    run_id     TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    step       INTEGER NOT NULL,
    epoch      INTEGER,
    value      REAL NOT NULL,
    logged_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, key, step)
);

CREATE TABLE artifacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                         -- checkpoint, plot, report, log, ...
    path        TEXT NOT NULL,
    sha256      TEXT,
    size_bytes  INTEGER,
    created_at  TEXT NOT NULL,
    UNIQUE (run_id, path)
);

CREATE TABLE datasets (
    card_id      TEXT PRIMARY KEY,
    local_path   TEXT,
    status       TEXT NOT NULL,
    fingerprint  TEXT,
    updated_at   TEXT NOT NULL
);

CREATE TABLE models (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    version          INTEGER NOT NULL,
    card_id          TEXT,
    source_run_id    TEXT REFERENCES runs(id) ON DELETE SET NULL,
    checkpoint_path  TEXT,
    metrics          TEXT NOT NULL DEFAULT '{}',       -- JSON
    created_at       TEXT NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE notebook_links (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook_path  TEXT NOT NULL,
    experiment_id  TEXT REFERENCES experiments(id) ON DELETE CASCADE,
    run_id         TEXT REFERENCES runs(id) ON DELETE CASCADE,
    created_at     TEXT NOT NULL,
    CHECK (experiment_id IS NOT NULL OR run_id IS NOT NULL)
);

CREATE TABLE jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('run', 'task')),
    title       TEXT NOT NULL,
    run_id      TEXT REFERENCES runs(id) ON DELETE SET NULL,
    status      TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed',
                    'cancelled', 'interrupted')),
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    ended_at    TEXT,
    error       TEXT
);
CREATE INDEX jobs_status ON jobs(status);
