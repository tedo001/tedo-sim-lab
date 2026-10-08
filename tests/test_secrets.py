"""Credentials never print, never land in logs, and come from env or keyring only."""

from __future__ import annotations

import logging

import pytest

from core.common import CredentialError, CredentialStore, Secret, mask_text, setup_logging
from core.common.masking import MASK, SecretMaskingFilter, register_secret
from core.common.secrets import KEYRING_SERVICE


def test_secret_never_prints_itself() -> None:
    secret = Secret("s3cr3t-value")
    assert "s3cr3t" not in repr(secret)
    assert "s3cr3t" not in str(secret)
    assert "s3cr3t" not in f"{secret}"
    assert secret.reveal() == "s3cr3t-value"


def test_environment_comes_before_keyring(keyring_backend) -> None:
    keyring_backend.set_password(KEYRING_SERVICE, "HF_TOKEN", "from-keyring")
    store = CredentialStore(env={"HF_TOKEN": "from-env"}, backend=keyring_backend)
    assert store.get("HF_TOKEN").reveal() == "from-env"
    assert store.source("HF_TOKEN") == "env"


def test_keyring_is_used_when_env_is_empty(keyring_backend) -> None:
    keyring_backend.set_password(KEYRING_SERVICE, "HF_TOKEN", "from-keyring")
    store = CredentialStore(env={"HF_TOKEN": ""}, backend=keyring_backend)
    assert store.get("HF_TOKEN").reveal() == "from-keyring"
    assert store.source("HF_TOKEN") == "keyring"


def test_missing_credential() -> None:
    store = CredentialStore(env={}, backend=None)
    assert store.get("KAGGLE_KEY") is None
    assert store.source("KAGGLE_KEY") == "missing"
    assert not store.keyring_available


def test_set_and_delete_go_to_the_keyring(keyring_backend) -> None:
    store = CredentialStore(env={}, backend=keyring_backend)
    store.set("ROBOFLOW_API_KEY", "rf-value-1234")
    assert keyring_backend.store[(KEYRING_SERVICE, "ROBOFLOW_API_KEY")] == "rf-value-1234"
    store.delete("ROBOFLOW_API_KEY")
    assert store.get("ROBOFLOW_API_KEY") is None


def test_without_a_keyring_storing_explains_the_alternative() -> None:
    store = CredentialStore(env={}, backend=None)
    with pytest.raises(CredentialError, match="environment variable"):
        store.set("HF_TOKEN", "anything")


def test_a_broken_keyring_reads_as_missing() -> None:
    class Broken:
        def get_password(self, service, username):
            raise RuntimeError("locked")

    store = CredentialStore(env={}, backend=Broken())
    assert store.get("HF_TOKEN") is None


def test_handed_out_values_are_masked_everywhere(keyring_backend) -> None:
    store = CredentialStore(env={"KAGGLE_KEY": "kaggle-value-9876"}, backend=keyring_backend)
    store.get("KAGGLE_KEY")
    assert mask_text("calling api with kaggle-value-9876 now") == f"calling api with {MASK} now"


@pytest.mark.parametrize("text", [
    "token hf_abcdefghijklmnopqrstuvwxyz123",
    "push with ghp_abcdefghijklmnopqrstuvwxyz1234",
    "github_pat_11ABCDEFG0123456789_abcdefghij",
    "Authorization: Bearer abc.def.ghijklmnop",
    "KAGGLE_KEY=0123abcd",
    "api_key: 'my-value'",
    "ROBOFLOW_API_KEY = rf123456",
    'password="hunter22"',
])
def test_token_shapes_are_masked(text: str) -> None:
    masked = mask_text(text)
    assert MASK in masked
    for fragment in ("hf_abc", "ghp_abc", "github_pat_11", "abc.def", "0123abcd", "my-value",
                     "rf123456", "hunter22"):
        assert fragment not in masked


@pytest.mark.parametrize("text", [
    "max_tokens: 512",
    "tokenizer: bert-base-uncased",
    "commit 3f2a9c8d1e4b5a6978c0d1e2f3a4b5c6d7e8f9a0",
    "spec sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "token classification task",
])
def test_ordinary_text_is_left_alone(text: str) -> None:
    assert mask_text(text) == text


def test_log_records_are_masked(caplog) -> None:
    register_secret("super-secret-value")
    record = logging.LogRecord("tedo.test", logging.INFO, __file__, 1,
                               "connecting with %s", ("super-secret-value",), None)
    SecretMaskingFilter().filter(record)
    assert record.getMessage() == f"connecting with {MASK}"


def test_tracebacks_are_masked() -> None:
    register_secret("leaky-value-42")
    try:
        raise ValueError("bad credential leaky-value-42")
    except ValueError:
        import sys
        record = logging.LogRecord("tedo.test", logging.ERROR, __file__, 1, "failed", None,
                                   sys.exc_info())
    SecretMaskingFilter().filter(record)
    assert "leaky-value-42" not in record.exc_text
    assert MASK in record.exc_text


def test_log_file_never_contains_the_secret(tmp_path) -> None:
    register_secret("file-secret-777")
    log = setup_logging(tmp_path, "INFO", console=False)
    log.info("using file-secret-777 and HF_TOKEN=hf_zzzzzzzzzzzzzzzzzzzzzzzz")
    logging.getLogger("third.party").warning("echo file-secret-777")
    setup_logging(None, console=False)  # flush and release the file
    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "file-secret-777" not in content
    assert "hf_zzzz" not in content
    assert content.count(MASK) >= 3
