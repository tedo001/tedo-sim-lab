"""Text and audio datasets in their sources' layouts, the TF-IDF and transformer text runners, the
spectrogram path through the PyTorch runner, and predictions with the runs they leave."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from core.catalog import load_cards  # noqa: E402
from core.common.cancel import CancelToken  # noqa: E402
from core.common.paths import AppPaths  # noqa: E402
from core.common.vocab import Task  # noqa: E402
from core.experiment_engine.spec import dump_spec  # noqa: E402
from core.experiment_engine.worker import run_evaluation, run_experiment  # noqa: E402
from labs.nlp.datasets import clean_post  # noqa: E402
from labs.nlp.presets import TEXT_PRESETS, text_spec, transformer_spec  # noqa: E402
from labs.nlp.runner import TextClassifierRunner, telling_words  # noqa: E402
from tests.fixtures import fake_data  # noqa: E402
from tests.labs.test_classical_ml import Recorder  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    return paths


def _run(paths: AppPaths, spec, name: str = "run"):
    run_dir = paths.experiments / name
    dump_spec(spec, run_dir / "experiment.yaml")
    result = run_experiment(run_dir, Recorder(), CancelToken())
    assert result.status == "completed", result.error
    return run_dir, result


def test_text_datasets_read_their_archives(workspace) -> None:
    datasets, _models = load_cards(workspace)
    root = workspace.datasets
    for card_id in ("movie_reviews", "sms_spam", "twenty_newsgroups"):
        assert datasets.status(card_id) == "not_downloaded"
    fake_data.movie_reviews(root)
    fake_data.sms_spam(root)
    groups = datasets.get("twenty_newsgroups").class_names
    fake_data.newsgroups(root, groups)
    texts, labels = datasets.adapter("movie_reviews").load_texts(root, "all")
    assert len(texts) == 80 and sorted(set(labels)) == [0, 1] and datasets.status("movie_reviews") == "ready"
    texts, labels = datasets.adapter("sms_spam").load_texts(root, "all")
    assert len(texts) == 40 and labels.count(1) == 20 and "prize" in texts[-1]
    news = datasets.adapter("twenty_newsgroups")
    train, y_train = news.load_texts(root, "train")
    test, _y = news.load_texts(root, "test")
    assert len(train) == 60 and len(test) == 40 and sorted(set(y_train)) == list(range(20))
    assert "From:" not in train[0] and "quoted" not in train[0] and "signature" not in train[0]
    assert clean_post("H: x\n\nbody\n-- \nsig") == "body"


def test_tfidf_run_scores_and_names_the_telling_words(workspace) -> None:
    fake_data.movie_reviews(workspace.datasets)
    runner = TextClassifierRunner()
    spec = text_spec("reviews", "movie_reviews", "tfidf_logistic", min_df=1)
    assert runner.validate(spec) == []
    assert "does not do text" in " ".join(runner.validate(text_spec("x", "movie_reviews", "random_forest")))
    assert "not a TF-IDF model" in " ".join(runner.validate(text_spec("x", "movie_reviews",
                                                                      "bert_base_uncased")))
    assert "is not a text dataset" in runner.validate(text_spec("x", "iris", "tfidf_logistic"))[0]
    run_dir, result = _run(workspace, spec)
    results = json.loads((run_dir / "results.json").read_text())
    assert result.metrics["test_acc"] > 0.9 and results["classes"] == ["neg", "pos"]
    assert results["importance_kind"] == "words" and results["words"]["pos"] and results["words"]["neg"]
    positive = {word for word, _w in results["words"]["pos"]}
    assert positive & {"wonderful", "brilliant", "great", "beautiful", "superb", "best", "loved"}
    again = run_evaluation(run_dir, Recorder(), CancelToken(),
                           checkpoint=run_dir / "checkpoints" / "model.joblib",
                           split="test")
    assert again.metrics["test_acc"] == pytest.approx(result.metrics["test_acc"])
    from labs.inference.runs import predict

    answer = predict({"run": str(run_dir), "inputs": ["a brilliant, wonderful film"]}, lambda _: None,
                     lambda: False, workspace.workspace)
    assert answer["predictions"][0]["classes"][0] == "pos" and answer["model"]["kind"] == "text"


@pytest.mark.parametrize("model", ["tfidf_linear_svm", "tfidf_naive_bayes"])
def test_other_tfidf_models_and_own_splits(workspace, model) -> None:
    datasets, _models = load_cards(workspace)
    fake_data.newsgroups(workspace.datasets, datasets.get("twenty_newsgroups").class_names)
    run_dir, result = _run(workspace, text_spec("news", "twenty_newsgroups", model, min_df=1))
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["own_split"] and info["test_rows"] == 40 and result.metrics["test_acc"] > 0.5
    import joblib

    words = telling_words(joblib.load(run_dir / "checkpoints" / "model.joblib"), info["classes"])
    assert len(words) == 20


def test_text_presets_are_valid_specs() -> None:
    for preset in TEXT_PRESETS:
        spec = preset.make()
        assert spec.task == Task.TEXT_CLASSIFICATION and spec.data.dataset == preset.dataset
        assert spec.model.model == preset.model


def test_transformer_fine_tune_with_a_downloaded_encoder(workspace) -> None:
    pytest.importorskip("transformers")
    from labs.nlp.transformer import TransformerTextRunner
    from tests.fixtures.tiny_models import encoder

    fake_data.movie_reviews(workspace.datasets)
    runner = TransformerTextRunner()
    spec = transformer_spec("bert", "movie_reviews", "distilbert_base_uncased", epochs=2, lr=1e-3,
                            max_length=32)
    assert any("Model Zoo" in p for p in runner.validate(spec))
    encoder(workspace.models / "huggingface" / "distilbert__distilbert-base-uncased")
    assert runner.validate(spec) == []
    assert TextClassifierRunner().supports(spec) is False and runner.supports(spec)
    run_dir, result = _run(workspace, spec)
    assert (run_dir / "checkpoints" / "best" / "config.json").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file() and result.metrics["epochs_completed"] == 2
    confusion = json.loads((run_dir / "test_confusion.json").read_text())
    assert confusion["classes"] == ["neg", "pos"] and np.asarray(confusion["matrix"]).sum() == 16
    again = run_evaluation(run_dir, Recorder(), CancelToken(), checkpoint=run_dir / "checkpoints" / "best",
                           split="test")
    assert again.metrics["test_acc"] == pytest.approx(result.metrics["test_acc"])


def test_spoken_digits_train_as_spectrograms_and_explain(workspace) -> None:
    pytest.importorskip("torch")
    from labs.audio.presets import AUDIO_PRESETS
    from labs.common.explore import image_points
    from labs.common.samples import samples
    from labs.inference.runs import predict
    from labs.inference.visualize import visualize

    datasets, _models = load_cards(workspace)
    assert datasets.status("spoken_digits") == "not_downloaded"
    folder = fake_data.spoken_digits(workspace.datasets)
    adapter = datasets.adapter("spoken_digits")
    assert datasets.status("spoken_digits") == "ready"
    assert len(adapter.load(workspace.datasets, "train")) == 2700 and len(adapter.load(workspace.datasets,
                                                                                       "test")) == 300
    assert image_points(adapter, workspace.datasets, limit=50).values.shape == (50, 64 * 64)
    assert samples(adapter, workspace.datasets, limit=3)["kind"] == "images"
    spec = AUDIO_PRESETS[0].make()
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"epochs": 2}),
                                   "data": spec.data.model_copy(update={"subset": 400})})
    run_dir, result = _run(workspace, spec)
    assert result.metrics["test_acc"] > 0.5
    wav = str(folder / "7_theo_2.wav")
    answer = predict({"run": str(run_dir), "inputs": [wav]}, lambda _: None, lambda: False,
                     workspace.workspace)
    assert answer["model"]["kind"] == "audio" and len(answer["predictions"][0]["classes"]) == 10
    view = visualize({"run": str(run_dir), "input": {"path": wav}}, lambda _: None, lambda: False,
                     workspace.workspace)
    assert view["input"]["shape"] == [64, 64] and view["layers"] and "gradcam" in view
