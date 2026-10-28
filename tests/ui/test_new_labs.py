"""The NLP / LLM, Audio, Multimodal and Quantum ML labs, Model Testing and the Dataset Hub's new
paths, driven through the window's widgets with tiny local models in the real inference worker."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures import fake_data, tiny_models
from tests.ui.test_experiment_service import wait_for

pytest.importorskip("transformers")
pytest.importorskip("torch")


def _hub(ctx, repo: str, make) -> None:
    from core.model_registry.cards import hub_folder

    make(hub_folder(ctx.paths.workspace, repo))


def _png(path, seed: int = 0):
    from PIL import Image

    Image.fromarray((np.random.default_rng(seed).random((40, 48, 3)) * 255).astype("uint8")).save(path)
    return path


def test_pretrained_models_show_their_state(ctx, qtbot) -> None:
    from app.ui.pages.lab_parts import DownloadedModels, PretrainedModels
    from core.common.vocab import Task

    table = PretrainedModels(ctx, {Task.TEXT_GENERATION})
    qtbot.addWidget(table)
    assert table.table.rowCount() == 3
    assert ctx.catalog.models.status("smollm2_135m_instruct") == "not_downloaded"
    chooser = DownloadedModels(ctx, {Task.TEXT_GENERATION})
    assert chooser.repo is None
    _hub(ctx, "HuggingFaceTB/SmolLM2-135M-Instruct", lambda f: tiny_models.causal_lm(f))
    chooser.refresh()
    assert chooser.repo == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert ctx.catalog.models.status("smollm2_135m_instruct") == "ready"


def test_nlp_chat_tokenizer_and_embeddings(ctx, qtbot) -> None:
    from app.ui.pages.nlp import NlpLlmPage

    _hub(ctx, "HuggingFaceTB/SmolLM2-135M-Instruct", lambda f: tiny_models.causal_lm(f))
    _hub(ctx, "sentence-transformers/all-MiniLM-L6-v2", lambda f: tiny_models.encoder(f))
    page = NlpLlmPage(ctx)
    qtbot.addWidget(page)
    page.show_view("chat")
    assert page.chat.isVisibleTo(page) and not page.presets.isVisibleTo(page)
    page.chat.tokens.setValue(16)
    page.chat.prompt.setPlainText("hello there")
    page.chat.send()
    qtbot.waitUntil(lambda: len(page.chat.messages) == 2 or "Failed" in page.chat.status.text(),
                    timeout=120_000)
    assert page.chat.messages[1]["role"] == "assistant" and "tokens" in page.chat.status.text()
    page.tokens.split()
    qtbot.waitUntil(lambda: "tokens for" in page.tokens.summary.text(), timeout=60_000)
    page.embed.model.setCurrentIndex(page.embed.model.findData("sentence-transformers/all-MiniLM-L6-v2"))
    page.embed.compare()
    qtbot.waitUntil(lambda: "dimensions" in page.embed.summary.text(), timeout=60_000)
    assert len(page.embed.ranking.rows()) == 5


def test_audio_lab_pictures_filters_and_transcription(ctx, qtbot) -> None:
    from app.ui.pages.audio import AudioPage

    page = AudioPage(ctx)
    qtbot.addWidget(page)
    assert page.values.value_text("Pitch").startswith("44")  # the 440 Hz tone
    assert page.spectrum.image is not None and page.mfcc.image is not None
    page.high_cut.setValue(200)
    page.apply_filter()
    assert np.abs(page.signal).max() < 0.2 * np.abs(page.original).max()
    _hub(ctx, "openai/whisper-tiny", lambda f: tiny_models.whisper(f))
    page.speech_model.refresh()
    page.transcribe()
    qtbot.waitUntil(lambda: page.transcript.text() not in ("Listening…", ""), timeout=120_000)
    assert not page.transcript.text().startswith("Failed"), page.transcript.text()


def test_multimodal_zero_shot_search_and_caption(ctx, qtbot, tmp_path) -> None:
    from app.ui.pages.multimodal import MultimodalPage

    _hub(ctx, "openai/clip-vit-base-patch32", lambda f: tiny_models.clip(f))
    _hub(ctx, "Salesforce/blip-image-captioning-base", lambda f: tiny_models.blip(f))
    page = MultimodalPage(ctx)
    qtbot.addWidget(page)
    page.refresh()
    folder = tmp_path / "pictures"
    folder.mkdir()
    for seed in range(5):
        _png(folder / f"p{seed}.png", seed)
    page.zero_image.set_path(str(folder / "p0.png"))
    page.classify()
    qtbot.waitUntil(lambda: page.zero_note.text().startswith(("Best", "Failed")), timeout=120_000)
    assert page.zero_note.text().startswith("Best") and len(page.zero_result.rows()) == 6
    page.search_folder = str(folder)
    page.search()
    qtbot.waitUntil(lambda: page.search_note.text().startswith(("The best", "Failed")), timeout=120_000)
    assert "of 5 images" in page.search_note.text() and page.map.isVisibleTo(page)
    page.caption_image.set_path(str(folder / "p1.png"))
    page.describe_image()
    qtbot.waitUntil(lambda: page.caption.text() != "Looking…", timeout=120_000)
    assert not page.caption.text().startswith("Failed"), page.caption.text()


def test_model_testing_detection_and_spectrogram_layers(ctx, qtbot, tmp_path) -> None:
    from app.ui.pages.testing import TestingPage
    from labs.audio.presets import AUDIO_PRESETS

    _hub(ctx, "hustvl/yolos-tiny", lambda f: tiny_models.detector(f))
    page = TestingPage(ctx)
    qtbot.addWidget(page)
    page.open_model("yolos_tiny")
    assert page.subject.kind == "detect" and not page.tabs.buttons["visualise"].isEnabled()
    page.inputs.set_path(_png(tmp_path / "scene.png"))
    page.inputs.threshold.setValue(0.05)
    page.try_it()
    qtbot.waitUntil(lambda: "found" in page.outputs.headline.text()
                    or "Failed" in page.outputs.headline.text(),
                    timeout=120_000)
    assert "found" in page.outputs.headline.text()

    folder = fake_data.spoken_digits(ctx.paths.datasets)
    spec = AUDIO_PRESETS[1].make()
    spec = spec.model_copy(update={"training": spec.training.model_copy(update={"epochs": 1}),
                                   "data": spec.data.model_copy(update={"subset": 200})})
    run_id = ctx.experiments.launch(spec)
    assert wait_for(qtbot, ctx, run_id, ("completed", "failed")).status == "completed"
    page.open_run(run_id)
    assert page.subject.kind == "audio" and page.tabs.buttons["visualise"].isEnabled()
    qtbot.waitUntil(lambda: page.target.count() == 11, timeout=60_000)
    page.inputs.set_path(folder / "3_lucas_1.wav")
    page.try_it()
    qtbot.waitUntil(lambda: page.outputs.headline.text().startswith(("Predicted", "Failed")), timeout=60_000)
    assert page.outputs.headline.text().startswith("Predicted")
    page.tabs.select("visualise")
    page.visualise()
    qtbot.waitUntil(lambda: page.maps.column.count() > 3 or "Failed" in page.vis_note.text(), timeout=60_000)
    assert page.maps.column.count() > 3 and page.vis_card.isVisibleTo(page)


def test_dataset_hub_previews_and_starts_experiments(ctx, qtbot) -> None:
    from app.ui.pages.dataset_hub import DatasetHubPage

    page = DatasetHubPage(ctx)
    qtbot.addWidget(page)
    catalogue = page.views["catalogue"]
    kinds = {ctx.catalog.datasets.get(card_id).modality.value for card_id in catalogue.ids}
    assert kinds == {"image", "tabular", "text", "audio", "multimodal"}
    catalogue.select("movie_reviews")
    assert not catalogue.builder_button.isEnabled() and catalogue.lab_button.text() == "Open NLP / LLM"
    fake_data.movie_reviews(ctx.paths.datasets)
    catalogue.select("iris")
    catalogue.select("movie_reviews")
    assert catalogue.builder_button.isEnabled()
    qtbot.waitUntil(lambda: "texts" in catalogue.samples.status.text(), timeout=30_000)
    drafts = []
    ctx.experiments.draft_changed.connect(drafts.append)
    catalogue._use_in_builder()
    assert drafts and drafts[0].task.value == "text_classification"
    assert drafts[0].data.dataset == "movie_reviews"
    catalogue.select("two_moons")
    assert catalogue.lab_button.text() == "Open Quantum ML"


def test_builder_round_trips_every_kind(ctx, qtbot) -> None:
    from app.ui.pages.builder.form import SpecForm
    from labs.audio.presets import AUDIO_PRESETS
    from labs.nlp.presets import TEXT_PRESETS
    from labs.quantum_ml.presets import QUANTUM_PRESETS

    form = SpecForm(ctx)
    qtbot.addWidget(form)
    for preset in (AUDIO_PRESETS[0], TEXT_PRESETS[0], TEXT_PRESETS[2], QUANTUM_PRESETS[0],
                   QUANTUM_PRESETS[1]):
        spec = preset.make()
        form.load(spec)
        again = form.spec()
        assert again.task == spec.task and again.data.dataset == spec.data.dataset, preset.id
        assert again.model.model == spec.model.model and again.training.kind == spec.training.kind, preset.id
        assert again.model.params == spec.model.params, preset.id
    form.load(QUANTUM_PRESETS[0].make())
    assert form.kind == "quantum" and form.quantum.cards.isVisibleTo(form)
    assert not form.torch.cards.isVisibleTo(form)
    form.load(TEXT_PRESETS[0].make())
    assert form.kind == "text" and form.text.cards.isVisibleTo(form)
