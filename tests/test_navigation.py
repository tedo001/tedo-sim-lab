"""The navigation table matches the brief: sections, pages, honest labels."""

from __future__ import annotations

from app.navigation import NAV, SECTIONS, page, pages_in

REQUIRED = {
    "home": ["Home"],
    "labs": ["Computer Vision", "Classical ML", "Deep Learning", "NLP / LLM", "Audio",
             "Multimodal", "Simulation Lab"],
    "data": ["Dataset Hub", "Model Zoo", "Model Registry"],
    "experiments": ["Experiment Builder", "Training", "Evaluation", "Benchmarking",
                    "Compare Experiments", "MLflow"],
    "tools": ["Terminal", "Jupyter Notebook", "Google Colab", "Plugin Store", "Hardware Monitor",
              "Documentation"],
}


def test_every_required_page_is_in_its_section() -> None:
    for section, titles in REQUIRED.items():
        present = [spec.title for spec in pages_in(section)]
        for title in titles:
            assert title in present, f"{title} missing from {section}"


def test_sections_are_in_order() -> None:
    assert [key for key, _ in SECTIONS][:5] == ["home", "labs", "data", "experiments", "tools"]
    order = [spec.section for spec in NAV if spec.in_sidebar]
    assert order == sorted(order, key=[key for key, _ in SECTIONS].index)


def test_ids_are_unique_and_lookups_work() -> None:
    ids = [spec.id for spec in NAV]
    assert len(ids) == len(set(ids))
    assert page("training").title == "Training"


def test_unbuilt_pages_say_when_and_what() -> None:
    for spec in NAV:
        assert spec.summary, spec.id
        if spec.built:
            continue
        assert spec.planned_for in {"v0.1", "v0.2", "v0.3", "v0.4", "v0.5", "TBD"}, spec.id
        assert spec.features, f"{spec.id} must list what it will do"
        if spec.planned_for == "v0.1":
            assert spec.build_phase in range(2, 11), spec.id


def test_future_domains_are_placeholders() -> None:
    assert page("audio").planned_for == "v0.2"
    assert page("nlp_llm").planned_for == "v0.3"
    assert page("multimodal").planned_for == "v0.4"
    assert page("quantum_ml").planned_for == "TBD"
    assert page("simulation_lab").planned_for == "TBD"


def test_built_pages_have_factories_and_icons_exist() -> None:
    from app.ui.icons import available_icons
    from app.ui.pages import PAGE_FACTORIES

    icons = available_icons()
    for spec in NAV:
        assert spec.icon in icons, f"{spec.id}: no icon {spec.icon}.svg"
        assert (spec.id in PAGE_FACTORIES) == spec.built, spec.id
