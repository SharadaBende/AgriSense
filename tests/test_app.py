"""Smoke-test every page of the Streamlit app headlessly."""
import pytest
from streamlit.testing.v1 import AppTest

from src import config as C

pytestmark = pytest.mark.skipif(not (C.ROOT / "models" / "final_model.joblib").exists(), reason="train first")
PAGES = ["Predict & what-if", "Risk profile", "Risk map", "Model & method"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(C.ROOT / "app" / "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"


def test_whatif_slider_changes_prediction():
    at = AppTest.from_file(str(C.ROOT / "app" / "app.py"), default_timeout=90).run()
    base = [m.value for m in at.metric if m.label == "Model prediction"][0]
    at.slider[0].set_value(50).run()
    assert not at.exception
    low = [m.value for m in at.metric if m.label == "Model prediction"][0]
    assert base != low
