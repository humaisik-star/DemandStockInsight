"""End-to-end pipeline + assistant chat-loop tests.

Artifact tests are skipped if train_model.py / predict.py haven't produced their
outputs yet, so the suite stays green on a fresh checkout.
"""

from pathlib import Path
from types import SimpleNamespace as NS

import joblib
import numpy as np
import pandas as pd
import pytest

from assistant import run_turn

MODEL_PATH = Path("models/demand_model.joblib")
METRICS_PATH = Path("results/model_metrics.csv")
PRED_PATH = Path("results/predictions.csv")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_model_loads_and_predicts():
    from src.features import build_features, split_X_y

    model = joblib.load(MODEL_PATH)
    raw = pd.read_csv("data/demand_forecasting.csv").head(4000)
    feat = build_features(raw, dropna=True)
    X, _, _, _ = split_X_y(feat)
    preds = model.predict(X.head(50))
    assert len(preds) == 50
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="metrics not generated yet")
def test_metrics_report_shape():
    report = pd.read_csv(METRICS_PATH)
    assert {"model", "RMSE", "MAE", "R2"}.issubset(report.columns)
    # Every model should beat a trivial R2 of 0.
    assert (report["R2"] > 0).all()


@pytest.mark.skipif(not PRED_PATH.exists(), reason="predictions not generated yet")
def test_predictions_non_negative():
    preds = pd.read_csv(PRED_PATH)
    assert (preds["Predicted_Demand"] >= 0).all()


# --------------------------------------------------------------------------- #
# Assistant chat-loop (fake Azure client, no network)
# --------------------------------------------------------------------------- #
class _FakeToolCall:
    def __init__(self, name, args):
        self.id = "call_1"
        self.function = NS(name=name, arguments=args)


class _FakeCompletions:
    """Returns a tool call on the first turn, then a text answer."""

    def __init__(self):
        self.step = 0

    def create(self, **kwargs):
        self.step += 1
        if self.step == 1:
            msg = NS(content=None, tool_calls=[_FakeToolCall("list_series", "{}")])
        else:
            msg = NS(content="There are 2 stores available.", tool_calls=None)
        return NS(choices=[NS(message=msg)])


class _FakeClient:
    def __init__(self):
        self.chat = NS(completions=_FakeCompletions())


def test_run_turn_resolves_tool_calls():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "list the stores"},
    ]
    answer = run_turn(_FakeClient(), "fake-deploy", messages)

    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    # The tool result was threaded back before the final answer.
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert "stores" in tool_msg["content"]
    assert answer == "There are 2 stores available."
