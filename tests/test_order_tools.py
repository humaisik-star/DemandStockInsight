"""Tests for the stock-management agent tools (order place / undo / list)."""

import pandas as pd
import pytest

import src.assistant_tools as at


@pytest.fixture(autouse=True)
def inject_inventory():
    at._inv_cache = pd.DataFrame({
        "Store ID": ["S001", "S002", "S003", "S004"],
        "Product ID": ["P0001", "P0002", "P0003", "P0004"],
        "abc_class": ["A", "A", "B", "C"],
        "alert_status": ["CRITICAL", "CRITICAL", "REORDER", "OK"],
        "current_inventory": [5, 8, 40, 200],
        "days_of_cover": [0.5, 1.5, 4.0, 30.0],
    })
    at.set_client_orders([])
    yield
    at._inv_cache = None
    at.set_client_orders([])


def test_siparis_ver_picks_most_critical_first():
    out = at.siparis_ver(top_n=2)
    assert out["count"] == 2
    skus = [(s["store_id"], s["product_id"]) for s in out["action"]["skus"]]
    # Only CRITICAL/REORDER are eligible; lowest days-of-cover first.
    assert skus == [("S001", "P0001"), ("S002", "P0002")]
    assert out["action"]["op"] == "mark"


def test_siparis_ver_excludes_ok_status():
    out = at.siparis_ver(top_n=10)               # 4 SKUs but one is OK
    ids = [s["product_id"] for s in out["action"]["skus"]]
    assert "P0004" not in ids                     # the OK SKU is never ordered
    assert len(ids) == 3


def test_siparis_ver_specific_sku():
    out = at.siparis_ver(store_id="S002", product_id="P0002")
    assert out["count"] == 1
    assert out["action"]["skus"] == [{"store_id": "S002", "product_id": "P0002"}]


def test_siparis_ver_status_filter():
    out = at.siparis_ver(top_n=10, status="REORDER")
    ids = [s["product_id"] for s in out["action"]["skus"]]
    assert ids == ["P0003"]


def test_siparis_geri_al_single_and_all():
    one = at.siparis_geri_al(store_id="S001", product_id="P0001")
    assert one["action"] == {"op": "unmark", "skus": [{"store_id": "S001", "product_id": "P0001"}]}
    everything = at.siparis_geri_al(hepsi=True)
    assert everything["action"] == {"op": "unmark", "all": True}
    assert "error" in at.siparis_geri_al()        # nothing specified


def test_verilen_siparisleri_listele_reads_client_state():
    at.set_client_orders([{"store_id": "S001", "product_id": "P0001", "ts": 1}])
    out = at.verilen_siparisleri_listele()
    assert out["count"] == 1
    assert out["orders"][0]["product_id"] == "P0001"
    assert out["action"]["op"] == "open_orders"
