import pandas as pd
from zalma_ai.data_gen import generate
from zalma_ai.rag import CatalogAssistant, NO_INFO
from zalma_ai.forecast import purchase_plan
from zalma_ai.tickets import TicketClassifier


def test_all(tmp_path):
    d = generate(tmp_path)
    a = CatalogAssistant(d / "catalog")
    assert "амплификатор" in a.ask("амплификатор для ПЦР диагностики")["answer"].lower()
    assert a.ask("экскаватор гусеничный")["answer"] == NO_INFO
    plan = purchase_plan(pd.read_csv(d / "sales.csv"), pd.read_csv(d / "inventory.csv"))
    assert len(plan) == 8 and plan["mape_%"].mean() < 25
    clf = TicketClassifier().fit(pd.read_csv(d / "tickets.csv"))
    assert clf.metrics["accuracy_type"] > 0.9
    assert clf.predict("Нужно обучить лаборантов работе с микроскопом")["type"] == "обучение"


if __name__ == "__main__":
    import tempfile, pathlib
    test_all(pathlib.Path(tempfile.mkdtemp()))
    print("OK")
