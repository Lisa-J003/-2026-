"""Командная строка: python -m zalma_ai <команда>"""
import argparse
import pandas as pd
from .data_gen import DATA, ROOT, generate
from .rag import CatalogAssistant
from .forecast import purchase_plan
from .tickets import TicketClassifier

REPORTS = ROOT / "reports"


def _ensure_data():
    if not (DATA / "sales.csv").exists():
        generate()


def cmd_ask(q):
    _ensure_data()
    r = CatalogAssistant(DATA / "catalog").ask(q)
    print(r["answer"])


def cmd_forecast():
    _ensure_data()
    plan = purchase_plan(pd.read_csv(DATA / "sales.csv"), pd.read_csv(DATA / "inventory.csv"))
    REPORTS.mkdir(exist_ok=True)
    plan.to_csv(REPORTS / "purchase_plan.csv", index=False, encoding="utf-8-sig")
    print(plan.to_string(index=False))
    print(f"\nСредний MAPE: {plan['mape_%'].mean():.1f}%  (цель ≤ 25%)  -> reports/purchase_plan.csv")


def cmd_ticket(text):
    _ensure_data()
    clf = TicketClassifier().fit(pd.read_csv(DATA / "tickets.csv"))
    print("Метрики:", clf.metrics)
    print(clf.predict(text))


def cmd_demo():
    _ensure_data()
    print("=== 1. Ассистент по каталогам ===")
    for q in ["Какой амплификатор подойдёт для лаборатории молекулярной диагностики?",
              "Как хранить ПЦР-реагенты?", "Сколько стоит экскаватор?"]:
        print(f"\nВопрос: {q}")
        cmd_ask(q)
    print("\n=== 2. Прогноз спроса и план закупок ===")
    cmd_forecast()
    print("\n=== 3. Классификация заявок ===")
    cmd_ticket("Автоклав не включается, лаборатория простаивает, нужен срочный ремонт")


def main():
    p = argparse.ArgumentParser(prog="zalma_ai", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-data", help="сгенерировать демо-данные")
    sub.add_parser("forecast", help="прогноз спроса и план закупок")
    sub.add_parser("demo", help="показать все модули")
    a = sub.add_parser("ask", help="вопрос по каталогу"); a.add_argument("question")
    t = sub.add_parser("ticket", help="классифицировать заявку"); t.add_argument("text")
    args = p.parse_args()
    if args.cmd == "init-data":
        print("Данные созданы в", generate())
    elif args.cmd == "ask":
        cmd_ask(args.question)
    elif args.cmd == "forecast":
        cmd_forecast()
    elif args.cmd == "ticket":
        cmd_ticket(args.text)
    else:
        cmd_demo()
