"""Классификация сервисных заявок: тип, срочность, ответственное подразделение."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DEPARTMENT = {"монтаж": "Сервисный отдел (монтаж)",
              "гарантия": "Сервисный отдел (гарантия)",
              "ремонт": "Сервисный отдел (ремонт)",
              "обучение": "Методическая и учебная поддержка"}


def _pipe():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         LogisticRegression(max_iter=1000))


class TicketClassifier:
    def fit(self, df: pd.DataFrame):
        tr, te = train_test_split(df, test_size=0.25, random_state=0, stratify=df["type"])
        self.type_model = _pipe().fit(tr.text, tr.type)
        self.prio_model = _pipe().fit(tr.text, tr.priority)
        self.metrics = {
            "accuracy_type": round(accuracy_score(te.type, self.type_model.predict(te.text)), 3),
            "accuracy_priority": round(accuracy_score(te.priority, self.prio_model.predict(te.text)), 3),
        }
        return self

    def predict(self, text: str) -> dict:
        t = self.type_model.predict([text])[0]
        conf = float(self.type_model.predict_proba([text]).max())
        return {"type": t, "priority": self.prio_model.predict([text])[0],
                "department": DEPARTMENT[t], "confidence": round(conf, 2),
                "needs_human_review": conf < 0.5}
