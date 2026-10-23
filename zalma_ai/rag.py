"""Ассистент менеджера: поиск по каталогам и инструкциям (retrieval, без выдумывания).

Ответ строится только из найденных фрагментов документов со ссылкой на источник.
Если релевантных фрагментов нет, возвращается «информации нет».
Поиск: TF-IDF по символьным n-граммам (устойчив к падежам русского языка).
Класс можно расширить: передать найденные фрагменты в LLM как контекст.
"""
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

NO_INFO = "Информации в загруженных документах нет. Уточните у поставщика или в отделе."


class CatalogAssistant:
    def __init__(self, catalog_dir, min_score: float = 0.22):
        self.min_score = min_score
        self.chunks = []  # (источник, заголовок, текст)
        for f in sorted(Path(catalog_dir).glob("*.txt")):
            paras = [p.strip() for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()]
            title = paras[0].splitlines()[0]
            for p in paras[1:] or paras:  # заголовок входит в каждый фрагмент
                self.chunks.append((f.name, title, f"{title}: {p}"))
        if not self.chunks:
            raise ValueError(f"В {catalog_dir} нет .txt документов")
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(p for _, _, p in self.chunks)

    def ask(self, question: str, k: int = 3) -> dict:
        scores = (self.matrix @ self.vec.transform([question]).T).toarray().ravel()
        best = np.argsort(scores)[::-1][:k]
        hits = [(self.chunks[i], float(scores[i])) for i in best if scores[i] >= self.min_score]
        if not hits:
            return {"answer": NO_INFO, "sources": []}
        answer = "\n\n".join(f"[{src}] {text}" for (src, _, text), _ in hits)
        return {"answer": answer, "sources": [{"file": s, "score": round(sc, 3)} for (s, _, _), sc in hits]}
