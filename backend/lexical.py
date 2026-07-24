"""어휘 검색(BM25) + 한국어 토크나이저.

dense 임베딩은 의역("아기 열"↔"영유아 발열")에 강하지만, 수치·약물명·월령
("38도", "6개월 미만", "아세트아미노펜") 같은 표면형 일치에는 약하다. BM25를
함께 써 두 신호를 보완한다(하이브리드 검색).

kiwipiepy가 있으면 형태소 토큰화를, 없으면 공백/정규식 토큰화로 폴백한다.
"""
from __future__ import annotations
import re


class Tokenizer:
    def __init__(self):
        self._kiwi = None
        try:
            from kiwipiepy import Kiwi
            self._kiwi = Kiwi()
            self.kind = "kiwi"
        except Exception:
            self.kind = "regex"

    def tokenize(self, text: str) -> list[str]:
        if self._kiwi is not None:
            # 내용어 위주(조사/어미 제외)로 토큰화하여 BM25 변별력을 높인다
            toks = []
            for t in self._kiwi.tokenize(text):
                if t.tag[0] in ("N", "V", "S", "X", "M") and len(t.form) > 1:
                    toks.append(t.form)
                elif t.tag.startswith("SN"):  # 숫자
                    toks.append(t.form)
            return toks or [w for w in re.findall(r"[\w가-힣]+", text)]
        return [w for w in re.findall(r"[\w가-힣]+", text.lower()) if len(w) > 1]


class BM25Index:
    """BM25Okapi 래퍼. rank-bm25 미설치 시 빈 점수(폴백)."""

    def __init__(self, corpus: list[str], tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._bm25 = None
        self.available = False
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [tokenizer.tokenize(t) for t in corpus]
            # 전부 빈 토큰이면 BM25가 의미 없음
            if any(tokenized):
                self._bm25 = BM25Okapi(tokenized)
                self.available = True
        except Exception:
            self.available = False

    def scores(self, query: str):
        """질의에 대한 전체 문서 BM25 점수 배열(numpy). 미가용 시 None."""
        if not self.available:
            return None
        import numpy as np
        toks = self.tokenizer.tokenize(query)
        if not toks:
            return np.zeros(self._bm25.corpus_size, dtype=np.float32)
        return np.asarray(self._bm25.get_scores(toks), dtype=np.float32)
