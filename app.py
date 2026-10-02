# -*- coding: utf-8 -*-
"""
海马体记忆服务（AML Cycle 2 参赛版 v0.1）
核心：Add 写入 → 分层组织 → Search 按相关性返回记忆证据
对齐 Agent Memory Leaderboard Add/Search 契约：
- Add: POST, 同步返回 200 + success/request_id/user_id/session_id 原样回传
- Search: POST, 返回 data[]（id/content/score/created_at），数量 ≤ top_k，按相关性降序
- Health: GET, 无鉴权 2xx
- 鉴权: Bearer / X-Api-Key / Token（Memory System Key）
设计要点：
- user_id 严格隔离（Search 只检索同 user_id 记忆）
- BM25 + embedding 融合（RRF），阈值截断实现"少而准"
- 低分返回空数组（对应 abstention 拒答语义）
"""
import json
import math
import os
import re
import sqlite3
import time
import hashlib
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union

# ---------- 配置 ----------
BASE_DIR = Path(__file__).resolve().parent
# 注意：SQLite 数据库必须放本地盘（/var/tmp），放云盘同步目录会 disk I/O error / malformed
DB_PATH = os.environ.get("HPC_DB", "/var/tmp/hippocampus.db")
EMBED_MODEL = os.environ.get("HPC_EMBED_MODEL", "embedding-3")
EMBED_DIM = int(os.environ.get("HPC_EMBED_DIM", "1024"))
ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
MIN_SCORE = float(os.environ.get("HPC_MIN_SCORE", "0.25"))  # 相关性阈值（拒答线）
RERANK = os.environ.get("HPC_RERANK", "0") == "1"  # LLM 相关性拒答（abstention 型）
RERANK_CANDS = int(os.environ.get("HPC_RERANK_CANDS", "20"))  # 拒答判定候选数
RERANK_MODEL = os.environ.get("HPC_RERANK_MODEL", "glm-4-flash")
RERANK_URL = os.environ.get("HPC_RERANK_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
TOPK_LIMIT = 100
KEY_PATH = Path(os.environ.get("HPC_KEY_FILE", str(BASE_DIR / ".memory_key")))
STOPWORDS = set("""a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with i you your we our he she they them me my mine yours ours""".split())
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")


def get_embed_key():
    # 优先：环境变量直传（Render 等 PaaS 部署，无本地文件）
    direct = os.environ.get("HPC_EMBED_KEY", "").strip()
    if direct:
        return direct
    if KEY_PATH.exists():
        return KEY_PATH.read_text().strip()
    return os.environ.get("ZHIPU_API_KEY", "")


def get_memory_key():
    if os.environ.get("HPC_MEMORY_KEY"):
        return os.environ["HPC_MEMORY_KEY"]
    if (BASE_DIR / ".memory_key").exists():
        return (BASE_DIR / ".memory_key").read_text().strip()
    return ""


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(str(text)) if t.lower() not in STOPWORDS and len(t) > 1]


def embed_texts(texts, key):
    """批量 embedding（智谱 embedding-3）"""
    body = json.dumps({"model": EMBED_MODEL, "input": texts, "dimensions": EMBED_DIM}).encode("utf-8")
    req = urllib.request.Request(
        ZHIPU_URL, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = sorted(data["data"], key=lambda x: x["index"])
    return [r["embedding"] for r in rows]


# ---------- 存储与索引 ----------
class MemoryStore:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT,
            role TEXT,
            content TEXT NOT NULL,
            created_at TEXT,
            vec BLOB
        )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON memories(user_id)")
        self.conn.commit()
        self._df_cache = None  # 文档频率缓存

    def add(self, user_id, session_id, role, content, created_at):
        mid = hashlib.sha1(f"{user_id}|{session_id}|{role}|{content}|{created_at}".encode("utf-8")).hexdigest()[:24]
        cur = self.conn.execute("SELECT 1 FROM memories WHERE id=?", (mid,))
        if cur.fetchone():
            return mid
        self.conn.execute(
            "INSERT INTO memories (id, user_id, session_id, role, content, created_at) VALUES (?,?,?,?,?,?)",
            (mid, user_id, session_id, role, content, created_at))
        self.conn.commit()
        self._df_cache = None
        return mid

    def get_all(self, user_id):
        cur = self.conn.execute(
            "SELECT id, session_id, role, content, created_at FROM memories WHERE user_id=? ORDER BY rowid",
            (user_id,))
        return cur.fetchall()

    def set_vec(self, mid, vec):
        self.conn.execute("UPDATE memories SET vec=? WHERE id=?", (json.dumps(vec), mid))
        self.conn.commit()

    def get_vecs(self, user_id):
        cur = self.conn.execute("SELECT id, vec FROM memories WHERE user_id=? AND vec IS NOT NULL", (user_id,))
        return [(r[0], json.loads(r[1])) for r in cur.fetchall()]


store = MemoryStore(DB_PATH)
bm25_state = {"docs": None, "idf": None}


def build_bm25(rows):
    docs = [tokenize(r[3]) for r in rows]
    N = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf = {t: math.log(1 + (N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    return docs, idf, df


def bm25_score(qt, doc, idf, avgdl, N, k1=1.5, b=0.75):
    tf = Counter(doc)
    dl = len(doc)
    s = 0.0
    for t in qt:
        f = tf.get(t, 0)
        if f == 0 or t not in idf:
            continue
        s += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(1.0, avgdl)))
    return s


def cos_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def rrf(rank_lists, k=60):
    scores = {}
    for rl in rank_lists:
        for rank, i in enumerate(rl):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

RERANK_SYSTEM = """你是记忆检索系统的判定器。系统根据用户查询检索出一批候选记忆消息。
判断这些候选消息中是否含有【与查询相关、可用于回答查询的实质信息】。
判定规则：
- 1 = 有：候选消息包含与查询相关的实质内容，可用于回答查询。包括需要综合多条消息推导的情况；也包括"我是否提过/说过/做过某事"类查询中，候选里出现过相关事件或陈述的情况。
- 0 = 无：候选消息只是提到相同词汇或主题，但没有任何与查询实质相关的内容；或与查询完全无关。
对于"库里没有相关信息"的查询（abstention），候选会显得主题沾边但答不上来，此时判 0。
只输出 JSON：{"has_answer": 0 或 1}，不要任何额外文字。"""


def rerank_relevant(query, cand_texts, key):
    """LLM 整组判定候选是否含可直接回答查询的信息；返回 True/False；失败返回 None（调用方降级）。"""
    if not cand_texts:
        return False
    NL = chr(10)
    lines = NL.join(f"[{i + 1}] {c[:300]}" for i, c in enumerate(cand_texts))
    user = f"用户查询：{query}{NL}{NL}候选记忆消息列表：{NL}{lines}"
    body = json.dumps({
        "model": RERANK_MODEL,
        "messages": [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        RERANK_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < 0:
            return None
        obj = json.loads(content[start:end + 1])
        v = obj.get("has_answer")
        if v not in (0, 1):
            return None
        return bool(v)
    except Exception:
        return None


# ---------- API 模型 ----------
class AddMessage(BaseModel):
    role: str
    timestamp: Optional[int] = None
    content: Union[str, List[dict]]


class AddRequest(BaseModel):
    request_id: str
    messages: List[AddMessage]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: Union[str, List[dict]]
    options: Optional[List[str]] = None
    user_id: str
    top_k: int = 100


class SearchItem(BaseModel):
    id: str
    content: str
    score: float = 0.0
    created_at: Optional[str] = None


class SearchResponse(BaseModel):
    data: List[SearchItem]


app = FastAPI(title="Hippocampus Memory Service", version="0.1.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": {"reason": f"internal: {exc}"}})


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.method == "GET" and request.url.path == "/health":
        return await call_next(request)
    mkey = get_memory_key()
    if mkey:
        auth = request.headers.get("Authorization", "")
        xkey = request.headers.get("X-Api-Key", "")
        token = request.headers.get("Token", "")
        if not (auth == f"Bearer {mkey}" or xkey == mkey or token == mkey):
            return JSONResponse(status_code=401, content={"detail": {"reason": "unauthorized"}})
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add", response_model=AddResponse)
def add(req: AddRequest):
    now = datetime.now(timezone.utc).isoformat()
    embed_key = get_embed_key()
    try:
        texts = []
        metas = []
        for m in req.messages:
            content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            if not content.strip():
                continue
            created = datetime.fromtimestamp(m.timestamp / 1000, tz=timezone.utc).isoformat() if m.timestamp else now
            mid = store.add(req.user_id, req.session_id, m.role, content, created)
            texts.append(content)
            metas.append(mid)
        # 批量 embedding（失败则降级纯 BM25）
        if embed_key and texts:
            try:
                vecs = embed_texts(texts, embed_key)
                for mid, v in zip(metas, vecs):
                    store.set_vec(mid, v)
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail={"reason": f"add failed: {e}"})
    return AddResponse(success=True, request_id=req.request_id, user_id=req.user_id, session_id=req.session_id)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    query = req.query if isinstance(req.query, str) else json.dumps(req.query, ensure_ascii=False)
    top_k = max(1, min(req.top_k, TOPK_LIMIT))
    rows = store.get_all(req.user_id)
    if not rows:
        return SearchResponse(data=[])
    # BM25
    docs, idf, _ = build_bm25(rows)
    avgdl = sum(len(d) for d in docs) / max(1, len(docs))
    qt = tokenize(query)
    bm_scores = [(bm25_score(qt, docs[i], idf, avgdl, len(docs)), i) for i in range(len(docs))]
    bm_scores = [(s, i) for s, i in bm_scores if s > 0]
    bm_scores.sort(key=lambda x: -x[0])
    bm_rank = [i for _, i in bm_scores]
    # embedding 余弦（记录绝对相似度供拒答判断）
    vec_rank = []
    vec_sim = {}
    embed_key = get_embed_key()
    if embed_key:
        vec_ok = False
        for _attempt in range(2):
            try:
                qvec = embed_texts([query], embed_key)[0]
                vecs = store.get_vecs(req.user_id)
                if vecs:
                    id2idx = {r[0]: i for i, r in enumerate(rows)}
                    sims = []
                    for mid, v in vecs:
                        idx = id2idx.get(mid)
                        if idx is not None:
                            s = cos_sim(qvec, v)
                            sims.append((s, idx))
                            vec_sim[idx] = s
                    sims.sort(key=lambda x: -x[0])
                    vec_rank = [i for _, i in sims[:200]]
                vec_ok = True
                break
            except Exception:
                continue
        if not vec_ok:
            # embedding 失败：fail-closed 返回空（拒答优先，避免 abstention/无关查询崩）
            return SearchResponse(data=[])
    # 融合
    rank_lists = [rl for rl in (bm_rank, vec_rank) if rl]
    if not rank_lists:
        return SearchResponse(data=[])
    merged = rrf(rank_lists, k=60)
    max_sc = merged[0][1] if merged else 1.0
    # LLM 相关性拒答：候选整组判定（abstention/无关 → 返回空；真相关/失败降级 → 走原逻辑）
    if RERANK and embed_key:
        cands = merged[:RERANK_CANDS]
        cand_texts = [rows[i][3] for i, _ in cands]
        has = rerank_relevant(query, cand_texts, embed_key)
        if has is False:
            return SearchResponse(data=[])
    # 拒答：绝对 cosine 低于阈值则截断（无 embedding 时用归一化 RRF）
    has_vec = bool(vec_sim)
    out = []
    for i, raw in merged:
        if has_vec:
            sim = vec_sim.get(i, 0.0)
            if sim < MIN_SCORE:
                continue
        else:
            norm = raw / max_sc if max_sc else 0.0
            if norm < MIN_SCORE:
                continue
        score = round(raw / max_sc, 4) if max_sc else 0.0
        r = rows[i]
        out.append(SearchItem(id=r[0], content=r[3], score=score, created_at=r[4]))
        if len(out) >= top_k:
            break
    return SearchResponse(data=out)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("HPC_PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port)
