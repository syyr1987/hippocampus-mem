# -*- coding: utf-8 -*-
"""海马体服务契约自测：模拟 AML 平台调用 Add → Search，验证格式/隔离/阈值"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE.parent / "BEAM" / "src"
CASES = BASE.parent / "BEAM" / "cases"

PORT = int(os.environ.get("HPC_TEST_PORT", "8099"))
BASE_URL = f"http://127.0.0.1:{PORT}"
KEY = "test-key-123"


def req(method, path, body=None, key=KEY):
    url = BASE_URL + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json",
                                        "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def load_case(case_dir):
    chat = json.loads((Path(case_dir) / "chat.json").read_text())
    probing = json.loads((Path(case_dir) / "probing_questions.json").read_text())
    return chat, probing


def msgs_from_chat(chat, max_msgs=200):
    """把 chat.json 拍平成 AML messages[]（chat = [batch{turns:[[msg...],...]}, ...]）"""
    out = []
    for batch in chat:
        for turn in batch.get("turns", []):
            for m in turn:
                if not isinstance(m, dict) or not m.get("content"):
                    continue
                out.append({
                    "role": m.get("role", "user"),
                    "timestamp": m.get("timestamp", int(time.time() * 1000)),
                    "content": m["content"],
                })
                if len(out) >= max_msgs:
                    return out
    return out


def run_case(case_dir, uid):
    print(f"\n=== case: {Path(case_dir).name} uid={uid} ===")
    chat, probing = load_case(case_dir)
    msgs = msgs_from_chat(chat)
    print(f"messages={len(msgs)} probing_questions={len(probing)}")

    # Add（分块：每 20 条一批，验证分段契约）
    added = 0
    for i in range(0, len(msgs), 20):
        chunk = msgs[i:i + 20]
        body = {
            "request_id": f"test:run1:{Path(case_dir).name}:chunk-{i//20}",
            "messages": chunk,
            "user_id": uid,
            "session_id": f"test:session:{i//20}",
        }
        st, resp = req("POST", "/add", body)
        assert st == 200 and resp.get("success") is True, f"add failed: {st} {resp}"
        assert resp["request_id"] == body["request_id"], "request_id mismatch"
        assert resp["user_id"] == uid, "user_id mismatch"
        assert resp["session_id"] == body["session_id"], "session_id mismatch"
        added += len(chunk)
    print(f"add ok: {added} msgs")

    # Health（无鉴权）
    r = urllib.request.urlopen(f"{BASE_URL}/health", timeout=10)
    assert r.status == 200, "health not 200"
    print("health ok")

    # 鉴权检查
    st, _ = req("POST", "/add", {"request_id": "x", "messages": [], "user_id": uid, "session_id": "s"}, key="wrong")
    assert st == 401, f"auth should 401, got {st}"
    print("auth ok (401 on wrong key)")

    # Search：逐题（probing = {category: [question, ...]}）
    hit = miss = empty = 0
    total_q = 0
    for cat, qlist in probing.items():
        for q in qlist:
            if not isinstance(q, dict):
                continue
            total_q += 1
            query = q.get("question", "") or q.get("query", "")
            options = q.get("options")
            body = {"query": query, "user_id": uid, "top_k": 100}
            if options:
                body["options"] = options
            st, resp = req("POST", "/search", body)
            assert st == 200, f"search failed: {st} {resp}"
            data = resp.get("data", [])
            assert isinstance(data, list), "data not list"
            assert len(data) <= 100, f"top_k exceeded: {len(data)}"
            for item in data:
                assert item.get("id") and item.get("content"), "item missing id/content"
            if data:
                hit += 1
                # 验证排序单调
                scores = [x.get("score", 0) for x in data]
                assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "not sorted desc"
            else:
                empty += 1
    print(f"search ok: hit={hit} empty={empty} total={total_q}")

    # 隔离检查：另一个 user_id 应查不到
    st, resp = req("POST", "/search", {"query": "anything", "user_id": "OTHER-USER", "top_k": 10})
    assert resp["data"] == [], f"isolation broken: {resp['data'][:2]}"
    print("isolation ok")


if __name__ == "__main__":
    run_case(CASES / "chats_100K_1", "test:uid:100k1")
    run_case(CASES / "chats_100K_2", "test:uid:100k2")
    print("\nALL CONTRACT TESTS PASSED")
