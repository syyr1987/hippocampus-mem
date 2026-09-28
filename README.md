# 海马体记忆服务（AML Cycle 2 参赛版 v0.1）

对齐 Agent Memory Leaderboard（AML）第二期 Add/Search 契约的自托管记忆服务。

## 架构
- **FastAPI + SQLite**：单机轻量，同步 Add/Search
- **海马体组织**：BM25 + 智谱 embedding-3 融合（RRF），相关性排序返回
- **user_id 隔离**：Search 只检索同 user_id 记忆，杜绝跨样本泄漏
- **阈值截断**：绝对 cosine 低于阈值不返回（对应 abstention 拒答语义）

## 接口契约（对齐 AML 官方 api-guide）
| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 无鉴权，返回 2xx |
| `/add` | POST | 同步写入，返回 200 + success/request_id/user_id/session_id 原样回传 |
| `/search` | POST | 返回 data[]（id/content/score/created_at），≤top_k，相关性降序 |

鉴权：`Authorization: Bearer <Memory-System-Key>` / `X-Api-Key` / `Token`，任一匹配即可。
Memory System Key 通过环境变量 `HPC_MEMORY_KEY` 或 `./.memory_key` 文件设置。

## 启动
```bash
pip install -r requirements.txt
./start.sh 8123 your-memory-key
```

环境变量：
- `HPC_PORT` 监听端口（默认 8090）
- `HPC_EMBED_KEY` 智谱 embedding API Key 原文（PaaS/容器部署推荐，优先级最高）
- `HPC_KEY_FILE` 智谱 key 文件路径（本地部署；与 HPC_EMBED_KEY 二选一）
- `HPC_MEMORY_KEY` Memory System Key（鉴权，必须）
- `HPC_DB` SQLite 路径（**必须放本地盘**，默认 `/var/tmp/hippocampus.db`；放云盘同步目录会 disk I/O error）
- `HPC_MIN_SCORE` 拒答阈值（默认 0.5；实测 embedding-3 余弦分布整体偏高，0.25 挡不住无关查询，2026-09-28 上调）

## 契约自测
```bash
# 先起服务（HPC_MEMORY_KEY=test-key-123），再：
HPC_TEST_PORT=8123 python3 contract_test.py
# 覆盖：Add 分块/回传一致性、Health、鉴权 401、Search 格式/top_k/排序、user_id 隔离
```

## 摸底结论（recon_20260928/ 下四轮 probe）
- 检索达标：BM25+embedding top100 对 BEAM 探针题 18/18 命中
- 组装是共同瓶颈：答案模型（glm-4-flash/deepseek-chat）在矛盾/事件/推理题全崩，基础题 60-80%
- 拉分点：基础题满分 + abstention 拒答 + 少而准返回
- v0.1 已知限制：单一 cosine 阈值挡不住"主题共现但关系缺失"型 abstention（待 Full 前迭代）

## 部署（公网）
服务默认监听 0.0.0.0，可配合反向代理或隧道对外提供 API：
```bash
./start.sh 8123 your-memory-key
# 示例：cloudflared quick tunnel 暴露公网
cloudflared tunnel --url http://127.0.0.1:8123 --no-autoupdate --protocol http2
```
注意：quick tunnel 无 uptime 保证，正式评测期建议使用命名隧道或固定公网。

### Render（推荐，境外 PaaS，URL 固定）
仓库已含 `render.yaml` + `Procfile`。在 Render Dashboard 用 GitHub 仓库 `syyr1987/hippocampus-mem` 创建 Blueprint / Web Service，然后配置两个环境变量（不进仓库）：
- `HPC_EMBED_KEY`：智谱 embedding API Key 原文
- `HPC_MEMORY_KEY`：Memory System Key（与 AML 申请绑定的一致）
