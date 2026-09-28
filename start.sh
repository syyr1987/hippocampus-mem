#!/bin/bash
# 海马体记忆服务 - 一键启动
# 用法: ./start.sh [port] [memory_key]
# 依赖: python3 + fastapi/uvicorn/pydantic（pip install -r requirements.txt）
set -e
cd "$(dirname "$0")"

PORT="${1:-8123}"
MEMORY_KEY="${2:-$(cat .memory_key 2>/dev/null || echo '')}"

# 智谱 embedding key 通过环境变量 HPC_KEY_FILE 指定（指向包含 API key 的文件）
if [ -z "$HPC_MEMORY_KEY" ]; then
  export HPC_MEMORY_KEY="$MEMORY_KEY"
fi
export HPC_PORT="$PORT"

echo "== 海马体记忆服务启动 =="
echo "  port:       $PORT"
echo "  embed key:  ${HPC_KEY_FILE:-未设置(仅 BM25 模式)}"
echo "  memory key: ${HPC_MEMORY_KEY:+已设置}${HPC_MEMORY_KEY:-未设置(无鉴权)}"
echo "  日志: nohup 输出到 hippocampus.log"
nohup python3 app.py > hippocampus.log 2>&1 &
echo "  PID: $!"
sleep 2
curl -s "http://127.0.0.1:${PORT}/health" && echo " <- health OK"
