#!/usr/bin/env bash
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_gpu_queue.sh" --cluster chuc "$@"
