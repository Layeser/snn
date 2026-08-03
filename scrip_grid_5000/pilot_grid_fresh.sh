#!/usr/bin/env bash
# Alias — nettoyage local mode auto (voir g5k_fresh.sh).
exec "$(dirname "${BASH_SOURCE[0]}")/g5k_fresh.sh" --local
