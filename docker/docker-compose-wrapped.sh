#!/bin/bash
# Decrypts secrets.enc.env via sops/age and runs docker compose with those
# credentials exported, so systemd (which never sources .envrc) still gets
# POSTGRES_PASSWORD and friends.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/sops/age/keys.txt}"
eval "$(sops -d --output-type dotenv secrets.enc.env | sed 's/^/export /')"

exec docker compose "$@"
