#!/bin/bash
# Copie les 9 scripts de campagne dans scrip_run/ pour l'orchestrateur.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/scrip_run"
EXP="$ROOT/experiences"

mkdir -p "$DEST"

copy() {
    cp "$EXP/$1" "$DEST/$(basename "$1")"
    echo "  -> scrip_run/$(basename "$1")"
}

echo "=== Campagne chicoree (Lille) — CIFAR-10 grille 2×2 ==="
copy cifar10/grid_hp_linear_optuna_t27_200ep.sh
copy cifar10/grid_hp_optuna_t27_200ep.sh
copy cifar10/grid_sdt_optuna_t27_200ep.sh
copy cifar10/grid_statten_optuna_t27_200ep.sh

echo "=== Campagne chuc (Lille) — CIFAR-10-DVS LR sweep ==="
copy cifar10-dvs/subset_lr1e-4.sh
copy cifar10-dvs/subset_lr1e-5.sh
copy cifar10-dvs/subset_lr1e-6.sh
copy cifar10-dvs/subset_lr1e-7.sh

echo "=== Campagne sirius (Lyon) — CIFAR-10-DVS Optuna ==="
copy cifar10-dvs/subset_optuna_20x30.sh

echo ""
echo "Pret. Depuis la racine du repo :"
echo "  git push && make pilot-grid    # une passe"
echo "  make grid-watch                # surveillance continue"
