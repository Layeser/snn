# Grid'5000 — lancement des expériences HP-STAtten

Toutes les commandes utilisent le préfixe **`g5k-`** (`make g5k-help`).

| Mot-clé | Signification |
|---------|---------------|
| **auto** | File OAR nuit, sans date fixe (machine locale) |
| **book** | Réserver un créneau `-r` (frontale) |
| **run** | Lancer les files sur nœud (au créneau) |
| **restart** | Nettoyer + relancer |
| **clean** | Nettoyer seulement |
| **check** | Afficher la file sans lancer |
| **test** | Préparer scripts smoke |

---

## Aide rapide

```bash
make g5k-help
```

---

## Mode AUTO — 1 job OAR par dossier cluster

**Dossier** : `scrip_run/lille/chicoree/`, `chuc/`, `lyon/sirius/`

Déposez **plusieurs `.sh`** dans chaque dossier → **`make g5k-auto` soumet 1 job OAR par dossier** (pas 1 par script).

Sur le nœud, `run_gpu_queue.sh` :
- parallelise jusqu'à **N GPU** (4 chicorée/chuc, 8 sirius — automatique)
- enchaîne la file dès qu'un GPU se libère

Exemple campagne complète :

```
scrip_run/lille/chicoree/exp1.sh … exp6.sh  →  1 job OAR (4 GPU parallèles + file)
scrip_run/lille/chuc/exp_a.sh … exp_d.sh    →  1 job OAR
scrip_run/lyon/sirius/exp_x.sh …            →  1 job OAR
```

→ **3 jobs OAR** au total pour Lille + Lyon.

### Workflow recommandé

```bash
# 1. Une seule fois : envoie tous les .sh vers les frontales (oarsub)
make g5k-auto
# → Les jobs tournent sur Grid'5000 même si votre PC s'éteint.

# 2. (Optionnel, PC allumé) Suivre + rapatrier les résultats en local
make g5k-auto-follow
# → Ne resoumet PAS de nouveaux jobs, seulement oarstat + téléchargement outputs/
```

| Commande | Soumet ? | Rôle |
|----------|----------|------|
| `g5k-auto` | **Oui** (1 fois) | 1 job OAR par dossier cluster → file GPU sur le nœud |
| `g5k-auto-follow` | **Non** | Suivre jobs en cours, rapatrier `outputs/` quand fini |
| `g5k-auto-restart` | Oui | Nettoyer + resoumettre |
| `g5k-auto-clean` | Non | Nettoyer seulement |

```bash
# Reprendre le suivi sans resoumettre (jobs déjà lancés)
make g5k-auto-follow

# Nettoyer + soumettre à nouveau
make g5k-auto-restart
```

---

## Mode MANUEL — créneau `-r` (frontale)

**Dossiers** : `chicoree_experiences/`, `chuc_experiences/`, `sirius_experiences/`

### 1. Réserver (`book`)

```bash
# flille — chicorée + chuc
make g5k-book-lille \
  RESERVE_START="2026-08-04 19:00:00" \
  RESERVE_END="2026-08-05 09:00:00" \
  RESERVE_TAG=04

# flyon — sirius
make g5k-book-lyon \
  RESERVE_START="2026-08-04 19:00:00" \
  RESERVE_END="2026-08-05 09:00:00" \
  RESERVE_TAG=04
```

JOB_ID → `scrip_grid_5000/manual_jobs.env`

### 2. Lancer au créneau (`run`)

```bash
make g5k-run-lille    # flille
make g5k-run-lyon     # flyon
```

### 3. Nettoyer + relancer (`restart`)

```bash
make g5k-restart-lille
make g5k-restart-lyon
```

Supprime archives + logs, puis lance les `.sh` présents dans `*_experiences/`.

### 4. Vérifier la file (`check`)

```bash
make g5k-check-lille
make g5k-check-lyon
```

---

## Tests smoke (jour)

```bash
make g5k-test-chicoree    # 6 scripts → chicoree_experiences/
make g5k-test-sirius      # 10 scripts → sirius_experiences/
# puis oarsub -I -t day ... et bash scrip_grid_5000/run_*_queue.sh
```

---

## Tableau complet

| Commande | Où | Action |
|----------|-----|--------|
| `g5k-auto` | local | 1 job OAR par dossier cluster (file GPU auto) |
| `g5k-auto-follow` | local | Suivre + rapatrier (sans resoumettre) |
| `g5k-auto-restart` | local | Nettoie + soumet `scrip_run/` |
| `g5k-auto-clean` | local | Nettoie seulement |
| `g5k-book-lille` | flille | Réserve chicorée + chuc |
| `g5k-book-lyon` | flyon | Réserve sirius |
| `g5k-run-lille` | flille | Lance files manuelles Lille |
| `g5k-run-lyon` | flyon | Lance file sirius |
| `g5k-restart-lille` | flille | Nettoie + lance Lille |
| `g5k-restart-lyon` | flyon | Nettoie + lance Lyon |
| `g5k-clean-manual` | partout | Nettoie archives/logs manuel |
| `g5k-check-lille` | flille | Dry-run file Lille |

---

## Format script d'expérience

```bash
RUN_NAME="mon_run_unique"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
cd HPSTAtten
python -m scripts.train --config ... --save-dir "$OUTPUT_DIR"
```

Gabarit : `experiences/template.sh`

---

## Quel mode ?

| Situation | Commande |
|-----------|----------|
| Nuit libre sur le Gantt, date fixe | `g5k-book-*` + `g5k-run-*` |
| Dès qu'un GPU se libère (nuit, sans date fixe) | `g5k-auto` |
| Test file GPU en journée | `g5k-test-chicoree` + `run_chicoree_queue.sh` |

Docs : `Notes/gpureser.md`, `Notes/gpures_sirius.md`

---

## Anciens noms (alias)

| Ancien | Nouveau |
|--------|---------|
| `pilot-grid-watch` | `g5k-auto-follow` |
| `pilot-grid-fresh` | `g5k-auto-restart` |
| `manual-reserve-lille` | `g5k-book-lille` |
| `manual-run-lille-fresh` | `g5k-restart-lille` |
