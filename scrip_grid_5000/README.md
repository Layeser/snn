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
# 0. Copiez vos .sh dans scrip_run/<site>/<cluster>/ (local, gitignoré)

# 1. Aligner code outils + frontales
make g5k-fresh

# 2. Soumettre (upload SCP des .sh + oarsub)
make g5k-auto
# → Les jobs tournent sur Grid'5000 même si votre PC s'éteint.

# 3. (Optionnel) Suivre + rapatrier les résultats en local
make g5k-auto-follow
```

| Commande | Soumet ? | Rôle |
|----------|----------|------|
| `g5k-fresh` | Non | Nettoie local **+** flille/flyon ; `git restore scrip_grid_5000/` sur les frontales |
| `g5k-auto` | **Oui** (1 fois) | 1 job OAR par dossier cluster → file GPU sur le nœud |
| `g5k-auto-follow` | **Non** | Suivre jobs en cours, rapatrier `outputs/` quand fini |
| `g5k-auto-restart` | Oui | `g5k-fresh` + resoumettre |
| `g5k-auto-clean` | Non | Nettoyer **local** seulement (alias `g5k-fresh --local`) |

```bash
# Reprendre le suivi sans resoumettre (jobs déjà lancés)
make g5k-auto-follow

# Nouvelle campagne : fresh partout puis soumettre
make g5k-auto-restart
```

## Nettoyage unifié (`g5k-fresh`)

Avant chaque campagne, pour repartir propre et **identique sur local + frontales** :

```bash
git push                              # vos .sh dans scrip_run/ ou *_experiences/
make g5k-fresh                        # depuis votre PC
```

**Local** : `run_status.json`, archives, `outputs/`  
**flille + flyon** (SSH) : `git pull --ff-only` + `git restore scrip_grid_5000/` + même nettoyage

Vos `.sh` actifs (non archivés) sont **conservés**. Seuls les artefacts d'exécution sont supprimés.

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

Chaque cluster utilise **son JOB_ID** → `oarsub -C` → `run_gpu_queue.sh` (4/4/8 GPU + file).

**Campagne réelle** (recommandé — même dossiers que `g5k-auto`) :

```bash
# PC local — déposer vos .sh dans scrip_run/lille/chicoree/, chuc/, lyon/sirius/
make g5k-sync-scrip-run          # envoie les scripts sur flille + flyon

# Au créneau (flille / flyon), quand state = Running :
make g5k-check-lille-scrip       # dry-run : compte les scripts + vérifie JOB_ID
make g5k-run-lille-scrip         # chicorée (JOB_CHICOREE) + chuc (JOB_CHUC)

make g5k-check-lyon-scrip
make g5k-run-lyon-scrip            # sirius (JOB_SIRIUS)
```

**Alternative** — dossiers `*_experiences/` :

```bash
make g5k-run-lille    # flille — chicoree_experiences/ + chuc_experiences/
make g5k-run-lyon     # flyon — sirius_experiences/
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

Smoke tests pour valider la chaîne complète **sans lancer une vraie campagne**.

### Ce que chaque smoke valide

| Mode | Commande | Valide |
|------|----------|--------|
| **Auto** | `make g5k-test-auto-smoke` | 22 scripts → 3 jobs OAR, file GPU complète (4+4+8), **sans intervention** |
| **Manuel chicorée** | `g5k-test-chicoree` + `oarsub -I` + queue | 6 jobs → 4 GPU + file (`chicoree_experiences/`) |
| **Manuel chuc** | `g5k-test-chuc` + `oarsub -I` + queue | 6 jobs → 4 GPU + file (`chuc_experiences/`) |
| **Manuel sirius** | `g5k-test-sirius` + `oarsub -I` + queue | 10 jobs → 8 GPU + file (`sirius_experiences/`) |

> **Ne pas confondre** : `g5k-test-chicoree` prépare le mode **manuel** sur flille.  
> Pour tout automatiser depuis le PC : **`make g5k-test-auto-smoke`**.

---

### Smoke AUTO — une commande (PC local)

Génère **22 scripts** dans `scrip_run/`, les **envoie par SCP**, soumet **3 jobs OAR** (type day) :

| Cluster | Scripts | Parallélisme |
|---------|---------|--------------|
| chicorée | 6 | 4 GPU + 2 en file |
| chuc | 6 | 4 GPU + 2 en file |
| sirius | 10 | 8 GPU + 2 en file |

```bash
make g5k-test-auto-smoke

# (Optionnel) Suivre et rapatrier
make g5k-auto-smoke-watch
```

Étapes séparées :

```bash
make g5k-test-auto      # génère scrip_run/ uniquement
make g5k-auto-smoke     # soumet (SCP + oarsub)
```

Vérifier sur les frontales (après soumission) :

```bash
ls ~/internship/snn/scrip_grid_5000/scrip_run/lille/chicoree/smoke_*.sh   # 6
ls ~/internship/snn/scrip_grid_5000/scrip_run/lille/chuc/smoke_*.sh       # 6
ls ~/internship/snn/scrip_grid_5000/scrip_run/lyon/sirius/smoke_*.sh      # 10
oarstat -u kasekou
```

> **Ne pas utiliser `g5k-auto` / `g5k-test-auto-smoke`** si vous voulez garder vos réservations `-r` :
> ces commandes créent de **nouveaux** jobs OAR (file FIFO). Voir ci-dessous.

---

### Réserver le smoke depuis le PC (`g5k-book-smoke`)

Crée **3 réservations `-r`** (chicorée, chuc, sirius) via SSH — sans être connecté aux frontales.

**1. Config** — copier et éditer le créneau (~20 min) :

```bash
cp scrip_grid_5000/reserve_smoke.yaml.example scrip_grid_5000/reserve_smoke.yaml
# éditer reserve_start, reserve_end (ou duration_minutes), tag
```

Exemple `reserve_smoke.yaml` :

```yaml
reserve_start: "2026-08-04 09:00:00"
reserve_end: "2026-08-04 09:20:00"   # ou laisser vide + duration_minutes: 20
tag: smoke04
```

**2. Réserver (PC local) :**

```bash
make g5k-book-smoke-check   # dry-run
make g5k-book-smoke          # soumet sur lille + lyon
```

Les `JOB_ID` sont enregistrés dans `scrip_grid_5000/manual_jobs.env` (local + frontales).

**3. Au créneau** → `make g5k-run-smoke-reserved-lille` / `g5k-run-smoke-reserved-lyon` (sur frontales).

---

### Smoke sur réservations `-r` existantes (sans `oardel`)

Utilise vos **JOB_ID déjà réservés** (`manual_jobs.env`) — **aucun nouvel `oarsub`**.

**1. Vérifier / renseigner les IDs** (sur chaque frontale, une fois) :

```bash
# flille — éditer scrip_grid_5000/manual_jobs.env
JOB_CHICOREE=2179957    # votre ID chicorée
JOB_CHUC=2179958        # votre ID chuc

# flyon
JOB_SIRIUS=2056054      # votre ID sirius
```

**2. Au créneau (≈ 9h), quand `state = Running` :**

```bash
# flille — vérifie d'abord
oarstat -fj $JOB_CHICOREE | grep state
oarstat -fj $JOB_CHUC | grep state
make g5k-run-smoke-reserved-lille

# flyon
oarstat -fj $JOB_SIRIUS | grep state
make g5k-run-smoke-reserved-lyon
```

Dry-run (sans lancer) :

```bash
make g5k-run-smoke-reserved-lille-check
make g5k-run-smoke-reserved-lyon-check
```

**Ce que ça fait :** génère 22 scripts smoke dans `scrip_run/`, se connecte à vos jobs `-r` via `oarsub -C`, lance `run_gpu_queue.sh` (4+4+8 GPU + file).

**Campagne réelle ensuite (auto, sans `-r`) :** déposez vos `.sh` dans `scrip_run/` → `make g5k-auto` (nouveaux jobs FIFO nuit).

---

### Smoke MANUEL — sur la frontale (réservation interactive)

Teste **file GPU + parallélisme** sur le nœud (`run_gpu_queue.sh`).  
Les scripts `*_experiences/*.sh` sont **gitignorés** : lancez `make g5k-test-*` **sur flille ou flyon** (ou copiez les scripts à la main).

#### Chicorée — 4 GPU + file (6 jobs)

```bash
# Sur flille
cd ~/internship/snn && git pull --ff-only
make g5k-test-chicoree
ls scrip_grid_5000/chicoree_experiences/smoke_*.sh   # 6 fichiers

# Réservation interactive day 15 min (4 GPU)
oarsub -I -p chicoree -t exotic -t day \
  -l host=1/gpu=4,walltime=0:15:00 -q default

# Sur le nœud
bash scrip_grid_5000/run_chicoree_queue.sh

# Suivi
tail -f outputs/chicoree_queue/scheduler.log
ls scrip_grid_5000/chicoree_experiences/archive/done/
```

#### Chuc — 4 GPU + file (6 jobs)

```bash
# Sur flille
cd ~/internship/snn && git pull --ff-only
make g5k-test-chuc
ls scrip_grid_5000/chuc_experiences/smoke_*.sh   # 6 fichiers

oarsub -I -p chuc -t day \
  -l host=1/gpu=4,walltime=0:15:00 -q default

bash scrip_grid_5000/run_chuc_queue.sh
tail -f outputs/chuc_queue/scheduler.log
ls scrip_grid_5000/chuc_experiences/archive/done/
```

#### Sirius — 8 GPU + file (10 jobs)

```bash
# Sur flyon
cd ~/internship/snn && git pull --ff-only
make g5k-test-sirius
ls scrip_grid_5000/sirius_experiences/smoke_*.sh   # 10 fichiers

oarsub -I -p sirius -t exotic -t day \
  -l host=1/gpu=8,walltime=0:20:00 -q default

bash scrip_grid_5000/run_sirius_queue.sh
tail -f outputs/sirius_queue/scheduler.log
ls scrip_grid_5000/sirius_experiences/archive/done/
```

---

### Commandes smoke (récap)

| Commande | Où | Action |
|----------|-----|--------|
| `g5k-test-auto-smoke` | local | Prépare + soumet smoke auto |
| `g5k-test-auto` | local | Copie 3 scripts → `scrip_run/` |
| `g5k-auto-smoke` | local | Soumet 3 jobs smoke (config jour) |
| `g5k-auto-smoke-watch` | local | Suivre + rapatrier |
| `g5k-test-chicoree` | flille | 6 scripts → `chicoree_experiences/` |
| `g5k-test-chuc` | flille | 6 scripts → `chuc_experiences/` |
| `g5k-test-sirius` | flyon | 10 scripts → `sirius_experiences/` |

Docs détaillées : `Notes/gpureser.md`, `Notes/gpures_sirius.md`

---

## Tableau complet

| Commande | Où | Action |
|----------|-----|--------|
| `g5k-auto` | local | 1 job OAR par dossier cluster (file GPU auto) |
| `g5k-auto-follow` | local | Suivre + rapatrier (sans resoumettre) |
| `g5k-auto-restart` | local | `g5k-fresh` + soumettre |
| `g5k-auto-clean` | local | Nettoyage local seulement |
| `g5k-fresh` | local → SSH | Local + flille + flyon alignés sur git |
| `g5k-book-lille` | flille | Réserve chicorée + chuc |
| `g5k-book-lyon` | flyon | Réserve sirius |
| `g5k-book-smoke` | local | Réserve smoke 3 clusters (`reserve_smoke.yaml`) |
| `g5k-book-smoke-check` | local | Dry-run réservations smoke |
| `g5k-run-lille` | flille | Lance files manuelles Lille |
| `g5k-run-lyon` | flyon | Lance file sirius |
| `g5k-restart-lille` | flille | `g5k-fresh` + lancer files Lille |
| `g5k-restart-lyon` | flyon | `g5k-fresh` + lancer file Lyon |
| `g5k-clean-manual` | local | Alias nettoyage local (`g5k-fresh --local`) |
| `g5k-check-lille` | flille | Dry-run file Lille |
| `g5k-test-auto-smoke` | local | Prépare + soumet smoke auto |
| `g5k-test-auto` | local | Prépare smoke auto → `scrip_run/` |
| `g5k-auto-smoke` | local | Soumet smoke auto (jour, 1 GPU/cluster) |
| `g5k-auto-smoke-watch` | local | Suivre smoke auto |
| `g5k-test-chicoree` | flille | 6 scripts → `chicoree_experiences/` |
| `g5k-test-chuc` | flille | 6 scripts → `chuc_experiences/` |
| `g5k-test-sirius` | flyon | 10 scripts → `sirius_experiences/` |

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
| Test orchestrateur auto (jour) | `make g5k-test-auto-smoke` |
| Test file GPU chicorée (manuel) | `g5k-test-chicoree` + `run_chicoree_queue.sh` |
| Test file GPU chuc (manuel) | `g5k-test-chuc` + `run_chuc_queue.sh` |
| Test file GPU sirius (manuel) | `g5k-test-sirius` + `run_sirius_queue.sh` |

Docs : `Notes/gpureser.md`, `Notes/gpures_sirius.md`

---

## Anciens noms (alias)

| Ancien | Nouveau |
|--------|---------|
| `pilot-grid-watch` | `g5k-auto-follow` |
| `pilot-grid-fresh` | `g5k-auto-restart` |
| `manual-reserve-lille` | `g5k-book-lille` |
| `manual-run-lille-fresh` | `g5k-restart-lille` |
