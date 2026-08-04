# Guideline — Lancer des entraînements sur Grid'5000

Ce document décrit le workflow complet pour lancer plusieurs entraînements en parallèle depuis votre **machine locale**, via l'orchestrateur Python (`pilot_grid/`).

---

## Vue d'ensemble

```
LOCAL (dev + pilotage)  →  GitHub  →  Grid'5000 (exécution GPU)
```

- Vous développez et versionnez en **local**.
- Vous poussez le code sur **GitHub** (`git push`).
- L'orchestrateur, lancé en **local**, se connecte à Grid'5000, fait `git pull`, soumet les jobs, suit leur avancement et rapatrie les résultats.

Avec **2 sites** et `max_jobs_per_site: 2` → **4 entraînements en parallèle** (2 jobs × 1 GPU par site).

---



## Setup initial (une seule fois)



### Sur Grid'5000 (frontale)

```bash
ssh access.grid5000.fr
ssh lille   # ou votre site

git clone <URL_DE_VOTRE_REPO> ~/internship/snn
cd ~/internship/snn
make setup
make download-data
# Pour CIFAR-10-DVS uniquement (conversion longue, une fois) :
# make prepare-cifar10-dvs
```

Vérifiez que le dossier s'appelle bien `~/internship/snn` (doit correspondre à `remote_project_dir` dans `pilot_grid/config.yaml`).

### Sur votre machine locale

```bash
git clone <URL_DE_VOTRE_REPO> snn   # si pas déjà fait
cd snn

make setup
make download-data   # optionnel en local si vous n'entraînez qu'en distant

pip install paramiko pyyaml   # dépendances de l'orchestrateur

# Clé SSH sans mot de passe vers Grid'5000
ssh-copy-id kasekou@access.grid5000.fr
ssh access.grid5000.fr   # doit passer sans mot de passe
```



### Configuration de l'orchestrateur

Éditez `scrip_grid_5000/pilot_grid/config.yaml` :

```yaml
user: kasekou
remote_project_dir: internship/snn

sites:
  - lille
  - lyon          # ajoutez un 2e site
max_jobs_per_site: 2

git_enabled: true
git_branch: main
```

---



## Ressources OAR — tout centraliser dans `config.yaml`

**Recommandé** : régler file, type, GPU et walltime dans `pilot_grid/config.yaml` — **pas besoin
de modifier les 9 scripts** dans `scrip_run/` ou `experiences/`.

```yaml
sites:
  - lille
max_jobs_per_site: 2
walltime: "10:00:00"
oar_type: day              # day | night
oar_queue: besteffort      # besteffort | default
oar_resources: host=1/gpu=1
```


| Champ           | Rôle                   | Exemples                         |
| --------------- | ---------------------- | -------------------------------- |
| `oar_resources` | GPU / nœuds (`-l`)     | `host=1/gpu=1` (1 GPU, mono-GPU) |
| `oar_type`      | Créneau horaire (`-t`) | `day`, `night`                   |
| `oar_queue`     | File OAR (`-q`)        | `besteffort`, `default`          |
| `walltime`      | Durée max du job       | `"10:00:00"`                     |


**Basculer besteffort ↔ default** : une seule ligne à changer :

```yaml
oar_queue: besteffort   # jour, préemptible
# oar_queue: default    # nuit / prioritaire
```

Les lignes `# OAR_option` dans les scripts sont **optionnelles** (surcharge locale seulement
si le champ config est vide).

### GPU : `gpu=1` ou `gpu=2` ?


| Configuration                    | Effet                                                                 |
| -------------------------------- | --------------------------------------------------------------------- |
| `gpu=1` + `max_jobs_per_site: 2` | **2 entraînements distincts** en parallèle par site, chacun sur 1 GPU |
| `gpu=2` + `max_jobs_per_site: 2` | **1 entraînement** qui réserve 2 GPU (multi-GPU / torchrun)           |


Notre entraînement HPSTAtten est **mono-GPU** (`python -m scripts.train`). Donc :

- `gpu=1` → 2 jobs/site = 2 expériences différentes qui tournent en même temps.
- `gpu=2` → inutile sauf si vous modifiez le code pour du multi-GPU.

> **Important** : le `#` devant `OAR_option` est **obligatoire** — c'est le format lu par
> l'orchestrateur (pas une ligne « désactivée »). Pour changer de file, modifiez la
> valeur (`default` → `besteffort`), sans retirer le `#`.

```bash
# OAR_option -q default    # réservation classique
# OAR_option -q besteffort # jobs préemptibles (peuvent être tués)
```

---



## Organisation des dossiers

```
scrip_grid_5000/
├── experiences/          # BIBLIOTHÈQUE — vos recettes d'expériences (versionné git)
│   ├── template.sh
│   ├── cifar10/
│   │   └── ablation_lr_step.sh
│   └── cifar10-dvs/
│       └── train.sh
├── scrip_run/            # FILE D'ATTENTE — ce qu'on lance maintenant (non versionné)
│   └── (copier ici les .sh à exécuter)
└── pilot_grid/           # Orchestrateur Python
```


| Dossier        | Rôle                                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `experiences/` | Stocker et organiser vos expériences (par dataset, campagne, hyperparamètres). L'orchestrateur **ne lit pas** ce dossier. |
| `scrip_run/`   | File d'attente active. L'orchestrateur **scanne uniquement** les `.sh` à la racine de ce dossier et les lance.            |


**Pourquoi deux dossiers ?**

- `experiences/` = carnet de recettes (permanent, réutilisable).
- `scrip_run/` = plateau de commandes du jour. Les fichiers y sont **déplacés** puis **archivés** après lancement — ils disparaissent de la file.

---



## Workflow : créer et lancer une expérience



### 1. Créer l'expérience (local)

Copiez le gabarit et adaptez-le :

```bash
cp scrip_grid_5000/experiences/template.sh \
   scrip_grid_5000/experiences/cifar10/mon_experiment.sh
```

Dans le fichier, modifiez **3 zones** :

```bash
# ---- Ressources OAR (décommenter) ----
# OAR_option -q default
# OAR_option -l host=1/gpu=1

# ---- Identité du run (OBLIGATOIRE, nom unique) ----
RUN_NAME="cifar10_mon_experiment"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ---- Commande d'entraînement ----
cd HPSTAtten
python -m scripts.train \
    --config config/ablation/step.yml \
    --dataset cifar10 \
    --save-dir "$OUTPUT_DIR"
```

Rangez par dataset (`cifar10/`, `cifar10-dvs/`, etc.) selon vos besoins.

### 2. Pousser sur GitHub

```bash
git add scrip_grid_5000/experiences/
git commit -m "Add experiment ..."
git push origin main
```

L'orchestrateur fera `git pull` sur Grid'5000 avant chaque soumission.

### 3. Mettre en file d'attente

Copiez **uniquement** les expériences à lancer **maintenant** :

```bash
cp scrip_grid_5000/experiences/cifar10/mon_experiment.sh scrip_grid_5000/scrip_run/
# Répéter pour chaque expérience à lancer (jusqu'à N fichiers)
```



### 4. Lancer l'orchestrateur (depuis la racine du repo)

```bash
cd /chemin/vers/snn
python scrip_grid_5000/pilot_grid/main.py
```

Pour automatiser le suivi (vérif statut + récupération + lancement du suivant) :

```bash
watch -n 300 "python scrip_grid_5000/pilot_grid/main.py"
```

---



## Intervalle `watch -n` : faut-il l'augmenter pour des runs de 2 h+ ?

**Non.** `-n 300` = l'orchestrateur **vérifie toutes les 5 minutes**, pas la durée des jobs.


| Paramètre                            | Signification                                |
| ------------------------------------ | -------------------------------------------- |
| `watch -n 300`                       | Relance l'orchestrateur toutes les **5 min** |
| `walltime: "10:00:00"` (config.yaml) | Durée **max** d'un job OAR sur Grid'5000     |


Un entraînement de 2 h tourne pendant 2 h ; l'orchestrateur vérifie en parallèle toutes les 5 min s'il est fini.

Recommandations :

- `300` (5 min) : bon compromis par défaut.
- `600` (10 min) : si vous voulez moins de connexions SSH.
- **Ne pas** mettre `-n 7200` : inutile, vous attendriez 2 h entre chaque vérif.

À chaque tournée, l'orchestrateur :

1. Vérifie les jobs en cours.
2. Si un job est **terminé** → rapatrie les résultats dans `outputs/<RUN_NAME>/`.
3. S'il reste des places libres et des `.sh` dans `scrip_run/` → lance le suivant.

---



## Capacité parallèle (exemple 2 sites)

```
config.yaml :
  sites: [lille, lyon]
  max_jobs_per_site: 2

→ 4 slots GPU en parallèle (2 × Lille + 2 × Lyon)
```

Si vous déposez **6** fichiers dans `scrip_run/` :

- Tournée 1 : lance 4 jobs.
- Quand un job finit : récupération + lance le 5ᵉ, puis le 6ᵉ.

---



## Suivi

**En local :**

```bash
cat scrip_grid_5000/pilot_grid/run_status.json
ls outputs/
```

**Sur Grid'5000 (optionnel) :**

```bash
ssh access.grid5000.fr
ssh lille
oarstat -u kasekou
```

---



## Checklist avant un lancement

- [ ] `~/internship/snn` prêt sur Grid'5000 (`make setup`, `make download-data`)
- [ ] `config.yaml` : login, sites, `max_jobs_per_site`
- [ ] SSH sans mot de passe vers `access.grid5000.fr`
- [ ] Chaque `.sh` a un `RUN_NAME` **unique**
- [ ] Chaque `.sh` a les lignes `# OAR_option -q ...` et `# OAR_option -l host=1/gpu=1` renseignées
- [ ] Code poussé sur GitHub (`git push`)
- [ ] Fichiers copiés dans `scrip_run/`
- [ ] Lancement depuis la **racine** du repo
- [ ] Machine locale allumée (pour `watch` ou relances manuelles)

---



## Résumé en une phrase

**Local** : créez vos expériences dans `experiences/`, poussez sur GitHub, copiez celles à lancer dans `scrip_run/`, puis lancez `watch -n 300 "python scrip_grid_5000/pilot_grid/main.py"` — l'orchestrateur synchronise le code, soumet jusqu'à 4 jobs (2 sites × 2), récupère les résultats et enchaîne la file automatiquement.