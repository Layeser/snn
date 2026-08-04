# Besteffort Grid5000

Lance des expériences sur **n'importe quel GPU disponible** (file `besteffort`), sans réserver chicorée, chuc ou sirius.

## Workflow (100 % dynamique)

**Lancer sur la frontale** (flille pour Lille, flyon pour Lyon) :

```bash
cd ~/internship/snn
git pull
make besteffort-watch-lille    # sur flille
make besteffort-watch-lyon     # sur flyon
```

Pas de paramiko : `besteffort_local.py` utilise uniquement Python stdlib + `oarsub` local.

1. Copiez vos `.sh` dans :
   ```
   besteffort_lille/*.sh
   besteffort_lyon/*.sh
   ```
2. `git pull` sur la frontale
3. Lancez :

```bash
make besteffort-watch-lille    # Lille seulement
make besteffort-watch-lyon     # Lyon seulement
make besteffort-watch          # les deux sites
```

**Aucune liste de noms en dur** : l'orchestrateur découvre tous les `*.sh` à chaque cycle. Ajoutez ou retirez des scripts quand vous voulez ; au prochain cycle (10 min), les nouveaux sont soumis.

```bash
make besteffort-list           # voir la file locale
make besteffort-check-lille    # 1 tour sans soumettre (alias --follow-only)
make besteffort-fresh        # reset l'état JSON (scripts conservés)
```

## Comportement

- **1 job OAR = 1 GPU = 1 `.sh`** — pas de `-p sirius|chicoree` (cluster libre)
- `-t exotic` requis sur Lyon pour accéder aux GPU ; `-t night` en plus sur Lille
- Préemption → **resoumission auto**
- Entraînement → reprise `save/last.pt`
- Fin OK → script déplacé dans `besteffort_<site>/archive/done/`

Chaque script doit définir un `RUN_NAME` unique et `OUTPUT_DIR`.

## Config

`grid5k/config.yaml` — login, walltime, `max_jobs_per_site` (par défaut 8).

Indépendant de `scrip_grid_5000/`.
