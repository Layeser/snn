# Besteffort Grid5000

Lance des expériences sur **n'importe quel GPU disponible** (file `besteffort`), sans réserver chicorée, chuc ou sirius.

## Workflow (100 % dynamique)

1. Copiez vos `.sh` dans le dossier du site :
   ```
   besteffort_lille/*.sh
   besteffort_lyon/*.sh
   ```
2. `git push` (sync code sur les frontales)
3. Lancez la surveillance :

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

- **1 job OAR = 1 GPU = 1 `.sh`** — pas de `-p cluster`
- Préemption → **resoumission auto**
- Entraînement → reprise `save/last.pt`
- Fin OK → script déplacé dans `besteffort_<site>/archive/done/`

Chaque script doit définir un `RUN_NAME` unique et `OUTPUT_DIR`.

## Config

`grid5k/config.yaml` — login, walltime, `max_jobs_per_site` (par défaut 8).

Indépendant de `scrip_grid_5000/`.
