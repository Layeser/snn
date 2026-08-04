# File d'attente — mode **auto** (`make g5k-auto`)

Déposez vos scripts ici (un `.sh` par expérience) — **copie manuelle** depuis `experiences/` :

> **Note** : `scrip_run/lille/` et `lyon/` sont **gitignorés**.  
> `make g5k-auto` **envoie vos `.sh` par SCP** sur la frontale avant chaque `oarsub`.  
> Pas besoin de `git push` les scripts de campagne (seulement le code outils).

```
scrip_run/
  lille/
    chicoree/    ← 1 job OAR, file GPU 4 parallèles + suite automatique
    chuc/        ← 1 job OAR, file GPU 4 parallèles
  lyon/
    sirius/      ← 1 job OAR, file GPU 8 parallèles
```

**Important** : `make g5k-auto` soumet **1 job OAR par dossier cluster**, pas 1 par `.sh`.
Le parallélisme et l'enchaînement de la file sont gérés sur le nœud par `run_gpu_queue.sh`.

Le nom du sous-dossier = cluster OAR (`-p chicoree`, `chuc`, `sirius`).

**Mode manuel** (réservation `-r`) : utilisez plutôt `chicoree_experiences/`, `chuc_experiences/`, `sirius_experiences/`.

Documentation complète : **`scrip_grid_5000/README.md`**
