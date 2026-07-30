# File d'attente Grid'5000 — déposez vos scripts ici
#
#   scrip_run/
#     lille/
#       chicoree/    ← 1 script = 1 job OAR (tous soumis ; OAR file si saturé)
#       chuc/
#     lyon/
#       sirius/
#
# Le nom du sous-dossier = cluster OAR (-p chicoree, chuc, sirius…).
# Options par cluster : pilot_grid/cluster_defaults.yaml
# Surcharge locale : # OAR_option dans le .sh
#
# Nouvelle expérience :
#   Mode bundle (recommandé campagne) : prepare_campaign_queue.sh → 3 jobs
#   Mode unitaire : cp experiences/cifar10/mon_run.sh scrip_run/lille/chicoree/
#   make pilot-grid
#
# (Ce dossier est gitignored — lancez prepare_campaign_queue.sh pour la campagne courante.)
