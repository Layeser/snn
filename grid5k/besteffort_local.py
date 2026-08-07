#!/usr/bin/env python3
"""Orchestrateur besteffort — à lancer SUR la frontale (flille / flyon).

Règle : 1 expérience = 1 job OAR max.
- W / R / F  → attendre (ne pas resoumettre)
- Terminé OK → archiver le script
- KILLED     → resoumettre (reprise last.pt)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SENTINELLE = "=== EXPERIENCE TERMINEE AVEC SUCCES ==="
GRID5K_DIR = Path(__file__).resolve().parent
REPO_ROOT = GRID5K_DIR.parent

HOST_TO_SITE = {"flille": "lille", "flyon": "lyon"}


def load_config(path: Path) -> dict:
    cfg: dict = {}
    if not path.exists():
        return cfg
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([\w_]+):\s*(.+)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
            cfg[key] = int(val) if val.isdigit() else val
    return cfg


def cfg_get(cfg: dict, key: str, default):
    return cfg.get(key, default)


def oar_type_flags(cfg: dict, site: str) -> str:
    raw = cfg.get(f"oar_types_{site}") or cfg_get(cfg, "oar_types", "exotic,night")
    types = [t.strip() for t in str(raw).split(",") if t.strip()]
    return " ".join(f"-t {t}" for t in types)


def run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def detect_site() -> str | None:
    return HOST_TO_SITE.get(os.environ.get("HOSTNAME", "").split(".")[0])


def scripts_dir(site: str) -> Path:
    return REPO_ROOT / f"besteffort_{site}"


def rel_key(site: str, name: str) -> str:
    return f"{site}/{name}"


def parse_output_dir(script_path: Path) -> tuple[str | None, str | None]:
    run_name = output_dir = None
    for line in script_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("RUN_NAME="):
            run_name = line.split("=", 1)[1].strip("'\"")
        elif "OUTPUT_DIR=" in line and not line.startswith("#"):
            output_dir = line.split("=", 1)[1].strip("'\"")
    if not run_name or not output_dir:
        return None, None
    tpl = output_dir.replace("$RUN_NAME", run_name).replace("${RUN_NAME}", run_name)
    return run(f'echo "{tpl}"').stdout.strip(), run_name


def oar_job_state(job_id: str) -> str:
    """Etat OAR fiable via oarstat -j (Running, Waiting, Terminated, ou ABSENT)."""
    r = run(f"oarstat -j {job_id} 2>/dev/null")
    if r.returncode != 0 or not r.stdout.strip():
        return "ABSENT"
    for line in r.stdout.splitlines():
        if "state" in line.lower() and "=" in line:
            val = line.split("=", 1)[1].strip()
            if val:
                return val
    return "UNKNOWN"


def find_oar_logs(job_id: str, log_dir: str, project_abs: str) -> list[Path]:
    """Cherche OAR.{id}.stdout/stderr dans OUTPUT_DIR puis racine projet."""
    candidates = [
        Path(log_dir) / f"OAR.{job_id}.stdout",
        Path(log_dir) / f"OAR.{job_id}.stderr",
        Path(project_abs) / f"OAR.{job_id}.stdout",
        Path(project_abs) / f"OAR.{job_id}.stderr",
        Path.home() / f"OAR.{job_id}.stdout",
        Path.home() / f"OAR.{job_id}.stderr",
    ]
    return [p for p in candidates if p.is_file()]


def job_outcome(job_id: str, log_dir: str, project_abs: str) -> str:
    """FINI | KILLED | EN_COURS | INCONNU — ne jamais confondre W avec ERREUR."""
    logs = find_oar_logs(job_id, log_dir, project_abs)
    for p in logs:
        if "stderr" in p.name:
            text = p.read_text(errors="replace")
            if f"Job {job_id} KILLED" in text or "Killed" in text:
                return "KILLED"
    for p in logs:
        if "stdout" in p.name:
            text = p.read_text(errors="replace")
            if SENTINELLE in text:
                return "FINI"
    if logs:
        return "INCONNU"
    return "PAS_DE_LOG"


def oar_submit(cfg: dict, script_abs: str, site: str, name: str) -> str | None:
    proj = Path(cfg["_project_abs"])
    queue = cfg_get(cfg, "oar_queue", "besteffort")
    resources = cfg_get(cfg, "oar_resources", "host=1/gpu=1")
    walltime = cfg_get(cfg, "walltime", "4:00:00")
    types = oar_type_flags(cfg, site)
    runner = proj / "grid5k" / "run_experiment.sh"
    job_name = f"be_{site}_{Path(name).stem}"[:50]
    cmd = (
        f'cd "{proj}" && oarsub -q {queue} {types} '
        f'-l "{resources},walltime={walltime}" '
        f'-n "{job_name}" '
        f'"{runner} {script_abs}"'
    )
    print(f"[OAR] {Path(script_abs).name}")
    r = run(cmd)
    for flux in (r.stdout, r.stderr):
        for line in flux.splitlines():
            if "OAR_JOB_ID=" in line:
                jid = line.split("=", 1)[1].strip()
                print(f"  → job {jid}")
                return jid
    print(f"[OAR] Echec : {r.stderr.strip() or r.stdout.strip()}")
    return None


def git_pull(cfg: dict) -> bool:
    if cfg_get(cfg, "git_enabled", "true") in ("false", "0", "no"):
        return True
    branch = cfg_get(cfg, "git_branch", "main")
    proj = cfg["_project_abs"]
    r = run(
        f'cd "{proj}" && git fetch --all --prune && '
        f'git checkout {branch} && git pull --ff-only origin {branch}'
    )
    if r.returncode != 0:
        print("[git] Echec pull — suivi seulement, pas de nouvelle soumission.")
        return False
    rev = run(f'cd "{proj}" && git rev-parse --short HEAD').stdout.strip()
    print(f"[git] {rev}")
    return True


def archive_script(script: Path):
    done = script.parent / "archive" / "done"
    done.mkdir(parents=True, exist_ok=True)
    dest = done / script.name
    if dest.exists():
        dest = done / f"{time.strftime('%Y%m%d_%H%M%S')}_{script.name}"
    shutil.move(str(script), str(dest))
    print(f"  → archive {dest.name}")


def tracked_scripts(site: str, state: dict) -> list[tuple[str, Path, dict]]:
    """Scripts .sh du dossier avec leur entrée d'état."""
    d = scripts_dir(site)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.sh")):
        key = rel_key(site, p.name)
        out.append((key, p, state.get(key, {})))
    return out


def needs_new_job(info: dict) -> bool:
    """True seulement si aucun job OAR n'est associé à cette expérience."""
    if info.get("etape") == "TERMINE" and info.get("statut_oar") == "FINI":
        return False
    if info.get("job_id") and info.get("etape") == "ETAPE_2":
        return False
    return True


def follow_one(cfg: dict, key: str, script: Path, info: dict, state: dict) -> dict:
    job_id = info.get("job_id")
    site = info.get("site", key.split("/")[0])
    project = cfg["_project_abs"]

    if not job_id:
        return state

    oar_state = oar_job_state(job_id)
    print(f"[{key}] job {job_id} → {oar_state}")

    # Encore dans OAR : ne rien faire
    if oar_state in ("Running", "Waiting", "Finishing", "Suspended", "ToResume"):
        info.update({
            "etape": "ETAPE_2",
            "statut_oar": oar_state,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        state[key] = info
        return state

    out_dir, _ = parse_output_dir(script)
    log_dir = out_dir or project
    outcome = job_outcome(job_id, log_dir, project)

    if outcome == "FINI":
        archive_script(script)
        state[key] = {
            "etape": "TERMINE",
            "statut_oar": "FINI",
            "site": site,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return state

    if outcome == "KILLED":
        print(f"[{key}] Preemption — resoumission (reprise last.pt)")
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        new_id = oar_submit(cfg, str(script.resolve()), site, script.name)
        if new_id:
            state[key] = {
                "etape": "ETAPE_2",
                "job_id": new_id,
                "site": site,
                "script": str(script.resolve()),
                "statut_oar": "Lancement",
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        return state

    if oar_state == "Terminated" and outcome == "INCONNU":
        print(f"[{key}] Terminé sans succès — voir logs OAR.{job_id}.* (pas de resoumission auto)")
        state[key] = {
            "etape": "TERMINE",
            "statut_oar": "ERREUR",
            "site": site,
            "job_id": job_id,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return state

    # Job disparu de oarstat mais pas encore de logs (file d'attente / démarrage)
    if oar_state == "ABSENT" and outcome == "PAS_DE_LOG":
        print(f"[{key}] job {job_id} absent de oarstat, pas de log — on attend")
        info["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state[key] = info
        return state

    info["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state[key] = info
    return state


def submit_one(cfg: dict, site: str, key: str, script: Path, state: dict) -> dict:
    script_abs = str(script.resolve())
    out_dir, _ = parse_output_dir(script)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    run(f'chmod +x "{script_abs}" "{cfg["_project_abs"]}/grid5k/run_experiment.sh"')
    job_id = oar_submit(cfg, script_abs, site, script.name)
    if not job_id:
        return state
    state[key] = {
        "etape": "ETAPE_2",
        "job_id": job_id,
        "site": site,
        "script": script_abs,
        "statut_oar": "Lancement",
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return state


def active_job_count(state: dict, site: str) -> int:
    n = 0
    for k, v in state.items():
        if not k.startswith(f"{site}/"):
            continue
        if v.get("etape") == "ETAPE_2" and v.get("job_id"):
            n += 1
    return n


def run_site(cfg: dict, site: str, state: dict, follow_only: bool) -> dict:
    scripts = tracked_scripts(site, state)
    if not scripts:
        print(f"[{site}] Aucun .sh dans besteffort_{site}/")
        return state

    print(f"\n>>> {site} — {len(scripts)} experience(s)")

    if not follow_only and not git_pull(cfg):
        follow_only = True

    # 1) Suivre les jobs existants (ne jamais resoumettre si job_id present)
    for key, script, info in scripts:
        if info.get("job_id") and info.get("etape") == "ETAPE_2":
            state = follow_one(cfg, key, script, info, state)

    if follow_only:
        return state

    # 2) Soumettre UNIQUEMENT les scripts sans job OAR actif
    slots = int(cfg_get(cfg, "max_jobs_per_site", 8)) - active_job_count(state, site)
    submitted = 0
    for key, script, info in scripts:
        if slots <= 0:
            break
        fresh = state.get(key, info)
        if not needs_new_job(fresh):
            continue
        if fresh.get("statut_oar") == "ERREUR":
            print(f"[{key}] ERREUR precedente — ignoré (besteffort-fresh pour relancer)")
            continue
        print(f"[{site}] Nouvelle soumission : {script.name}")
        state = submit_one(cfg, site, key, script, state)
        submitted += 1
        slots -= 1

    if submitted == 0:
        print(f"[{site}] Rien a soumettre ({active_job_count(state, site)} job(s) deja actif(s))")

    return state


def cmd_cleanup(cfg: dict, site: str, state: dict, apply: bool):
    """Annule les jobs besteffort orphelins (pas dans run_status.json)."""
    user = cfg["_user"]
    tracked = {
        str(v["job_id"])
        for k, v in state.items()
        if k.startswith(f"{site}/") and v.get("job_id")
    }
    r = run(f"oarstat -u {user}")
    orphans = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        jid, q = parts[0], parts[-1]
        st = parts[-2]
        if q != cfg_get(cfg, "oar_queue", "besteffort"):
            continue
        if jid not in tracked and st in ("W", "F", "R"):
            orphans.append(jid)

    if not orphans:
        print(f"[{site}] Aucun job besteffort orphelin.")
        return
    print(f"[{site}] {len(orphans)} job(s) orphelin(s) : {', '.join(orphans)}")
    if not apply:
        print("Dry-run — relancer avec --cleanup-apply pour oardel")
        return
    for jid in orphans:
        run(f"oardel {jid}")
        print(f"  oardel {jid}")


def main():
    parser = argparse.ArgumentParser(description="Besteffort local (frontale Grid5000)")
    parser.add_argument("--site", choices=["lille", "lyon"], default=None)
    parser.add_argument("--config", default=str(GRID5K_DIR / "config.yaml"))
    parser.add_argument("--follow-only", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Lister jobs orphelins")
    parser.add_argument("--cleanup-apply", action="store_true", help="Annuler jobs orphelins")
    args = parser.parse_args()

    site = args.site or detect_site()
    if not site:
        print("Erreur : --site lille|lyon requis.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(Path(args.config))
    cfg["_user"] = os.environ.get("USER") or cfg_get(cfg, "user", "")
    cfg["_project_abs"] = str(Path.home() / cfg_get(cfg, "remote_project_dir", "internship/snn"))

    state_path = REPO_ROOT / cfg_get(cfg, "state_file", "besteffort_state/run_status.json")
    state = load_state(state_path)

    if args.cleanup or args.cleanup_apply:
        cmd_cleanup(cfg, site, state, apply=args.cleanup_apply)
        return

    print("=" * 50)
    print(f"Besteffort — {site} | 1 job max / experience")
    print("=" * 50)

    os.chdir(cfg["_project_abs"])
    state = run_site(cfg, site, state, args.follow_only)
    save_state(state_path, state)
    print(f"\nEtat : {state_path}")


if __name__ == "__main__":
    main()
