#!/usr/bin/env python3
"""Orchestrateur besteffort — à lancer SUR la frontale (flille / flyon).

Sans paramiko ni SSH : oarsub directement depuis la frontale.
Usage :
  python3 grid5k/besteffort_local.py --site lille
  python3 grid5k/besteffort_local.py --site lyon --follow-only
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
ETAPES_ACTIVES = ("ETAPE_1", "ETAPE_2", "ETAPE_3")
GRID5K_DIR = Path(__file__).resolve().parent
REPO_ROOT = GRID5K_DIR.parent

HOST_TO_SITE = {
    "flille": "lille",
    "flyon": "lyon",
}


def load_config(path: Path) -> dict:
    cfg: dict = {}
    if not path.exists():
        return cfg
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\w+):\s*(.+)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
            if val.isdigit():
                cfg[key] = int(val)
            else:
                cfg[key] = val
    return cfg


def cfg_get(cfg: dict, key: str, default):
    return cfg.get(key, default)


def oar_type_flags(cfg: dict, site: str) -> str:
    """Construit les -t exotic -t night … sans -p cluster."""
    raw = cfg.get(f"oar_types_{site}") or cfg_get(cfg, "oar_types", "exotic,night")
    types = [t.strip() for t in str(raw).split(",") if t.strip()]
    return " ".join(f"-t {t}" for t in types)


def run(cmd: str, *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, check=check)


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
    host = os.environ.get("HOSTNAME", "").split(".")[0]
    return HOST_TO_SITE.get(host)


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
    r = run(f'echo "{tpl}"')
    return r.stdout.strip(), run_name


def oar_status(user: str, job_id: str) -> str:
    r = run(f"oarstat -u {user}")
    for line in r.stdout.splitlines():
        if str(job_id) in line.split():
            parts = line.split()
            if len(parts) >= 5:
                return parts[-2]
    return "ABSENT"


def oar_submit(cfg: dict, script_abs: str, site: str) -> str | None:
    proj = Path(cfg["_project_abs"])
    queue = cfg_get(cfg, "oar_queue", "besteffort")
    resources = cfg_get(cfg, "oar_resources", "host=1/gpu=1")
    walltime = cfg_get(cfg, "walltime", "4:00:00")
    types = oar_type_flags(cfg, site)
    runner = proj / "grid5k" / "run_experiment.sh"
    cmd = (
        f'cd "{proj}" && oarsub -q {queue} {types} '
        f'-l "{resources},walltime={walltime}" '
        f'"{runner} {script_abs}"'
    )
    print(f"[OAR] {script_abs}")
    r = run(cmd)
    for flux in (r.stdout, r.stderr):
        for line in flux.splitlines():
            if "OAR_JOB_ID=" in line:
                jid = line.split("=", 1)[1].strip()
                print(f"  → job {jid}")
                return jid
    print(f"[OAR] Echec : {r.stderr.strip() or r.stdout.strip()}")
    return None


def job_result(job_id: str, log_dir: str) -> str:
    for suffix in ("stderr", "stdout"):
        p = Path(log_dir) / f"OAR.{job_id}.{suffix}"
        if p.exists():
            tail = p.read_text()[-2000:]
            if f"Job {job_id} KILLED" in tail:
                return "KILLED"
    stdout = Path(log_dir) / f"OAR.{job_id}.stdout"
    if stdout.exists() and SENTINELLE in stdout.read_text()[-4000:]:
        return "FINI"
    return "ERREUR"


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
        print("[git] Echec pull — pas de nouvelle soumission.")
        print(r.stderr.strip() or r.stdout.strip())
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


def list_pending(site: str, state: dict) -> list[tuple[str, Path]]:
    d = scripts_dir(site)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.sh")):
        key = rel_key(site, p.name)
        if state.get(key, {}).get("etape") in ETAPES_ACTIVES:
            continue
        out.append((key, p))
    return out


def active_count(state: dict, site: str) -> int:
    return sum(
        1 for k, v in state.items()
        if k.startswith(f"{site}/") and v.get("etape") in ETAPES_ACTIVES
    )


def submit_one(cfg: dict, site: str, key: str, script: Path, state: dict) -> dict:
    script_abs = str(script.resolve())
    out_dir, _ = parse_output_dir(script)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    run(f'chmod +x "{script_abs}" "{cfg["_project_abs"]}/grid5k/run_experiment.sh"')
    job_id = oar_submit(cfg, script_abs, site)
    if not job_id:
        return state
    state[key] = {
        "etape": "ETAPE_2",
        "job_id": job_id,
        "site": site,
        "script": script_abs,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return state


def follow_one(cfg: dict, key: str, info: dict, state: dict) -> dict:
    job_id = info.get("job_id")
    script = Path(info.get("script", ""))
    site = info.get("site", "")
    user = cfg["_user"]

    if not job_id or not script.exists():
        return state

    statut = oar_status(user, job_id)
    print(f"[{key}] job {job_id} → {statut}")

    if statut in ("R", "W", "F"):
        info["statut_oar"] = statut
        info["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state[key] = info
        return state

    if statut != "ABSENT":
        return state

    out_dir, _ = parse_output_dir(script)
    log_dir = out_dir or cfg["_project_abs"]
    result = job_result(job_id, str(log_dir))

    if result == "FINI":
        archive_script(script)
        state[key] = {"etape": "TERMINE", "statut_oar": "FINI", "site": site}
    elif result == "KILLED":
        print(f"[{key}] Preemption → resoumission")
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        new_id = oar_submit(cfg, str(script.resolve()), site)
        if new_id:
            info.update({
                "etape": "ETAPE_2",
                "job_id": new_id,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            state[key] = info
    else:
        print(f"[{key}] Erreur (logs OAR.{job_id}.* dans {log_dir})")
        state[key] = {"etape": "TERMINE", "statut_oar": "ERREUR", "site": site}
    return state


def run_site(cfg: dict, site: str, state: dict, follow_only: bool) -> dict:
    active = [(k, v) for k, v in state.items()
              if v.get("site") == site and v.get("etape") in ETAPES_ACTIVES]
    pending = list_pending(site, state)

    if not active and not pending:
        print(f"[{site}] Rien a faire.")
        return state

    print(f"\n>>> {site} — actifs {len(active)}, en attente {len(pending)}")

    if not follow_only and not git_pull(cfg):
        follow_only = True

    for key, info in active:
        state = follow_one(cfg, key, info, state)

    if follow_only:
        return state

    slots = int(cfg_get(cfg, "max_jobs_per_site", 8)) - active_count(state, site)
    for key, script in pending[: max(0, slots)]:
        print(f"[{site}] Soumission : {script.name}")
        state = submit_one(cfg, site, key, script, state)

    return state


def main():
    parser = argparse.ArgumentParser(description="Besteffort local (frontale Grid5000)")
    parser.add_argument("--site", choices=["lille", "lyon"], default=None)
    parser.add_argument("--config", default=str(GRID5K_DIR / "config.yaml"))
    parser.add_argument("--follow-only", action="store_true")
    args = parser.parse_args()

    site = args.site or detect_site()
    if not site:
        print("Erreur : precisez --site lille|lyon (hors frontale flille/flyon).", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(Path(args.config))
    user = os.environ.get("USER") or cfg_get(cfg, "user", "")
    remote_dir = cfg_get(cfg, "remote_project_dir", "internship/snn")
    cfg["_user"] = user
    cfg["_project_abs"] = str(Path.home() / remote_dir)

    state_path = REPO_ROOT / cfg_get(cfg, "state_file", "besteffort_state/run_status.json")
    state = load_state(state_path)

    print("=" * 50)
    print(f"Besteffort local — {site} ({os.environ.get('HOSTNAME', '?')})")
    print(f"File : besteffort_{site}/*.sh")
    print("=" * 50)

    os.chdir(cfg["_project_abs"])
    state = run_site(cfg, site, state, args.follow_only)
    save_state(state_path, state)
    print(f"\nEtat : {state_path}")


if __name__ == "__main__":
    main()
