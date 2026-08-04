#!/usr/bin/env python3
"""Orchestrateur besteffort Grid5000 — autonome, sans scrip_grid_5000.

1 job OAR par .sh, n'importe quel nœud GPU (-q besteffort, sans -p cluster).
Reprise automatique après préemption (KILLED → resoumission ; entraînement via last.pt).

Usage :
  python grid5k/besteffort_pilot.py
  python grid5k/besteffort_pilot.py --sites lyon
  python grid5k/besteffort_pilot.py --follow-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import paramiko
import yaml

SENTINELLE_FIN = "=== EXPERIENCE TERMINEE AVEC SUCCES ==="
ETAPES_ACTIVES = ("ETAPE_1", "ETAPE_2", "ETAPE_3")
GRID5K_DIR = Path(__file__).resolve().parent
REPO_ROOT = GRID5K_DIR.parent


class Config:
    def __init__(self, data: dict):
        self.user = data["user"]
        self.ssh_gateway = data.get("ssh_gateway", "access.grid5000.fr")
        self.remote_project_dir = data["remote_project_dir"]
        self.sites = data.get("sites", ["lille", "lyon"])
        self.max_jobs_per_site = int(data.get("max_jobs_per_site", 8))
        self.walltime = data.get("walltime", "4:00:00")
        self.oar_queue = data.get("oar_queue", "besteffort")
        self.oar_resources = data.get("oar_resources", "host=1/gpu=1")
        self.oar_types_lille = data.get("oar_types_lille", "exotic,night")
        self.oar_types_lyon = data.get("oar_types_lyon", "exotic")
        self.state_file = data.get("state_file", "besteffort_state/run_status.json")
        self.local_outputs_dir = data.get("local_outputs_dir", "outputs")
        self.git_enabled = bool(data.get("git_enabled", True))
        self.git_branch = data.get("git_branch", "main")
        self.git_repo = data.get("git_repo", "")

    def remote_abs(self) -> str:
        return f"/home/{self.user}/{self.remote_project_dir}"

    def remote_home(self) -> str:
        return f"$HOME/{self.remote_project_dir}"

    def run_script(self) -> str:
        return f"{self.remote_home()}/grid5k/run_experiment.sh"

    def state_path(self) -> Path:
        p = Path(self.state_file)
        return p if p.is_absolute() else REPO_ROOT / p

    def scripts_dir(self, site: str) -> Path:
        return REPO_ROOT / f"besteffort_{site}"

    def outputs_dir(self) -> Path:
        p = Path(self.local_outputs_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    def oar_type_flags(self, site: str) -> str:
        raw = getattr(self, f"oar_types_{site}", None) or "exotic,night"
        types = [t.strip() for t in str(raw).split(",") if t.strip()]
        return " ".join(f"-t {t}" for t in types)


def load_config(path: Path) -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f) or {})


# --- SSH ---

def ssh_connect(user: str, host: str, gateway: str):
    try:
        bastion = paramiko.SSHClient()
        bastion.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        bastion.connect(hostname=gateway, username=user)
        transport = bastion.get_transport()
        channel = transport.open_channel("direct-tcpip", (host, 22), ("localhost", 0))
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, username=user, sock=channel)
        return bastion, client
    except Exception as exc:
        print(f"[SSH] Erreur connexion {host} : {exc}")
        return None, None


def ssh_run(client, cmd: str) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode(), stderr.read().decode()


def scp_upload(client, local: str, remote: str) -> bool:
    try:
        sftp = client.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        return True
    except Exception as exc:
        print(f"[SCP] Echec upload {local} → {remote} : {exc}")
        return False


def scp_download(client, remote: str, local: str) -> bool:
    try:
        sftp = client.open_sftp()
        sftp.get(remote, local)
        sftp.close()
        return True
    except Exception as exc:
        print(f"[SCP] Echec download {remote} : {exc}")
        return False


# --- État ---

def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def rel_key(site: str, script_name: str) -> str:
    return f"{site}/{script_name}"


# --- Scripts ---

def list_pending(cfg: Config, site: str, state: dict) -> list[tuple[str, Path]]:
    d = cfg.scripts_dir(site)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.sh")):
        key = rel_key(site, p.name)
        info = state.get(key, {})
        # En cours / en attente OAR → déjà géré par follow_script
        if info.get("etape") in ETAPES_ACTIVES:
            continue
        # Tout .sh présent dans le dossier et non actif → à soumettre
        # (nouveau script, ou script remis après archivage / besteffort-fresh)
        out.append((key, p))
    return out


def active_count(state: dict, site: str) -> int:
    return sum(
        1 for k, v in state.items()
        if k.startswith(f"{site}/") and v.get("etape") in ETAPES_ACTIVES
    )


def parse_output_dir(client, remote_script: str) -> tuple[str | None, str | None]:
    _, out, _ = ssh_run(client, f"cat {remote_script}")
    run_name = output_dir = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("RUN_NAME="):
            run_name = line.split("=", 1)[1].strip("'\"")
        elif "OUTPUT_DIR=" in line and not line.startswith("#"):
            output_dir = line.split("=", 1)[1].strip("'\"")
    if not run_name or not output_dir:
        return None, None
    tpl = output_dir.replace("$RUN_NAME", run_name).replace("${RUN_NAME}", run_name)
    _, resolved, _ = ssh_run(client, f'echo "{tpl}"')
    return resolved.strip(), run_name


# --- OAR ---

def oar_status(client, user: str, job_id: str) -> str:
    _, out, _ = ssh_run(client, f"oarstat -u {user}")
    for line in out.splitlines():
        if str(job_id) in line.split():
            parts = line.split()
            if len(parts) >= 5:
                return parts[-2]
    return "ABSENT"


def oar_submit(client, cfg: Config, remote_script: str, site: str) -> str | None:
    resources = f"{cfg.oar_resources},walltime={cfg.walltime}"
    types = cfg.oar_type_flags(site)
    cmd = (
        f'cd "{cfg.remote_abs()}" && '
        f'oarsub -q {cfg.oar_queue} {types} -l "{resources}" '
        f'"{cfg.run_script()} {remote_script}"'
    )
    print(f"[OAR] {cmd}")
    code, out, err = ssh_run(client, cmd)
    for flux in (out, err):
        for line in flux.splitlines():
            if "OAR_JOB_ID=" in line:
                jid = line.split("=", 1)[1].strip()
                print(f"→ Job {jid}")
                return jid
    print(f"[OAR] Echec soumission (code={code})")
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())
    return None


def job_finished(client, job_id: str, output_dir: str) -> str:
    for suffix in ("stderr", "stdout"):
        path = f"{output_dir}/OAR.{job_id}.{suffix}"
        _, tail, _ = ssh_run(client, f"tail -n 15 {path} 2>/dev/null")
        if f"Job {job_id} KILLED" in tail:
            return "KILLED"
    _, tail, _ = ssh_run(client, f"tail -n 20 {output_dir}/OAR.{job_id}.stdout 2>/dev/null")
    if SENTINELLE_FIN in tail:
        return "FINI"
    return "ERREUR"


# --- Git sync ---

def git_sync(client, cfg: Config) -> bool:
    if not cfg.git_enabled:
        return True
    proj = cfg.remote_abs()
    parent = os.path.dirname(proj)
    if cfg.git_repo:
        ssh_run(
            client,
            f'mkdir -p "{parent}" && [ -e "{proj}/.git" ] || '
            f'git clone {cfg.git_repo} "{proj}"',
        )
    code, out, err = ssh_run(
        client,
        f'cd "{proj}" && git fetch --all --prune && '
        f'git checkout {cfg.git_branch} && git pull --ff-only origin {cfg.git_branch}',
    )
    if code != 0:
        print("[git] Echec sync — pas de nouvelle soumission.")
        if err.strip():
            print(err.strip())
        return False
    _, rev, _ = ssh_run(client, f'cd "{proj}" && git rev-parse --short HEAD')
    print(f"[git] commit deploye : {rev.strip()}")
    return True


# --- Pipeline ---

def submit_script(client, cfg: Config, site: str, key: str, local: Path, state: dict) -> dict:
    remote_rel = f"besteffort_{site}/{local.name}"
    remote = f"{cfg.remote_abs()}/{remote_rel}"
    remote_dir = os.path.dirname(remote)
    ssh_run(client, f'mkdir -p "{remote_dir}"')
    if not scp_upload(client, str(local), remote):
        return state
    ssh_run(client, f'chmod +x "{remote}" "{cfg.remote_abs()}/grid5k/run_experiment.sh"')

    out_dir, _ = parse_output_dir(client, remote)
    if out_dir:
        ssh_run(client, f'mkdir -p "{out_dir}"')

    job_id = oar_submit(client, cfg, remote, site)
    if not job_id:
        return state

    state[key] = {
        "etape": "ETAPE_2",
        "statut_oar": "Lancement",
        "job_id": job_id,
        "site": site,
        "chemin_local": str(local),
        "chemin_distant": remote,
        "derniere_verification": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return state


def follow_script(client, cfg: Config, key: str, info: dict, state: dict) -> dict:
    job_id = info.get("job_id")
    remote = info.get("chemin_distant", "")
    site = info.get("site", "")
    local = Path(info.get("chemin_local", ""))

    if info.get("etape") == "ETAPE_1":
        return submit_script(client, cfg, site, key, local, state)

    if info.get("etape") != "ETAPE_2" or not job_id:
        return state

    statut = oar_status(client, cfg.user, job_id)
    print(f"[{key}] job {job_id} → {statut}")

    if statut in ("R", "W", "F"):
        info["statut_oar"] = statut
        info["derniere_verification"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state[key] = info
        return state

    if statut != "ABSENT":
        return state

    out_dir, run_name = parse_output_dir(client, remote)
    if not out_dir:
        out_dir = cfg.remote_abs()
    result = job_finished(client, job_id, out_dir)

    if result == "FINI":
        fetch_results(client, cfg, out_dir, run_name)
        archive_script(local)
        state[key] = {
            "etape": "TERMINE",
            "statut_oar": "FINI",
            "site": site,
            "chemin_local": str(local.parent / "archive" / "done" / local.name),
            "derniere_verification": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    elif result == "KILLED":
        print(f"[{key}] Preemption → resoumission...")
        if out_dir:
            ssh_run(client, f'mkdir -p "{out_dir}"')
        new_id = oar_submit(client, cfg, remote, site)
        if new_id:
            info.update({
                "etape": "ETAPE_2",
                "statut_oar": "Lancement",
                "job_id": new_id,
                "derniere_verification": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            state[key] = info
    else:
        print(f"[{key}] Erreur job (voir logs OAR.{job_id}.*)")
        state[key] = {
            "etape": "TERMINE",
            "statut_oar": "ERREUR",
            "site": site,
            "derniere_verification": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    return state


def fetch_results(client, cfg: Config, remote_out: str, run_name: str | None):
    if not run_name:
        return
    out_local = cfg.outputs_dir()
    out_local.mkdir(parents=True, exist_ok=True)
    archive = f"/tmp/{run_name}.tar"
    parent = os.path.dirname(remote_out)
    code, _, _ = ssh_run(client, f'tar -cf {archive} -C "{parent}" "{run_name}"')
    if code != 0:
        return
    local_tar = out_local / f"{run_name}.tar"
    if scp_download(client, archive, str(local_tar)):
        with tarfile.open(local_tar) as tar:
            tar.extractall(out_local)
        local_tar.unlink(missing_ok=True)
        print(f"→ Resultats : {out_local / run_name}")


def archive_script(local: Path):
    done = local.parent / "archive" / "done"
    done.mkdir(parents=True, exist_ok=True)
    dest = done / local.name
    if dest.exists():
        dest = done / f"{time.strftime('%Y%m%d_%H%M%S')}_{local.name}"
    if local.exists():
        shutil.move(str(local), str(dest))
        print(f"→ Script archive : {dest}")


def run_site(cfg: Config, site: str, state: dict, follow_only: bool) -> dict:
    active = [(k, v) for k, v in state.items()
              if v.get("site") == site and v.get("etape") in ETAPES_ACTIVES]
    pending = list_pending(cfg, site, state)

    if not active and not pending:
        print(f"[{site}] Rien a faire.")
        return state

    print(f"\n>>> {site.upper()} — actifs {len(active)}, en attente {len(pending)}")
    bastion, client = ssh_connect(cfg.user, site, cfg.ssh_gateway)
    if not client:
        return state

    try:
        git_ok = git_sync(client, cfg)

        for key, info in active:
            state = follow_script(client, cfg, key, info, state)

        if follow_only or not git_ok:
            return state

        slots = cfg.max_jobs_per_site - active_count(state, site)
        for key, local in pending[: max(0, slots)]:
            print(f"[{site}] Nouvelle soumission : {local.name}")
            state = submit_script(client, cfg, site, key, local, state)

    finally:
        if client:
            client.close()
        if bastion:
            bastion.close()
    return state


def main():
    parser = argparse.ArgumentParser(description="Orchestrateur besteffort Grid5000")
    parser.add_argument("--config", default=str(GRID5K_DIR / "config.yaml"))
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--follow-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    sites = args.sites or cfg.sites
    state_path = cfg.state_path()
    state = load_state(state_path)

    print("=" * 55)
    print("Besteffort Grid5000 — 1 GPU/exp., machine quelconque")
    print(f"Sites : {sites} | max {cfg.max_jobs_per_site} jobs/site")
    print(f"File  : besteffort_<site>/*.sh")
    print("=" * 55)

    for site in sites:
        state = run_site(cfg, site, state, args.follow_only)

    save_state(state_path, state)
    print("\nEtat :", state_path)
    print("=" * 55)


if __name__ == "__main__":
    main()
