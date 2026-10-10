#!/usr/bin/env python3
import sys, os, json, uuid, hashlib, tempfile, shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta

VERSION = "8.0"
ROOT = Path(os.environ.get("MRR_ROOT", "/work/resource/multi-robot-registration"))
RESPONSES = ROOT / "responses"
DEFAULT_TTL_SECONDS = 1800
DEFAULT_STOP_SECONDS = 600

def now_dt():
    return datetime.now(timezone.utc)

def now_iso():
    return now_dt().isoformat()

def parse_time(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()

def emit(obj, code=0):
    print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(code)

def load_stdin():
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except Exception as e:
        emit({"status":"error","code":"INVALID_INPUT_JSON","error":str(e)}, 2)

def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def fsync_dir(path):
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass

def atomic_write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        fsync_dir(p.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def create_json_once(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        fsync_dir(p.parent)
        return "created"
    except FileExistsError:
        existing = read_json(p)
        if canonical(existing) == canonical(data):
            return "exists_same"
        raise RuntimeError(f"immutable file conflict: {p}")

def current_path():
    return ROOT / "current.json"

def round_dir(round_id):
    return ROOT / "rounds" / round_id

def config_path(round_id, version):
    return round_dir(round_id) / f"config-v{version}.json"

def records_dir(round_id):
    return round_dir(round_id) / "records"

def runtime_dir(round_id):
    return round_dir(round_id) / "runtime"

def lock_path(round_id):
    return runtime_dir(round_id) / "interaction-lock.json"

def interactions_dir(round_id):
    return runtime_dir(round_id) / "interactions"

def snapshots_dir(round_id):
    return round_dir(round_id) / "snapshots"

def final_path(round_id):
    return round_dir(round_id) / "final.json"

def response_path(request_id):
    return RESPONSES / f"{request_id}.json"

def archive_path():
    return ROOT / "archive.json"

def setup_round_dirs(round_id):
    for p in [
        RESPONSES, records_dir(round_id), interactions_dir(round_id),
        snapshots_dir(round_id), round_dir(round_id) / "received" / "snapshots"
    ]:
        p.mkdir(parents=True, exist_ok=True)

def load_current():
    return read_json(current_path())

def save_current(cur):
    cur["updated_at"] = now_iso()
    atomic_write_json(current_path(), cur)

def load_config(cur):
    path = cur.get("config_path")
    if not path:
        raise RuntimeError("config_path missing")
    cfg = read_json(path)
    if not isinstance(cfg, dict):
        raise RuntimeError("config file missing or invalid")
    if cfg.get("round_id") != cur.get("round_id") or int(cfg.get("config_version", -1)) != int(cur.get("config_version", -2)):
        raise RuntimeError("config/current mismatch")
    return cfg

def ensure_config(round_id, version, raw_cfg):
    cfg = dict(raw_cfg or {})
    cfg["round_id"] = round_id
    cfg["config_version"] = int(version)
    p = config_path(round_id, int(version))
    result = create_json_once(p, cfg)
    return cfg, str(p), result

def current_lock(cur):
    if not cur:
        return None
    return read_json(lock_path(cur["round_id"]))

def session_path(round_id, session_id):
    return interactions_dir(round_id) / f"{session_id}.json"

def mark_session(round_id, session_id, status, reason=None):
    p = session_path(round_id, session_id)
    s = read_json(p, {})
    if not s:
        return
    s["status"] = status
    s["updated_at"] = now_iso()
    s["ended_at"] = now_iso()
    if reason:
        s["reason"] = reason
    atomic_write_json(p, s)

def mark_lock(round_id, lock, status, reason=None):
    if not lock:
        return
    lock = dict(lock)
    lock["status"] = status
    lock["updated_at"] = now_iso()
    lock["ended_at"] = now_iso()
    if reason:
        lock["reason"] = reason
    atomic_write_json(lock_path(round_id), lock)

def lock_is_live(lock):
    if not lock or lock.get("status") != "in_progress":
        return False
    exp = parse_time(lock.get("expires_at"))
    upd = parse_time(lock.get("updated_at"))
    now = now_dt()
    if exp and now > exp:
        return False
    if upd and now > upd + timedelta(seconds=int(lock.get("ttl_seconds", DEFAULT_TTL_SECONDS))):
        return False
    return True

def abort_live_interaction(cur, reason):
    lock = current_lock(cur)
    if lock and lock.get("status") == "in_progress":
        sid = lock.get("session_id")
        if sid:
            mark_session(cur["round_id"], sid, "aborted", reason)
        mark_lock(cur["round_id"], lock, "aborted", reason)

def validate_pending(cur):
    pv = cur.get("pending_config_version")
    pp = cur.get("pending_config_path")
    if pv is None or not pp:
        raise RuntimeError("pending config fields missing")
    cfg = read_json(pp)
    if not isinstance(cfg, dict):
        raise RuntimeError("pending config missing or invalid")
    if cfg.get("round_id") != cur.get("round_id"):
        raise RuntimeError("pending config round mismatch")
    if int(cfg.get("config_version", -1)) != int(pv):
        raise RuntimeError("pending config version mismatch")
    if int(pv) <= int(cur.get("config_version", 0)):
        raise RuntimeError("pending config is not newer")
    return cfg

def count_records(round_id, file_names=None):
    rdir = records_dir(round_id)
    names = list(file_names) if file_names is not None else sorted(p.name for p in rdir.glob("*.json"))
    valid, invalid, events = [], [], []
    for name in names:
        p = rdir / name
        try:
            obj = read_json(p)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            if obj.get("round_id") != round_id:
                raise ValueError("round mismatch")
            if not obj.get("event_id"):
                raise ValueError("event_id missing")
            valid.append(name)
            events.append(obj)
        except Exception as e:
            invalid.append({"file":name,"reason":str(e)})
    return {
        "record_count": len(names),
        "valid_record_count": len(valid),
        "invalid_record_count": len(invalid),
        "valid_files": valid,
        "invalid_files": invalid,
        "events": events,
    }

def ensure_final(cur):
    if not cur or cur.get("state") != "stopped":
        return None
    p = final_path(cur["round_id"])
    existing = read_json(p)
    if existing:
        return existing
    counts = count_records(cur["round_id"])
    final = {
        "round_id": cur["round_id"],
        "role": cur.get("role"),
        "self": cur.get("self"),
        "leader": cur.get("leader"),
        "stop_mode": cur.get("stop_mode"),
        "stop_requested_at": cur.get("stop_requested_at"),
        "frozen_at": now_iso(),
        "config_version": cur.get("config_version"),
        "record_count": counts["record_count"],
        "valid_record_count": counts["valid_record_count"],
        "invalid_record_count": counts["invalid_record_count"],
        "record_files": sorted(counts["valid_files"] + [x["file"] for x in counts["invalid_files"]]),
        "invalid_files": counts["invalid_files"],
    }
    try:
        create_json_once(p, final)
    except RuntimeError:
        pass
    return read_json(p)

def converge():
    cur = load_current()
    if not cur:
        return None
    rid = cur.get("round_id")
    if not rid:
        return cur

    lock = current_lock(cur)
    if lock and lock.get("status") == "in_progress" and not lock_is_live(lock):
        sid = lock.get("session_id")
        if sid:
            mark_session(rid, sid, "aborted", "stale interaction after interruption")
        mark_lock(rid, lock, "aborted", "stale interaction after interruption")
        lock = current_lock(cur)

    if cur.get("state") == "stopping":
        deadline = parse_time(cur.get("stop_deadline"))
        if deadline and now_dt() > deadline:
            abort_live_interaction(cur, "graceful stop deadline exceeded")
            cur["state"] = "stopped"
            cur["registration_active"] = False
            cur["accepting_new"] = False
            save_current(cur)
            ensure_final(cur)
            return cur
        if not lock_is_live(current_lock(cur)):
            cur["state"] = "stopped"
            cur["registration_active"] = False
            cur["accepting_new"] = False
            save_current(cur)
            ensure_final(cur)
            return cur

    if cur.get("state") == "config_updating" and not lock_is_live(current_lock(cur)):
        try:
            validate_pending(cur)
            cur["config_version"] = int(cur["pending_config_version"])
            cur["config_path"] = cur["pending_config_path"]
            cur.pop("pending_config_version", None)
            cur.pop("pending_config_path", None)
            cur["config_applied_at"] = now_iso()
            cur["state"] = "registering"
            cur["registration_active"] = True
            cur["accepting_new"] = True
            cur.pop("last_config_update_error", None)
            save_current(cur)
        except Exception as e:
            try:
                load_config(cur)
                cur.pop("pending_config_version", None)
                cur.pop("pending_config_path", None)
                cur["last_config_update_error"] = str(e)
                cur["state"] = "registering"
                cur["registration_active"] = True
                cur["accepting_new"] = True
                save_current(cur)
            except Exception as e2:
                cur["last_config_update_error"] = f"{e}; previous config invalid: {e2}"
                cur["accepting_new"] = False
                save_current(cur)

    return load_current()

def validate_envelope(req):
    if not isinstance(req, dict):
        raise ValueError("request must be object")
    if req.get("marker") != "MRR_CONTROL_V1" or req.get("protocol") != "mrr-v1":
        raise ValueError("marker/protocol mismatch")
    for k in ["command","request_id","round_id","leader","config_version","payload"]:
        if k not in req:
            raise ValueError(f"{k} missing")
    if req["command"] not in {"START","STATUS","SNAPSHOT","UPDATE","STOP","CLEANUP"}:
        raise ValueError("unsupported command")
    return req

def cached_response(req):
    p = response_path(req["request_id"])
    if not p.exists():
        return None
    c = read_json(p)
    if not c:
        return None
    if c.get("request_hash") != digest(req):
        return {"protocol":"mrr-v1","request_id":req["request_id"],"command":req["command"],
                "round_id":req["round_id"],"responder":c.get("response",{}).get("responder"),
                "status":"conflict","effect":"not_applied","error":"request_id reused with different content"}
    return c.get("response")

def cache_response(req, resp):
    data = {
        "request_id": req["request_id"],
        "request_hash": digest(req),
        "command": req["command"],
        "round_id": req["round_id"],
        "saved_at": now_iso(),
        "response": resp,
    }
    try:
        create_json_once(response_path(req["request_id"]), data)
    except RuntimeError:
        existing = read_json(response_path(req["request_id"]))
        if not existing or existing.get("request_hash") != digest(req):
            raise
    return resp

def responder_from(req, cur=None, local_role=None):
    payload = req.get("payload") or {}
    target = payload.get("target")
    if cur and cur.get("self"):
        if target and target.get("peer_id") and target.get("peer_id") != cur["self"].get("peer_id"):
            raise RuntimeError("target peer mismatch")
        return cur["self"]
    if target and target.get("peer_id"):
        return {"peer_id":target.get("peer_id"),"name":target.get("name")}
    if local_role == "leader":
        return req.get("leader")
    raise RuntimeError("payload.target.peer_id required for first START")

def base_response(req, responder, **extra):
    d = {
        "protocol":"mrr-v1",
        "request_id":req["request_id"],
        "command":req["command"],
        "round_id":req["round_id"],
        "responder":responder,
        "status":"ok",
        "effect":"applied",
        "config_version":req.get("config_version"),
        "runtime_state":None,
        "registration_active":False,
        "data_cutoff":now_iso(),
        "error":None,
    }
    d.update(extra)
    return d

def verify_round_and_leader(req, cur):
    if not cur:
        raise RuntimeError("no active round")
    if cur.get("round_id") != req["round_id"]:
        raise RuntimeError("round mismatch")
    stored_leader = (cur.get("leader") or {}).get("peer_id")
    req_leader = (req.get("leader") or {}).get("peer_id")
    if stored_leader and req_leader != stored_leader:
        raise RuntimeError("leader mismatch")
    responder_from(req, cur)

def handle_start(req, role):
    converge()
    cur = load_current()
    rid = req["round_id"]
    ver = int(req["config_version"])
    payload = req.get("payload") or {}
    cfg_raw = payload.get("config")
    if not isinstance(cfg_raw, dict):
        raise RuntimeError("payload.config required")
    self_info = responder_from(req, cur if cur and cur.get("round_id")==rid else None, role)
    leader = req["leader"]
    if cur and cur.get("state") != "stopped" and cur.get("round_id") != rid:
        raise RuntimeError("another active round exists")
    if cur and cur.get("round_id") == rid:
        verify_round_and_leader(req, cur)
        cfg_expected = dict(cfg_raw)
        cfg_expected["round_id"] = rid
        cfg_expected["config_version"] = ver
        existing = read_json(config_path(rid, ver))
        if existing and canonical(existing) == canonical(cfg_expected) and int(cur.get("config_version",0)) == ver:
            return base_response(req, cur["self"], effect="already_applied",
                                 runtime_state=cur.get("state"),
                                 registration_active=bool(cur.get("registration_active")))
        if existing and canonical(existing) != canonical(cfg_expected):
            raise RuntimeError("same version config conflict")
    setup_round_dirs(rid)
    cfg, cpath, _ = ensure_config(rid, ver, cfg_raw)
    newcur = {
        "round_id":rid,
        "role":role,
        "self":self_info,
        "leader":leader,
        "state":"registering",
        "registration_active":True,
        "accepting_new":True,
        "config_version":ver,
        "config_path":cpath,
        "created_at":now_iso(),
    }
    if role == "leader":
        newcur["team_members"] = payload.get("team_members") or []
        newcur["member_start_status"] = {}
    save_current(newcur)
    return base_response(req, self_info, runtime_state="registering", registration_active=True)

def status_payload(cur):
    counts = count_records(cur["round_id"])
    return {
        "runtime_state":cur.get("state"),
        "registration_active":bool(cur.get("registration_active")),
        "accepting_new":bool(cur.get("accepting_new")),
        "config_version":cur.get("config_version"),
        "record_count":counts["record_count"],
        "valid_record_count":counts["valid_record_count"],
        "invalid_record_count":counts["invalid_record_count"],
        "last_config_update_error":cur.get("last_config_update_error"),
    }

def handle_status(req, role):
    cur = converge()
    verify_round_and_leader(req, cur)
    info = status_payload(cur)
    return base_response(req, cur["self"], effect="already_applied", **info)

def make_snapshot(cur, snapshot_id, mode):
    sp = snapshots_dir(cur["round_id"]) / f"{snapshot_id}.json"
    existing = read_json(sp)
    if existing:
        if existing.get("round_id") != cur["round_id"] or existing.get("mode") != mode:
            raise RuntimeError("snapshot_id conflict")
        return existing
    if mode == "final":
        if cur.get("state") != "stopped":
            raise RuntimeError("final snapshot requires stopped state")
        final = ensure_final(cur)
        names = list(final.get("record_files") or [])
    else:
        names = sorted(p.name for p in records_dir(cur["round_id"]).glob("*.json"))
    snap = {
        "snapshot_id":snapshot_id,
        "round_id":cur["round_id"],
        "mode":mode,
        "created_at":now_iso(),
        "files":names,
        "snapshot_count":len(names),
    }
    create_json_once(sp, snap)
    return snap

def handle_snapshot(req, role):
    cur = converge()
    verify_round_and_leader(req, cur)
    p = req.get("payload") or {}
    sid = p.get("snapshot_id")
    if not sid:
        raise RuntimeError("snapshot_id required")
    cursor = int(p.get("cursor",0))
    limit = int(p.get("limit",50))
    mode = p.get("mode","current")
    if limit < 1 or limit > 200:
        raise RuntimeError("limit must be 1..200")
    snap = make_snapshot(cur, sid, mode)
    names = snap["files"]
    if cursor < 0 or cursor > len(names):
        raise RuntimeError("cursor out of range")
    chunk = names[cursor:cursor+limit]
    records, invalid = [], []
    for name in chunk:
        try:
            obj = read_json(records_dir(cur["round_id"]) / name)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            if obj.get("round_id") != cur["round_id"]:
                raise ValueError("round mismatch")
            if not obj.get("event_id"):
                raise ValueError("event_id missing")
            records.append(obj)
        except Exception as e:
            invalid.append({"file":name,"reason":str(e)})
    next_cursor = cursor + len(chunk)
    done = next_cursor >= len(names)
    return base_response(
        req, cur["self"], effect="already_applied",
        runtime_state=cur.get("state"),
        registration_active=bool(cur.get("registration_active")),
        config_version=cur.get("config_version"),
        snapshot_id=sid,
        snapshot_count=len(names),
        cursor=cursor,
        returned_count=len(records),
        page_item_count=len(chunk),
        page_valid_record_count=len(records),
        page_invalid_record_count=len(invalid),
        next_cursor=next_cursor,
        done=done,
        records=records,
        invalid_files=invalid,
        data_incomplete=bool(invalid),
    )

def handle_update(req, role):
    cur = converge()
    verify_round_and_leader(req, cur)
    p = req.get("payload") or {}
    cfg_raw = p.get("config")
    if not isinstance(cfg_raw, dict):
        raise RuntimeError("payload.config required")
    ver = int(req["config_version"])
    curver = int(cur.get("config_version",0))
    expected = dict(cfg_raw)
    expected["round_id"] = req["round_id"]
    expected["config_version"] = ver
    if ver < curver:
        raise RuntimeError("config version rollback is not allowed")
    if ver == curver:
        existing = read_json(cur.get("config_path"))
        if existing and canonical(existing)==canonical(expected):
            return base_response(req, cur["self"], effect="already_applied",
                                 runtime_state=cur.get("state"),
                                 registration_active=bool(cur.get("registration_active")),
                                 config_version=curver)
        raise RuntimeError("same config_version has different content")
    cfg, cpath, _ = ensure_config(req["round_id"], ver, cfg_raw)
    if cur.get("role") == "leader" and "team_members" in p:
        cur["team_members"] = p.get("team_members") or []
    if lock_is_live(current_lock(cur)):
        cur["pending_config_version"] = ver
        cur["pending_config_path"] = cpath
        cur["state"] = "config_updating"
        cur["accepting_new"] = False
        save_current(cur)
        return base_response(req, cur["self"], status="pending",
                             runtime_state="config_updating",
                             registration_active=True,
                             config_version=curver,
                             target_config_version=ver)
    cur["config_version"] = ver
    cur["config_path"] = cpath
    cur["state"] = "registering"
    cur["registration_active"] = True
    cur["accepting_new"] = True
    cur["config_applied_at"] = now_iso()
    save_current(cur)
    return base_response(req, cur["self"], runtime_state="registering",
                         registration_active=True, config_version=ver)

def handle_stop(req, role):
    cur = converge()
    verify_round_and_leader(req, cur)
    p = req.get("payload") or {}
    mode = p.get("mode","graceful")
    if mode not in {"graceful","immediate"}:
        raise RuntimeError("mode must be graceful or immediate")
    seconds = int(p.get("deadline_seconds", DEFAULT_STOP_SECONDS))
    deadline = parse_time(p.get("stop_deadline")) or (now_dt() + timedelta(seconds=seconds))
    cur["stop_mode"] = mode
    cur["stop_requested_at"] = now_iso()
    cur["stop_deadline"] = deadline.isoformat()
    cur["accepting_new"] = False

    if mode == "immediate":
        abort_live_interaction(cur, "immediate stop requested")
        cur["registration_active"] = False
        cur["state"] = "stopped"
        save_current(cur)
        final = ensure_final(cur)
        info = status_payload(cur)
        return base_response(req, cur["self"], runtime_state="stopped",
                             registration_active=False, config_version=cur.get("config_version"),
                             **{k:v for k,v in info.items() if k not in {"runtime_state","registration_active","config_version"}})

    if lock_is_live(current_lock(cur)):
        cur["state"] = "stopping"
        cur["registration_active"] = True
        save_current(cur)
        info = status_payload(cur)
        return base_response(req, cur["self"], status="pending",
                             runtime_state="stopping", registration_active=True,
                             config_version=cur.get("config_version"),
                             **{k:v for k,v in info.items() if k not in {"runtime_state","registration_active","config_version"}})
    cur["state"] = "stopped"
    cur["registration_active"] = False
    save_current(cur)
    ensure_final(cur)
    info = status_payload(cur)
    return base_response(req, cur["self"], runtime_state="stopped",
                         registration_active=False, config_version=cur.get("config_version"),
                         **{k:v for k,v in info.items() if k not in {"runtime_state","registration_active","config_version"}})

def remove_tree(path):
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)

def handle_cleanup(req, role):
    p = req.get("payload") or {}
    target = p.get("target") or {}
    target_peer = target.get("peer_id")
    if not target_peer:
        raise RuntimeError("payload.target.peer_id required")

    cur = load_current()
    if cur and cur.get("round_id") != req["round_id"]:
        raise RuntimeError("another current round exists")

    if cur:
        verify_round_and_leader(req, cur)
        if cur.get("state") != "stopped":
            raise RuntimeError("cleanup requires stopped state")
        responder = cur.get("self")
    else:
        responder = {"peer_id": target_peer, "name": target.get("name")}

    rid = req["round_id"]
    rd = round_dir(rid)
    already = (not rd.exists()) and not (cur and cur.get("round_id") == rid)

    if role == "leader":
        arc = read_json(archive_path())
        entry = ((arc or {}).get("rounds") or {}).get(rid) if isinstance(arc, dict) else None
        if not isinstance(entry, dict) or entry.get("data_complete") is not True:
            raise RuntimeError("verified complete archive required before leader cleanup")
        expected_digest = p.get("archive_digest")
        actual_digest = digest(entry)
        if expected_digest and expected_digest != actual_digest:
            raise RuntimeError("archive digest mismatch")
    else:
        if p.get("archive_confirmed") is not True or not p.get("archive_digest"):
            raise RuntimeError("leader archive confirmation required before member cleanup")

    if rd.exists():
        remove_tree(rd)
    cp = current_path()
    cur2 = read_json(cp)
    if isinstance(cur2, dict) and cur2.get("round_id") == rid:
        cp.unlink(missing_ok=True)
    if RESPONSES.exists():
        remove_tree(RESPONSES)
    rounds_root = ROOT / "rounds"
    try:
        if rounds_root.exists() and not any(rounds_root.iterdir()):
            rounds_root.rmdir()
    except Exception:
        pass
    fsync_dir(ROOT)
    return base_response(req, responder, status="ok",
                         effect="already_applied" if already else "applied",
                         runtime_state="cleaned", registration_active=False,
                         cleaned=True, archive_digest=p.get("archive_digest"))

def control(req, role):
    try:
        validate_envelope(req)
    except Exception as e:
        emit({"protocol":"mrr-v1","status":"error","effect":"not_applied","error":str(e)})
    cmd = req["command"]
    # CLEANUP intentionally does not use response cache: it deletes response caches and must not recreate one.
    if cmd != "CLEANUP":
        RESPONSES.mkdir(parents=True, exist_ok=True)
        cached = cached_response(req)
        if cached is not None:
            emit(cached)
    try:
        cmd = req["command"]
        if cmd == "START":
            resp = handle_start(req, role)
        elif cmd == "STATUS":
            resp = handle_status(req, role)
        elif cmd == "SNAPSHOT":
            resp = handle_snapshot(req, role)
        elif cmd == "UPDATE":
            resp = handle_update(req, role)
        elif cmd == "STOP":
            resp = handle_stop(req, role)
        elif cmd == "CLEANUP":
            resp = handle_cleanup(req, role)
        else:
            raise RuntimeError("unsupported command")
    except Exception as e:
        cur = load_current()
        responder = (cur or {}).get("self")
        try:
            responder = responder or responder_from(req, cur, role)
        except Exception:
            responder = None
        resp = base_response(req, responder, status="conflict" if "conflict" in str(e).lower() or "mismatch" in str(e).lower() else "error",
                             effect="not_applied", error=str(e),
                             runtime_state=(cur or {}).get("state"),
                             registration_active=bool((cur or {}).get("registration_active")))
    if req.get("command") != "CLEANUP":
        cache_response(req, resp)
    emit(resp)

def inspect_employee():
    cur = converge()
    if not cur or cur.get("state") == "stopped" or not cur.get("registration_active"):
        emit({"status":"no_activity"})
    if cur.get("role") not in {"leader","member"}:
        emit({"status":"error","code":"INVALID_ROLE"})
    if cur.get("state") != "registering" or not cur.get("accepting_new"):
        emit({"status":"unavailable","state":cur.get("state"),"accepting_new":bool(cur.get("accepting_new"))})
    try:
        cfg = load_config(cur)
    except Exception as e:
        emit({"status":"error","code":"CONFIG_INVALID","error":str(e)})
    emit({"status":"ready","round_id":cur["round_id"],"role":cur["role"],
          "config_version":cur["config_version"],"config":cfg})

def begin_interaction(inp):
    cur = converge()
    if not cur or cur.get("state") != "registering" or not cur.get("registration_active") or not cur.get("accepting_new"):
        emit({"status":"unavailable","state":(cur or {}).get("state")})
    lock = current_lock(cur)
    if lock_is_live(lock):
        emit({"status":"busy","session_id":lock.get("session_id"),"expires_at":lock.get("expires_at")})
    cfg = load_config(cur)
    ttl = int((cfg.get("runtime") or {}).get("interaction_ttl_seconds", DEFAULT_TTL_SECONDS))
    sid = uuid.uuid4().hex
    t = now_dt()
    session = {
        "session_id":sid,
        "round_id":cur["round_id"],
        "status":"in_progress",
        "config_version":cur["config_version"],
        "config_path":cur["config_path"],
        "started_at":t.isoformat(),
        "updated_at":t.isoformat(),
        "expires_at":(t+timedelta(seconds=ttl)).isoformat(),
        "data":inp.get("data") or {},
    }
    lock = {
        "session_id":sid,
        "status":"in_progress",
        "started_at":t.isoformat(),
        "updated_at":t.isoformat(),
        "expires_at":(t+timedelta(seconds=ttl)).isoformat(),
        "ttl_seconds":ttl,
        "config_version":cur["config_version"],
    }
    create_json_once(session_path(cur["round_id"], sid), session)
    atomic_write_json(lock_path(cur["round_id"]), lock)
    saved = read_json(session_path(cur["round_id"], sid))
    saved_lock = read_json(lock_path(cur["round_id"]))
    if saved.get("session_id") != sid or saved_lock.get("session_id") != sid:
        emit({"status":"error","code":"BEGIN_VERIFY_FAILED"})
    emit({"status":"ok","session_id":sid,"expires_at":lock["expires_at"],
          "round_id":cur["round_id"],"config_version":cur["config_version"],"config":cfg})

def session_update(inp):
    cur = converge()
    if not cur:
        emit({"status":"error","code":"NO_ACTIVITY"})
    sid = inp.get("session_id")
    if not sid:
        emit({"status":"error","code":"SESSION_ID_REQUIRED"})
    lock = current_lock(cur)
    if not lock_is_live(lock) or lock.get("session_id") != sid:
        emit({"status":"error","code":"SESSION_NOT_ACTIVE"})
    s = read_json(session_path(cur["round_id"], sid))
    if not s or s.get("status") != "in_progress":
        emit({"status":"error","code":"SESSION_NOT_ACTIVE"})
    patch = inp.get("data") or {}
    data = dict(s.get("data") or {})
    for k,v in patch.items():
        data[k] = v
    ttl = int(lock.get("ttl_seconds", DEFAULT_TTL_SECONDS))
    t = now_dt()
    s["data"] = data
    s["updated_at"] = t.isoformat()
    s["expires_at"] = (t+timedelta(seconds=ttl)).isoformat()
    lock["updated_at"] = t.isoformat()
    lock["expires_at"] = s["expires_at"]
    atomic_write_json(session_path(cur["round_id"], sid), s)
    atomic_write_json(lock_path(cur["round_id"]), lock)
    emit({"status":"ok","session_id":sid,"expires_at":s["expires_at"]})

def validate_answers(cfg, answers):
    if not isinstance(answers, dict):
        raise ValueError("answers must be object")
    errors = []
    for q in cfg.get("questions") or []:
        qid = q.get("id")
        if not qid:
            continue
        val = answers.get(qid)
        if q.get("required", False) and (val is None or val == "" or val == []):
            errors.append(f"{qid}: required")
            continue
        if val is None:
            continue
        typ = q.get("type","text")
        if typ == "single_choice":
            if val not in (q.get("options") or []):
                errors.append(f"{qid}: invalid option")
        elif typ == "multi_choice":
            if not isinstance(val, list):
                errors.append(f"{qid}: must be list")
            else:
                opts = set(q.get("options") or [])
                bad = [x for x in val if x not in opts]
                if bad:
                    errors.append(f"{qid}: invalid options {bad}")
                max_n = q.get("max_selections")
                if max_n is not None and len(val) > int(max_n):
                    errors.append(f"{qid}: too many selections")
        elif typ == "number":
            try:
                num = float(val)
                if q.get("min") is not None and num < float(q["min"]):
                    errors.append(f"{qid}: below min")
                if q.get("max") is not None and num > float(q["max"]):
                    errors.append(f"{qid}: above max")
            except Exception:
                errors.append(f"{qid}: not a number")
        elif typ == "text":
            if q.get("required", False) and not str(val).strip():
                errors.append(f"{qid}: empty text")
    if errors:
        raise ValueError("; ".join(errors))

def commit_record(inp):
    cur = converge()
    if not cur:
        emit({"status":"error","code":"NO_ACTIVITY"})
    sid = inp.get("session_id")
    lock = current_lock(cur)
    if not sid or not lock_is_live(lock) or lock.get("session_id") != sid:
        emit({"status":"error","code":"SESSION_NOT_ACTIVE"})
    s = read_json(session_path(cur["round_id"], sid))
    if not s or s.get("status") != "in_progress":
        emit({"status":"error","code":"SESSION_NOT_ACTIVE"})
    allowed = False
    if cur.get("state") in {"registering","config_updating"}:
        allowed = True
    elif cur.get("state") == "stopping" and cur.get("stop_mode") == "graceful":
        deadline = parse_time(cur.get("stop_deadline"))
        allowed = not deadline or now_dt() <= deadline
    if not allowed:
        emit({"status":"error","code":"ROUND_NOT_WRITABLE","state":cur.get("state")})
    cfg = read_json(s.get("config_path"))
    if not cfg or cfg.get("round_id") != cur["round_id"] or int(cfg.get("config_version",-1)) != int(s.get("config_version",-2)):
        emit({"status":"error","code":"SESSION_CONFIG_INVALID"})
    answers = inp.get("answers") or {}
    try:
        validate_answers(cfg, answers)
    except Exception as e:
        emit({"status":"error","code":"ANSWER_VALIDATION_FAILED","error":str(e)})
    employee = inp.get("employee") or {}
    stable_id = employee.get("stable_id")
    if not stable_id:
        emit({"status":"error","code":"EMPLOYEE_STABLE_ID_REQUIRED"})
    event_id = uuid.uuid4().hex
    event = {
        "event_id":event_id,
        "round_id":cur["round_id"],
        "robot_id":(cur.get("self") or {}).get("peer_id"),
        "robot_name":(cur.get("self") or {}).get("name"),
        "employee":employee,
        "operation":inp.get("operation","register"),
        "answers":answers,
        "registered_at":inp.get("registered_at") or now_iso(),
        "saved_at":now_iso(),
        "config_version":int(s["config_version"]),
        "session_id":sid,
    }
    p = records_dir(cur["round_id"]) / f"{event_id}.json"
    try:
        create_json_once(p, event)
    except Exception as e:
        emit({"status":"error","code":"RECORD_WRITE_FAILED","error":str(e)})
    saved = read_json(p)
    checks = ["event_id","round_id","employee","answers","config_version"]
    if any(canonical(saved.get(k)) != canonical(event.get(k)) for k in checks):
        emit({"status":"error","code":"RECORD_VERIFY_FAILED"})
    mark_session(cur["round_id"], sid, "completed")
    mark_lock(cur["round_id"], lock, "completed")
    cur2 = converge()
    emit({"status":"ok","event_id":event_id,"saved_at":event["saved_at"],
          "round_state":(cur2 or {}).get("state"),
          "config_version":(cur2 or {}).get("config_version")})

def abort_interaction(inp):
    cur = converge()
    if not cur:
        emit({"status":"ok","effect":"nothing_to_abort"})
    sid = inp.get("session_id")
    reason = inp.get("reason","interaction aborted")
    lock = current_lock(cur)
    if lock and lock.get("status") == "in_progress" and (not sid or lock.get("session_id") == sid):
        sid2 = lock.get("session_id")
        mark_session(cur["round_id"], sid2, "aborted", reason)
        mark_lock(cur["round_id"], lock, "aborted", reason)
    cur2 = converge()
    emit({"status":"ok","round_state":(cur2 or {}).get("state")})

def store_page(inp):
    cur = converge()
    if not cur or cur.get("role") != "leader":
        emit({"status":"error","code":"LEADER_ONLY"})
    target = inp.get("target_peer_id")
    resp = inp.get("response")
    if not target or not isinstance(resp, dict):
        emit({"status":"error","code":"TARGET_AND_RESPONSE_REQUIRED"})
    if (resp.get("responder") or {}).get("peer_id") != target:
        emit({"status":"conflict","code":"RESPONDER_MISMATCH"})
    if resp.get("round_id") != cur.get("round_id"):
        emit({"status":"conflict","code":"ROUND_MISMATCH"})
    sid = resp.get("snapshot_id")
    cursor = resp.get("cursor")
    rid = resp.get("request_id")
    if not sid or cursor is None or not rid:
        emit({"status":"error","code":"NOT_A_SNAPSHOT_PAGE"})
    p = round_dir(cur["round_id"]) / "received" / "snapshots" / target / sid / f"page-{cursor}-{rid}.json"
    try:
        create_json_once(p, resp)
    except RuntimeError:
        existing = read_json(p)
        if canonical(existing) != canonical(resp):
            emit({"status":"conflict","code":"PAGE_FILE_CONFLICT"})
    saved = read_json(p)
    keys = ["request_id","snapshot_id","cursor","next_cursor","page_item_count","page_valid_record_count","page_invalid_record_count"]
    if any(canonical(saved.get(k)) != canonical(resp.get(k)) for k in keys):
        emit({"status":"error","code":"PAGE_VERIFY_FAILED"})
    emit({"status":"ok","path":str(p),"snapshot_id":sid,"cursor":cursor})

def leader_note(inp):
    cur = converge()
    if not cur or cur.get("role") != "leader":
        emit({"status":"error","code":"LEADER_ONLY"})
    peer = inp.get("peer_id")
    kind = inp.get("kind")
    if not peer or not kind:
        emit({"status":"error","code":"PEER_AND_KIND_REQUIRED"})
    notes = dict(cur.get("member_runtime_status") or {})
    one = dict(notes.get(peer) or {})
    one[kind] = {"updated_at":now_iso(),"data":inp.get("data")}
    notes[peer] = one
    cur["member_runtime_status"] = notes
    save_current(cur)
    emit({"status":"ok","peer_id":peer,"kind":kind})

def snapshot_bundle(round_id, peer_id, snapshot_id):
    base = round_dir(round_id) / "received" / "snapshots" / peer_id / snapshot_id
    raw_pages = []
    if base.exists():
        for p in base.glob("page-*.json"):
            r = read_json(p)
            if isinstance(r, dict):
                raw_pages.append(r)
    raw_pages.sort(key=lambda r: (int(r.get("cursor", -1)), str(r.get("request_id", ""))))

    errors = []
    by_cursor = {}
    for r in raw_pages:
        if r.get("round_id") != round_id:
            errors.append("round mismatch")
            continue
        if (r.get("responder") or {}).get("peer_id") != peer_id:
            errors.append("responder mismatch")
            continue
        if r.get("snapshot_id") != snapshot_id:
            errors.append("snapshot mismatch")
            continue
        try:
            c = int(r.get("cursor"))
        except Exception:
            errors.append("invalid cursor")
            continue
        if c in by_cursor:
            if canonical(by_cursor[c]) != canonical(r):
                errors.append(f"duplicate cursor {c}")
            continue
        by_cursor[c] = r

    pages = []
    events = []
    invalid_count = 0
    expected_cursor = 0
    snapshot_count = None
    done = False
    seen = set()
    while expected_cursor in by_cursor and expected_cursor not in seen:
        seen.add(expected_cursor)
        r = by_cursor[expected_cursor]
        pages.append(r)
        sc = int(r.get("snapshot_count") or 0)
        if snapshot_count is None:
            snapshot_count = sc
        elif sc != snapshot_count:
            errors.append("snapshot_count changed across pages")
        item = int(r.get("page_item_count") or 0)
        valid = int(r.get("page_valid_record_count") or 0)
        invalid = int(r.get("page_invalid_record_count") or 0)
        if item != valid + invalid:
            errors.append(f"page count mismatch at cursor {expected_cursor}")
        records = r.get("records") or []
        if len(records) != valid:
            errors.append(f"records length mismatch at cursor {expected_cursor}")
        events.extend(records)
        invalid_count += invalid
        nxt = int(r.get("next_cursor") or 0)
        if nxt != expected_cursor + item:
            errors.append(f"next_cursor mismatch at cursor {expected_cursor}")
            break
        done = bool(r.get("done"))
        expected_cursor = nxt
        if done:
            break

    if 0 not in by_cursor:
        errors.append("missing cursor 0")
    if not pages or not done:
        errors.append("final page missing")
    processed = sum(int(r.get("page_item_count") or 0) for r in pages)
    if snapshot_count is None:
        snapshot_count = 0
    if processed != snapshot_count:
        errors.append("processed count does not equal snapshot_count")
    if invalid_count:
        errors.append("snapshot contains invalid record files")
    complete = not errors
    return {
        "events":events, "pages":pages, "invalid_count":invalid_count,
        "snapshot_count":snapshot_count, "processed_count":processed,
        "complete":complete, "errors":errors,
    }

def collect_snapshot_events(round_id, peer_id, snapshot_id):
    b = snapshot_bundle(round_id, peer_id, snapshot_id)
    return b["events"], b["pages"], b["invalid_count"]

def effective_latest(events):
    by_event = {}
    for e in events:
        if isinstance(e, dict) and e.get("event_id"):
            by_event[e["event_id"]] = e
    valid = list(by_event.values())
    def key(e):
        return (parse_time(e.get("registered_at")) or datetime.min.replace(tzinfo=timezone.utc), str(e.get("event_id")))
    by_emp = {}
    for e in sorted(valid, key=key):
        sid = ((e.get("employee") or {}).get("stable_id"))
        if sid:
            by_emp[sid] = e
    return list(by_emp.values()), valid

def aggregate(inp):
    cur = converge()
    if not cur or cur.get("role") != "leader":
        emit({"status":"error","code":"LEADER_ONLY"})
    local_mode = inp.get("local_mode","current")
    if local_mode == "final":
        final = ensure_final(cur)
        if not final:
            emit({"status":"error","code":"LOCAL_NOT_STOPPED"})
        local = count_records(cur["round_id"], final.get("record_files") or [])
    else:
        local = count_records(cur["round_id"])
    events = list(local["events"])
    remote_summary = []
    for x in inp.get("remote_snapshots") or []:
        peer = x.get("peer_id")
        sid = x.get("snapshot_id")
        ev, pages, inv = collect_snapshot_events(cur["round_id"], peer, sid)
        events.extend(ev)
        remote_summary.append({
            "peer_id":peer,"snapshot_id":sid,"page_count":len(pages),
            "received_event_count":len(ev),"invalid_file_count":inv
        })
    effective, unique_events = effective_latest(events)
    emit({
        "status":"ok",
        "round_id":cur["round_id"],
        "unique_event_count":len(unique_events),
        "effective_employee_count":len(effective),
        "effective_records":effective,
        "remote_summary":remote_summary,
        "local_invalid_record_count":local["invalid_record_count"],
    })

def archived_record(e):
    return {
        "employee": e.get("employee"),
        "answers": e.get("answers") or {},
        "registered_at": e.get("registered_at"),
        "saved_at": e.get("saved_at"),
        "robot_id": e.get("robot_id"),
        "config_version": e.get("config_version"),
    }

def build_answer_statistics(cfg, effective):
    out = {}
    for q in cfg.get("questions") or []:
        qid = q.get("id")
        if not qid:
            continue
        typ = q.get("type", "text")
        values = [(e.get("answers") or {}).get(qid) for e in effective]
        present = [v for v in values if v is not None and v != "" and v != []]
        item = {"type":typ,"answered":len(present),"missing":len(values)-len(present)}
        if typ == "single_choice":
            counts = {}
            for v in present:
                k = str(v)
                counts[k] = counts.get(k, 0) + 1
            item["value_counts"] = counts
        elif typ == "multi_choice":
            counts = {}
            for v in present:
                for x in (v if isinstance(v, list) else [v]):
                    k = str(x)
                    counts[k] = counts.get(k, 0) + 1
            item["selection_counts"] = counts
        elif typ == "number":
            nums = []
            for v in present:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            if nums:
                item["min"] = min(nums)
                item["max"] = max(nums)
                item["average"] = sum(nums) / len(nums)
        out[qid] = item
    return out

def archive_final(inp):
    cur = converge()
    if not cur or cur.get("role") != "leader":
        emit({"status":"error","code":"LEADER_ONLY"})
    if cur.get("state") != "stopped":
        emit({"status":"error","code":"LOCAL_NOT_STOPPED"})

    final = ensure_final(cur)
    if not final:
        emit({"status":"error","code":"LOCAL_FINAL_MISSING"})
    local = count_records(cur["round_id"], final.get("record_files") or [])
    events = list(local["events"])
    data_complete = local["invalid_record_count"] == 0

    expected_members = [x for x in (cur.get("team_members") or []) if isinstance(x, dict) and x.get("peer_id")]
    supplied = {}
    for x in inp.get("remote_snapshots") or []:
        if isinstance(x, dict) and x.get("peer_id") and x.get("snapshot_id"):
            supplied[x["peer_id"]] = x["snapshot_id"]

    member_stop_status = inp.get("member_stop_status") or {}
    robot_data = [{
        "peer_id":(cur.get("self") or {}).get("peer_id"),
        "name":(cur.get("self") or {}).get("name"),
        "role":"leader",
        "record_count":local["record_count"],
        "valid_record_count":local["valid_record_count"],
        "invalid_record_count":local["invalid_record_count"],
        "data_complete":local["invalid_record_count"] == 0,
    }]

    expected_peer_ids = {m["peer_id"] for m in expected_members}
    for m in expected_members:
        peer = m["peer_id"]
        sid = supplied.get(peer)
        if not sid:
            data_complete = False
            robot_data.append({"peer_id":peer,"name":m.get("name"),"role":"member",
                               "data_complete":False,"errors":["final snapshot missing"]})
            continue
        b = snapshot_bundle(cur["round_id"], peer, sid)
        events.extend(b["events"])
        peer_errors = list(b["errors"])
        peer_complete = bool(b["complete"])
        stop_info = member_stop_status.get(peer) or {}
        expected_count = stop_info.get("record_count")
        if expected_count is not None:
            try:
                if int(expected_count) != int(b["snapshot_count"]):
                    peer_errors.append("final snapshot_count does not equal STOP record_count")
                    peer_complete = False
            except Exception:
                peer_errors.append("invalid STOP record_count")
                peer_complete = False
        if stop_info and stop_info.get("runtime_state") != "stopped":
            peer_errors.append("member STOP state not stopped")
            peer_complete = False
        if not peer_complete:
            data_complete = False
        robot_data.append({
            "peer_id":peer,"name":m.get("name"),"role":"member",
            "snapshot_id":sid,"snapshot_count":b["snapshot_count"],
            "received_event_count":len(b["events"]),
            "invalid_record_count":b["invalid_count"],
            "data_complete":peer_complete,"errors":peer_errors,
        })

    extra_supplied = sorted(set(supplied) - expected_peer_ids)
    if extra_supplied:
        data_complete = False
        robot_data.append({"role":"unexpected","data_complete":False,"peer_ids":extra_supplied,
                           "errors":["snapshot supplied for robot not in team_members"]})

    effective, unique_events = effective_latest(events)
    try:
        cfg = load_config(cur)
    except Exception as e:
        emit({"status":"error","code":"CONFIG_INVALID","error":str(e)})

    entry = {
        "round_id":cur["round_id"],
        "activity_name":cfg.get("activity_name"),
        "started_at":cur.get("created_at"),
        "ended_at":final.get("frozen_at"),
        "archived_at":now_iso(),
        "final_config_version":cur.get("config_version"),
        "config":cfg,
        "leader":cur.get("self"),
        "team_members":expected_members,
        "unique_success_event_count":len(unique_events),
        "effective_employee_count":len(effective),
        "statistics":build_answer_statistics(cfg, effective),
        "results":[archived_record(e) for e in effective],
        "robot_data":robot_data,
        "member_stop_status":member_stop_status,
        "data_complete":bool(data_complete),
    }

    ap = archive_path()
    arc = read_json(ap)
    if not isinstance(arc, dict):
        arc = {"schema":"mrr-archive-v1","rounds":{}}
    if arc.get("schema") != "mrr-archive-v1":
        emit({"status":"error","code":"ARCHIVE_SCHEMA_CONFLICT"})
    rounds = arc.get("rounds")
    if not isinstance(rounds, dict):
        rounds = {}
    existing = rounds.get(cur["round_id"])
    if isinstance(existing, dict) and existing.get("data_complete") is True:
        emit({
            "status":"ok","archived":True,"effect":"already_applied",
            "round_id":cur["round_id"],"data_complete":True,
            "archive_digest":digest(existing),
            "summary":{
                "effective_employee_count":existing.get("effective_employee_count"),
                "unique_success_event_count":existing.get("unique_success_event_count"),
                "statistics":existing.get("statistics"),
                "robot_data":existing.get("robot_data"),
            }
        })
    rounds[cur["round_id"]] = entry
    arc["rounds"] = rounds
    arc["updated_at"] = now_iso()
    atomic_write_json(ap, arc)
    saved = read_json(ap)
    saved_entry = ((saved or {}).get("rounds") or {}).get(cur["round_id"])
    if not isinstance(saved_entry, dict) or digest(saved_entry) != digest(entry):
        emit({"status":"error","code":"ARCHIVE_VERIFY_FAILED"})
    emit({
        "status":"ok","archived":True,"effect":"applied",
        "round_id":cur["round_id"],"data_complete":bool(data_complete),
        "archive_digest":digest(saved_entry),
        "summary":{
            "effective_employee_count":len(effective),
            "unique_success_event_count":len(unique_events),
            "statistics":entry["statistics"],
            "robot_data":robot_data,
        }
    })

def archive_read(inp):
    arc = read_json(archive_path())
    if not isinstance(arc, dict):
        emit({"status":"ok","schema":"mrr-archive-v1","rounds":[]})
    rid = inp.get("round_id")
    rounds = arc.get("rounds") or {}
    if rid:
        item = rounds.get(rid)
        if item is None:
            emit({"status":"not_found","round_id":rid})
        emit({"status":"ok","round":item})
    index = []
    for k, v in rounds.items():
        if isinstance(v, dict):
            index.append({
                "round_id":k,"activity_name":v.get("activity_name"),
                "started_at":v.get("started_at"),"ended_at":v.get("ended_at"),
                "effective_employee_count":v.get("effective_employee_count"),
                "data_complete":v.get("data_complete"),
            })
    index.sort(key=lambda x: str(x.get("ended_at") or x.get("started_at") or ""))
    emit({"status":"ok","schema":arc.get("schema"),"updated_at":arc.get("updated_at"),"rounds":index})

def cli():
    if len(sys.argv) < 2:
        emit({"status":"error","code":"COMMAND_REQUIRED"}, 2)
    cmd = sys.argv[1]
    if cmd == "--version":
        print(VERSION)
        return
    inp = load_stdin()
    if cmd == "control-member":
        control(inp, "member")
    elif cmd == "control-leader":
        control(inp, "leader")
    elif cmd == "inspect":
        inspect_employee()
    elif cmd == "begin":
        begin_interaction(inp)
    elif cmd == "session-update":
        session_update(inp)
    elif cmd == "commit":
        commit_record(inp)
    elif cmd == "abort":
        abort_interaction(inp)
    elif cmd == "store-page":
        store_page(inp)
    elif cmd == "leader-note":
        leader_note(inp)
    elif cmd == "aggregate":
        aggregate(inp)
    elif cmd == "archive-final":
        archive_final(inp)
    elif cmd == "archive-read":
        archive_read(inp)
    elif cmd == "converge":
        cur = converge()
        emit({"status":"ok","current":cur})
    else:
        emit({"status":"error","code":"UNKNOWN_COMMAND","command":cmd}, 2)

if __name__ == "__main__":
    cli()

