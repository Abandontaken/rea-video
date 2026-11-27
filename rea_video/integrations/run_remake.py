#!/usr/bin/env python3
"""Execute rea-video remake jobs with capsule-cinema's video generation tools.

Runs inside capsule-cinema's Python environment (its .env provides API keys).
Writes results.json next to jobs.json; rea-video `assemble` consumes it."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


_FALLBACK_ENGINE_ALIASES = {
    "gemini-omni-flash": "gemini-omni-1.2-flash",
    "veo-3.1": "veo3",
}


def _resolve_engine_name(raw_engine: str, vgt) -> str:
    import os

    routes = {}
    for item in os.getenv("REA_VIDEO_ENGINE_ROUTES", "").split(","):
        if "=" in item:
            a, b = item.split("=", 1)
            if a.strip() and b.strip():
                routes[vgt.normalize_video_engine(a.strip())] = vgt.normalize_video_engine(b.strip())
    norm = vgt.normalize_video_engine(raw_engine)
    if norm in routes:
        return routes[norm]
    try:
        from src.video_engines import registered_video_engines

        reg = registered_video_engines()
    except Exception:
        reg = {}
    if reg and norm not in reg and norm in _FALLBACK_ENGINE_ALIASES and _FALLBACK_ENGINE_ALIASES[norm] in reg:
        return _FALLBACK_ENGINE_ALIASES[norm]
    return norm


def _adapt_unit_params(eng_tool, raw_params: dict, model_options: dict | None, job_id: str) -> dict:
    """Adapt UniversalVideoGenerationSchema-shaped params to a specific engine tool's signature & schema."""
    import inspect

    sig = inspect.signature(eng_tool._run)
    schema = getattr(eng_tool, "args_schema", None)
    schema_fields = set(getattr(schema, "model_fields", {}) or {})
    allowed = (schema_fields | set(sig.parameters)) - {"self", "extra", "kwargs"}
    has_strict_schema = bool(schema_fields)

    p = {k: v for k, v in raw_params.items() if k != "engine"}
    if "output_path" in (schema_fields | set(sig.parameters)) and "output_path" not in p and p.get("output_dir"):
        p["output_path"] = str(Path(p["output_dir"]) / f"{job_id}.mp4")
    if "task_id" in allowed and "task_id" not in p and p.get("output_path"):
        op = Path(p["output_path"])
        tf = op.with_suffix(".task.json")
        if not op.is_file() and tf.is_file():
            saved_tid = (json.loads(tf.read_text(encoding="utf-8")) or {}).get("task_id")
            if saved_tid:
                p["task_id"] = saved_tid

    # Map frame keys for tools that name them first_frame / last_frame (e.g. yy2img / gemini-omni-1.2-flash)
    if "first_frame" in allowed and "first_frame" not in p:
        first = p.get("start_image_path") or p.get("image_path")
        if first:
            p["first_frame"] = first
    if "last_frame" in allowed and "last_frame" not in p and p.get("end_image_path"):
        p["last_frame"] = p["end_image_path"]
    if "image_path" in allowed and "image_path" not in p and p.get("start_image_path"):
        p["image_path"] = p["start_image_path"]

    m_opts = dict(model_options or {})
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
    if has_strict_schema or not accepts_var_kw:
        p = {k: v for k, v in p.items() if k in allowed}
        for k, v in m_opts.items():
            if k in allowed:
                p[k] = v
        if "options" in allowed and isinstance(p.get("options"), (dict, type(None))):
            rest = {k: v for k, v in m_opts.items() if k not in allowed}
            if rest:
                p["options"] = {**(p.get("options") or {}), **rest}
        elif "extra" in sig.parameters and sig.parameters["extra"].kind != inspect.Parameter.VAR_KEYWORD:
            rest = {k: v for k, v in m_opts.items() if k not in allowed}
            if rest:
                p["extra"] = {**(p.get("extra") or {}), **rest}
    else:
        p.update(m_opts)
    return p


def _video_path(res) -> str | None:
    """Find the generated file in the various result shapes capsule-cinema tools return."""
    if isinstance(res, str):
        return res if res.lower().endswith((".mp4", ".mov", ".webm")) and Path(res).is_file() else None
    if not isinstance(res, dict):
        return None
    for key in ("output_path", "video_path", "local_path", "path", "file_path"):
        v = res.get(key)
        if isinstance(v, str) and Path(v).is_file():
            return v
    for key in ("result", "data", "video"):
        v = _video_path(res.get(key)) if res.get(key) is not None else None
        if v:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--only", default="", help="comma-separated shots or units: 3,5 / shot_003 / G2")
    ap.add_argument("--capsule-root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs_path = Path(args.jobs).expanduser().resolve()
    spec = json.loads(jobs_path.read_text(encoding="utf-8"))
    only, only_units = set(), set()
    for x in args.only.split(","):
        x = x.strip()
        if not x:
            continue
        u = re.fullmatch(r"G\d+", x, re.I)
        if u:
            only_units.add(x.upper())
            continue
        m = re.fullmatch(r"(?:shot_?)?0*(\d+)", x, re.I)
        if not m:
            sys.exit(f"--only 看不懂 {x!r}：写镜头序号（3）、镜头 id（shot_003）或生成单元（G2），逗号分隔")
        only.add(int(m.group(1)))
    results_path = jobs_path.with_name("results.json")
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {"results": {}}

    root = Path(args.capsule_root).expanduser().resolve()
    planned = Path(spec.get("capsule_root") or "").expanduser().resolve()
    if planned != root:
        sys.exit(f"jobs.json 的 capsule_root（{planned}）与 --capsule-root（{root}）不一致；确认来源可信后用 --capsule-root 指定")
    out_root = (root / "output").resolve()
    for j in spec["jobs"]:
        od = Path(j.get("params", {}).get("output_dir", "")).expanduser().resolve()
        if j.get("executor") == "capsule_cinema" and not od.is_relative_to(out_root):
            sys.exit(f"shot {j.get('shot')}: output_dir 不在 {out_root} 下，拒绝执行")

    def picked(j: dict) -> bool:
        if not (only or only_units):
            return True
        return j.get("unit") in only_units or bool(only & set(j.get("shots") or [j["shot"]]))

    todo = [j for j in spec["jobs"] if j["executor"] == "capsule_cinema" and picked(j)]
    if (only or only_units) and not todo:
        have = sorted(j["shot"] for j in spec["jobs"] if j["executor"] == "capsule_cinema")
        sys.exit(f"--only {args.only} 里没有需要生成的镜头；可生成的镜头序号：{have}")
    print(f"{len(todo)} generation job(s); run dir {spec['run_dir']}")
    if args.dry_run:
        for j in todo:
            p = dict(j["params"])
            p["prompt"] = p["prompt"][:120] + ("…" if len(p["prompt"]) > 120 else "")
            print(json.dumps({"job": j.get("id", j["shot"]), "shots": j.get("shots") or [j["shot"]], "label": j["label"], **p}, ensure_ascii=False))
            if j.get("warning"):
                print(f"  ! {j['warning']}")
        return

    sys.path.insert(0, str(root / "lib"))
    sys.path.insert(0, str(root / "scripts"))
    from env_loader import load_video_agent_env  # noqa: E402

    load_video_agent_env(root)
    from custom_tools.video_generation import video_generation_tool as vgt  # noqa: E402

    tool = vgt.UniversalVideoGenerationTool()
    failed = []
    for j in todo:
        Path(j["params"]["output_dir"]).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        label = j["id"] if j.get("unit") else f"shot {j['shot']}"
        print(f"{label}: {j['params']['generation_type']} via {j['params']['engine']} ...", flush=True)
        try:
            if j.get("unit"):  # a unit's references go straight to the engine's own tool
                eng_name = _resolve_engine_name(j["params"]["engine"], vgt)
                eng_tool = vgt._tool_for_engine(eng_name)
                p = _adapt_unit_params(eng_tool, j["params"], j.get("model_options"), j["id"])
                res = eng_tool._run(**p)
            else:
                res = tool._run(**j["params"])
        except Exception as exc:  # keep going, record the failure
            res = {"error": str(exc)}
        video = _video_path(res)
        key = j["id"] if j.get("unit") else str(j["shot"])  # a unit's clip covers several shots: assemble splits it
        results["results"][key] = {"id": j["id"], "video": video, "raw": res if not video else None,
                                   "seconds": round(time.time() - t0, 1), "trim_to": j["trim_to"],
                                   **({k: j[k] for k in ("unit", "part", "parts", "shots")} if j.get("unit") else {})}
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  -> {video or res}", flush=True)
        if not video:
            failed.append(key)
    print(f"results: {results_path}")
    try:
        from src import clip_check

        print(clip_check.summary(clip_check.check_jobs(str(jobs_path))))
    except Exception:
        pass
    if failed:
        sys.exit(f"{len(failed)}/{len(todo)} 个镜头没生成出视频：{', '.join(failed)}（详情见 results.json）")


if __name__ == "__main__":
    main()
