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


def _frame_struct_vec(bgr) -> "np.ndarray":
    import cv2
    import numpy as np

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(cv2.GaussianBlur(gray, (0, 0), 2.0), (24, 42), interpolation=cv2.INTER_AREA)
    gx = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)).ravel()
    gy = np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)).ravel()
    hog = np.concatenate([gx, gy])
    hog = hog / (float(np.linalg.norm(hog)) + 1e-5)
    luma = small.ravel() - float(small.mean())
    luma = luma / (float(np.linalg.norm(luma)) + 1e-5)
    v = np.concatenate([0.70 * hog, 0.70 * luma])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-5 else np.zeros_like(v)


def _split_multishot_unit(video_file: Path, shot_ids: list[int], trim_to: float, out_dir: Path,
                          ref_shots: list[dict] | None = None, bdir: Path | None = None) -> dict[str, str]:
    """Split a single-call multi-shot unit clip at its actual pixel hard cuts and align sub-clips to target shots.

    Multi-shot generators occasionally shift cut timing or insert an extra internal sub-cut inside one shot. Rather than
    blindly taking top-(N-1) histogram jumps (which cascades an off-by-one shift across later shots when K > N), this
    detects all true hard cuts (K >= N sub-clips) and runs monotonic DP alignment on normalized time + appearance-invariant
    structural similarity so every target shot gets one clean, jump-free sub-clip.
    """
    import subprocess
    import cv2
    import numpy as np

    n_target = len(shot_ids)
    if n_target < 2 or not video_file.is_file():
        return {}
    cap = cv2.VideoCapture(str(video_file))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or total_frames < n_target * 4:
        cap.release()
        return {}
    end_frame = min(total_frames, max(n_target * 4, int(round(trim_to * fps)))) if trim_to > 0 else total_frames
    small_frames: list = []
    hists: list = []
    while len(small_frames) < end_frame:
        ok, fr = cap.read()
        if not ok:
            break
        sm = cv2.resize(fr, (108, 192))
        small_frames.append(sm)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(h, h)
        hists.append(h)
    cap.release()
    end_frame = len(small_frames)
    if end_frame < n_target * 4:
        return {}

    # Combined cut strength: HSV Bhattacharyya jump + normalized pixel MAD spike above local neighbourhood
    mads = np.array([float(np.mean(np.abs(small_frames[i].astype(np.int16) - small_frames[i - 1].astype(np.int16))))
                     for i in range(1, end_frame)], dtype=np.float32)
    bhat = np.array([float(cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA))
                     for i in range(1, end_frame)], dtype=np.float32)
    # Local contrast ratio on pixel MAD catches hard cuts even between two similarly coloured dark scenes
    med_k = np.array([float(np.median(mads[max(0, i - 5):min(len(mads), i + 6)])) for i in range(len(mads))], dtype=np.float32)
    spike_ratio = mads / (med_k + 2.5)
    scores: list[tuple[int, float]] = []
    for i in range(3, end_frame - 3):
        sc = float(0.55 * bhat[i - 1] + 0.45 * min(1.0, mads[i - 1] / 60.0))
        is_real_cut = (mads[i - 1] >= 28.0 and spike_ratio[i - 1] >= 2.6 and bhat[i - 1] >= 0.16) or bhat[i - 1] >= 0.36
        scores.append((i, sc, is_real_cut))

    min_gap = max(6, int(round(0.35 * fps)))
    # First collect all genuine hard cuts (up to n_target + 3 to absorb occasional extra sub-cuts)
    strong_cuts: list[int] = []
    for f_idx, sc, is_real in sorted(scores, key=lambda x: -x[1]):
        if not is_real and len(strong_cuts) >= n_target - 1:
            break
        if sc < 0.22:
            break
        if all(abs(f_idx - p) >= min_gap for p in strong_cuts):
            strong_cuts.append(f_idx)
            if len(strong_cuts) >= n_target + 3:
                break
    if len(strong_cuts) < n_target - 1:
        return {}

    cand_bounds = [0] + sorted(strong_cuts) + [end_frame]
    subclips = [(a, b) for a, b in zip(cand_bounds[:-1], cand_bounds[1:]) if b - a >= 2]
    k_sub = len(subclips)
    if k_sub < n_target:
        return {}

    if k_sub == n_target:
        chosen = subclips
    else:
        # Monotonic DP: match each of the N target shots to its best clean sub-clip (by structural frame match + normalized time)
        gen_vecs = [_frame_struct_vec(small_frames[(a + b) // 2]) for a, b in subclips]
        gen_mids = [0.5 * (a + b) / float(end_frame) for a, b in subclips]
        ref_vecs: list = [None] * n_target
        ref_mids = [(i + 0.5) / float(n_target) for i in range(n_target)]
        ref_durs = [1.0 / float(n_target)] * n_target
        if ref_shots and len(ref_shots) == n_target:
            u_t0 = float(ref_shots[0]["start"])
            u_span = max(0.1, float(ref_shots[-1]["end"]) - u_t0)
            ref_mids = [(0.5 * (float(s["start"]) + float(s["end"])) - u_t0) / u_span for s in ref_shots]
            ref_durs = [max(0.02, (float(s["end"]) - float(s["start"])) / u_span) for s in ref_shots]
            if bdir is not None:
                for idx_i, s in enumerate(ref_shots):
                    ff = (s.get("assets") or {}).get("first_frame")
                    p_ff = (Path(bdir) / ff) if ff else None
                    if p_ff and p_ff.is_file():
                        im = cv2.imread(str(p_ff))
                        if im is not None:
                            ref_vecs[idx_i] = _frame_struct_vec(im)

        gen_durs = [max(0.02, (b - a) / float(end_frame)) for a, b in subclips]
        dp = np.full((n_target, k_sub), -1e9, dtype=np.float64)
        parent = np.full((n_target, k_sub), -1, dtype=np.int32)
        for j in range(k_sub):
            for i in range(n_target):
                if j < i or (k_sub - 1 - j) < (n_target - 1 - i):
                    continue
                sim = float(np.dot(ref_vecs[i], gen_vecs[j])) if ref_vecs[i] is not None else 0.5
                t_pen = 0.65 * abs(gen_mids[j] - ref_mids[i])
                dur_ratio = min(gen_durs[j], ref_durs[i]) / max(gen_durs[j], ref_durs[i])
                node_sc = sim - t_pen + 0.35 * dur_ratio
                if i == 0:
                    dp[i, j] = node_sc
                else:
                    best_prev = int(np.argmax(dp[i - 1, :j]))
                    dp[i, j] = dp[i - 1, best_prev] + node_sc
                    parent[i, j] = best_prev
        cur_j = int(np.argmax(dp[n_target - 1]))
        pick_idx = []
        for i in range(n_target - 1, -1, -1):
            pick_idx.append(cur_j)
            cur_j = int(parent[i, cur_j])
        chosen = [subclips[j] for j in reversed(pick_idx)]

    out_dir.mkdir(parents=True, exist_ok=True)
    clips: dict[str, str] = {}
    for s_id, (a, b) in zip(shot_ids, chosen):
        dst = out_dir / f"shot_{int(s_id):03d}.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video_file),
             "-vf", f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst)],
            check=True,
        )
        clips[str(s_id)] = str(dst)
    return clips


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
        if video and j.get("unit") and int(j.get("parts") or 1) == 1 and len(j.get("shots") or []) > 1:
            try:
                b_json = Path(spec["breakdown"]) if spec.get("breakdown") else None
                b_data = json.loads(b_json.read_text(encoding="utf-8")) if b_json and b_json.is_file() else {}
                by_idx = {int(s["index"]): s for s in (b_data.get("shots") or [])}
                r_shots = [by_idx[int(i)] for i in j["shots"] if int(i) in by_idx]
                per_shot = _split_multishot_unit(
                    Path(video), list(j["shots"]), float(j["trim_to"]), Path(video).parent / "shots",
                    ref_shots=r_shots if len(r_shots) == len(j["shots"]) else None,
                    bdir=b_json.parent if b_json else None,
                )
                for s_key, s_vid in per_shot.items():
                    results["results"][s_key] = {"id": f"shot_{int(s_key):03d}", "video": s_vid, "from_unit": j["unit"]}
            except Exception:
                pass
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
