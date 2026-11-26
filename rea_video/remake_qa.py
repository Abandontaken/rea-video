"""Generative remake fidelity QA: per-shot camera trajectory, subject pose dynamics, framing layout and smoothness.

`layers.compare()` scores flat code-rendered graphics by foreground colour masks. For generated / filmed / 3D shots
where characters, colours or outfits intentionally change (`swaps`), pixel colour diff is meaningless while motion and
framing drift are critical. This module measures four appearance-invariant axes per replaced shot:
  1. smoothness    — absence of frozen duplicate frames (`step=0` warp stutter) when the source shot is moving
  2. camera_score  — agreement of 3D-parallax-aware `camera_traj` (catches inverted push-in vs pull-out or lost pans)
  3. motion_score  — agreement of global-motion-compensated `pose_change` & flow energy (catches hallucinated turns on
                     anchored shots or frozen stills on active action shots)
  4. layout_score  — coarse gradient-envelope cosine similarity on start/end frames (invariant to colour/identity swaps,
                     sensitive to framing collapse such as wide profile -> centred close-up)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .features import global_motion, kinematics_summary, motion_summary
from .i18n import T
from .media import iter_frames, video_info
from .util import write_json

QA_W = 160

_OPPOSITE_TRAJ = {
    frozenset(("push_in", "pull_out")),
    frozenset(("pan_left", "pan_right")),
    frozenset(("tilt_up", "tilt_down")),
}


def _decode_by_shots(video_path: str | Path, shots: list[dict], width: int = QA_W) -> tuple[dict[int, list[np.ndarray]], dict[int, list[float]], float]:
    info = video_info(str(video_path))
    fps = float(info.fps) if info.fps and 0 < info.fps <= 60 else 30.0
    ordered = sorted(shots, key=lambda s: float(s["start"]))
    frames_by_shot: dict[int, list[np.ndarray]] = {int(s["index"]): [] for s in ordered}
    times_by_shot: dict[int, list[float]] = {int(s["index"]): [] for s in ordered}
    if not ordered:
        return frames_by_shot, times_by_shot, fps
    margin = 0.6 / fps  # stay strictly inside shot boundaries so VFR / codec rounding never samples adjacent shots
    k = 0
    for fi, bgr in enumerate(iter_frames(str(video_path), info, width, fps=fps)):
        t = fi / fps
        while k < len(ordered) and t >= float(ordered[k]["end"]) - margin:
            k += 1
        if k >= len(ordered):
            break
        sh = ordered[k]
        if t >= float(sh["start"]) + margin:
            idx = int(sh["index"])
            frames_by_shot[idx].append(bgr.copy())
            times_by_shot[idx].append(t)
    return frames_by_shot, times_by_shot, fps


def _shot_kinematics(frames: list[np.ndarray], times: list[float], fps: float) -> dict:
    if len(frames) < 2:
        return {"frozen_ratio": 0.0, "flow_p90": 0.0, "camera_traj": "static", "net_zoom_pct": 0.0,
                "pose_change": 0.0, "motion_regime": "near_static"}
    trim = 1 if len(frames) > 8 else 0
    fr = frames[trim:len(frames) - trim] if trim else frames
    ts = times[trim:len(times) - trim] if trim else times
    if len(fr) < 2:
        fr, ts = frames, times
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
    mads = [float(cv2.absdiff(grays[i], grays[i - 1]).mean()) for i in range(1, len(grays))]
    frozen_ratio = float(np.mean([m < 0.15 for m in mads])) if mads else 0.0
    step_idx = np.unique(np.linspace(1, len(grays) - 1, num=min(16, len(grays) - 1)).round().astype(int))
    mot = []
    for i in step_idx:
        gm = global_motion(grays[i - 1], grays[i])
        if gm is not None:
            mot.append(gm)
    ms = motion_summary(mot, fps)
    keep_idx = np.unique(np.linspace(0, len(fr) - 1, num=min(12, len(fr))).round().astype(int))
    sub_fr = [fr[i] for i in keep_idx]
    sub_ts = [ts[i] for i in keep_idx]
    kin = kinematics_summary(sub_fr, sub_ts, ms)
    return {
        "frozen_ratio": round(frozen_ratio, 3),
        "flow_p90": round(float(ms.get("flow_p90") or 0.0), 3),
        **kin,
    }


def _coarse_layout_vec(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    gh = max(16, int(round(24 * h / max(1, w))))
    small = cv2.resize(cv2.GaussianBlur(gray, (0, 0), max(1.5, w / 40.0)), (24, gh), interpolation=cv2.INTER_AREA)
    gx = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)).ravel()
    gy = np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)).ravel()
    v = np.concatenate([gx, gy])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-5 else np.zeros_like(v)


def _layout_similarity(src_frames: list[np.ndarray], rem_frames: list[np.ndarray]) -> float:
    if not src_frames or not rem_frames:
        return 1.0
    pairs = [(src_frames[0], rem_frames[0])]
    if len(src_frames) >= 3 and len(rem_frames) >= 3:
        pairs.append((src_frames[-1], rem_frames[-1]))
    sims = []
    for a, b in pairs:
        va = _coarse_layout_vec(a)
        vb = _coarse_layout_vec(cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA))
        if va.any() and vb.any():
            sims.append(float(np.clip(np.dot(va, vb), 0.0, 1.0)))
    return round(float(np.mean(sims)), 3) if sims else 1.0


def evaluate_remake(breakdown: dict, remake_video: str | Path, out_dir: Path | None = None,
                    legend: list[tuple] | None = None, replaced_only: bool = True) -> dict:
    """Compare a remake video shot-by-shot against the source breakdown on camera, pose kinematics, framing and smoothness."""
    src_path = Path(breakdown["source"]["path"])
    rem_path = Path(remake_video)
    if not src_path.is_file() or not rem_path.is_file():
        return {"ok": False, "reason": "missing_video", "shots": []}
    how_map = {int(i): h for i, h in (legend or [])}
    eval_shots = [s for s in breakdown.get("shots") or []
                  if not (replaced_only and how_map and how_map.get(int(s["index"])) == "original")]
    if not eval_shots:
        return {"ok": True, "evaluated_shots": 0, "score": None, "shots": [], "issues": []}

    src_fr, src_ts, src_fps = _decode_by_shots(src_path, eval_shots)
    rem_fr, rem_ts, rem_fps = _decode_by_shots(rem_path, eval_shots)
    rows: list[dict] = []
    issues: list[str] = []
    for sh in eval_shots:
        idx = int(sh["index"])
        sf, st = src_fr.get(idx) or [], src_ts.get(idx) or []
        rf, rt = rem_fr.get(idx) or [], rem_ts.get(idx) or []
        if len(sf) < 2 or len(rf) < 2:
            continue
        sk = _shot_kinematics(sf, st, src_fps)
        rk = _shot_kinematics(rf, rt, rem_fps)
        shot_issues: list[str] = []

        # 1. Smoothness (freeze-frame check)
        if sk["frozen_ratio"] < 0.08 and rk["frozen_ratio"] > 0.18 and sk["motion_regime"] != "near_static":
            smooth = float(np.clip(1.0 - (rk["frozen_ratio"] - 0.10) * 2.0, 0.0, 1.0))
            shot_issues.append(T("冻结重复帧偏多（{r:.0%}）", "excess frozen duplicate frames ({r:.0%})", r=rk["frozen_ratio"]))
        else:
            smooth = 1.0

        # 2. Camera trajectory agreement
        s_tr, r_tr = sk["camera_traj"], rk["camera_traj"]
        if s_tr == r_tr:
            cam_sc = 1.0
        elif frozenset((s_tr, r_tr)) in _OPPOSITE_TRAJ:
            cam_sc = 0.0
            shot_issues.append(T("运镜方向相反（原片 {a}，成片 {b}）", "opposite camera move (source {a}, remake {b})", a=s_tr, b=r_tr))
        elif "static" in (s_tr, r_tr):
            strong = abs(sk["net_zoom_pct"]) >= 4.0 or abs(rk["net_zoom_pct"]) >= 4.0 or s_tr not in ("static", "handheld") or r_tr not in ("static", "handheld")
            cam_sc = 0.45 if strong else 0.75
            if strong:
                shot_issues.append(T("运镜不一致（原片 {a}，成片 {b}）", "camera move mismatch (source {a}, remake {b})", a=s_tr, b=r_tr))
        else:
            cam_sc = 0.65

        # 3. Subject pose & motion-energy fidelity
        sp, rp = sk["pose_change"], rk["pose_change"]
        if sk["motion_regime"] in ("anchored_pose", "camera_only", "near_static") and sp < 0.25 and rp >= sp + 0.20:
            mot_sc = float(np.clip(1.0 - 2.0 * (rp - sp), 0.1, 0.65))
            shot_issues.append(T("主体姿态大幅偏离（原片稳姿 {a:.2f}，成片转身/位移 {b:.2f}）",
                                 "unwanted pose/orientation jump (source anchored {a:.2f}, remake {b:.2f})", a=sp, b=rp))
        elif sk["motion_regime"] == "articulated_action" and sp >= 0.30 and rp < 0.45 * sp:
            mot_sc = 0.35
            shot_issues.append(T("主体动作幅度不足（原片 {a:.2f}，成片 {b:.2f}）",
                                 "under-animated subject action (source {a:.2f}, remake {b:.2f})", a=sp, b=rp))
        else:
            mot_sc = float(np.clip(1.0 - abs(rp - sp) * 1.1, 0.4, 1.0))
        if sk["flow_p90"] > 0.65 and rk["flow_p90"] < 0.25 * sk["flow_p90"]:
            mot_sc = min(mot_sc, 0.45)
            shot_issues.append(T("物理动态偏弱（风动/粒子/运动量不足）", "weak secondary motion energy"))

        # 4. Appearance-invariant framing layout similarity
        lay_sc = _layout_similarity(sf, rf)
        if lay_sc < 0.52:
            shot_issues.append(T("构图景别或主体朝向偏移（轮廓布局相似度 {s:.2f}）",
                                 "framing or silhouette layout drift (similarity {s:.2f})", s=lay_sc))

        total = round(0.30 * cam_sc + 0.35 * mot_sc + 0.25 * lay_sc + 0.10 * smooth, 3)
        for msg in shot_issues:
            issues.append(f"#{idx}: {msg}")
        rows.append({
            "shot": idx,
            "start": sh["start"],
            "end": sh["end"],
            "score": total,
            "camera_score": round(cam_sc, 3),
            "motion_score": round(mot_sc, 3),
            "layout_score": lay_sc,
            "smoothness": round(smooth, 3),
            "source": {"camera_traj": s_tr, "net_zoom_pct": sk["net_zoom_pct"], "motion_regime": sk["motion_regime"],
                       "pose_change": sp, "frozen_ratio": sk["frozen_ratio"]},
            "remake": {"camera_traj": r_tr, "net_zoom_pct": rk["net_zoom_pct"], "motion_regime": rk["motion_regime"],
                       "pose_change": rp, "frozen_ratio": rk["frozen_ratio"]},
            "issues": shot_issues,
        })

    mean_sc = round(float(np.mean([r["score"] for r in rows])), 3) if rows else None
    res = {
        "ok": True,
        "evaluated_shots": len(rows),
        "score": mean_sc,
        "camera_score": round(float(np.mean([r["camera_score"] for r in rows])), 3) if rows else None,
        "motion_score": round(float(np.mean([r["motion_score"] for r in rows])), 3) if rows else None,
        "layout_score": round(float(np.mean([r["layout_score"] for r in rows])), 3) if rows else None,
        "smoothness": round(float(np.mean([r["smoothness"] for r in rows])), 3) if rows else None,
        "issues": issues,
        "shots": rows,
    }
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(out_dir) / "qa.json", res)
    return res


def qa_markdown(qa: dict) -> list[str]:
    """Markdown section summarizing per-shot remake fidelity for `remake/status.md`."""
    rows = qa.get("shots") or []
    if not rows:
        return []
    L = ["", T("## 还原度质检（运镜 · 姿态 · 构图 · 流畅度）", "## Fidelity QA (camera · pose · layout · smoothness)"), "",
         T("- 综合得分：**{s:.0%}**（运镜 {c:.0%} · 姿态动态 {m:.0%} · 构图轮廓 {l:.0%} · 流畅度 {f:.0%}）",
           "- Overall fidelity: **{s:.0%}** (camera {c:.0%} · pose/motion {m:.0%} · framing layout {l:.0%} · smoothness {f:.0%})",
           s=qa.get("score") or 0, c=qa.get("camera_score") or 0, m=qa.get("motion_score") or 0,
           l=qa.get("layout_score") or 0, f=qa.get("smoothness") or 0), ""]
    L += [T("| 镜头 | 得分 | 原片运镜 / 姿态 | 成片运镜 / 姿态 | 构图相似 | 诊断 |",
            "| Shot | Score | Source cam / pose | Remake cam / pose | Layout | Diagnosis |"),
          "|---|---|---|---|---|---|"]
    for r in rows:
        s_info = f"`{r['source']['camera_traj']}` / `{r['source']['motion_regime']}` ({r['source']['pose_change']:.2f})"
        r_info = f"`{r['remake']['camera_traj']}` / `{r['remake']['motion_regime']}` ({r['remake']['pose_change']:.2f})"
        diag = "；".join(r["issues"]) if r.get("issues") else T("通过", "OK")
        L.append(f"| #{r['shot']} | {r['score']:.0%} | {s_info} | {r_info} | {r['layout_score']:.2f} | {diag} |")
    return L
