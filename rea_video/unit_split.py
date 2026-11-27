"""Split one generated multi-shot clip back into its shots.

A multi-shot call returns one clip for several source shots. The model's cuts rarely land exactly where the plan put
them, and it sometimes adds an extra cut inside a shot. So the clip is cut where its own pixels cut, and the pieces are
grouped, in order, into as many shots as were planned, so a shot the model cut in two keeps both halves. Only a very
short glitch at a shot's edge is left out.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

SHORT_SIDE = 108  # decode size for cut detection


def _struct_vec(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Colour-blind layout signature: blurred gradients + mean-removed luma, unit length."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(cv2.GaussianBlur(gray, (0, 0), 2.0), size, interpolation=cv2.INTER_AREA)
    grad = np.concatenate([np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)).ravel(),
                           np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)).ravel()])
    grad /= float(np.linalg.norm(grad)) + 1e-5
    luma = small.ravel() - float(small.mean())
    luma /= float(np.linalg.norm(luma)) + 1e-5
    v = np.concatenate([grad, luma])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-5 else np.zeros_like(v)


def _read_small(video_file: Path, limit: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_file))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while len(frames) < limit:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        s = SHORT_SIDE / float(min(h, w))
        frames.append(cv2.resize(fr, (max(2, int(round(w * s))), max(2, int(round(h * s)))), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, fps


def find_cuts(frames: list[np.ndarray], fps: float, want: int) -> list[int]:
    """Frame indices where a new shot starts: real hard cuts first, then the strongest jumps until `want` are found."""
    n = len(frames)
    if n < 8:
        return []
    hists = []
    for f in frames:
        h = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(h, h)
        hists.append(h)
    mads = np.array([float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)))) for i in range(1, n)], np.float32)
    bhat = np.array([float(cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)) for i in range(1, n)], np.float32)
    # a cut is a spike against its neighbourhood, which also catches cuts between two similar dark scenes
    med = np.array([float(np.median(mads[max(0, i - 5):i + 6])) for i in range(len(mads))], np.float32)
    spike = mads / (med + 2.5)
    cands = []
    for i in range(3, n - 3):
        d = i - 1
        score = 0.55 * float(bhat[d]) + 0.45 * min(1.0, float(mads[d]) / 60.0)
        real = (mads[d] >= 28.0 and spike[d] >= 2.6 and bhat[d] >= 0.16) or bhat[d] >= 0.36
        cands.append((score, real, i))
    min_gap = max(6, int(round(0.35 * (fps or 24.0))))
    cuts: list[int] = []
    for score, real, i in sorted(cands, reverse=True):
        if score < 0.22 or (not real and len(cuts) >= want):
            break
        if all(abs(i - c) >= min_gap for c in cuts):
            cuts.append(i)
            if len(cuts) >= want + 4:
                break
    return sorted(cuts)


def group_pieces(pieces: list[tuple[int, int]], n_shots: int, ref_mid: list[float], ref_dur: list[float],
                 ref_vecs: list[np.ndarray | None], piece_vecs: list[np.ndarray]) -> list[list[tuple[int, int]]] | None:
    """Group consecutive pieces into `n_shots` runs, best match on time position, length and layout. Every piece goes
    to exactly one shot, so nothing in the clip is lost."""
    k = len(pieces)
    if k < n_shots:
        return None
    total = float(pieces[-1][1] - pieces[0][0]) or 1.0
    t0 = pieces[0][0]

    def score(i: int, s: int, e: int) -> float:
        a, z = pieces[s][0], pieces[e][1]
        mid, dur = (0.5 * (a + z) - t0) / total, (z - a) / total
        longest = max(range(s, e + 1), key=lambda j: pieces[j][1] - pieces[j][0])
        sim = float(np.dot(ref_vecs[i], piece_vecs[longest])) if ref_vecs[i] is not None else 0.5
        return sim - 0.65 * abs(mid - ref_mid[i]) + 0.35 * min(dur, ref_dur[i]) / max(dur, ref_dur[i], 1e-6)

    neg = -1e18
    best = [[neg] * k for _ in range(n_shots)]  # best[i][e]: shots 0..i placed, shot i ends at piece e
    back = [[-1] * k for _ in range(n_shots)]
    for e in range(k - n_shots + 1):
        best[0][e] = score(0, 0, e)
    for i in range(1, n_shots):
        for e in range(i, k - (n_shots - 1 - i)):
            for s in range(i, e + 1):
                prev = best[i - 1][s - 1]
                if prev <= neg:
                    continue
                v = prev + score(i, s, e)
                if v > best[i][e]:
                    best[i][e], back[i][e] = v, s
    if best[n_shots - 1][k - 1] <= neg:
        return None
    groups, e = [], k - 1
    for i in range(n_shots - 1, -1, -1):
        s = back[i][e] if i > 0 else 0
        groups.append(pieces[s:e + 1])
        e = s - 1
    return groups[::-1]


def split_unit(video_file: Path, shot_ids: list[int], trim_to: float, out_dir: Path,
               ref_shots: list[dict] | None = None, bdir: Path | None = None) -> dict[str, str]:
    """{shot id: clip} cut from one multi-shot clip, or {} when its cuts can't be matched to the shots."""
    n = len(shot_ids)
    if n < 2 or not video_file.is_file():
        return {}
    cap = cv2.VideoCapture(str(video_file))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0 or total < n * 4:
        return {}
    limit = min(total, max(n * 4, int(round(trim_to * fps)))) if trim_to > 0 else total
    frames, _ = _read_small(video_file, limit)
    if len(frames) < n * 4:
        return {}
    cuts = find_cuts(frames, fps, n - 1)
    if len(cuts) < n - 1:
        return {}
    bounds = [0] + cuts + [len(frames)]
    pieces = list(zip(bounds[:-1], bounds[1:]))

    h, w = frames[0].shape[:2]
    vec_size = (24, max(8, int(round(24 * h / float(w)))))
    piece_vecs = [_struct_vec(frames[(a + z) // 2], vec_size) for a, z in pieces]
    ref_mid = [(i + 0.5) / n for i in range(n)]
    ref_dur = [1.0 / n] * n
    ref_vecs: list[np.ndarray | None] = [None] * n
    if ref_shots and len(ref_shots) == n:
        u0 = float(ref_shots[0]["start"])
        span = max(0.1, float(ref_shots[-1]["end"]) - u0)
        ref_mid = [(0.5 * (float(s["start"]) + float(s["end"])) - u0) / span for s in ref_shots]
        ref_dur = [max(0.02, (float(s["end"]) - float(s["start"])) / span) for s in ref_shots]
        for i, s in enumerate(ref_shots):
            ff = (s.get("assets") or {}).get("first_frame")
            p = Path(bdir) / ff if (bdir and ff) else None
            im = cv2.imread(str(p)) if p and p.is_file() else None
            if im is not None:
                ref_vecs[i] = _struct_vec(im, vec_size)
    groups = [[pc] for pc in pieces] if len(pieces) == n else group_pieces(pieces, n, ref_mid, ref_dur, ref_vecs, piece_vecs)
    if not groups:
        return {}
    # a very short piece at the edge of a shot is a glitch the model put in (a flash, a stray frame run): leave it out
    glitch = max(3, int(round(0.4 * fps)))
    spans = []
    for g in groups:
        g = list(g)
        while len(g) > 1 and g[0][1] - g[0][0] < glitch and g[0][1] - g[0][0] < 0.35 * (g[-1][1] - g[0][0]):
            g.pop(0)
        while len(g) > 1 and g[-1][1] - g[-1][0] < glitch and g[-1][1] - g[-1][0] < 0.35 * (g[-1][1] - g[0][0]):
            g.pop()
        spans.append((g[0][0], g[-1][1]))

    out_dir.mkdir(parents=True, exist_ok=True)
    clips: dict[str, str] = {}
    for sid, (a, z) in zip(shot_ids, spans):
        dst = out_dir / f"shot_{int(sid):03d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_file),
                        "-vf", f"trim=start_frame={a}:end_frame={z},setpts=PTS-STARTPTS",
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst)], check=True)
        clips[str(sid)] = str(dst)
    return clips
