"""Tests for the optional Perceptron detection pass, offline: the HTTP calls are replaced by fakes.

Covers photo/view alignment (against MapAnything's own crop when it is installed), people blurring,
box painting (order, kept wall/floor/ceiling, depth gate), name mapping (model + fallback), response
parsing, and the generic objects that reach plan.json.
Run: python test_detect.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import perception as P
from palette import Palette, builtin_surface_entries
from roomplan import ADE_TO_CATEGORY, TEMPLATE_INFO, TEMPLATES, build_plan

rng = np.random.default_rng(0)
ID2LABEL = {0: "wall", 1: "floor", 2: "ceiling", 3: "table", 4: "person", 5: "chair"}


def photo(w, h):
    """Smooth random photo (so resampling differences stay small, like a real one)."""
    small = rng.uniform(0, 255, (h // 16 + 1, w // 16 + 1, 3)).astype(np.uint8)
    return Image.fromarray(small).resize((w, h), Image.BICUBIC)


def test_alignment():
    tmp = Path(tempfile.mkdtemp())
    for (w, h), view_size in [((3024, 4032), (392, 518)), ((2524, 1378), (392, 518)), ((1600, 1200), (518, 392))]:
        img = photo(w, h)
        img.save(tmp / "p.png")
        try:
            from mapanything.utils.cropping import crop_resize_if_necessary
            view = np.asarray(crop_resize_if_necessary(img, resolution=view_size)[0])
            src = "mapanything"
        except ImportError:
            l, t, r, b = P.view_crop(img.size, view_size)
            view = np.asarray(img.crop((round(l), round(t), round(r), round(b))).resize(view_size, Image.LANCZOS))
            src = "reference"
        v = {"rgb": view}
        up, k = P.upload_image(tmp / "p.png", v, None, log=lambda m: (_ for _ in ()).throw(AssertionError(m)))
        assert abs(up.width * k - view_size[0]) < 1e-6 and abs(up.height * k - view_size[1]) < 1.5, (up.size, k)
        assert max(up.size) <= 1536
        print(f"alignment OK ({src}) {w}x{h} -> {view_size}, upload {up.size}")
    # A view that is not this photo falls back to the view pixels.
    logs = []
    up, k = P.upload_image(tmp / "p.png", {"rgb": np.asarray(photo(518, 392))}, None, log=logs.append)
    assert logs and up.size == (1036, 784) and k == 0.5
    print("misaligned view falls back OK")


def test_blur():
    tmp = Path(tempfile.mkdtemp())
    img = photo(400, 300)
    img.save(tmp / "p.png")
    view = np.asarray(img.resize((200, 150), Image.LANCZOS))
    mask = np.zeros((150, 200), bool)
    mask[50:100, 120:170] = True
    up, _ = P.upload_image(tmp / "p.png", {"rgb": view}, mask)
    a, b = np.asarray(up, float), np.asarray(img, float)
    assert np.abs(a[110:190, 250:330] - b[110:190, 250:330]).mean() > 5, "person area must change"
    assert np.abs(a[:60, :150] - b[:60, :150]).max() == 0, "far from people nothing changes"
    print("blur OK")


def test_paint():
    lm = np.full((100, 100), 3)          # table everywhere
    lm[:20] = 0                          # wall on top rows
    big, small = (10, (0, 0, 80, 80)), (11, (30, 30, 50, 50))
    id2 = {**ID2LABEL, 10: "det:sofa", 11: "det:lamp"}
    out = P.paint_detections(lm, [small, big], id2)
    assert (out[:20] == 0).all(), "wall stays"
    assert (out[30:50, 30:50] == 11).all(), "smaller box on top, whatever the input order"
    assert (out[20:30, 0:80] == 10).all() and (out[80:, :] == 3).all() and (out[20:, 80:] == 3).all()
    assert (lm[30:50, 30:50] == 3).all(), "input untouched"
    # Depth gate: the box's centre is 2 m away, the left half of the box is a shelf 6 m away.
    depth = np.full((100, 100), 2.0)
    depth[:, :20] = 6.0
    out = P.paint_detections(lm, [big], id2, depth=depth)
    assert (out[20:80, 20:80] == 10).all() and (out[20:80, :20] == 3).all()

    # Review regression: a chair frame (thin, 2 m) seen against a wall (3.5 m) that fills most of the box.
    lm = np.zeros((100, 100), int)
    depth = np.full((100, 100), 3.5)
    for sl in [np.s_[20:80, 30:34], np.s_[20:80, 66:70], np.s_[50:54, 30:70]]:   # legs and seat
        lm[sl], depth[sl] = 5, 2.0
    lm[85:] = 1
    out = P.paint_detections(lm, [(10, (25, 15, 75, 90))], id2, depth=depth)
    assert (out[lm == 5] == 10).all() and (out[lm == 0] == 0).all() and (out[lm == 1] == 1).all()

    # Review regression: masked points are (0,0,0) in MapAnything; they must not set the object depth.
    pts = np.zeros((100, 100, 3))
    pts[..., 2] = 4.0
    pts[30:70, 30:70, 2] = 2.0
    mask = np.ones((100, 100), bool)
    mask[30:70, 30:52] = False             # most of the core has no reconstruction
    pts[~mask] = 0.0
    v = {"points": pts, "mask": mask, "cam2world": np.eye(4)}
    d = P.camera_depth(v)
    assert np.isnan(d[~mask]).all() and np.allclose(d[mask & (pts[..., 2] == 2.0)], 2.0)
    lm = np.full((100, 100), 3)
    out = P.paint_detections(lm, [(10, (30, 30, 70, 70))], id2, depth=d)
    assert (out[30:70, 52:70] == 10).all() and (out[:30] == 3).all()

    # Review regression: a pixel the small box rejects stays with the big box below, not SegFormer.
    depth = np.full((100, 100), 2.0)       # sofa at 2 m fills its box
    depth[40:60, 40:50] = 1.0              # the lamp (1 m) only fills the left half of its box
    lm = np.full((100, 100), 3)
    out = P.paint_detections(lm, [(10, (10, 10, 90, 90)), (11, (40, 40, 60, 60))], id2, depth=depth)
    assert (out[40:60, 40:50] == 11).all() and (out[40:60, 50:60] == 10).all()

    # Wall-mounted things may take wall pixels and rugs floor pixels (SegFormer often calls them wall/floor).
    lm = np.zeros((100, 100), int)
    lm[70:] = 1
    out = P.paint_detections(lm, [(10, (10, 10, 40, 40)), (11, (50, 10, 80, 40), {"wall"}), (12, (0, 75, 60, 100), {"floor"})], id2)
    assert (out[10:40, 10:40] == 0).all() and (out[10:40, 50:80] == 11).all() and (out[75:, :60] == 12).all()
    assert (out[75:, 60:] == 1).all()
    print("paint OK")


def test_multiview_fusion():
    # A close image contributes many pixels, but should have only one 3D label vote.
    views, maps = [], []
    for count, label in ((100, 5), (1, 3), (1, 3)):
        views.append({"points": np.zeros((1, count, 3), np.float32),
                      "rgb": np.zeros((1, count, 3), np.uint8),
                      "mask": np.ones((1, count), bool), "conf": np.ones((1, count))})
        maps.append(np.full((1, count), label, np.int32))
    _, _, cats = P.fuse(views, maps, ID2LABEL, {"chair": "chair", "table": "table"})
    assert cats.tolist() == ["table"], cats
    print("multi-view fusion OK")


def test_object_matching():
    def obs(view, center):
        points = rng.normal(center, 0.04, (120, 3)).astype(np.float32)
        return {"category": "chair", "view": view, "points": points,
                "colors": np.full((120, 3), 80, np.uint8)}
    groups = P.match_object_observations([obs(0, (1, 0.5, 1)), obs(1, (1.03, 0.5, 1.02)),
                                          obs(0, (1.8, 0.5, 1))])
    assert len(groups) == 1 and groups[0]["views"] == [0, 1]
    assert np.linalg.norm(groups[0]["center"] - [1, 0.5, 1]) < 0.08
    print("object matching OK")


def test_mapping(monkeypatch_post):
    assert P.synonym_template("Office Chairs", TEMPLATES, ADE_TO_CATEGORY) == "chair"
    assert P.synonym_template("person", TEMPLATES, ADE_TO_CATEGORY) == "ignore"
    assert P.synonym_template("bookcase", TEMPLATES, ADE_TO_CATEGORY) == "bookshelf"
    assert P.synonym_template("coffee table", TEMPLATES, ADE_TO_CATEGORY) == "coffee_table"
    assert P.synonym_template("backpack", TEMPLATES, ADE_TO_CATEGORY) == "generic"
    assert P.synonym_template("man sitting", TEMPLATES, ADE_TO_CATEGORY) == "ignore"
    assert P.synonym_template("students at desk", TEMPLATES, ADE_TO_CATEGORY) == "ignore"
    for name, expected in {"podium": "podium", "cork board": "bulletin_board", "computer monitor": "monitor",
                           "ceiling fan": "ceiling_fan", "stack of chairs": "chair_stack", "shower": "shower",
                           "range hood": "range_hood", "coat hooks": "wall_hooks", "nightstand": "nightstand",
                           "dresser": "dresser", "projection screen": "projection_screen"}.items():
        assert P.synonym_template(name, TEMPLATES, ADE_TO_CATEGORY) == expected, name

    # The model answers for some names, with one invalid choice; the rest fall back to synonyms.
    def post(path, body, key):
        assert path == "/chat/completions" and body["response_format"]["json_schema"]["strict"]
        enum = body["response_format"]["json_schema"]["schema"]["properties"]["mappings"]["items"]["properties"]["template"]["enum"]
        assert "unknown" in enum and "ignore" in enum and "chair" in enum
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"mappings": [
            {"name": "reading lamp", "template": "lamp"}, {"name": "globe", "template": "unknown"},
            {"name": "student", "template": "ignore"}, {"name": "couch", "template": "spaceship"}]})}}]}
    monkeypatch_post(post)
    m = P.map_names(["reading lamp", "globe", "student", "couch", "armchair"], TEMPLATES, ADE_TO_CATEGORY, "k", print)
    assert m == {"reading lamp": ("lamp", "model"), "globe": ("generic", "model"), "student": ("ignore", "model"),
                 "couch": ("sofa", "synonyms"), "armchair": ("chair", "synonyms")}, m

    def broken(path, body, key):
        raise RuntimeError("HTTP 500")
    monkeypatch_post(broken)
    m = P.map_names(["globe", "man"], TEMPLATES, ADE_TO_CATEGORY, "k", print)
    assert m == {"globe": ("generic", "synonyms"), "man": ("ignore", "synonyms")}
    print("mapping OK")


def test_retries():
    """Timeouts, 429 and 5xx are retried; other 4xx are not."""
    import time
    import urllib.error
    import urllib.request
    P._post = REAL_POST
    calls, real_open = [], urllib.request.urlopen
    sleep = time.sleep
    time.sleep = lambda s: None

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": 1}'
    script = []

    def fake_open(req, timeout):
        calls.append(req.full_url)
        e = script.pop(0)
        if e == "ok":
            return Resp()
        if e == "timeout":
            raise TimeoutError("The read operation timed out")
        raise urllib.error.HTTPError(req.full_url, e, "x", {}, __import__("io").BytesIO(b"{}"))
    urllib.request.urlopen = fake_open
    try:
        script[:] = ["timeout", 503, "ok"]
        assert P._post("/detect", {}, "k") == {"ok": 1} and len(calls) == 3
        calls.clear(); script[:] = [400]
        try:
            P._post("/detect", {}, "k"); raise AssertionError("400 must raise")
        except RuntimeError:
            assert len(calls) == 1
        calls.clear(); script[:] = ["timeout"] * 3
        try:
            P._post("/detect", {}, "k"); raise AssertionError("3 timeouts must raise")
        except RuntimeError as ex:
            assert len(calls) == 3 and "timed out" in str(ex)
    finally:
        urllib.request.urlopen, time.sleep = real_open, sleep
    print("retries OK")


def test_end_to_end(monkeypatch_post):
    """Two fake views of one scene through perceptron_pass, fuse and build_plan."""
    import os
    H, W = 60, 80
    views, maps = [], []
    for _ in range(2):
        z = np.full((H, W), 3.0)
        z[:12] = 3.0
        ys, xs = np.mgrid[0:H, 0:W]
        pts = np.stack([(xs - W / 2) / W * 4, 2.5 - ys / H * 2.5, z], -1)   # a flat scene 3 m ahead
        views.append({"rgb": np.full((H, W, 3), 128, np.uint8), "points": pts, "conf": np.ones((H, W)),
                      "mask": np.ones((H, W), bool), "cam2world": np.eye(4)})
        lm = np.full((H, W), 3)   # table in the middle, wall on top, floor below
        lm[:12] = 0
        lm[45:] = 1
        maps.append(lm)

    def post(path, body, key):
        if path == "/detect":
            assert body["media"]["image_url"].startswith("data:image/jpeg;base64,")
            assert "categories" not in body and body["config"]["output"] == "box"
            return {"finish_reason": "complete", "detections": [
                {"mention": "Globe, blue, on a stand", "point_box": {"top_left": {"x": 40, "y": 60}, "bottom_right": {"x": 20, "y": 100}}},
                {"mention": "student", "point_box": {"top_left": {"x": 0, "y": 0}, "bottom_right": {"x": 50, "y": 50}}}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"mappings": [
            {"name": "globe", "template": "unknown"}, {"name": "student", "template": "ignore"}]})}}]}
    monkeypatch_post(post)
    os.environ["PERCEPTRON_API_KEY"] = "test"
    ade_before = dict(ADE_TO_CATEGORY)
    out, id2, a2c, rep = P.perceptron_pass([], views, maps, ID2LABEL, ADE_TO_CATEGORY, TEMPLATES, print,
                                           find_people=lambda img: [])
    assert ADE_TO_CATEGORY == ade_before and 6 not in ID2LABEL, "inputs must not be modified"
    gid = next(k for k, v in id2.items() if v == "det:globe")
    assert a2c["det:globe"] == "generic:globe" and a2c["det:student"] == "other"
    # Fallback path sends 2x view pixels, so boxes come back halved: x 10..20, y 30..50; floor rows keep "floor".
    assert (out[0][30:45, 10:20] == gid).all() and (out[0][45:50, 10:20] == 1).all()
    assert (out[0][:30] != gid).all(), "ignored detections paint nothing"
    assert rep["names"]["globe"] == {"template": "generic", "source": "model", "count": 2}
    assert rep["photos"][0]["detections"][0]["box"] == [10.0, 30.0, 20.0, 50.0]
    assert rep["photos"][0]["detections"][0]["mention"] == "Globe, blue, on a stand"
    pts, rgb, cats = P.fuse(views, out, id2, a2c)
    assert "generic:globe" in set(cats)

    # Same scene with real photo files (4x the view size): the boxes come back in upload pixels and must
    # land on the same view pixels, and the uploaded image must be blurred where SegFormer saw a person.
    tmp = Path(tempfile.mkdtemp())
    big = photo(W * 4, H * 4)
    big.save(tmp / "a.png")
    views[0]["rgb"] = np.asarray(big.resize((W, H), Image.LANCZOS))
    maps[0][5:20, 60:75] = 4                # a person
    sent = []

    def post2(path, body, key):
        if path == "/detect":
            import base64, io
            sent.append(Image.open(io.BytesIO(base64.b64decode(body["media"]["image_url"].split(",", 1)[1]))).convert("RGB"))
            k = sent[-1].width / W
            return {"finish_reason": "complete", "detections": [
                {"mention": "globe", "point_box": {"top_left": {"x": 10 * k, "y": 30 * k}, "bottom_right": {"x": 20 * k, "y": 50 * k}}}]}
        return post(path, body, key)
    monkeypatch_post(post2)
    found = []

    def find_people(img):  # stands in for the local detector: one person SegFormer missed, bottom left
        found.append(img.size)
        return [(10, 150, 60, 230)]
    out, id2, a2c, rep = P.perceptron_pass([tmp / "a.png"], views[:1], maps[:1], ID2LABEL, ADE_TO_CATEGORY, TEMPLATES,
                                           print, find_people=find_people)
    assert found == [sent[0].size], "the local detector sees the full-resolution upload"
    assert sent[0].size == (W * 4, H * 4), sent[0].size
    assert rep["photos"][0]["detections"][0]["box"] == [10.0, 30.0, 20.0, 50.0]
    a, b = np.asarray(sent[0], float), np.asarray(big, float)
    blurred, untouched = np.abs(a[20:80, 240:300] - b[20:80, 240:300]).mean(), np.abs(a[100:140, 100:200] - b[100:140, 100:200]).mean()  # clear of both people
    assert blurred > 5 * untouched and untouched < 3, (blurred, untouched)   # untouched only differs by JPEG
    det_blur = np.abs(a[155:225, 15:55] - b[155:225, 15:55]).mean()
    elsewhere = np.abs(a[150:, 120:] - b[150:, 120:]).mean()
    assert det_blur > 5 * elsewhere and elsewhere < 3, (det_blur, elsewhere)
    print("end-to-end OK")


def test_plan_generic():
    """Synthetic room: a generic thing on the floor is built with its name, one on a table is skipped."""
    pts, cols, cats = [], [], []

    def add(n, lo, hi, rgb, cat):
        pts.append(rng.uniform(lo, hi, (n, 3)))
        cols.append(np.clip(np.array(rgb) + rng.normal(0, 6, (n, 3)), 0, 255))
        cats.extend([cat] * n)
    Wm, Dm, Hm = 4.0, 3.2, 2.6
    add(20000, (0, 0, 0), (Wm, 0, Dm), (165, 125, 80), "floor")
    add(12000, (0, Hm, 0), (Wm, Hm, Dm), (240, 240, 236), "ceiling")
    for lo, hi in [((0, 0, 0), (Wm, Hm, 0)), ((0, 0, 0), (0, Hm, Dm)), ((Wm, 0, 0), (Wm, Hm, Dm)), ((0, 0, Dm), (Wm, Hm, Dm))]:
        add(8000, lo, hi, (150, 175, 205), "wall")
    add(1500, (2.6, 0.7, 1.0), (3.6, 0.75, 1.8), (110, 80, 50), "table")
    add(800, (1.0, 0.0, 1.5), (1.6, 1.4, 2.0), (30, 60, 160), "generic:globe stand")
    add(300, (3.0, 0.76, 1.2), (3.3, 1.0, 1.4), (200, 200, 200), "generic:laptop bag")
    flip = np.diag([1, -1, -1.0])
    plan = build_plan(np.concatenate(pts) @ flip.T, np.concatenate(cols), np.array(cats),
                      Palette(builtin_surface_entries()), log=print)
    gen = [o for o in plan["objects"] if o["type"] == "generic"]
    assert len(gen) == 1 and gen[0]["name"] == "globe stand" and gen[0]["color"].endswith("blue"), gen
    assert gen[0]["h"] == 2 and gen[0]["w"] >= 1, "generic boxes are capped at 2 blocks"
    json.dumps(plan)
    print("plan generic OK", gen[0])


def test_cache_and_prompt(monkeypatch_post):
    """A cached upload is never sent twice; the mapping prompt describes each template."""
    sent = []

    def post(path, body, key):
        sent.append((path, body))
        if path == "/detect":
            return {"finish_reason": "complete", "detections": [
                {"mention": "hanging lamp", "point_box": {"top_left": {"x": 1, "y": 2}, "bottom_right": {"x": 3, "y": 4}}}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
            {"mappings": [{"name": "hanging lamp", "template": "chandelier"}]})}}]}
    monkeypatch_post(post)
    tmp = Path(tempfile.mkdtemp())
    img = photo(64, 48)
    a = P.detect_objects(img, "k", "none", tmp)
    b = P.detect_objects(img, "k", "none", tmp)
    assert a == b and len([1 for p, _ in sent if p == "/detect"]) == 1, "second call must come from the cache"
    P.detect_objects(img, "k", "high", tmp)
    assert len([1 for p, _ in sent if p == "/detect"]) == 2, "another reasoning setting is another answer"
    assert sent[0][1]["config"]["reasoning_effort"] == "none"
    m = P.map_names(["hanging lamp"], TEMPLATE_INFO, ADE_TO_CATEGORY, "k", print)
    prompt = sent[-1][1]["messages"][0]["content"]
    assert m == {"hanging lamp": ("chandelier", "model")}
    assert "- chandelier: a chandelier, pendant" in prompt and "- lamp: a floor or table lamp" in prompt
    assert "window" not in P.MAY_TAKE and "door" not in P.MAY_TAKE and P.MAY_TAKE["painting"] == {"wall"}
    rep = {"photos": [{"detections": [{"template": "window", "box": [2.0, 1.0, 5.0, 4.0]},
                                       {"template": "chair", "box": [0.0, 0.0, 8.0, 8.0]}]}]}
    mk = P.detection_masks(rep, [{"rgb": np.zeros((6, 8, 3))}])[0]
    assert mk.sum() == 9 and mk[1:4, 2:5].all() and P.detection_masks(None, []) is None
    print("cache + prompt OK")


def test_known_height_and_chandelier():
    """A room reconstructed at 0.4x is restored by --known-height; a chandelier hangs at its real height."""
    pts, cols, cats = [], [], []

    def add(n, lo, hi, rgb, cat):
        pts.append(rng.uniform(lo, hi, (n, 3)))
        cols.append(np.clip(np.array(rgb) + rng.normal(0, 6, (n, 3)), 0, 255))
        cats.extend([cat] * n)
    Wm, Dm, Hm = 10.0, 6.0, 8.0                     # the real room
    add(20000, (0, 0, 0), (Wm, 0, Dm), (120, 120, 120), "floor")
    add(15000, (0, Hm, 0), (Wm, Hm, Dm), (90, 70, 50), "ceiling")
    for lo, hi in [((0, 0, 0), (Wm, Hm, 0)), ((0, 0, 0), (0, Hm, Dm)), ((Wm, 0, 0), (Wm, Hm, Dm)), ((0, 0, Dm), (Wm, Hm, Dm))]:
        add(9000, lo, hi, (200, 190, 160), "wall")
    add(600, (4.6, 4.5, 2.6), (5.4, 5.6, 3.4), (230, 170, 60), "chandelier")   # body 4.5-5.6 m up, rod above
    P_true = np.concatenate(pts)
    flip = np.diag([1, -1, -1.0])
    base = dict(colors=np.concatenate(cols), cats=np.array(cats), surface_palette=Palette(builtin_surface_entries()),
                scale=1.0, log=lambda m: None)
    plan = build_plan((0.4 * P_true) @ flip.T, known_height=Hm, **base)
    assert abs(plan["metres"]["y"] - Hm) < 0.2 and abs(max(plan["metres"]["x"], plan["metres"]["z"]) - Wm) < 0.6, plan["metres"]
    wrong = build_plan((0.4 * P_true) @ flip.T, **base)
    assert abs(wrong["metres"]["y"] - 0.4 * Hm) < 0.2, "without it the 0.4x scale stays"
    ch = [o for o in plan["objects"] if o["type"] == "chandelier"]
    assert len(ch) == 1, plan["objects"]
    c = ch[0]
    assert 5 <= c["y"] <= 6 and 1 <= c["h"] <= 2 and c["y"] + c["h"] - 1 < plan["size"]["y"], c   # rows 1..H, 1 m each
    assert 1 <= c["w"] <= 3 and 1 <= c["d"] <= 3
    print("known height + chandelier OK", plan["metres"], c)


def test_vanilla_first(monkeypatch_post):
    """A name that is a vanilla block is that block; only the rest goes to the model and our templates."""
    import vanilla
    asked = []

    def post(path, body, key):
        asked.append(body["messages"][0]["content"].split("Names:\n", 1)[1].splitlines())
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
            {"mappings": [{"name": "chair", "template": "chair"}]})}}]}
    monkeypatch_post(post)
    cat = dict(vanilla.BUILTIN)
    m = P.map_names(["lanterns", "chair", "barrel", "glass", "person"], TEMPLATE_INFO, ADE_TO_CATEGORY, "k", print,
                    vanilla_cat=cat)
    assert m["lanterns"] == ("vanilla:lantern:model", "vanilla") and m["barrel"] == ("vanilla:barrel:cube", "vanilla")
    assert m["chair"] == ("chair", "model") and asked == [["chair", "glass", "person"]], (m, asked)
    assert vanilla.match("stone wall", cat) is None and vanilla.match("flowerpot", cat) == ("flower_pot", False)

    # In the plan: a hanging lantern keeps its height, a barrel on the floor fills its footprint.
    pts, cols, cats = [], [], []

    def add(n, lo, hi, rgb, cat_):
        pts.append(rng.uniform(lo, hi, (n, 3)))
        cols.append(np.clip(np.array(rgb) + rng.normal(0, 6, (n, 3)), 0, 255))
        cats.extend([cat_] * n)
    add(20000, (0, 0, 0), (5, 0, 4), (120, 120, 120), "floor")
    add(12000, (0, 3, 0), (5, 3, 4), (200, 200, 200), "ceiling")
    for lo, hi in [((0, 0, 0), (5, 3, 0)), ((0, 0, 0), (0, 3, 4)), ((5, 0, 0), (5, 3, 4)), ((0, 0, 4), (5, 3, 4))]:
        add(6000, lo, hi, (180, 180, 170), "wall")
    add(80, (2.4, 2.3, 1.9), (2.6, 2.6, 2.1), (240, 190, 90), "vanilla:lantern:model:lantern")
    add(400, (0.6, 0.0, 0.6), (1.2, 0.9, 1.2), (130, 90, 50), "vanilla:barrel:cube:barrel")
    flip = np.diag([1, -1, -1.0])
    plan = build_plan(np.concatenate(pts) @ flip.T, np.concatenate(cols), np.array(cats),
                      Palette(builtin_surface_entries()), scale=2.0, log=print)
    van = {o["block"]: o for o in plan["objects"] if o["type"] == "vanilla"}
    assert set(van) == {"minecraft:lantern", "minecraft:barrel"}, plan["objects"]
    ln, br = van["minecraft:lantern"], van["minecraft:barrel"]
    assert ln["hanging"] and not ln["cube"] and ln["y"] >= 5 and ln["name"] == "lantern", ln   # 2.6 m up at 2 blocks/m
    assert br["cube"] and br["y"] == 1 and br["h"] >= 1, br
    print("vanilla first OK", ln, br)


def test_new_templates():
    """Whiteboard, exit sign and fire extinguisher go on their wall at believable heights; the projector hangs from
    the ceiling facing the whiteboard; the fan stands on the floor. The fallback dictionary knows their names."""
    for name, t in [("whiteboard", "whiteboard"), ("chalkboard", "whiteboard"), ("projector", "projector"),
                    ("exit sign", "exit_sign"), ("fire extinguisher", "fire_extinguisher"), ("tower fan", "fan"),
                    ("fan", "fan")]:
        assert P.synonym_template(name, TEMPLATES, ADE_TO_CATEGORY) == t, (name, P.synonym_template(name, TEMPLATES, ADE_TO_CATEGORY))
    pts, cols, cats = [], [], []

    def add(n, lo, hi, rgb, cat):
        pts.append(rng.uniform(lo, hi, (n, 3)))
        cols.append(np.clip(np.array(rgb) + rng.normal(0, 4, (n, 3)), 0, 255))
        cats.extend([cat] * n)
    Wm, Dm, Hm = 8.0, 6.0, 3.0
    add(20000, (0, 0, 0), (Wm, 0, Dm), (90, 90, 90), "floor")
    add(15000, (0, Hm, 0), (Wm, Hm, Dm), (230, 230, 230), "ceiling")
    for lo, hi in [((0, 0, 0), (Wm, Hm, 0)), ((0, 0, 0), (0, Hm, Dm)), ((Wm, 0, 0), (Wm, Hm, Dm)), ((0, 0, Dm), (Wm, Hm, Dm))]:
        add(8000, lo, hi, (225, 222, 215), "wall")
    add(900, (2.0, 0.9, 0.0), (4.4, 2.1, 0.05), (245, 245, 245), "whiteboard")        # on the z=0 wall
    add(60, (6.5, 2.3, 0.0), (6.8, 2.5, 0.05), (60, 200, 80), "exit_sign")           # high, same wall
    add(80, (0.0, 0.3, 4.0), (0.05, 0.9, 4.2), (200, 30, 30), "fire_extinguisher")    # low, on the x=0 wall
    add(150, (3.0, 2.55, 3.0), (3.4, 2.75, 3.4), (240, 240, 240), "projector")        # under the ceiling
    add(300, (6.0, 0.0, 4.0), (6.3, 0.9, 4.3), (30, 30, 30), "fan")                   # on the floor
    flip = np.diag([1, -1, -1.0])
    plan = build_plan(np.concatenate(pts) @ flip.T, np.concatenate(cols), np.array(cats),
                      Palette(builtin_surface_entries()), scale=1.25, log=lambda m: None)
    ops = {o["type"]: o for o in plan["openings"]}
    objs = {o["type"]: o for o in plan["objects"]}
    H = plan["size"]["y"]
    wb, ex, fe = ops["whiteboard"], ops["exit_sign"], ops["fire_extinguisher"]
    assert wb["w"] >= 2 and wb["h"] >= 1 and wb["color"] == "white" and wb["side"] == ex["side"] != fe["side"], (wb, ex, fe)
    assert ex["y"] > wb["y"] and ex["y"] >= H - 2 and fe["y"] <= 1, (ex, fe, H)
    pj, fan = objs["projector"], objs["fan"]
    assert pj["y"] >= H - 1 and pj["facing"] == wb["side"], (pj, wb)
    assert fan["w"] == fan["d"] == 1 and 1 <= fan["h"] <= 2, fan
    # Without the detector: SegFormer calls a whiteboard a "screen" (tv). A bright, colourless one is a whiteboard;
    # a dark one stays a TV.
    for rgb, want in [((235, 236, 232), "whiteboard"), ((25, 25, 30), "tv")]:
        pts2, cols2, cats2 = [p.copy() for p in pts], [c.copy() for c in cols], list(cats)
        keep = np.array([c != "whiteboard" for c in cats2])
        P2 = np.concatenate(pts2)[keep]; C2 = np.concatenate(cols2)[keep]; K2 = list(np.array(cats2)[keep])
        scr = rng.uniform((2.0, 0.9, 0.0), (4.4, 2.1, 0.05), (900, 3))
        P2 = np.concatenate([P2, scr]); C2 = np.concatenate([C2, np.tile(rgb, (900, 1))]); K2 += ["tv"] * 900
        p2 = build_plan(P2 @ flip.T, C2, np.array(K2), Palette(builtin_surface_entries()), scale=1.25, log=lambda m: None)
        got = [o["type"] for o in p2["openings"] if o["type"] in ("whiteboard", "tv")]
        assert got == [want], (rgb, got)
    # Trash cans keep their colour (black trash, blue recycling).
    pts3, cols3, cats3 = list(pts), list(cols), list(cats)
    for (lo, hi), rgb in [(((5.0, 0.0, 5.5), (5.3, 0.7, 5.8)), (25, 25, 28)), (((5.6, 0.0, 5.5), (5.9, 0.7, 5.8)), (40, 70, 170))]:
        pts3.append(rng.uniform(lo, hi, (250, 3))); cols3.append(np.tile(rgb, (250, 1)).astype(float)); cats3 += ["trash"] * 250
    p3 = build_plan(np.concatenate(pts3) @ flip.T, np.concatenate(cols3), np.array(cats3), Palette(builtin_surface_entries()),
                    scale=1.25, log=lambda m: None)
    bins = sorted(o["color"] for o in p3["objects"] if o["type"] == "trash")
    assert bins == ["black", "blue"], bins
    print("new templates OK", {k: (v.get("side"), v.get("y"), v.get("h")) for k, v in ops.items()}, pj, fan)


if __name__ == "__main__":
    real_post = REAL_POST = P._post

    def monkeypatch_post(fn):
        P._post = fn
    try:
        test_alignment()
        test_blur()
        test_paint()
        test_multiview_fusion()
        test_object_matching()
        test_mapping(monkeypatch_post)
        test_retries()
        test_cache_and_prompt(monkeypatch_post)
        test_known_height_and_chandelier()
        test_vanilla_first(monkeypatch_post)
        test_new_templates()
        test_end_to_end(monkeypatch_post)
        test_plan_generic()
    finally:
        P._post = real_post
    print("OK")
