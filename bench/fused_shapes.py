import mlx_lm  # first
import json, os
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2 import fused_ref as FR
D = os.path.expanduser("~/loopling-opt/runs/visits_p3")
idx = json.load(open(D + "/index.json"))
model, _ = load(os.path.expanduser("~/loopling/models/") + idx["model"])
layers = list(model.model.language_model.layers)
seen = set()
for v in idx["visits"]:
    kind = ("F" if v["full"] else "S") + ("b" if v["boundary"] else "") + ("m" if v["has_msg"] else "") + ("B" if v["has_bs"] else "")
    if kind in seen: continue
    seen.add(kind)
    t = mx.load(D + "/" + v["file"])
    T = FR.visit_ref(layers[v["layer"]], t["h"], t.get("bs"), t.get("msg"), [t["cs%d" % i] for i in range(4)], t["attn_out"], t.get("tau"))
    print("== kind %s layer %d" % (kind, v["layer"]), "bs", None if "bs" not in t else t["bs"].shape, "tau", None if "tau" not in t else (t["tau"].shape, t["tau"].dtype))
    print("  " + "  ".join("%s%s:%s" % (k, tuple(x.shape), str(x.dtype).split(".")[-1]) for k, x in T.items() if x is not None))
