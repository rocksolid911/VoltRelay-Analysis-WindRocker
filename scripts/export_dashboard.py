import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "outputs/dashboard_data.json").read_text())
for s in data["stations"]:
    for k in ("summer_temp", "charge_min", "cm", "tickets_per_1k", "summer_fail"):
        if s.get(k) is None or s[k] != s[k]: s[k] = 0
tpl = (ROOT / "scripts/dashboard_template.html").read_text(encoding="utf-8")
html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=str).replace("</", "<" + "\\/"))
out = ROOT / "dashboard/VoltRelay_Dashboard_WindRocker.html"; out.write_text(html, encoding="utf-8")
print(out, round(out.stat().st_size / 1e3), "KB")
