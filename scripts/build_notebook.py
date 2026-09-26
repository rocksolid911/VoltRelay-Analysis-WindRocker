import sys, json, time, nbformat
from pathlib import Path
from nbclient import NotebookClient
sys.path.insert(0, str(Path(__file__).parent))
import importlib, nb_cells; importlib.reload(nb_cells)
ROOT = Path(__file__).resolve().parent.parent
narr_path = ROOT / "scripts/narrative.json"
narr = json.loads(narr_path.read_text(encoding="utf-8")) if narr_path.exists() else {}
nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}
for kind, src in nb_cells.CELLS:
    for key, txt in narr.items():
        src = src.replace(f"<!--{key}-->", txt)
    nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
out = ROOT / "VoltRelay_Analysis_WindRocker.ipynb"
t0 = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    nbformat.write(nb, out)
    print(f"written {out.name} in {time.time()-t0:.0f}s")
