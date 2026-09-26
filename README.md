# VoltRelay Energy: Service, Retention & Margin Analysis
**Team WindRocker** · Gradient Learnings Data Analytics Hackathon

| Notebook | Open |
|---|---|
| Main analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rocksolid911/VoltRelay-Analysis-WindRocker/blob/main/VoltRelay_Analysis_WindRocker.ipynb) |
| Robustness checks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rocksolid911/VoltRelay-Analysis-WindRocker/blob/main/VoltRelay_Robustness_Checks_WindRocker.ipynb) |

Both notebooks are saved with their outputs, so results can be read without running them. To re-run in Colab, see [Running the notebook in Google Colab](#running-the-notebook-in-google-colab).

**Question:** what is really driving VoltRelay's service failures, new-rider churn and eroding per-swap margins, and where should the next budget go?

**Answer in one line:** three fixable operational problems. Gen1 chargers overheat in Jaipur, Delhi NCR and Hyderabad summers. Three bad battery lots from one supplier (Kyron KY-2407/08/09) degraded 2.4× faster and erased the price-rise margin; the supplier's later lots are fine. New riders who hit 2+ failed swaps in their first fortnight churn 1.7× more.

## Deliverables
| File | What it is |
|---|---|
| `VoltRelay_Analysis_WindRocker.ipynb` | Colab notebook: data understanding → validation scorecard → 9 cleaning steps → EDA → six core questions → deep-dives → findings & recommendations. Runs top to bottom (~30 s on the raw CSVs). |
| `VoltRelay_Robustness_Checks_WindRocker.ipynb` | Companion notebook that stress-tests every headline claim (confounding, exposure bias, double counting, behavioural response) and ends with a claim-by-claim scorecard. Revised numbers go to `outputs/audit_metrics.json`. Runs top to bottom (~20 s). |
| `dashboard/VoltRelay_Dashboard_WindRocker.html` | Self-contained interactive dashboard (open in any browser): filters, 6 question tabs, recommendations, data-quality tab, light/dark. |
| `report/VoltRelay_Report_WindRocker.pdf` | 12-page analysis report (problem, approach, insights, findings, recommendations, robustness appendix). |
| `video/VoltRelay_WindRocker_3min.mp4` (+ `.srt`) | 3-minute narrated video: problem → approach → insights → recommendations. |
| `content/video_script.md` | Timed 3-minute video script with screen cues. |
| `outputs/metrics.json` · `outputs/figures/` | Every number and chart used above (single source of truth). |

## Running the notebook in Google Colab
1. Put this folder in Google Drive as `MyDrive/DataAnalyticsHackathon/`, with the data in `MyDrive/DataAnalyticsHackathon/Hackathon Data Set _ Gradient/` (`.csv` or `.csv.gz` both work).
2. Open the notebook in Colab → **Runtime → Run all** → allow the Drive mount.
3. Dependencies (`duckdb`, `pandas`, `pyarrow`, `statsmodels`, `scipy`, `matplotlib`) are preinstalled on Colab. The notebook installs `duckdb` if it is missing.

Locally: `pip install duckdb pandas pyarrow statsmodels scipy matplotlib jinja2`, then run the notebook from this folder. `scripts/to_parquet.py` optionally builds a Parquet cache for faster reloads.

## Rebuilding the other artefacts
```
python scripts/build_notebook.py      # regenerate + execute the notebook, refresh outputs/
python scripts/build_audit_notebook.py  # regenerate + execute the robustness notebook (run after build_notebook.py)
python scripts/export_dashboard.py    # inject outputs/dashboard_data.json into the dashboard
```
The report PDF is printed from `report/VoltRelay_Report.html` with headless Edge/Chrome (`--print-to-pdf`). The video is rebuilt with `python video/make_video.py` (slides in `video/slides.html`).

**Data is not included in this repository.** Place the organisers' CSV files in `Hackathon Data Set _ Gradient/` before running the notebooks.
