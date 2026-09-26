# 3-minute video script: Team WindRocker

**Order required by the brief:** business problem → analytical approach → most important insights → recommendations.
**Audience:** VoltRelay operations & CX leadership. **Target:** ~410 spoken words ≈ 2:55 at a calm 140 wpm.
**Setup:** screen-record the dashboard (`dashboard/VoltRelay_Dashboard_WindRocker.html`, light theme looks best on video) with your face in a corner (OBS / Loom / PowerPoint recording). Rehearse twice with a timer.

| Time | On screen | Say (word for word, or close) |
|---|---|---|
| **0:00 – 0:25** · Problem | Dashboard header + KPI row | "VoltRelay grew fast: completed swaps nearly tripled in eighteen months and monthly revenue went from six to almost twenty million rupees. But failed swaps spiked every summer, new riders stopped coming back, and each swap now loses more money than it did after last year's price rise. Leadership has four budget proposals on the table. Our job was to find out what is actually driving these outcomes before any money is spent." |
| **0:25 – 0:55** · Approach | Data-quality tab, then Performance tab | "We analysed all 3.9 million swap attempts, hourly station telemetry, battery records, rider profiles and support tickets in DuckDB and Python. First we cleaned: we removed two test stations that had billed almost four million rupees, fixed a firmware bug that shifted 139,000 timestamps by five and a half hours, and re-read ticket comments in English and Hinglish. Then we followed one chain: equipment, to service, to retention, to unit economics." |
| **0:55 – 1:30** · Insight 1: heat × Gen1 | Failures tab heatmap → Stations tab heat charts | "Insight one: failures are not spread evenly. They concentrate in Jaipur, Delhi and Hyderabad in summer. The telemetry shows why: above 45 degrees, a first-generation cabinet needs 156 minutes to charge a pack instead of 88, and runs out of charged batteries in 37 percent of hours. Just 36 Gen1 stations in those three cities cause 57.5 percent of all summer failures." |
| **1:30 – 2:00** · Insight 2: Kyron | Batteries tab SoH chart | "Insight two: three bad battery lots from one supplier. They lost health 2.4 times faster, giving riders 49 kilometres per swap instead of 61 and costing 82 rupees of wear per swap instead of 35. Those lots explain three quarters of this year's margin loss. Riders noticed: nine thousand tickets filed as 'other' were really battery complaints." |
| **2:00 – 2:25** · Insight 3: churn + pricing | Retention tab → Pricing tab | "Insight three: new riders who hit two or more failed swaps in their first two weeks churn at 15 percent instead of 9. Price plan, channel and partner don't matter. And the peak-pricing pilot? Plus eight rupees a swap, but almost no demand shifted. It's a revenue lever, not a congestion fix. Meanwhile the biggest partner, on a 28 percent discount, is the least profitable." |
| **2:25 – 2:58** · Recommendations | Recommendations tab | "So our recommendations: one, replace the 36 Gen1 cabinets in the three hot cities before next summer, which avoids about 22,600 failures a year. Two, pull the bad-lot packs that are still in service, replace them, and buy on lot testing, not supplier name. Three, protect every new rider's first fortnight with healthy packs and an automatic credit after any failure. Four, renegotiate ZipDrop instead of signing an exclusive. Fix the equipment and the batteries, and growth becomes profitable. Thank you." |

## Delivery tips
- Open with energy on "VoltRelay grew fast". The first five seconds decide whether judges lean in.
- **Point at the chart while you say the number** (hover tooltips in the dashboard make this easy).
- Pause for half a second after each insight number (156 minutes, 2.4×, 15 vs 9).
- Keep it under 3:00: if you run long, cut the ZipDrop sentence in 2:00–2:25 (it's in the report anyway).
- Export at 1080p. Name the file `VoltRelay_WindRocker_3min.mp4`.

## Slide-free fallback (if screen-recording is awkward)
Use the 8-slide carousel PDF as your slides. The slide order matches the script: 1 → hook/problem, 2 → paradox, 3 → heat, 4 → Kyron, 5 → churn, 6 → pricing/partner, 7 → budget verdict/recommendations, 8 → closing.
