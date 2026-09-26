"""Build the 3-minute narrated video from video/slides.html + content/video_script.md.

Steps: screenshot each slide (headless Edge) -> neural TTS per slide (edge-tts, with word
timings) -> caption chunks drawn with Pillow -> ffmpeg concat -> MP4 + SRT.
Run: .venv/Scripts/python.exe video/make_video.py
"""
import asyncio, json, subprocess, sys
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
B = ROOT / "build"
B.mkdir(exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
VOICE = sys.argv[1] if len(sys.argv) > 1 else "en-IN-PrabhatNeural"
RATE = "+25%"
PAD = 0.35           # silence after each slide (s)
W, H, FPS = 1920, 1080, 30

# One entry per slide, in order. Text is the script, lightly re-spelled for the TTS voice.
NARRATION = [
    "VoltRelay grew fast: completed swaps nearly tripled in eighteen months, and monthly revenue went from six to almost twenty million rupees.",
    "But failed swaps spiked every summer, new riders stopped coming back, and each swap now loses more money than it did after last year's price rise.",
    "Leadership has four budget proposals on the table. Our job was to find out what is actually driving these outcomes, before any money is spent.",
    "We analysed all 3.9 million swap attempts, hourly station telemetry, battery records, rider profiles and support tickets, in DuckDB and Python.",
    "First we cleaned: we removed two test stations that had billed almost four million rupees, fixed a firmware bug that shifted 139,000 timestamps by five and a half hours, and re-read ticket comments in English and Hinglish.",
    "Then we followed one chain: equipment, to service, to retention, to unit economics.",
    "Insight one: failures are not spread evenly. They concentrate in Jaipur, Delhi and Hyderabad, in summer.",
    "The telemetry shows why: above 45 degrees, a first-generation cabinet needs 156 minutes to charge a pack instead of 88, and runs out of charged batteries in 37 percent of hours.",
    "Just 36 Gen 1 stations in those three cities cause 57.5 percent of all summer failures.",
    "Insight two: three bad battery lots from one supplier. They lost health 2.4 times faster, giving riders 49 kilometres per swap instead of 61, and costing 82 rupees of wear per swap instead of 35.",
    "Those lots explain three quarters of this year's margin loss. Riders noticed: nine thousand tickets filed as 'other' were really battery complaints.",
    "Insight three: new riders who hit two or more failed swaps in their first two weeks churn at 15 percent instead of 9. Price plan, channel and partner don't matter.",
    "And the peak-pricing pilot? Plus eight rupees a swap, but almost no demand shifted. It's a revenue lever, not a congestion fix. Meanwhile the biggest partner, on a 28 percent discount, is the least profitable.",
    "So, our recommendations. One: replace the 36 Gen 1 cabinets in the three hot cities before next summer, which avoids about 22,600 failures a year. Two: pull the bad-lot packs that are still in service, replace them, and buy on lot testing, not supplier name.",
    "Three: protect every new rider's first fortnight, with healthy packs and an automatic credit after any failure. Four: renegotiate ZipDrop instead of signing an exclusive.",
    "Fix the equipment and the batteries, and growth becomes profitable. Thank you.",
]
# Caption display fixes (what the viewer reads vs what the voice says)
CAP_FIX = {"Gen 1": "Gen1"}

FONT = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 40)


def shoot(i):
    out = B / f"slide_{i:02d}.png"
    url = (ROOT / "slides.html").as_uri() + f"#s={i}"
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={W},{H}", "--virtual-time-budget=3000",
                    f"--screenshot={out}", url], check=True, capture_output=True)
    return out


async def tts(i, text):
    mp3 = B / f"nar_{i:02d}.mp3"
    words = []
    c = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="WordBoundary")
    with open(mp3, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append((ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7, ch["text"]))
    return mp3, words


def duration(path):
    r = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True)
    t = r.stderr.split("Duration: ")[1].split(",")[0]
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def chunk_words(text, words, max_chars=58):
    """Split the script text into caption lines, then time each line by mapping its
    first token onto the TTS word-boundary timeline proportionally."""
    toks = text.split()
    lines, cur = [], []
    for j, tok in enumerate(toks):
        cur.append(j)
        line = " ".join(toks[x] for x in cur)
        nxt = len(line) + 1 + len(toks[j + 1]) if j + 1 < len(toks) else 0
        # don't leave a 1-2 word orphan before a sentence end
        tail = toks[j + 1:j + 3]
        orphan = any(t[-1] in ".?:," for t in tail[:2]) and len(line) + 1 + len(" ".join(tail)) <= 70
        if tok[-1] in ".?:" or (tok[-1] == "," and len(line) > 30) or (nxt > max_chars and not orphan):
            lines.append(cur); cur = []
    if cur:
        lines.append(cur)
    N, M = len(toks), len(words)
    chunks = []
    for L in lines:
        ws = min(M - 1, round(L[0] / N * M))
        we = min(M - 1, max(ws, round((L[-1] + 1) / N * M) - 1))
        chunks.append([words[ws][0], words[we][1], " ".join(toks[x] for x in L)])
    chunks[0][0] = 0.0
    return chunks


def caption_img(slide_png, text, out):
    img = Image.open(slide_png).convert("RGB")
    if text:
        for a, b in CAP_FIX.items():
            text = text.replace(a, b)
        d = ImageDraw.Draw(img, "RGBA")
        tw = d.textlength(text, font=FONT)
        x0, y0 = (W - tw) / 2 - 30, H - 118
        d.rounded_rectangle([x0, y0, x0 + tw + 60, y0 + 74], 18, fill=(0, 0, 0, 205))
        d.text((W / 2, y0 + 37), text, font=FONT, fill=(255, 255, 255), anchor="mm")
    img.save(out)


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


async def main():
    n = len(NARRATION)
    frames, audio, srt, t0 = [], [], [], 0.0
    for i, text in enumerate(NARRATION, 1):
        png = shoot(i)
        mp3, words = await tts(i, text)
        dur = duration(mp3) + PAD
        chunks = chunk_words(text, words)
        # caption frames: fill the slide's whole duration
        edges = [0.0] + [c[0] for c in chunks[1:]] + [dur]
        for k, c in enumerate(chunks):
            f = B / f"f_{i:02d}_{k:02d}.png"
            caption_img(png, c[2], f)
            frames.append((f, edges[k + 1] - edges[k]))
            srt.append((t0 + c[0], t0 + min(c[1] + 0.25, edges[k + 1]), c[2]))
        wav = B / f"nar_{i:02d}.wav"
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(mp3), "-af", f"apad=pad_dur={PAD}",
                        "-ar", "48000", "-ac", "2", str(wav)], check=True)
        audio.append(wav)
        t0 += duration(wav)
        print(f"slide {i:2d}/{n}: {dur:5.1f}s  (total {t0:6.1f}s)")

    with open(B / "frames.txt", "w") as f:
        for p, d in frames:
            f.write(f"file '{p.as_posix()}'\nduration {d:.3f}\n")
        f.write(f"file '{frames[-1][0].as_posix()}'\n")
    with open(B / "audio.txt", "w") as f:
        for a in audio:
            f.write(f"file '{a.as_posix()}'\n")
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(B / "audio.txt"),
                    str(B / "narration.wav")], check=True)

    out = ROOT / "VoltRelay_WindRocker_3min.mp4"
    subprocess.run([FF, "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(B / "frames.txt"),
                    "-i", str(B / "narration.wav"),
                    "-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)], check=True)

    with open(ROOT / "VoltRelay_WindRocker_3min.srt", "w", encoding="utf-8") as f:
        for k, (a, b, t) in enumerate(srt, 1):
            for x, y in CAP_FIX.items():
                t = t.replace(x, y)
            f.write(f"{k}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n")
    print(f"\nwrote {out}  ({duration(out):.1f}s)")


asyncio.run(main())
