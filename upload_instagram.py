import os, time, requests

B = "https://graph.instagram.com"
uid, tok = os.environ["IG_USER_ID"], os.environ["IG_TOKEN"]
caption = open("output/caption.txt", encoding="utf-8").read()

r = requests.post(f"{B}/{uid}/media", data={"media_type": "REELS", "video_url": os.environ["VIDEO_URL"],
                                            "caption": caption, "access_token": tok})
r.raise_for_status()
cid = r.json()["id"]

for _ in range(40):
    s = requests.get(f"{B}/{cid}", params={"fields": "status_code", "access_token": tok}).json().get("status_code")
    if s == "FINISHED":
        break
    if s == "ERROR":
        raise SystemExit("Instagram konnte das Video nicht verarbeiten")
    time.sleep(10)
else:
    raise SystemExit("Zeitüberschreitung bei der Verarbeitung")

requests.post(f"{B}/{uid}/media_publish", data={"creation_id": cid, "access_token": tok}).raise_for_status()
print("Auf Instagram veröffentlicht")

name: daily
on:
  schedule:
    - cron: "0 15 * * *"
  workflow_dispatch:
permissions:
  contents: write
jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 40
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install requests numpy pillow edge-tts
      - run: python bear_facts.py
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      - name: Video veröffentlichen (öffentliche URL für Instagram)
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          TAG="v$(date +%Y%m%d-%H%M)"
          gh release create "$TAG" output/video.mp4 --title "$TAG" --notes-file output/caption.txt
          echo "VIDEO_URL=https://github.com/${{ github.repository }}/releases/download/$TAG/video.mp4" >> "$GITHUB_ENV"
      - name: Faktenliste speichern
        run: |
          git config user.name "bot"
          git config user.email "bot@users.noreply.github.com"
          git add facts_log.json
          git commit -m "log" || true
          git push
      - run: python upload_instagram.py
        env:
          IG_USER_ID: ${{ secrets.IG_USER_ID }}
          IG_TOKEN: ${{ secrets.IG_TOKEN }}
          VIDEO_URL: ${{ env.VIDEO_URL }}

#!/usr/bin/env python3
"""Täglicher Fakt-Clip: Wikipedia -> Gemini (Skript + Faktencheck) -> edge-tts -> Bär-Video.

Benötigt: pip install requests numpy pillow edge-tts   |   ffmpeg im PATH
Umgebung: GEMINI_API_KEY (gratis über Google AI Studio)
Dateien:  assets/bear_closed.png, assets/bear_open.png (gleiche Größe, transparent)
Ausgabe:  output/video.mp4, output/caption.txt, facts_log.json
"""
import asyncio, bisect, json, os, subprocess, textwrap, time
import edge_tts, numpy as np, requests
from PIL import Image, ImageDraw, ImageFont

KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FONT = os.getenv("FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
VOICE, W, H, FPS, MIN_SEC = "de-DE-ConradNeural", 1080, 1920, 24, 62
LOG, OUT = "facts_log.json", "output"


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"responseMimeType": "application/json"}},
        timeout=120)
    r.raise_for_status()
    time.sleep(6)  # Gratis-Limit einhalten
    return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])


def article():
    r = requests.get("https://de.wikipedia.org/w/api.php", headers={"User-Agent": "BearFacts/1.0"},
                     params={"action": "query", "generator": "random", "grnnamespace": 0,
                             "prop": "extracts", "explaintext": 1, "exlimit": 1, "format": "json"},
                     timeout=30).json()
    p = next(iter(r["query"]["pages"].values()))
    return p["title"], p.get("extract", "")[:6000]


def tts(text, path):
    asyncio.run(edge_tts.Communicate(text, VOICE).save(path))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.int16).astype(float)
    spf = 16000 // FPS
    n = len(a) // spf
    rms = np.sqrt((a[:n * spf].reshape(n, spf) ** 2).mean(1))
    return len(a) / 16000, rms > 0.3 * np.percentile(rms, 90)


def make_content(log):
    for _ in range(25):
        title, text = article()
        if title in log or len(text) < 3000:
            continue
        d = gemini(f"""Quelle (Wikipedia: {title}):\n{text}\n
Enthält die Quelle einen wirklich überraschenden Fakt, der ein breites Publikum interessiert?
Wenn nein: {{"ok": false}}
Wenn ja: Schreibe auf Deutsch einen Sprechtext mit 180-200 Wörtern für ein Kurzvideo: Hook (Frage oder
Behauptung), Hauptfakt, 2-3 Hintergrunddetails, Wendung, am Ende eine Frage an das Publikum, welches Thema
als Nächstes kommen soll. Verwende AUSSCHLIESSLICH Informationen aus der Quelle, erfinde nichts, keine Emojis,
keine Regieanweisungen, reiner Fließtext.
JSON: {{"ok": true, "script": "...", "caption": "Kurzbeschreibung mit 3 Hashtags"}}""")
        if not d.get("ok"):
            continue
        check = gemini(f"""Quelle:\n{text}\n\nSprechtext:\n{d['script']}\n
Nenne alle Aussagen im Sprechtext, die NICHT durch die Quelle gedeckt sind.
JSON: {{"unsupported": ["..."]}}""")
        if check.get("unsupported"):
            continue
        dur, mouth = tts(d["script"], f"{OUT}/voice.mp3")
        if dur < MIN_SEC:
            continue
        return title, d, dur, mouth
    raise SystemExit("Kein passender Fakt gefunden")


def render(script, dur, mouth):
    y = np.linspace(0, 1, H)[:, None, None]
    bg = Image.fromarray((np.array([30, 40, 90]) * (1 - y) + np.array([120, 50, 110]) * y)
                         .repeat(W, axis=1).astype(np.uint8))
    bears = [Image.open(f"assets/bear_{s}.png").convert("RGBA") for s in ("closed", "open")]
    bears = [b.resize((900, int(b.height * 900 / b.width))) for b in bears]
    words = script.split()
    chunks = [" ".join(words[i:i + 4]) for i in range(0, len(words), 4)]
    cum = np.cumsum([len(c) + 1 for c in chunks])
    starts = [0] + list(cum[:-1] / cum[-1] * dur)
    font = ImageFont.truetype(FONT, 84)
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", f"{OUT}/voice.mp3",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                          f"{OUT}/video.mp4"], stdin=subprocess.PIPE)
    for i in range(len(mouth)):
        f = bg.copy()
        b = bears[int(mouth[i])]
        f.paste(b, ((W - b.width) // 2, H - b.height - 150 + int(10 * np.sin(i / FPS * 6))), b)
        k = bisect.bisect(starts, i / FPS) - 1
        ImageDraw.Draw(f).multiline_text((W // 2, 520), textwrap.fill(chunks[k], 16), font=font, fill="white",
                                         stroke_width=8, stroke_fill="black", anchor="mm", align="center")
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    log = json.load(open(LOG)) if os.path.exists(LOG) else []
    title, d, dur, mouth = make_content(log)
    render(d["script"], dur, mouth)
    open(f"{OUT}/caption.txt", "w", encoding="utf-8").write(
        f"{d['caption']}\n\nQuelle: Wikipedia – {title}\n#KIgeneriert")
    json.dump(log + [title], open(LOG, "w"), ensure_ascii=False)
    print(f"Fertig: {OUT}/video.mp4 ({dur:.0f} s) – {title}")
