#!/usr/bin/env python3
"""Täglicher Fakt-Clip: Wikipedia -> Gemini (Skript + Faktencheck) -> edge-tts -> Bär-Video.

Benötigt: pip install requests numpy pillow edge-tts   |   ffmpeg im PATH
Umgebung: GEMINI_API_KEY (gratis über Google AI Studio)
Dateien:  assets/bear_closed.png, assets/bear_open.png (gleiche Größe, transparent)
Ausgabe:  output/video.mp4, output/caption.txt, facts_log.json
"""
import asyncio, bisect, json, os, random, subprocess, textwrap, time
import edge_tts, numpy as np, requests
from PIL import Image, ImageDraw, ImageFont

KEY = os.environ["GEMINI_API_KEY"]
MODELS = [os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), "gemini-3.1-flash-lite"]
FONT = os.getenv("FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
VOICE = os.getenv("VOICE", "de-DE-ConradNeural")
PITCH, RATE = "-12Hz", "-8%"  # tiefer und ruhiger
W, H, FPS, MIN_SEC = 1080, 1920, 24, 62
LOG, OUT = "facts_log.json", "output"


def gemini(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}}
    for attempt in range(3):
        for model in MODELS:  # bei Timeout, 404, 429 oder Serverfehler das nächste Modell probieren
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}",
                    json=body, timeout=60)
            except requests.RequestException as e:
                print(model, "nicht erreichbar:", type(e).__name__)
                continue
            if r.ok:
                try:
                    out = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                    time.sleep(6)  # Gratis-Limit einhalten
                    return out
                except (KeyError, ValueError):
                    print(model, "unbrauchbare Antwort")
                    continue
            print(model, r.status_code, r.text[:200])
        time.sleep(20)
    raise SystemExit("Gemini nicht erreichbar oder Limit erreicht")


def article(log):
    api, ua = "https://de.wikipedia.org/w/api.php", {"User-Agent": "BearFacts/1.0"}
    topic = open("topic.txt", encoding="utf-8").read().strip() if os.path.exists("topic.txt") else ""
    if topic:  # Wunschthema aus den Kommentaren (comments.py)
        q = {"list": "search", "srsearch": topic, "srlimit": 20, "srnamespace": 0}
    else:
        q = {"list": "random", "rnnamespace": 0, "rnlimit": 20}
    res = requests.get(api, headers=ua, timeout=30, params={"action": "query", "format": "json", **q}).json()["query"]
    titles = [h["title"] for h in res.get("search", res.get("random", [])) if h["title"] not in log]
    if not titles:
        return None, ""
    # Ein günstiger Aufruf wählt den besten Titel, statt jeden Artikel einzeln zu prüfen (spart Gratis-Kontingent)
    i = gemini("Titel:\n" + "\n".join(f"{n}: {t}" for n, t in enumerate(titles)) +
               "\n\nWelcher Artikel eignet sich am besten für einen überraschenden Wissensfakt für ein breites "
               'Publikum? JSON: {"index": 0}').get("index", 0)
    title = titles[i] if isinstance(i, int) and 0 <= i < len(titles) else titles[0]
    r = requests.get(api, headers=ua, timeout=30, params={
        "action": "query", "prop": "extracts", "explaintext": 1, "exlimit": 1, "titles": title, "format": "json"}).json()
    p = next(iter(r["query"]["pages"].values()))
    return title, p.get("extract", "")[:6000]


def tts(text, path):
    asyncio.run(edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH).save(path))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.int16).astype(float)
    spf = 16000 // FPS
    n = len(a) // spf
    rms = np.sqrt((a[:n * spf].reshape(n, spf) ** 2).mean(1))
    return len(a) / 16000, rms > 0.3 * np.percentile(rms, 90)


def make_content(log):
    for _ in range(8):
        title, text = article(log)
        if title is None or len(text) < 3000:
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
