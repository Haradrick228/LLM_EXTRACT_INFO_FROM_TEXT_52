"""Test RAG via REST API and send Q+A to Telegram chat."""
import requests, json, time, re

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Bot token comes from the environment / .env (BOT_TOKEN), never hard-coded.
TOKEN = os.environ["BOT_TOKEN"]
API = "http://127.0.0.1:8001/chat"

# Get chat_id
r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=10)
updates = [u for u in r.json().get("result", []) if "message" in u]
if not updates:
    print("ERROR: Send a message to @testvbudusheebot first!")
    exit(1)

chat_id = updates[-1]["message"]["chat"]["id"]
print(f"chat_id = {chat_id}")

QUESTIONS = [
    ("1", "Что такое серия образовательных материалов «Школа возможностей»?"),
    ("2", "Сколько произведений включает Хрестоматия?"),
    ("3", "Кто заложил методологию Хрестоматии и когда?"),
    ("4", "Кому адресована Хрестоматия?"),
    ("5", "Что такое раздел «Я не тормоз»?"),
    ("6", "Для какого возраста предназначен раздел «Я не тормоз»?"),
    ("7", "Какова миссия фонда «Вклад в будущее» в рамках программы «Развивающая среда»?"),
    ("8", "Как фонд привлекает родителей и педагогов к использованию материалов?"),
    ("9", "Что означает «живой, постоянно пополняемый ресурс»?"),
    ("10", "Что такое раздел «Хрестоматия 14–18 лет»?"),
    ("11", "Какую задачу решает «Школа возможностей»?"),
    ("12", "Как фонд описывает подготовку ребенка к жизни?"),
    ("13", "Какую роль играют задания и игры в Хрестоматии?"),
    ("14", "Как фонду помогает чтение и анализ литературы?"),
    ("15", "Какую роль играют внутренние ресурсы ребенка в программе?"),
]

def send(chat_id, text):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={
        "chat_id": chat_id, "text": text
    }, timeout=10)

def has_cjk(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))

send(chat_id, "🧪 *ТЕСТ RAG-СИСТЕМЫ — 15 вопросов*\nЭмбеддер: USER-bge-m3 | Рераркер: bge-reranker-v2-m3 | LLM: qwen2.5:7b\n———")

results = []

for num, q in QUESTIONS:
    print(f"[{num}] asking...")
    t0 = time.time()
    try:
        r = requests.post(API, json={"message": q}, timeout=90)
        ans = r.json().get("response", f"HTTP {r.status_code}") if r.ok else f"HTTP {r.status_code}"
    except Exception as e:
        ans = f"ERROR: {e}"
    elapsed = round(time.time() - t0, 1)
    cjk = has_cjk(ans)

    # Send to Telegram
    a_short = ans[:600] + ("..." if len(ans) > 600 else "")
    send(chat_id, f"❓ *{num}. {q}*\n\n💬 {a_short}\n\n⏱ {elapsed}s | {'⚠️ CJK' if cjk else '✅ OK'}")

    results.append({"n": num, "q": q, "a": ans, "t": elapsed, "cjk": cjk, "len": len(ans)})
    time.sleep(0.6)

# Summary
ok = sum(1 for r in results if not r["cjk"])
cjk = sum(1 for r in results if r["cjk"])
tt = sum(r["t"] for r in results)
tc = sum(r["len"] for r in results)

summary = f"""📊 *ИТОГИ (15 тестов)*

✅ OK: {ok}/15
⚠️ CJK: {cjk}/15
⏱ Среднее время: {tt/15:.1f}s
📝 Всего символов: {tc}"""

send(chat_id, summary)
print(f"\nDone. {ok} OK, {cjk} CJK, {tt:.0f}s total, {tt/15:.1f}s avg")

# Save results
import json as j
Path=__import__('pathlib').Path
out = Path(__file__).resolve().parent.parent / "analysis" / "reports" / "tg_test_15_results.json"
out.write_text(j.dumps({"config": {"embedder":"USER-bge-m3","reranker":"BAAI/bge-reranker-v2-m3"}, "summary": {"total":15,"ok":ok,"cjk":cjk,"avg_time":round(tt/15,1),"total_chars":tc}, "results":results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved: {out}")
