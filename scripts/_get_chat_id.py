import requests, time

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Bot token comes from the environment / .env (BOT_TOKEN), never hard-coded.
TOKEN = os.environ["BOT_TOKEN"]

print("LISTENING 35s... SEND MESSAGE NOW to @testvbudusheebot!")
# Poll repeatedly
for attempt in range(8):
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TOKEN}/getUpdates",
            params={"offset": -1, "timeout": 5},
            timeout=10
        )
        d = r.json()
        msgs = [u["message"] for u in d.get("result", []) if "message" in u]
        if msgs:
            for m in msgs:
                print(f"  chat_id={m['chat']['id']}  text={m.get('text','')[:60]}")
            cid = msgs[-1]["chat"]["id"]
            with open("scripts/_chat_id.txt", "w") as f:
                f.write(str(cid))
            print(f"SAVED chat_id={cid}")
            exit(0)
    except Exception as e:
        print(f"  err: {e}")
    time.sleep(4)

print("NO MESSAGES received")
