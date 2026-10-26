"""Drop cached configs that predate a measurement change, keeping _meta."""

import json

PATH = "results.json"
try:
    d = json.load(open(PATH, encoding="utf-8"))
except FileNotFoundError:
    d = {}

names = [k for k in d if not k.startswith("_")]
for k in names:
    del d[k]

json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2)
print(f"cleared {len(names)} cached configs")
