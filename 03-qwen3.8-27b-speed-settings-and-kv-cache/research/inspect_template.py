"""Extract the chat template from the GGUF and look for reasoning controls.

The tweet claims Qwen3.8 has graded reasoning ("default xhigh, set medium").
The chat template is the authority on which knobs actually exist, so read it
rather than trusting the claim.
"""

import re

from gguf import GGUFReader

import harness as H

r = GGUFReader(H.MODEL_GGUF)

tmpl = None
for field in r.fields.values():
    if "chat_template" in field.name:
        try:
            tmpl = field.contents()
        except Exception:
            pass
        if isinstance(tmpl, str):
            break

if not isinstance(tmpl, str):
    print("no chat template found")
    raise SystemExit(1)

print(f"chat template: {len(tmpl)} chars\n")

# Which variables does the template branch on?
vars_used = sorted(set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*(?:is defined|==|!=)", tmpl)))
print("template variables referenced:")
for v in vars_used:
    print("   ", v)

print("\nlines mentioning reasoning / thinking / effort:")
for line in tmpl.splitlines():
    low = line.lower()
    if any(k in low for k in ("think", "reason", "effort", "budget", "high", "medium", "low")):
        print("   ", line.strip()[:170])

with open("chat_template.jinja", "w", encoding="utf-8") as f:
    f.write(tmpl)
print("\nwrote chat_template.jinja")
