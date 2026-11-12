import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
for match in re.finditer(r"^```python\n# file: (\S+)\n(.*?)\n```$", text, flags=re.S | re.M):
    path = pathlib.Path(match.group(1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(match.group(2) + "\n", encoding="utf-8")
    print("scritto", path)
