"""Convert churn_pipeline.py (# %% cells) to churn_prediction.ipynb."""
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
src = (HERE / "churn_pipeline.py").read_text(encoding="utf-8")

# Split by cell markers
parts = re.split(r"^# %%(.*)$", src, flags=re.MULTILINE)
# parts: [pre, header1, body1, header2, body2, ...]
nb = nbf.v4.new_notebook()
cells = []

# Top-of-file docstring → markdown intro
intro = parts[0].strip()
if intro:
    cells.append(nbf.v4.new_markdown_cell(
        "# Customer Churn Prediction Using Machine Learning\n\n"
        "Telco Customer Churn dataset — Logistic Regression, Random Forest, SVM (RBF)."
    ))

for i in range(1, len(parts), 2):
    header = parts[i].strip()
    body = parts[i + 1].strip("\n")
    if header.startswith("[markdown]"):
        # markdown cell — body is comment-prefixed
        md = "\n".join(line.lstrip("# ").rstrip() for line in body.splitlines())
        cells.append(nbf.v4.new_markdown_cell(md))
    else:
        if header:
            cells.append(nbf.v4.new_markdown_cell(f"## {header}"))
        cells.append(nbf.v4.new_code_cell(body))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

out = HERE / "churn_prediction.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} ({len(cells)} cells)")
