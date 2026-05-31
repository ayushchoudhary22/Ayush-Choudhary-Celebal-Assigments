# Ayush Choudhary — Celebal Assignments

This repository contains weekly data-science assignments for the Celebal Technology internship.

## What this is
- A workspace of Jupyter notebooks and supporting files for week-by-week assignments.
- Each notebook corresponds to one week's exercises and should contain only code and minimal outputs when submitted.

## Project Structure
- `week1_Ayush Choudhary.ipynb`: Week 1 assignment notebook.
- `README.md`: This file — internship onboarding and submission instructions.

## Quick Start — Setup (Windows)
1. Install Python 3.8+ from the official website.
2. Open PowerShell in the project folder and create a virtual environment:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install commonly-used packages (adjust as needed):

```
pip install --upgrade pip
pip install numpy pandas matplotlib scipy jupyterlab nbformat nbconvert
```

4. Launch JupyterLab or open the notebook in VS Code:

```
jupyter lab
```

## How to Work on Assignments
- Duplicate or open the relevant notebook for the week and complete cells marked `# YOUR CODE HERE`.
- Run cells locally to verify outputs. Keep outputs small — avoid embedding large images or long text outputs.
- Save frequently and make small commits with descriptive messages.

## Git & Submission Guidelines
- Use a feature branch for any non-trivial edits: `git checkout -b week1/yourname`.
- Commit message format: `week1: add solutions — Ayush Choudhary`.
- Before pushing, clear notebook outputs to ensure GitHub can render the file and to keep repository size small.

To clear outputs locally (recommended):

```
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "week1_Ayush Choudhary.ipynb"
```

Or use the `nbstripout` tool to automatically strip outputs on commit (optional):

```
pip install nbstripout
nbstripout --install
```

Then push and open a Pull Request to the main repo (or share the GitHub link with your mentor).

## Coding & Notebook Style
- Keep functions small and well-named.
- Prefer vectorized NumPy/Pandas operations to Python loops when possible.
- Add short comments for non-obvious logic and keep markdown cells for explanations.

## Checklist Before Submission
- [ ] All required cells completed.
- [ ] No leftover debug prints or large outputs.
- [ ] Notebook passes basic runtime locally.
- [ ] Commit and push to a branch, then open a PR or share the repo link.

## Troubleshooting GitHub Notebook Render Errors
- If GitHub shows "An error occurred" when previewing the notebook, it usually means one of the outputs is large or contains complex HTML. Fix by clearing outputs and pushing again.
- Convert locally to HTML with `nbconvert` to confirm notebook validity:

```
python -m nbconvert --to html "week1_Ayush Choudhary.ipynb"
```

## Contacts & Help
- Mentor / Review: [Your mentor's name and email]
- For Git issues, ask on the Slack channel or raise a quick PR and tag your mentor.

---

If you want, I can also:
- Clean outputs in every notebook now to make the repo previewable on GitHub.
- Add a `requirements.txt` listing the exact package versions used.

Tell me which of those you'd like me to do next.