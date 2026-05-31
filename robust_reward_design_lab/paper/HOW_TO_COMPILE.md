# How to Compile the Paper

## Requirements
Install a TeX distribution:
- **Windows**: MiKTeX (https://miktex.org/) or TeX Live
- **Linux/macOS**: TeX Live (`sudo apt install texlive-full`)

## Compile Commands

```bash
cd paper/

# Full compilation (recommended — run 3 times for cross-references)
pdflatex robust_reward_design_paper.tex
bibtex robust_reward_design_paper
pdflatex robust_reward_design_paper.tex
pdflatex robust_reward_design_paper.tex
```

Or using `latexmk` (automatically handles all passes):
```bash
latexmk -pdf robust_reward_design_paper.tex
```

## Output
- `robust_reward_design_paper.pdf` — the compiled paper

## Notes on Figure Paths
The paper references figures using relative paths:
```
../results/notebook_run/<figure>.png
```
Make sure you have run the Jupyter notebook `Full_Experiment_Suite.ipynb`
first to generate all figures in `results/notebook_run/`.

## Author Photo Placeholders
Replace these placeholder files with actual photos (1in × 1.25in):
- `paper/placeholder_hieu.png`
- `paper/placeholder_viet.png`
- `paper/placeholder_binh.png`
- `paper/placeholder_duc.png`

If you don't have photos, remove or comment out the `\begin{IEEEbiography}` blocks.
