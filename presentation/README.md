# Presentation

`slides.tex` (Beamer, 16:9) pulls every number from `report/generated/macros.tex`
and every figure from `report/figures/`, exactly like the report, so the slides can
never disagree with it.

```bash
python -m src.run_all        # from the repo root: regenerates numbers + figures
make slides                  # -> presentation/slides.pdf
```

Submit `slides.pdf` on Moodle (the same PDF you present). Fill in `\TeamAuthors`
at the top of `slides.tex` first.

Flow (~12 slides, defended live after the deadline):
1. Problem and dataset (price regression + derived price tier)
2. Cleaning, features, the leakage columns removed
3. Our decision tree (impurity, vectorised split search, hyperparameters, depth curves)
4. Our Pegasos SVM (hinge primal, update rule, margin/support vectors; convergence; RFF extension)
5. Experimental setup (selection on validation, test once, CV, matched baselines)
6. Results table + one strong figure (errors concentrate at the tier threshold)
7. What the models use (importances, SVM weights)
8. Where the models fail + honest limitations
9. Bonus, who did what, next steps

Every team member should be able to explain any part of the code.
