# Contributing to GradLab!

Thank you for considering contributing to GradLab - I hope we can together build GradLab into a great educational tool for beginners!


## Ways to Contribute

GradLab began as a learning exercise for me to check my understanding of the core fundamental concepts when constructing neural networks. Now, my goal is to transform it into an open source project. I appreciate any small contributions in making GradLab more accessible for all! This can include:
- Adding further documentation
- Writing clearer tutorials
- Adding more focused demo files on key functions within the files in the `core/` folder
- Implementing new code in `core/` (i.e. a new `Layer` class or optimiser algorithm)
- Submitting bug reports and feature requests

Be sure to keep contributions focused to a single issue to make review easier!


## Ground Rules

* Initially, I loosely followed the [Google Python Style](https://google.github.io/styleguide/pyguide.html) when writing the docstring for my classes and functions. Ensure that code written in `core/` adheres to the same style guide 
* Ensure that any new code implemented is properly tested 
* Create issues for any major changes that you wish to make, so we can discuss the implementation (may host dicussions elsewhere in the future) 
* Keep feature versions as small as possible, preferably one new feature per version (as stated previously)
* Use clear PR titles and descriptions when starting a new PR - makes it easier for me to add the correct label
* Finally, and most importantly, have fun and use this as a learning opportunity!


## How to Contribute
1. **Fork & clone** the repo
2. **Create a branch**: `git checkout -b feat/my-change`
3. **Set up environment inside root folder**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pytest -m test  # run test suite
   ```
4. **Make changes** (follow [Google Python Style](https://google.github.io/styleguide/pyguide.html) for docstrings)
5. **PyLint** highlights no errors (only exception being the PyLint error: protected-access / W0212 when accessing a protected member)
6. **Test** before pushing
7. **Commit & push** your branch
   ```bash
   git add .
   git commit -m "feat: My Change"
   git push origin feat/my-change
   ```
8. **Open a Pull Request** on GitHub. Keep PRs focused on a single issue/feature

A good pull requests (i.e. patches, improvements, new features) remains focused in scope and avoids containing unrelated commits as mentioned previously. 

**NOTE**: Small fixes (typos, formatting, trivial docs) can be submitted directly as a PR without discussion - no need to open an issue.


## Reporting Bugs
Before reporting any bugs, be sure to search existing issues first in case someone else has already identified the same bug. If new, then open an issue and include the following:
  - Environment (OS, Python version)
  - Steps to reproduce bug
  - Expected vs actual behaviour

A good bug report shouldn't leave others needing to chase you up for more information. Please try to be as detailed as possible in your report.
Detailed bug reports go a long way in making sure the repo can run at its' absolute best - thank you!


## Suggesting Features
Open an issue describing:
- The problem your feature solves
- Why it’s useful for GradLab
- Any ideas for implementation

A good suggestion should be able to propose its' value to the project in a concise, detailed manner to help people understand its' potential.
Big or small, any suggestions that enhance the quality of the repository are welcome!

## Code Review Process
- PRs are reviewed by me (may look for maintainers in the future)
- Expect feedback within ~1 week
- Please respond to review comments; inactive PRs may be closed

## Quick Checklist Before PR
- [ ] Code follows style guide & is documented
- [ ] Tests added/updated
- [ ] All tests & linters pass
- [ ] PR is small and focused
