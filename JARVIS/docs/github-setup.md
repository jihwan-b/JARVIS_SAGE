# Pushing to GitHub

This repo is ready to push as-is.

```bash
cd JARVIS
git init
git add -A
git commit -m "JARVIS: tool-latent conditioned RL for zero-demo VLA fine-tuning"

# create the repo on github, then:
git remote add origin https://github.com/<you>/jarvis.git
git branch -M main
git push -u origin main
```

Or with the GitHub CLI in one shot:

```bash
gh repo create jarvis --public --source=. --remote=origin --push
```

## Notes

- `assets/` holds the demo GIFs and figures the README embeds (~20 MB total). GitHub renders them inline.
- `.gitignore` excludes weights, checkpoints, datasets, and `*.usd`; it keeps `eval/diversity/*.csv`.
- When you wire in the real code, drop your modules into the matching folders (`rl/`, `tool_latents/`, …) — the README's repo-layout and quickstart already reference those paths.
