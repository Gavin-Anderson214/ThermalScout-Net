# Publish ThermalScout-Net to GitHub

Your currently connected GitHub login is **Gavin-Anderson214**. No repository was created or uploaded automatically by this package.

## Step A — create an EMPTY private repository

1. Sign in to <https://github.com/new>.
2. Name the new repository **ThermalScout-Net**.
3. Set visibility to **Private** initially; you can make it public after reviewing file contents and dataset permissions.
4. Leave *Add a README*, *.gitignore*, and *Choose a license* **unchecked** (this package already contains a README and `.gitignore`).
5. Click **Create repository**.

## Step B — upload from VS Code Terminal (PowerShell)

Open the extracted **ThermalScout-Net_GitHub_Ready** folder in VS Code (**File > Open Folder**) and run:

```powershell
git --version
git init
git branch -M main
git add .
git status
git commit -m "Organize and document ThermalScout-Net research project"
git remote add origin https://github.com/Gavin-Anderson214/ThermalScout-Net.git
git push -u origin main
```

If Git requests your username/email, set them with `git config --global user.name "Your Name"` and `git config --global user.email "your-github-email"`. Authenticate in the browser if prompted. If `origin` already exists, run `git remote -v` before changing it; don't add it twice.

## Step C — routine project updates

```powershell
python3.13 project_manager.py update
git add .
git commit -m "Record new ThermalScout-Net experiment"
git push
```

Only the tracked reports, CSV metrics, scripts, and documentation go to GitHub. **The private raw frames and training weights are intentionally not pushed.**

## Git's role vs the project tracker

- **Git/GitHub** remembers code and report changes over time.
- **`project_manager.py`** updates your local results CSVs and living report when you run it.
- GitHub does not train your neural network or automatically collect accuracy results.
- For the eventual professor presentation, build a second tiny presentation-only folder, separate from the full research implementation.
