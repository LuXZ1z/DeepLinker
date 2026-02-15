# Push to GitHub and use only this branch as default

## 1. Create repo on GitHub (if not yet)

Create a **new empty repository** on GitHub (do not add README or .gitignore).

## 2. Add remote and push only `chore/github-cleanup`

Replace `YOUR_USERNAME` and `YOUR_REPO` with your GitHub username and repo name:

```bash
cd /path/to/Linker_MultiModal

# Add remote (use your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
# or: git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git

# Push only this branch (do NOT push master/main)
git push -u origin chore/github-cleanup
```

This pushes **only** the branch `chore/github-cleanup`. The `master` branch is not pushed.

## 3. Set this branch as the default on GitHub

So the repo "main" view is this branch, not master:

1. Open your repo on GitHub.
2. Go to **Settings** → **General**.
3. Under **Default branch**, click the switch/edit icon.
4. Choose **chore/github-cleanup** (or type it) and confirm.

After this, the front page of the repo and "Code" tab will show `chore/github-cleanup` by default. You never need to push or use `master` unless you want to.

## 4. (Optional) Rename branch to `main` on GitHub

If you prefer the default branch to be called `main`:

1. On GitHub: **Settings** → **General** → **Default branch** → set to `chore/github-cleanup` (as above).
2. Locally rename the branch and push:
   ```bash
   git branch -m chore/github-cleanup main
   git push -u origin main
   ```
3. On GitHub, set default branch to `main`, then delete the remote `chore/github-cleanup` if you like.

Either way, only this branch is the "total/main" branch shown on GitHub; the old `master` is not pushed and not default.
