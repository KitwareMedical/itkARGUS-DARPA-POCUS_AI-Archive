# Anatomic Reconstruction - POCUS AI

This repository contains all of Kitware's code and resources for the DARPA POCUS AI project.

## Repo Layout

This repo is organized as a mono-repo. That means every top-level directory roughly corresponds to a project. A description of the repo is provided below.

* [ARGUS](https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI/tree/main/ARGUS): Contains all relevant projects related to the U-Net based AR/AI approach.

### Workflow

If you are working on a project in this repo, here is an outline for minimizing confusion and merge conflicts.

- Never force push to the "main" branch. To ensure this is followed, "main" will become a protected branch.
- When pushing into the remote main branch, be sure to fetch the remote main branch first. This way, you can handle merges as necessary.
- Each project should be in its own folder. This will help reduce merge conflicts that may arise when others push to the main branch.
- When there is a conflict with the main branch, you may either choose to create a merge commit or rebase your work on top of the remote main branch.

An example workflow would be to perform development directly on the main branch, and rebase as necessary when pushing.

```
(main) $ git commit -A -m "my message"
(main) $ git fetch origin
(main) $ git rebase [-i] origin/main
(main) $ git push -u origin main
```

If you prefer working on local branches, then the workflow would be similar except at the beginning when you merge into master.

```
(topic) $ git commit -A -m "my message"
(topic) $ git checkout main
(main) $ git merge topic
(main) $ git fetch origin
(main) $ git rebase [-i] origin/main
(main) $ git push -u origin main
```

## Funding

This work was funded, in part, by DARPA agreement HR00112190077 “Anatomic Reconstruction for Multi-Task POCUS Automated Interpretation (AR for POCUS AI).
