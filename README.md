# Anatomic Reconstruction - POCUS AI

This repository contains all of Kitware's code and resources for the DARPA POCUS AI project.

## Repo Layout

This repo is organized as a mon-repo with a hierarchical structure. That means that a directory structure is provided to give logical organization to a set of projects that comprise the tools, experiments, and final ARGUS project. A description of the hierarchy is provided below.

* [Tools](https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI/tree/main/Tools): Contains the projects used to visualize, annotate, and understand the data.
* [Experiments](https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI/tree/main/Experiments): Is further divided into PreProcessing, UNet, VAE, ROI, and other subdirs based on the focus of each experiment.  Each experiment is then a project within one of those subdirs.
* [Data](https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI/tree/main/Data): Contains a hierarchy that represents the data organization of the data to be used by the Experiments and in testing the final ARGUS project.  The data is NOT provided with this project.  It must be obtained from the project leaders and has strict distribution limitations.
* [ARGUS](https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI/tree/main/ARGUS): Contains the projects related to the final anatomic reconstruction for generalizable ultrasound AI (ARGUS) project

### Workflow

If you are working on a project in this repo, here is an outline for minimizing confusion and merge conflicts.

- Never force push to the "main" branch. To ensure this is followed, "main" will become a protected branch.
- When pushing into the remote main branch, be sure to fetch the remote main branch first. This way, you can handle merges as necessary.
- Each project should be in its own subfolder within the appropriate hierarchy. This will help reduce merge conflicts that may arise when others push to the main branch.
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

This work was funded, in part, by DARPA agreement HR00112190077 "Anatomic Reconstruction for Multi-Task POCUS Automated Interpretation (AR for POCUS AI)" and by NIH NIBIB/NIGMS R01 EB021396 "Slicer+PLUS: Collaborative, open-source software for ultrasound analysis".
