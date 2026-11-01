# Releasing

Maintainer notes for cutting a new version and keeping the CI/CD setup up to date. Day-to-day
development checks (pre-commit, `ci.yml`, the docs build) are described in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#automated-checks).

- [Overview](#overview)
- [Docs deploy: `docs.yml`](#docs-deploy-docsyml)
- [One-off setup](#one-off-setup)
- [Release checklist](#release-checklist)
- [Maintenance](#maintenance)
- [When a release fails](#when-a-release-fails)

## Overview

The repository is a template: users clone it rather than install it from a package index.
A release is a tagged, documented snapshot of `main`:

| Event | `ci.yml` | `docs.yml` |
|---|---|---|
| Publish a GitHub release | | build + **deploy** |
| *Run workflow* button (`docs.yml`) | | build + **deploy** |

CI already ran on the commit when it reached `main`, so the release itself only deploys the docs.

## Docs deploy: `docs.yml`

On a published release, or *Run workflow*, the docs job builds the site with
`mkdocs build --strict`. It then publishes it to https://imendezguerra.github.io/modular_ml/
through the `github-pages` environment.

Deploying on releases rather than on every push to `main` keeps the site describing the
released version. A manual run from a feature branch is blocked by the `github-pages`
environment rules (only `main` and `v*` tags are allowed).

## One-off setup

1. **GitHub Pages:** in *Settings → Pages*, set *Source* to **GitHub Actions**.
   This creates a `github-pages` environment that only allows `main`. In *Settings → Environments
   → github-pages*, add the **tag** rule `v*` next to `main`, or the docs deploy on release fails.
2. **Zenodo (optional):** sign in to [Zenodo](https://zenodo.org) with GitHub and switch on
   this repository. Every GitHub release is then archived with a citable DOI, using the
   metadata in `CITATION.cff`. Add the DOI to `CITATION.cff` and the README afterwards.

## Release checklist

1. **Choose the version** using [semantic versioning](https://semver.org):
    - `MAJOR` for breaking changes, such as removing or renaming a function or config field.
    - `MINOR` for new features.
    - `PATCH` for bug fixes.
2. **Bump the version** in two places, in a PR to `main`:
    - `src/modular_ml/__init__.py` (`__version__`), which `pyproject.toml` reads.
    - `CITATION.cff` (`version:`), and add or update `date-released:` with the release date.
3. **Merge** once CI and the docs build are green.
4. **Tag the release commit** from the command line (or let GitHub create the tag in the next
   step):

    ```sh
    git checkout main && git pull                 # tag exactly what is on GitHub
    git tag -a vX.Y.Z -m "Release vX.Y.Z"         # annotated tag, e.g. v0.2.0
    git push origin vX.Y.Z
    ```

    Pushing a tag does **not** deploy anything. Only publishing a GitHub release does.
    Check existing tags with `git tag --list "v*"`.

5. **Release:** *Releases → Draft a new release*:
    - Choose the tag `vX.Y.Z`: the one you pushed, or type a new one to create it on `main`.
    - Write the release notes, flagging any breaking config changes.
    - Click **Publish release**.
6. **Verify** that the docs site shows the new version.

## Maintenance

Dependabot is off, so these updates are manual:

- **Action versions:** in both workflows (`actions/checkout@v7` and so on), check the actions'
  GitHub release pages a few times a year.
- **Hook versions:** run `pre-commit autoupdate`, then `pre-commit run --all-files`.
- **Python versions (each October, when a new Python is released and an old one reaches end of
  life):**
    - Update the test matrix in `ci.yml`.
    - Update `requires-python` in `pyproject.toml` and the `minimum-deps` Python version.
- **Minimum dependencies:** when you raise a lower bound in `pyproject.toml`, raise the matching
  pin in `ci/constraints-min.txt`.
- **MkDocs:** capped at `<2` because MkDocs 2.0 drops plugins (mkdocstrings).
  [Zensical](https://zensical.org) reads `mkdocs.yml` and is the likely migration path.

## When a release fails

- **Docs deploy fails on release:** the `github-pages` environment is missing the `v*` tag rule
  (one-off setup, step 1). Add it, then re-run the failed job.
- **Wrong commit or version released:** delete the release and its tag, fix `main`, and release
  again:
    1. Delete the release on GitHub: open it under *Releases* and click the trash icon.
    2. Delete the tag on GitHub and locally, then tag the fixed commit again:

        ```sh
        git push origin --delete vX.Y.Z
        git tag -d vX.Y.Z
        git checkout main && git pull
        git tag -a vX.Y.Z -m "Release vX.Y.Z"
        git push origin vX.Y.Z
        ```
