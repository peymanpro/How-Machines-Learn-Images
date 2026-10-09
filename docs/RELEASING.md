# Releasing

The release version is stored in `VERSION`. Each release must have a matching changelog entry and a file at `docs/RELEASE-NOTES-v<VERSION>.md`.

## Publish a release

1. Update `VERSION`, `CHANGELOG.md`, and the corresponding release notes.
2. Push the prepared release commit to `main` with a message beginning `release: publish v`.
3. The publish workflow runs the complete quality gate: pytest, synthetic training smoke checks, Ruff linting, Ruff formatting, and strict Mypy.
4. Only after all checks pass, the workflow creates the version tag and publishes the GitHub Release from the exact tested commit.

The release workflow rejects invalid semantic versions, missing release notes, and existing version tags. A failed check prevents publication.
