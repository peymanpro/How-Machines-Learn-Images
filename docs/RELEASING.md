# Release process

## What runs automatically

The [Quality workflow](../.github/workflows/quality.yml) runs on pushes to `main` and pull requests targeting `main`. It installs the pinned project dependencies and runs:

- the full pytest suite;
- synthetic training smoke checks;
- Ruff linting;
- Ruff formatting verification;
- strict Mypy checking.

A release should only be created after the Quality workflow succeeds on the exact commit intended for release.

## Publish a release manually

1. Update `VERSION`, `CHANGELOG.md`, and add matching release notes at `docs/RELEASE-NOTES-v<VERSION>.md`.
2. Commit and push those changes to `main`.
3. Open the repository's [Releases page](https://github.com/peymanpro/How-Machines-Learn-Images/releases) and select **Draft a new release**.
4. Create a new tag using the version in `VERSION` prefixed with `v` (for example, `v1.0.1`) and target the verified commit on `main`.
5. Use the matching release-notes file for the description, review the details, and select **Publish release**.

Publishing through the GitHub UI makes the publishing account explicit rather than attributing the release publication to the `github-actions` app.

## Important checks

- Do not reuse an existing tag or edit a published version to represent different source code.
- Verify the CI result belongs to the exact commit chosen for the tag.
- Describe small synthetic datasets as sanity checks, not as performance benchmarks.
- Keep the limitations and the framework-free NumPy scope clear in the release notes.
