# Project Contracts

## Stop Contract

Stop immediately when:
- A command fails unexpectedly.
- A test fails.
- Build, import, runtime, or environment behavior is inconsistent with expectations.
- Required evidence is missing.
- The cause of a failure is unknown.

Do not continue to the next task until the failure is understood and resolved.

## Error Reporting Contract

For every failure, record:
- Exact command
- Exact error output
- Relevant environment information
- Observed behavior
- Expected behavior
- Current hypothesis, clearly marked as a hypothesis

Never convert a hypothesis into a fact.

## Error Diagnosis Protocol

1. Observe.
2. Collect evidence.
3. Reproduce when safe.
4. Isolate the failure.
5. Identify the root cause.
6. Apply the smallest justified fix.
7. Re-run the failing check.
8. Re-run relevant regression checks.
9. Commit only after verification.

## Error Severity

### Critical
Security issue, data loss, corrupted repository state, unreproducible environment, or unknown failure.

Action: Stop immediately.

### High
Broken tests, broken build, incorrect numerical result, or unresolved runtime failure.

Action: Stop the current task.

### Medium
Quality-tool failure, documentation inconsistency, or non-blocking tooling issue.

Action: Resolve before completing the current task.

### Low
Formatting, wording, or cosmetic issue.

Action: Fix during the current task when practical.

## No Silent Recovery

Never hide, suppress, ignore, or silently work around a failure.

Any workaround must be explicit, justified, and verified.

## Git Safety / Commit Safety

- Never overwrite unrelated user work.
- Never use destructive Git commands without explicit justification.
- Inspect git status before commits.
- Keep commits small and logically scoped.
- Do not commit known-broken work unless explicitly documenting a deliberate checkpoint.
- The repository must remain the source of truth.

## Unknown Failure Rule

If the cause of a failure is unknown:

STOP.

Do not guess.
Do not continue.
Do not replace the failing mechanism with another approach merely to make the error disappear.

## Security Stop Rule

Stop immediately when a task exposes:
- credentials
- secrets
- private keys
- tokens
- sensitive personal data
- unsafe dependency or execution behavior

Do not commit exposed sensitive data.

## Scope Stop Rule

Do not expand the project beyond the current phase or task without an explicit decision.

Avoid premature abstractions and unrelated improvements.

## Completion Rule

A task is complete only when:
- Implementation exists.
- Relevant tests pass.
- Relevant quality checks pass.
- Expected output has been verified.
- Git state is understood.
- Documentation/state is updated when required.
- The change is committed.

## Continuation Rule After Failure

After a failure:
- collect evidence first;
- resolve the failure;
- rerun the failed check;
- rerun relevant regression checks;
- only then continue.

## Core Principle

Never hide a failure.
Never guess past a failure.
Never continue past an unresolved failure.
