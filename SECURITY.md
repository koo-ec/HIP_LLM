# Security policy

## Supported versions

Security fixes are applied to the latest commit on the default branch. This research repository does not currently maintain separate supported release branches.

## Reporting a vulnerability

Please do **not** disclose a suspected vulnerability in a public issue.

Use GitHub's private vulnerability-reporting facility for this repository where available:

1. Open the repository's **Security** tab.
2. Select **Advisories**.
3. Select **Report a vulnerability**.

If private reporting is unavailable, contact a repository maintainer privately through their verified institutional or GitHub contact channel. Include:

- the affected file, function and revision;
- a clear impact assessment;
- minimal reproduction steps or a proof of concept;
- any suggested mitigation; and
- whether the matter has been disclosed elsewhere.

Please allow the maintainers a reasonable period to investigate and prepare a fix before public disclosure.

## Security-sensitive areas

Particular care is required for:

- live provider API clients and credential handling;
- execution of generated benchmark code;
- loading untrusted NumPy, pickle or notebook artefacts;
- dependency and workflow changes; and
- accidental publication of prompts, responses or private datasets.

API keys must be supplied through environment variables and must never be committed. Live API tests are excluded from the default test and CI paths.
