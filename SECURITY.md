# Security policy

## Supported versions

The latest release on `master` receives security fixes. This is research
software and does not provide a hardened multi-user service.

## Reporting

Report vulnerabilities privately through GitHub's security-advisory interface.
Do not include sensitive proof-of-concept material in a public issue.

The local browser server should remain bound to `127.0.0.1`. It has no
authentication, quotas, or multi-tenant isolation. Treat uploaded or generated
artifacts as untrusted when moving them into other systems.
