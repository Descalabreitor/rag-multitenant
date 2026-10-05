# Routing service runbook

## Restarting the router

1. Drain the queue first.
2. Restart one replica at a time.
3. Wait for the health check: it needs about 90 seconds to turn green.

Reference: CANARY-ACME-RUNBOOK
