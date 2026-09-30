# rag-multitenant

A retrieval-augmented generation service for multiple organizations, where no user can ever retrieve, or see cited, a document they aren't allowed to read. Isolation lives in PostgreSQL Row-Level Security, so a bug in the application or a prompt injection in a document can't widen access.

The goal is to show, with numbers, what it takes to make permission-aware RAG trustworthy: a leak-test suite (cross-tenant queries, JWT tampering, connection-pool reuse, revocation, prompt injection, property-based tests), recall under restrictive filters, and the latency cost of doing it right.
