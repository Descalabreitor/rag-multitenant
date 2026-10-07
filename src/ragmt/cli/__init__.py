"""ragctl: the command-line client. It talks to the API over HTTP only.

`ragctl login` signs in through the browser (authorization code + PKCE S256
against the public `ragmt-cli` client, loopback redirect, RFC 8252) and keeps
the tokens in the user's config directory, readable by the user only.
`ragctl ask` sends a question to `POST /ask` and prints the answer and its
citations. `--dev-user` gets a token from the dev-only `ragmt-dev-password`
client instead (ADR 0005), for local demos.

Nothing here imports the server side: the CLI knows the API only by its JSON.
"""
