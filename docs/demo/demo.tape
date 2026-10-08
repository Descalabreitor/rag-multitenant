# The README demo: the same question asked as alice, dave and erin.
#
# Recorded with VHS (https://github.com/charmbracelet/vhs), which needs ttyd and
# ffmpeg on PATH. `make demo` brings the stack up, seeds it, starts the API on
# DEMO_PORT and runs `vhs docs/demo/demo.tape` with RAGMT_API_URL pointing at it.
# Run from the repository root: the output path is relative to it.
#
# Dev only: --dev-user gets a password-grant token from the ragmt-dev-password
# client, with KC_DEMO_USER_PASSWORD read from .env. Nothing is stored.

Output docs/demo.gif

Require ragctl

Set Shell "bash"
Set FontSize 15
Set Width 1280
Set Height 760
Set Padding 20
Set TypingSpeed 35ms
Set Theme "Builtin Dark"

Hide
Type "export PS1='> ' KC_DEMO_USER_PASSWORD=$(grep '^KC_DEMO_USER_PASSWORD=' .env | cut -d= -f2-) && clear"
Enter
Sleep 1s
Show

Type "# Same question, three users. alice: Acme, group finance."
Enter
Type `ragctl --dev-user alice ask "Summarise the budget."`
Enter
# A CPU completion with llama3.1:8b takes from a few seconds to a minute or two.
Wait@300s
Sleep 4s

Type "clear"
Enter
Type "# dave: Umbra, also a group called finance, other tenant."
Enter
Type `ragctl --dev-user dave ask "Summarise the budget."`
Enter
Wait@300s
Sleep 4s

Type "clear"
Enter
Type "# erin: Acme, no groups. Only the tenant-wide handbook is readable."
Enter
Type `ragctl --dev-user erin ask "Summarise the budget."`
Enter
Wait@300s
Sleep 5s
