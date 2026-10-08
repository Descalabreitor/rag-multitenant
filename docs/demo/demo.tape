# The README demo: the same question asked as alice, dave and erin.
#
# Recorded with VHS (https://github.com/charmbracelet/vhs), which needs ttyd and
# ffmpeg on PATH. `make demo` brings the stack up, seeds it, starts the API on
# DEMO_PORT and runs VHS on this tape with RAGMT_API_URL pointing at it and
# KC_DEMO_USER_PASSWORD taken from .env. Run from the repository root: the output
# path is relative to it.
#
# The commands typed below are the same in bash and PowerShell. `make demo`
# replaces the Set Shell line with DEMO_SHELL: on Windows, "bash" would start
# System32's WSL bash.exe rather than Git Bash.
#
# `make demo` also replaces the Output line: VHS 0.12 on Windows writes no GIF, so
# it records PNG frames and ffmpeg encodes docs/demo.gif from them (DEMO_FRAMERATE
# in the Makefile must match Set Framerate below). `vhs docs/demo/demo.tape` alone
# writes the GIF where VHS's own encoder works.
#
# Dev only: --dev-user gets a password-grant token from the ragmt-dev-password
# client. Nothing is stored.

Output docs/demo.gif

Require ragctl

Set Shell "bash"
Set Framerate 20
Set FontSize 22
Set Width 1280
Set Height 420
Set Padding 20
Set TypingSpeed 35ms
Set Theme "Builtin Dark"

Hide
Type "clear"
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
