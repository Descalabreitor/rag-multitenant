"""Tables and charts from a benchmark run: docs/results/eval.md and its PNGs.

`python -m eval.report` reads the newest docs/results/raw/eval-*/ (or `--run`)
written by `python -m eval.bench`, and overwrites docs/results/eval.md and
docs/results/eval-*.png. The interpretation lives in ADR 0009 (Results), not
here, so a re-run doesn't silently change what the ADR argues from.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from eval.bench import ACCESS, RAW_DIR
from eval.corpus import SHARES
from eval.database import ROOT

RESULTS = ROOT / "docs" / "results"
TENANTS = tuple(SHARES)  # largest first
TENANT_LABEL = {key: f"{share * 100:g}%" for key, share in SHARES.items()}
LEVELS = tuple(ACCESS[u] for u in (0, 2, 1))  # least to most access

RECALL_VARIANTS = (
    "hnsw_off",
    "hnsw_strict",
    "hnsw_relaxed",
    "partition_off",
    "partition_relaxed",
    "exact",
    "planner",
)
POLICY_VARIANTS = {
    "HNSW (relaxed_order)": ("hnsw_relaxed", "baseline_hnsw", "exists_hnsw"),
    "exact": ("exact", "baseline_exact", "exists_exact"),
    "planner's choice": ("planner", "baseline_planner", "exists_planner"),
}
POLICY_LABELS = (
    "RLS, acl_principals (ours)",
    "no RLS, explicit WHERE",
    "RLS, EXISTS on document_acl",
)

# The dataviz skill's reference palette (light), slots in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def latest_run() -> Path:
    runs = sorted(p for p in RAW_DIR.glob("eval-*") if (p / "meta.json").is_file())
    if not runs:
        raise SystemExit("no benchmark run in docs/results/raw/ (run python -m eval.bench)")
    return runs[-1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Run:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta: dict[str, Any] = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.labels = {v["name"]: v_label(v) for v in self.meta["variants"]}
        self.names = [v["name"] for v in self.meta["variants"]]
        self.samples = _read_csv(path / "samples.csv")
        self.plans = _read_csv(path / "plans.csv")

    def recall(self, variant: str, k: int, tenant: str, access: str | None = None) -> float:
        values = [
            float(s["recall"])
            for s in self.samples
            if s["variant"] == variant
            and int(s["k"]) == k
            and s["tenant"] == tenant
            and (access is None or s["access"] == access)
            and s["recall"]
            and (k != 5 or s["rep"] == "0")
        ]
        return statistics.fmean(values) if values else float("nan")

    def short(self, variant: str, k: int, tenant: str) -> tuple[int, int]:
        """Queries that returned fewer rows than exist for the user, of all queries."""
        rows = [
            s
            for s in self.samples
            if s["variant"] == variant
            and int(s["k"]) == k
            and s["tenant"] == tenant
            and (k != 5 or s["rep"] == "0")
        ]
        return sum(int(s["returned"]) < int(s["expected"]) for s in rows), len(rows)

    def latencies(
        self, variant: str, tenant: str | None = None, access: str | None = None
    ) -> list[float]:
        k = self.meta["k_latency"]
        return [
            float(s["ms"])
            for s in self.samples
            if s["variant"] == variant
            and int(s["k"]) == k
            and (tenant is None or s["tenant"] == tenant)
            and (access is None or s["access"] == access)
        ]


def _shown(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def v_label(variant: dict[str, Any]) -> str:
    return str(variant["label"])


def _fmt_recall(value: float) -> str:
    return "-" if value != value else f"{value:.3f}"


def _fmt_ms(values: Sequence[float]) -> str:
    if not values:
        return "-"
    return f"{percentile(values, 0.5):.2f} / {percentile(values, 0.95):.2f}"


def _spread(values: Iterable[int]) -> str:
    ordered = sorted(values)
    low, high = ordered[0], ordered[-1]
    return f"{low:,}" if low == high else f"{low:,}..{high:,}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


# --- Charts --------------------------------------------------------------------------


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _log_ms(axis: Any) -> None:
    """Plain numbers on a log axis (1, 10, 100), no minor labels."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def recall_chart(run: Run, k: int, out: Path) -> None:
    variants = [v for v in RECALL_VARIANTS if v in run.names]
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    x = range(len(TENANTS))
    for ax, level in zip(axes, LEVELS, strict=True):
        _style(ax)
        for i, variant in enumerate(variants):
            ys = [run.recall(variant, k, t, level) for t in TENANTS]
            # Nudged sideways so series that agree (often at 1.0) stay visible.
            offset = (i - (len(variants) - 1) / 2) * 0.05
            ax.plot(
                [xi + offset for xi in x],
                ys,
                color=SERIES[i],
                marker=MARKERS[i],
                markersize=7,
                linewidth=2,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=run.labels[variant],
            )
        ax.set_xticks(list(x), [TENANT_LABEL[t] for t in TENANTS])
        ax.set_title(f"user with {level}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("tenant's share of the chunks table", color=INK_2, fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel(f"mean recall@{k}", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"Recall@{k} by tenant size and access level ({int(run.meta['corpus']['total']):,} chunks)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def latency_chart(run: Run, out: Path) -> None:
    """p50 bars with a p95 tick, one panel per tenant size, log scale."""
    variants = [v for v in run.names]
    fig, axes = plt.subplots(
        1, len(TENANTS), figsize=(14, 6), sharey=True, sharex=True, facecolor=SURFACE
    )
    y = range(len(variants))
    for ax, tenant in zip(axes, TENANTS, strict=True):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        p50 = [percentile(run.latencies(v, tenant), 0.5) for v in variants]
        p95 = [percentile(run.latencies(v, tenant), 0.95) for v in variants]
        ax.barh(list(y), p50, color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.scatter(p95, list(y), marker="|", s=160, color=INK, linewidths=2, zorder=3)
        ax.set_xscale("log")
        _log_ms(ax.xaxis)
        ax.set_title(f"tenant {TENANT_LABEL[tenant]}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("ms (log)", color=INK_2, fontsize=9)
    axes[0].set_yticks(list(y), [run.labels[v] for v in variants])
    axes[0].invert_yaxis()
    fig.suptitle(
        f"Retrieval query latency, k={run.meta['k_latency']}: bar = p50, tick = p95",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def policy_chart(run: Run, out: Path) -> None:
    """RLS vs no RLS vs EXISTS: p50 per tenant size, one panel per plan."""
    modes = [(m, vs) for m, vs in POLICY_VARIANTS.items() if all(v in run.names for v in vs)]
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    axes = list(axes) if len(modes) > 1 else [axes]
    width = 0.26
    for ax, (mode, variants) in zip(axes, modes, strict=True):
        _style(ax)
        for i, variant in enumerate(variants):
            xs = [t + (i - 1) * width for t in range(len(TENANTS))]
            p50 = [percentile(run.latencies(variant, t), 0.5) for t in TENANTS]
            p95 = [percentile(run.latencies(variant, t), 0.95) for t in TENANTS]
            ax.bar(
                xs,
                p50,
                width,
                color=SERIES[i],
                edgecolor=SURFACE,
                linewidth=2,
                label=POLICY_LABELS[i],
            )
            ax.scatter(xs, p95, marker="_", s=120, color=INK, linewidths=2, zorder=3)
        ax.set_yscale("log")
        _log_ms(ax.yaxis)
        ax.set_xticks(range(len(TENANTS)), [TENANT_LABEL[t] for t in TENANTS])
        ax.set_title(mode, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("tenant's share of the chunks table", color=INK_2, fontsize=9)
    axes[0].set_ylabel("ms (log): bar = p50, tick = p95", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Cost of the policy: RLS vs no RLS vs a normalised EXISTS policy",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --- Markdown ------------------------------------------------------------------------


def markdown(run: Run) -> str:
    meta = run.meta
    corpus = meta["corpus"]
    host = meta["host"]
    server = meta["server"]
    docker = host.get("docker") or {}
    k_lat = meta["k_latency"]
    lines: list[str] = [
        "# Retrieval benchmarks",
        "",
        f"Generated by `python -m eval.report` from `{_shown(run.path)}`"
        f" (git {meta['git']}). `make eval` reproduces it; ADR 0009 (Results) says what"
        " the numbers change. Method: `eval/bench.py`, corpus: `eval/corpus.py`.",
        "",
        "## Setup",
        "",
        f"- **Host:** {host['cpu']}, {host.get('logical_cpus')} logical CPUs,"
        f" {host.get('memory_gib')} GiB RAM, {host['os']}.",
        f"- **Docker:** {docker.get('server')} on {docker.get('os')},"
        f" VM with {docker.get('cpus')} CPUs and {docker.get('memory_gib')} GiB.",
        f"- **Database:** PostgreSQL {server['server_version']}, pgvector {server['pgvector']},"
        f" the compose image with its defaults: shared_buffers {server['shared_buffers']},"
        f" work_mem {server['work_mem']}, effective_cache_size {server['effective_cache_size']},"
        f" max_parallel_workers_per_gather {server['max_parallel_workers_per_gather']},"
        f" jit {server['jit']}. Sizes: "
        + ", ".join(f"{k} {v}" for k, v in sorted(server["sizes"].items()))
        + ".",
        f"- **Client:** Python {host['python']}, numpy {host['numpy']}, asyncpg over"
        " localhost into the container.",
        f"- **Corpus:** `eval.corpus`, {int(corpus['total']):,} chunks, seed {corpus['seed']},"
        f" {corpus['dim']} dimensions, {corpus['topics']} topics shared by all tenants,"
        f" spread {corpus['spread']}, {corpus['chunks_per_document']} chunks per document,"
        f" {corpus['users']} users per tenant, ACL mix (tenant:*, group, user)"
        f" {corpus['acl_mix']}. Four filler tenants make up the rest of the table."
        f" {corpus['queries_per_tenant']} query vectors per tenant, each run for three users.",
        f"- **HNSW:** the index from the migrations (m 16, ef_construction 64), ef_search"
        f" {meta['hnsw']['ef_search']}, max_scan_tuples {meta['hnsw']['max_scan_tuples']}"
        " (the service's defaults). Scratch indexes are built the same way.",
        f"- **Runs:** per variant, one warm-up pass, one pass at k={meta['k_recall']} (recall@10),"
        f" {meta['reps']} timed passes at k={k_lat} (latency, recall@5 from the first)."
        " Latency is the client's wall time of the search statement only.",
        f"- **Duration:** {meta['seconds']['total'] / 60:.0f} min for this run"
        f" (corpus {'already loaded' if meta['corpus_was_loaded'] else 'loaded in this run'}).",
        "",
        "Chunks per tenant, and chunks each measured user may read:",
        "",
    ]
    lines += _table(
        ["tenant", "chunks", *LEVELS],
        (
            [
                f"{TENANT_LABEL[t]} (`{t}`)",
                f"{meta['tenants'][t]['chunks']:,}",
                *(f"{meta['readable'][f'{t}/{level}']:,}" for level in LEVELS),
            ]
            for t in TENANTS
        ),
    )
    lines += [
        "",
        "Variants (all run the retriever's statement; plans forced per transaction):",
        "",
    ]
    forced = meta["forced_settings"]
    lines += _table(
        ["name", "label", "table", "role", "plan", "iterative scan"],
        (
            [
                f"`{v['name']}`",
                v["label"],
                f"`{v['table']}`",
                v["role"],
                f"{v['plan']}"
                + (
                    f" ({', '.join(f'{k}={x}' for k, x in forced[v['plan']].items())})"
                    if forced[v["plan"]]
                    else ""
                ),
                v["iterative_scan"],
            ]
            for v in meta["variants"]
        ),
    )
    mismatches = meta["plan_mismatches"]
    lines += [
        "",
        "Forced plans: "
        + (
            "every forced variant used the plan it forces (checked with EXPLAIN on each tenant"
            " and user)."
            if not mismatches
            else f"**{len(mismatches)} cells did not use the forced plan**: "
            + "; ".join(mismatches)
        ),
        "",
    ]

    for k in (meta["k_latency"], meta["k_recall"]):
        lines += [f"## Recall@{k}", "", f"![Recall@{k}](eval-recall-at-{k}.png)", ""]
        variants = [v for v in RECALL_VARIANTS if v in run.names]
        lines += _table(
            ["variant", "access", *(TENANT_LABEL[t] for t in TENANTS)],
            (
                [run.labels[v], level, *(_fmt_recall(run.recall(v, k, t, level)) for t in TENANTS)]
                for v in variants
                for level in LEVELS
            ),
        )
        lines += [
            "",
            f"Queries that came back short (fewer than min(k, readable) rows), k={k}:",
            "",
        ]
        lines += _table(
            ["variant", *(TENANT_LABEL[t] for t in TENANTS)],
            (
                [run.labels[v], *("{}/{}".format(*run.short(v, k, t)) for t in TENANTS)]
                for v in variants
            ),
        )
        lines.append("")

    lines += [
        f"## Latency (k={k_lat})",
        "",
        "![Latency per variant](eval-latency.png)",
        "",
        f"p50 / p95 in ms, over {len(LEVELS)} users x {corpus['queries_per_tenant']} queries x"
        f" {meta['reps']} passes per cell:",
        "",
    ]
    lines += _table(
        ["variant", *(TENANT_LABEL[t] for t in TENANTS), "recall@5, all cells"],
        (
            [
                run.labels[v],
                *(_fmt_ms(run.latencies(v, t)) for t in TENANTS),
                _fmt_recall(statistics.fmean(run.recall(v, k_lat, t) for t in TENANTS)),
            ]
            for v in run.names
        ),
    )
    lines += ["", "Per access level, largest tenant (50%) and smallest (0.1%):", ""]
    lines += _table(
        [
            "variant",
            *(f"{TENANT_LABEL[t]}, {level}" for t in (TENANTS[0], TENANTS[-1]) for level in LEVELS),
        ],
        (
            [
                run.labels[v],
                *(
                    _fmt_ms(run.latencies(v, t, level))
                    for t in (TENANTS[0], TENANTS[-1])
                    for level in LEVELS
                ),
            ]
            for v in run.names
        ),
    )
    lines += [
        "",
        "### The cost of the policy",
        "",
        "![RLS vs no RLS vs EXISTS](eval-policy.png)",
        "",
    ]
    lines += [
        "p50 / p95 in ms. Same statement and plan; the baseline connects as the superuser"
        " (no RLS) and adds the policy as WHERE clauses with the principals computed by the"
        " client; the EXISTS variant reads `eval_scratch.chunks_normalized` with a policy that"
        " checks `document_acl` per row instead of `acl_principals`.",
        "",
    ]
    rows: list[list[str]] = []
    for mode, policy_variants in POLICY_VARIANTS.items():
        if not all(v in run.names for v in policy_variants):
            continue
        for label, v in zip(POLICY_LABELS, policy_variants, strict=True):
            rows.append([mode, label, *(_fmt_ms(run.latencies(v, t)) for t in TENANTS)])
    lines += _table(["plan", "access check", *(TENANT_LABEL[t] for t in TENANTS)], rows)

    lines += [
        "",
        "## Caveats",
        "",
        "- One machine, one run, Docker Desktop's VM, PostgreSQL's default memory settings"
        " (shared_buffers 128MB: the HNSW index doesn't fit, so warm runs read from the VM's"
        " page cache). Absolute milliseconds will differ elsewhere; the ratios are the point.",
        "- Synthetic vectors (clustered random unit vectors), not nomic-embed-text output."
        " Recall depends on the distribution; the comparison between variants is what carries"
        " over.",
        "- The scratch tables are static copies whose HNSW indexes were built in one pass;"
        " `public.chunks`' index was built row by row during the load. Under the same plan and"
        " filter, the EXISTS table's index reaches a slightly different recall than"
        " `public.chunks`' (see the recall@5 column), which bounds how much of a scratch"
        " variant's recall comes from the build rather than the design.",
        "- Eight tenants, so eight partitions (plus an empty default). Thousands of tenants"
        " would mean thousands of partitions and indexes, which this run doesn't measure.",
        "- Latency includes the localhost round trip and the parsing of the 768-float literal,"
        " which is the floor of every variant (about 3 ms here).",
        "",
        "## Plans",
        "",
        f"EXPLAIN (ANALYZE, BUFFERS) of the first query at k={k_lat}, per tenant (users"
        " collapsed when their plan is the same). Full JSON in `plans.jsonl`.",
        "",
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for p in run.plans:
        grouped[(p["variant"], p["tenant"], p["scan"])].append(p)
    rows = []
    for (variant, tenant, scan), plans in sorted(
        grouped.items(), key=lambda kv: (run.names.index(kv[0][0]), TENANTS.index(kv[0][1]))
    ):
        users = ", ".join(p["access"] for p in plans)
        removed = _spread(int(p["rows_removed"]) for p in plans)
        buffers = _spread(int(p["buffers"]) for p in plans)
        pruned = max(int(p["subplans_removed"]) for p in plans)
        rows.append(
            [
                f"`{variant}`",
                TENANT_LABEL[tenant],
                users,
                f"`{scan}`" + (f" ({pruned} partitions pruned)" if pruned else ""),
                removed,
                buffers,
            ]
        )
    lines += _table(["variant", "tenant", "users", "CTE scan", "rows removed", "buffers"], rows)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__)
    parser.add_argument("--run", type=Path, help="a docs/results/raw/eval-*/ directory")
    args = parser.parse_args(argv)
    run = Run(args.run or latest_run())
    for k in (run.meta["k_latency"], run.meta["k_recall"]):
        recall_chart(run, k, RESULTS / f"eval-recall-at-{k}.png")
    latency_chart(run, RESULTS / "eval-latency.png")
    policy_chart(run, RESULTS / "eval-policy.png")
    (RESULTS / "eval.md").write_text(markdown(run), encoding="utf-8")
    print(f"wrote {RESULTS / 'eval.md'} and its charts from {run.path}")


if __name__ == "__main__":
    main()
