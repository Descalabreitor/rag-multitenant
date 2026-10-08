"""The synthetic corpus generator (eval.corpus), in memory: no database."""

from collections import Counter
from uuid import uuid4

import numpy as np
import pytest

from eval.corpus import FILLERS, GROUPS, SHARES, Config, Corpus, chunk_counts, generate, tenant_ids

CONFIG = Config(total=20_000, seed=7, topics=8, chunks_per_document=5, queries=8, k=5)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return generate(CONFIG)


def test_chunk_counts_follow_the_shares() -> None:
    counts = chunk_counts(100_000)
    assert [counts[k] for k in SHARES] == [50_000, 10_000, 1_000, 100]
    assert sum(counts.values()) == 100_000
    assert {counts[k] for k in FILLERS} == {9_725}


def test_every_tenant_has_its_chunks_and_contiguous_ordinals(corpus: Corpus) -> None:
    expected = chunk_counts(CONFIG.total)
    assert {t.key: len(t.chunk_ids) for t in corpus.tenants} == expected
    for tenant in corpus.tenants:
        assert len(set(tenant.chunk_ids)) == len(tenant.chunk_ids)
        assert tenant.embeddings.shape == (len(tenant.chunk_ids), CONFIG.dim)
        for document in range(len(tenant.document_ids)):
            ordinals = tenant.chunk_ordinals[tenant.chunk_documents == document]
            assert list(ordinals) == list(range(len(ordinals)))
            assert 1 <= len(ordinals) <= CONFIG.chunks_per_document


def test_tenant_ids_depend_only_on_the_namespace(corpus: Corpus) -> None:
    assert {t.key: t.id for t in corpus.tenants} == tenant_ids(CONFIG.namespace)
    assert generate(Config(total=1000, seed=99)).tenants[0].id == tenant_ids()["share-50"]
    assert set(tenant_ids(uuid4()).values()).isdisjoint(tenant_ids().values())


def test_acl_mix_is_the_same_in_every_tenant(corpus: Corpus) -> None:
    for tenant in corpus.tenants:
        documents = len(tenant.document_ids)
        kinds = Counter(acl.split(":")[0] for acl in tenant.document_acls)
        assert sum(kinds.values()) == documents
        for kind, share in zip(("tenant", "group", "user"), CONFIG.acl_mix, strict=True):
            assert abs(kinds[kind] - documents * share) <= 1
        if documents >= 20:
            groups = {a for a in tenant.document_acls if a.startswith("group:")}
            assert groups == {f"group:{g}" for g in GROUPS}
            users = {a for a in tenant.document_acls if a.startswith("user:")}
            assert users <= {f"user:{u}" for u in tenant.users}
            assert len(users) > 1


def test_users_range_from_no_groups_to_all_groups(corpus: Corpus) -> None:
    for tenant in corpus.tenants:
        groups = {u: {g for s, g in tenant.memberships if s == u} for u in tenant.users}
        assert groups[tenant.users[0]] == set()
        assert groups[tenant.users[1]] == set(GROUPS)
        assert all(1 <= len(groups[u]) <= 2 for u in tenant.users[2:])
        # The narrow user reads strictly less than the broad one.
        assert tenant.readable(tenant.users[0]).sum() < tenant.readable(tenant.users[1]).sum()


def test_vectors_are_unit_and_clustered_by_topic(corpus: Corpus) -> None:
    tenant = corpus.tenants[0]
    assert tenant.embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(tenant.embeddings, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(corpus.centres, axis=1), 1.0, atol=1e-5)
    similarity = tenant.embeddings @ corpus.centres.T
    own = similarity[np.arange(len(similarity)), tenant.document_topics[tenant.chunk_documents]]
    # About 1 / sqrt(1 + spread**2) for the default spread of 1.
    assert abs(float(own.mean()) - 2**-0.5) < 0.01
    assert (np.argmax(similarity, axis=1) == tenant.document_topics[tenant.chunk_documents]).all()


def test_ground_truth_is_the_exact_top_k_of_what_the_user_may_read(corpus: Corpus) -> None:
    """Recomputed by brute force in float64, with the ACL checked chunk by chunk."""
    for tenant in corpus.tenants:
        position = {chunk_id: i for i, chunk_id in enumerate(tenant.chunk_ids)}
        for query in tenant.queries:
            principals = tenant.principals(query.user_sub)
            readable = [i for i, acl in enumerate(tenant.chunk_acls) if acl in principals]
            scores = tenant.embeddings[readable].astype(np.float64) @ query.vector.astype(
                np.float64
            )
            best = sorted(zip(-scores, readable, strict=True))[: CONFIG.k]
            assert query.readable == len(readable)
            assert [position[c] for c in query.expected] == [i for _, i in best]
            assert np.allclose(query.scores, [-s for s, _ in best])
            assert len(query.expected) == min(CONFIG.k, len(readable))


def test_queries_cover_every_user_and_sit_near_a_centre(corpus: Corpus) -> None:
    for tenant in corpus.tenants:
        assert {q.user_sub for q in tenant.queries} == set(tenant.users)
        for query in tenant.queries:
            assert query.tenant_id == tenant.id
            assert float(query.vector @ corpus.centres[query.topic]) > 0.8


def test_same_seed_gives_the_same_corpus(corpus: Corpus) -> None:
    again = generate(CONFIG)
    for a, b in zip(corpus.tenants, again.tenants, strict=True):
        assert a.document_ids == b.document_ids
        assert a.chunk_ids == b.chunk_ids
        assert a.document_acls == b.document_acls
        assert a.memberships == b.memberships
        assert np.array_equal(a.embeddings, b.embeddings)
        assert [q.expected for q in a.queries] == [q.expected for q in b.queries]
        assert all(
            np.array_equal(x.vector, y.vector) for x, y in zip(a.queries, b.queries, strict=True)
        )


def test_another_seed_gives_another_corpus(corpus: Corpus) -> None:
    other = generate(Config(**{**CONFIG.__dict__, "seed": CONFIG.seed + 1}))
    a, b = corpus.tenants[0], other.tenants[0]
    assert a.id == b.id
    assert set(a.chunk_ids).isdisjoint(b.chunk_ids)
    assert not np.array_equal(a.embeddings, b.embeddings)


@pytest.mark.parametrize(
    "changes",
    [{"total": 999}, {"users": 2}, {"acl_mix": (0.5, 0.5, 0.5)}, {"k": 0}],
)
def test_bad_configs_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Config(**changes)  # type: ignore[arg-type]
