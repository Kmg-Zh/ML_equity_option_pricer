from benchmarks import aggregate_by_buckets, build_default_m3_cases, run_teacher_benchmark


def test_default_case_builder_generates_mixed_dividend_labels() -> None:
    cases = build_default_m3_cases()
    labels = {case.dividend_label for case in cases}
    assert "flat_yield" in labels
    assert "piecewise_yield" in labels
    assert len(cases) > 0


def test_benchmark_rows_have_consistency_metrics() -> None:
    cases = build_default_m3_cases()[:4]
    rows = run_teacher_benchmark(
        cases,
        binomial_steps=120,
        lsm_time_steps=25,
        lsm_paths=800,
        lsm_seed=3,
        fdm_spot_steps=80,
        fdm_time_steps=80,
    )

    assert len(rows) == len(cases)
    for row in rows:
        assert row.abs_lsm_vs_binomial >= 0.0
        assert row.abs_fdm_vs_binomial >= 0.0
        assert row.abs_fdm_vs_lsm >= 0.0
        assert row.binomial_us > 0.0
        assert row.lsm_us > 0.0
        assert row.fdm_us > 0.0


def test_bucket_aggregation_is_non_empty() -> None:
    cases = build_default_m3_cases()[:8]
    rows = run_teacher_benchmark(
        cases,
        binomial_steps=80,
        lsm_time_steps=20,
        lsm_paths=600,
        lsm_seed=9,
        fdm_spot_steps=70,
        fdm_time_steps=70,
    )
    summary = aggregate_by_buckets(rows)

    assert len(summary) >= 1
    assert all(item.count >= 1 for item in summary)
