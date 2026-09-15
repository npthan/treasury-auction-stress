from __future__ import annotations

from treasury_auction_stress.data.normalize import (
    check_schema_drift,
    normalize_auctions,
    pages_to_records,
    to_raw_dataframe,
)
from treasury_auction_stress.data.quality import (
    build_quality_report,
    render_quality_report_markdown,
)


def _report(sample_page_body):
    records = pages_to_records({"pages": [{"body": sample_page_body}]})
    raw_df = to_raw_dataframe(records)
    drift = check_schema_drift(raw_df)
    normalized, anomalies = normalize_auctions(raw_df)
    return build_quality_report(normalized, anomalies, drift)


def test_quality_report_counts_match_fixture(sample_page_body):
    report = _report(sample_page_body)
    assert report["total_rows"] == len(sample_page_body["data"])
    # 2-year new issue, 10-year reopening, pending 30-year reopening
    assert report["nominal_coupon_row_count"] == 3
    assert report["excluded_records"]["bills"] == 1
    assert report["excluded_records"]["tips"] == 1
    assert report["excluded_records"]["frn"] == 1
    assert report["excluded_records"]["unaccounted_for"] == 0
    assert report["duplicate_key_row_count"] == 0
    assert len(report["pending_or_unsettled_nominal_auctions"]) == 1
    assert report["schema_drift"] == {"new_columns": [], "missing_columns": []}


def test_quality_report_excludes_pending_auction_from_coverage(sample_page_body):
    report = _report(sample_page_body)
    # the pending 30-year auction has no total_accepted yet, so it must not
    # count toward the settled-subset coverage denominator
    coverage = report["candidate_target_field_coverage_settled_nominal"]
    assert coverage["total_accepted"] == 1.0


def test_render_quality_report_markdown_is_nonempty_and_mentions_key_sections(sample_page_body):
    report = _report(sample_page_body)
    markdown = render_quality_report_markdown(report, "test run")
    assert "# Treasury Auction Data Quality Report" in markdown
    assert "Coverage by tenor" in markdown
    assert "Pending / unsettled nominal-coupon auctions" in markdown
    assert "912810UW6" not in markdown  # report is aggregate, not row-level dumps
