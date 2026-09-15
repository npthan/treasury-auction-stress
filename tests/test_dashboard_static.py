"""Static/structural regression tests for the Phase 8 dashboard's HTML,
CSS, and JS source. This project has no JavaScript test runner or
headless-browser dependency (the offline suite is pure Python/pytest, no
live network, no new dependency added without explicit review per
`docs/project_rules.md`) -- these tests parse the dashboard source files as text and
assert the exact invariants an external, real-browser review found broken,
so a regression in either fix would fail here even without a browser. See
the Phase 8 acceptance review, Sections 9-10, for the browser
findings these tests guard against. They cannot replace a rendered-browser
recheck; they only prove the specific fixes are present and intact.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

DASHBOARD_DIR = Path("dashboard")
STYLE_CSS = DASHBOARD_DIR / "style.css"
APP_JS = DASHBOARD_DIR / "app.js"
INDEX_HTML = DASHBOARD_DIR / "index.html"
README = Path("README.md")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================
# Defect A -- the `hidden` attribute must always win over any `display`
# declaration, and the load-error banner's success/failure state must be
# set explicitly in JS, not left to the initial HTML alone.
# ============================================================


def test_hidden_attribute_cannot_be_overridden_by_display():
    css = _read(STYLE_CSS)
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css), (
        "style.css must force `[hidden] { display: none !important; }` so no later "
        "`display` rule (e.g. `.banner { display: flex }`) can keep a hidden element visible -- "
        "this is exactly the bug an external browser review found on #load-error"
    )


def test_banner_wraps_instead_of_forcing_horizontal_overflow():
    css = _read(STYLE_CSS)
    banner_block = re.search(r"^\.banner\s*\{([^}]*)\}", css, re.MULTILINE)
    assert banner_block, ".banner rule not found in style.css"
    body = banner_block.group(1)
    assert re.search(r"flex-wrap:\s*wrap", body), (
        ".banner must wrap its flex items onto multiple lines -- without this, a banner mixing "
        "raw text with several <code> children (like #load-error) forces one wide row that can "
        "exceed a narrow viewport's width even when correctly hidden most of the time"
    )


def test_load_error_banner_has_no_unguarded_display_declaration():
    css = _read(STYLE_CSS)
    for selector, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        if "load-error" in selector and "[hidden]" not in selector:
            assert "display" not in body, (
                f"found a display declaration scoped to #load-error outside the guarded "
                f"[hidden] rule: {selector.strip()!r} -- this would let it win over [hidden] again"
            )


def test_document_declares_a_global_horizontal_overflow_backstop():
    css = _read(STYLE_CSS)
    html_body_rule = re.search(r"html,\s*body\s*\{([^}]*)\}", css)
    assert html_body_rule, "html, body rule not found in style.css"
    assert "overflow-x: hidden" in html_body_rule.group(1), (
        "html, body must declare overflow-x: hidden as a backstop against global horizontal "
        "overflow -- local scrolling (.chart-wrap/.table-wrap use their own overflow-x: auto) "
        "must remain the only way to scroll sideways"
    )
    for wrapper_class in (".chart-wrap", ".table-wrap"):
        wrapper_block = re.search(re.escape(wrapper_class) + r"\s*\{[^}]*\}", css)
        assert wrapper_block and "overflow-x: auto" in wrapper_block.group(0), (
            f"{wrapper_class} must keep its own intentional overflow-x: auto"
        )


def test_fetch_success_path_explicitly_hides_load_error_banner():
    js = _read(APP_JS)
    success_handler = re.search(r"\.then\(\(data\)\s*=>\s*\{(.*?)\}\)\s*\n\s*\.catch", js, re.DOTALL)
    assert success_handler, "could not find the fetch success .then((data) => {...}) handler in app.js"
    assert re.search(r'getElementById\("load-error"\)\.hidden\s*=\s*true', success_handler.group(1)), (
        "the fetch success path must explicitly set load-error.hidden = true, not rely only on "
        "the initial HTML hidden attribute never having been touched"
    )


def test_fetch_failure_path_explicitly_shows_load_error_and_hides_main():
    js = _read(APP_JS)
    catch_handler = re.search(r"\.catch\(\(err\)\s*=>\s*\{(.*?)\}\);", js, re.DOTALL)
    assert catch_handler, "could not find the fetch .catch((err) => {...}) handler in app.js"
    body = catch_handler.group(1)
    assert re.search(r'getElementById\("load-error"\)\.hidden\s*=\s*false', body)
    assert re.search(r'getElementById\("main"\)\.hidden\s*=\s*true', body)


# ============================================================
# Defect B -- the documented preview topology must actually serve both the
# dashboard's own data AND every internal header link without a 404.
# ============================================================

_SERVER_ROOT = Path.cwd()
_PAGE_URL = "http://localhost:8000/dashboard/index.html"


def _resolve_under_repo_root(href: str) -> Path:
    resolved = urljoin(_PAGE_URL, href)
    relative = resolved.replace("http://localhost:8000/", "", 1)
    return _SERVER_ROOT / relative


def test_documented_preview_topology_resolves_every_internal_link():
    html = _read(INDEX_HTML)
    hrefs = re.findall(r'<a[^>]+href="([^"]+)"', html)
    internal = [h for h in hrefs if "://" not in h and not h.startswith("#")]
    assert internal, "expected at least one internal link in index.html"
    for href in internal:
        local_path = _resolve_under_repo_root(href)
        assert local_path.is_file(), (
            f"{href!r} resolves to {local_path} under the documented preview topology "
            "(server root = repository root, page opened at /dashboard/) but that file does not exist"
        )


def test_header_links_point_at_durable_documentation_not_reports():
    """The header's internal navigation must point at tracked, durable
    documentation (README / docs), never at the untracked generated
    reports -- those live in the gitignored artifacts/ directory."""
    html = _read(INDEX_HTML)
    hrefs = re.findall(r'<a[^>]+href="([^"]+)"', html)
    assert "../README.md" in hrefs
    assert not any("reports/" in h or "artifacts/" in h for h in hrefs)


def test_dashboard_json_fetch_still_resolves_under_the_dashboard_directory():
    js = _read(APP_JS)
    match = re.search(r'fetch\("([^"]+)"\)', js)
    assert match, "could not find the dashboard's fetch(...) call in app.js"
    fetch_path = match.group(1)
    assert not fetch_path.startswith("/") and "://" not in fetch_path, (
        "the dashboard JSON fetch must stay a page-relative path (not absolute) so it keeps "
        "resolving under dashboard/ regardless of where the server's root is"
    )
    local_path = _resolve_under_repo_root(fetch_path)
    assert local_path.is_file(), f"{fetch_path!r} resolves to {local_path}, which does not exist"


def test_no_stale_dashboard_rooted_preview_command_remains():
    for path in (README, INDEX_HTML):
        text = _read(path)
        assert "cd dashboard && python3 -m http.server" not in text, (
            f"{path} still documents the broken dashboard-rooted preview command "
            "(server root = dashboard/, which 404s the internal header links)"
        )
    assert "http://localhost:8000/dashboard/" in _read(README), (
        "README must document opening the dashboard at the /dashboard/ subpath, not the bare root"
    )


def test_table_scroll_hint_present_and_narrow_screen_only():
    html = _read(INDEX_HTML)
    css = _read(STYLE_CSS)
    assert "table-scroll-hint" in html, "case-studies table should carry a scroll-affordance hint"
    assert ".table-scroll-hint { display: none;" not in css.replace("\n", " ") or re.search(
        r"\.table-scroll-hint\s*\{[^}]*display:\s*none", css
    ), "the hint must be hidden by default (shown only on narrow screens via a media query)"
    media_block = re.search(r"@media\s*\(max-width:\s*640px\)\s*\{(.*?)\n\}", css, re.DOTALL)
    assert media_block and re.search(r"\.table-scroll-hint\s*\{[^}]*display:\s*block", media_block.group(1)), (
        "the narrow-screen media query must reveal .table-scroll-hint"
    )


# ============================================================
# The NY Fed's Terms of Use condition redistribution of NY Fed content
# on including their suggested copyright/source-identifier line, in
# their suggested format, without implying endorsement. This dashboard
# page republishes derived/aggregated NY Fed Primary Dealer Statistics
# figures, so its footer must carry that line. See
# docs/data_sources_and_licensing.md Section 2 for the recorded terms.
# ============================================================


def test_dashboard_footer_carries_the_ny_fed_required_attribution():
    html = _read(INDEX_HTML)
    assert "page-footer" in html
    footer_match = re.search(r"<footer[^>]*>(.*?)</footer>", html, re.DOTALL)
    assert footer_match, "expected a <footer> element"
    footer_text = footer_match.group(1)
    assert "Federal Reserve Bank of New York" in footer_text
    assert "Terms of Use" in footer_text
    assert "newyorkfed.org" in footer_text


def test_dashboard_footer_does_not_imply_ny_fed_endorsement():
    html = _read(INDEX_HTML)
    footer_match = re.search(r"<footer[^>]*>(.*?)</footer>", html, re.DOTALL)
    assert footer_match
    footer_text = footer_match.group(1).lower()
    assert "not endorsed by" in footer_text or "does not imply endorsement" in footer_text


def test_dashboard_footer_discloses_mit_license_does_not_cover_third_party_data():
    """User decision 3 (Phase 9 acceptance review): the MIT License
    applies only to this project's own code/docs, never to the
    third-party source data it analyzes -- this must be stated
    somewhere a public visitor to the dashboard will actually see it,
    not only in README/docs."""
    html = _read(INDEX_HTML)
    footer_match = re.search(r"<footer[^>]*>(.*?)</footer>", html, re.DOTALL)
    assert footer_match
    footer_text = footer_match.group(1)
    assert "MIT" in footer_text
    assert "does not relicense" in footer_text or "not relicense" in footer_text
