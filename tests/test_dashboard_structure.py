from __future__ import annotations

import re

from scripts.build_html_report import page_html


def test_dashboard_has_accessible_product_navigation() -> None:
    html = page_html()

    assert 'class="skip-link" href="#main-content"' in html
    assert 'class="topnav" aria-label="Dashboard sections"' in html

    for anchor in (
        "overview",
        "rankings",
        "portfolio",
        "horizons",
        "charts",
        "validity",
    ):
        assert f'href="#{anchor}"' in html
        assert re.search(rf'id="{anchor}"', html)


def test_dashboard_hero_states_scope_without_live_claims() -> None:
    html = page_html()

    assert "Signals, portfolio risk, and evidence in one view." in html
    assert "Validation mode" in html
    assert "Static historical snapshot" in html
    assert "No real-money orders" in html
    assert "historical diagnostics" in html


def test_overview_cards_are_nonredundant_and_backtest_is_labeled() -> None:
    html = page_html()

    assert html.count('class="metric-card"') == 4
    assert "Selected research horizon" in html
    assert "Latest backtest signal" in html
    assert "historical walk-forward snapshot, not a live signal" in html
    assert "Evidence status" in html
