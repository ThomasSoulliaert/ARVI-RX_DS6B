"""ARVI-RX design system for the Streamlit front end.

Pure presentation layer: CSS injected into the page plus small HTML
snippet builders for the pieces native Streamlit widgets cannot render
(colored result badges, the confidence meter, the mandatory warning
banner, the class-distribution bars and the history table). None of
this touches prediction logic, storage, or the API — it only changes
how existing data is displayed.

Values below mirror the ARVI-RX Design System tokens/components
(tokens/colors.css, tokens/typography.css, tokens/spacing.css and the
Button/Card/ResultBadge/ConfidenceMeter/WarningBanner/DataTable/
SidebarNav components) so the app matches the approved mockups.
"""

from __future__ import annotations

import html
from typing import Any, Iterable

import streamlit as st

WARNING_TEXT = (
    "Prototype pédagogique. Non destiné au diagnostic. "
    "Validation par un professionnel qualifié requise."
)

RESULT_BADGE_CONFIG = {
    "normal": {
        "label": "Normal",
        "bg": "var(--result-normal-bg)",
        "fg": "var(--result-normal-fg)",
        "border": "var(--result-normal-border)",
        "dot": "#1c7a44",
    },
    "suspected_opacity": {
        "label": "Suspicion d'opacité",
        "bg": "var(--result-suspect-bg)",
        "fg": "var(--result-suspect-fg)",
        "border": "var(--result-suspect-border)",
        "dot": "#b7791f",
    },
    "uncertain": {
        "label": "Incertain",
        "bg": "var(--result-uncertain-bg)",
        "fg": "var(--result-uncertain-fg)",
        "border": "var(--result-uncertain-border)",
        "dot": "#8a94a2",
    },
}

_BADGE_SIZES = {
    "sm": {"font": "var(--text-xs)", "pad": "3px 9px", "dot": 6, "gap": 6},
    "md": {"font": "var(--text-sm)", "pad": "5px 12px", "dot": 7, "gap": 7},
    "lg": {"font": "var(--text-base)", "pad": "7px 16px", "dot": 9, "gap": 8},
}

THEME_CSS = """
:root {
  /* --- Brand blue (primary / action / trust) --- */
  --blue-50:  #eef4fb;
  --blue-100: #d8e7f6;
  --blue-200: #b4cfec;
  --blue-300: #86b0de;
  --blue-400: #4f8ccd;
  --blue-500: #2b72bd;
  --blue-600: #1e63ad;
  --blue-700: #1a5291;
  --blue-800: #164476;
  --blue-900: #133a63;

  /* --- Cool neutral (clinical gray) --- */
  --gray-0:   #ffffff;
  --gray-25:  #fafbfc;
  --gray-50:  #f4f6f9;
  --gray-100: #eceff3;
  --gray-200: #e0e5ec;
  --gray-300: #ccd3dd;
  --gray-400: #9fa9b7;
  --gray-500: #717c8c;
  --gray-600: #545f6e;
  --gray-700: #3a434f;
  --gray-800: #232a33;
  --gray-900: #141a20;

  /* --- Result-badge palette — reserved for result badges only --- */
  --result-normal-bg:     #e8f4ec;
  --result-normal-fg:     #1c7a44;
  --result-normal-border: #bfe0cb;
  --result-suspect-bg:     #fbf1de;
  --result-suspect-fg:     #97600f;
  --result-suspect-border: #f0d6a4;
  --result-uncertain-bg:     #eef1f5;
  --result-uncertain-fg:     #55606e;
  --result-uncertain-border: #d5dce4;

  /* --- Semantic aliases --- */
  --surface-canvas:   var(--gray-25);
  --surface-card:     var(--gray-0);
  --surface-sunken:   var(--gray-50);
  --surface-hover:    var(--gray-100);

  --border-hairline:  var(--gray-200);
  --border-strong:    var(--gray-300);
  --border-focus:     var(--blue-500);

  --text-strong:      var(--gray-900);
  --text-body:        var(--gray-700);
  --text-muted:       var(--gray-500);
  --text-faint:       var(--gray-400);
  --text-on-accent:   var(--gray-0);

  --action:           var(--blue-600);
  --action-hover:     var(--blue-700);
  --action-subtle-bg: var(--blue-50);
  --action-text:      var(--blue-700);

  --focus-ring:       rgba(43, 114, 189, 0.35);

  /* --- Alert tones (never green/amber — those are reserved for result badges) --- */
  --alert-info-bg:     var(--blue-50);
  --alert-info-border: var(--blue-100);
  --alert-info-fg:     var(--blue-900);
  --alert-error-bg:     #fbecec;
  --alert-error-border: #f0cfcd;
  --alert-error-fg:     #9c3b34;

  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, "Noto Sans", sans-serif;
  --font-mono: "SFMono-Regular", "SF Mono", ui-monospace, Menlo, Consolas,
               "Liberation Mono", monospace;

  --text-display: 30px;
  --text-h1:      24px;
  --text-h2:      19px;
  --text-h3:      16px;
  --text-base:    14px;
  --text-sm:      13px;
  --text-xs:      12px;
  --text-micro:   11px;

  --weight-regular:  400;
  --weight-medium:   500;
  --weight-semibold: 600;
  --weight-bold:     700;

  --leading-tight:  1.2;
  --leading-snug:   1.35;
  --leading-normal: 1.5;
  --leading-relaxed: 1.65;

  --tracking-tight:  -0.01em;
  --tracking-wide:   0.02em;
  --tracking-eyebrow: 0.08em;

  --space-1:  4px;
  --space-2:  8px;
  --space-3:  12px;
  --space-4:  16px;
  --space-5:  20px;
  --space-6:  24px;
  --space-8:  32px;

  --radius-xs:   4px;
  --radius-sm:   6px;
  --radius-md:   8px;
  --radius-lg:   12px;
  --radius-pill: 999px;

  --shadow-xs:   0 1px 2px rgba(20, 26, 32, 0.04);
  --shadow-sm:   0 1px 3px rgba(20, 26, 32, 0.06), 0 1px 2px rgba(20, 26, 32, 0.04);

  --duration-fast: 120ms;
  --duration-base: 180ms;
  --ease-standard: cubic-bezier(0.2, 0, 0.15, 1);

  --content-max: 1120px;
}

/* ============ App canvas & layout ============ */
.stApp, .stApp p, .stApp div, .stApp label,
.stApp span:not([data-testid="stIconMaterial"]) {
  font-family: var(--font-sans);
}
.stApp { background: var(--surface-canvas); }
[data-testid="stHeader"] { background: var(--surface-canvas); }
[data-testid="stMainBlockContainer"] {
  max-width: var(--content-max);
  padding-top: var(--space-8);
}

/* ============ Sidebar ============ */
[data-testid="stSidebar"] {
  background: var(--surface-card);
  border-right: 1px solid var(--border-hairline);
}
[data-testid="stSidebarUserContent"] { padding-top: var(--space-4); }

.arvi-sidebar-brand {
  display: flex; align-items: center; gap: 9px;
  padding: 2px 0 18px;
}
.arvi-sidebar-brand .arvi-chip {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; border-radius: var(--radius-sm);
  background: var(--action); color: var(--text-on-accent);
  font-size: var(--text-xs); font-weight: var(--weight-bold); letter-spacing: 0.02em;
}
.arvi-sidebar-brand .arvi-wordmark {
  font-size: var(--text-h3); font-weight: var(--weight-bold);
  color: var(--text-strong); letter-spacing: var(--tracking-tight);
}

/* Sidebar nav (st.sidebar.radio) styled as a nav rail */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
  font-size: var(--text-micro) !important;
  font-weight: var(--weight-semibold) !important;
  color: var(--text-muted) !important;
  text-transform: uppercase;
  letter-spacing: var(--tracking-eyebrow);
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
  gap: 2px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
  width: 100%;
  padding: 9px 10px;
  border-radius: var(--radius-md);
  transition: background var(--duration-fast) var(--ease-standard);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
  display: none;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
  background: var(--surface-hover);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {
  font-size: var(--text-base) !important;
  color: var(--text-body) !important;
  font-weight: var(--weight-medium) !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
  background: var(--action-subtle-bg);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) [data-testid="stMarkdownContainer"] p {
  color: var(--action-text) !important;
  font-weight: var(--weight-semibold) !important;
}

[data-testid="stSidebar"] hr { border-color: var(--border-hairline); }
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"] {
  border-radius: var(--radius-md) !important;
  border-color: var(--border-strong) !important;
}
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"]:focus-within {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 3px var(--focus-ring) !important;
}

/* ============ Typography ============ */
[data-testid="stHeading"] h1 {
  font-size: var(--text-display) !important;
  font-weight: var(--weight-bold) !important;
  color: var(--text-strong) !important;
  letter-spacing: var(--tracking-tight);
}
[data-testid="stCaptionContainer"] p {
  font-size: var(--text-sm) !important;
  color: var(--text-muted) !important;
}
h2 { font-size: var(--text-h2) !important; font-weight: var(--weight-semibold) !important; color: var(--text-strong) !important; }
h3 { font-size: var(--text-h3) !important; font-weight: var(--weight-semibold) !important; color: var(--text-strong) !important; }
hr { border-color: var(--border-hairline) !important; margin: var(--space-5) 0 !important; }

/* ============ Buttons ============ */
.stButton > button, button[kind="primary"], button[kind="secondary"] {
  border-radius: var(--radius-md) !important;
  font-weight: var(--weight-medium) !important;
  transition: background var(--duration-fast) var(--ease-standard),
              border-color var(--duration-fast) var(--ease-standard);
}
button[kind="primary"] {
  background: var(--action) !important;
  border-color: var(--action) !important;
}
button[kind="primary"]:hover:not(:disabled) {
  background: var(--action-hover) !important;
  border-color: var(--action-hover) !important;
}
button[kind="secondary"] {
  border-color: var(--border-strong) !important;
  color: var(--text-strong) !important;
}
button[kind="secondary"]:hover:not(:disabled) {
  background: var(--surface-hover) !important;
}

/* ============ Select ============ */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  border-radius: var(--radius-md) !important;
  border-color: var(--border-strong) !important;
  background: var(--surface-card) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 3px var(--focus-ring) !important;
}
[data-testid="stWidgetLabel"] p {
  font-size: var(--text-sm) !important;
  font-weight: var(--weight-medium) !important;
  color: var(--text-body) !important;
}

/* ============ File uploader (FileDrop) ============ */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface-sunken) !important;
  border: 1.5px dashed var(--border-strong) !important;
  border-radius: var(--radius-lg) !important;
  transition: background var(--duration-fast), border-color var(--duration-fast);
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--border-focus) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--text-strong) !important;
}

/* ============ Cards (st.container(border=True)) ============ */
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
  background: var(--surface-card) !important;
  border: 1px solid var(--border-hairline) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-xs) !important;
  padding: var(--space-5) !important;
}

/* ============ Metrics ============ */
[data-testid="stMetricLabel"] p {
  font-size: var(--text-micro) !important;
  font-weight: var(--weight-semibold) !important;
  color: var(--text-muted) !important;
  text-transform: uppercase;
  letter-spacing: var(--tracking-eyebrow);
}
[data-testid="stMetricValue"] {
  font-size: var(--text-h1) !important;
  font-weight: var(--weight-semibold) !important;
  color: var(--text-strong) !important;
  font-variant-numeric: tabular-nums;
}

/* ============ Alerts — never green/amber (reserved for result badges) ============ */
[data-testid="stAlertContainer"] {
  border-radius: var(--radius-md) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
  background: var(--alert-info-bg) !important;
  border: 1px solid var(--alert-info-border) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) p,
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) p,
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) p {
  color: var(--alert-info-fg) !important;
  font-weight: var(--weight-medium) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
  background: var(--alert-error-bg) !important;
  border: 1px solid var(--alert-error-border) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) p {
  color: var(--alert-error-fg) !important;
  font-weight: var(--weight-medium) !important;
}

/* ============ Expander (JSON block) ============ */
[data-testid="stExpander"] {
  border: 1px solid var(--border-hairline) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-xs) !important;
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  padding: var(--space-3) var(--space-4) !important;
  font-size: var(--text-sm) !important;
  font-weight: var(--weight-medium) !important;
  color: var(--text-body) !important;
}

/* ============ Custom HTML pieces ============ */
.arvi-eyebrow {
  display: block;
  font-size: var(--text-micro);
  font-weight: var(--weight-semibold);
  color: var(--text-muted);
  letter-spacing: var(--tracking-eyebrow);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.arvi-card-header { margin-bottom: var(--space-4); }
.arvi-card-header .arvi-card-title {
  margin: 0; font-size: var(--text-h3); font-weight: var(--weight-semibold); color: var(--text-strong);
}
.arvi-card-header .arvi-card-subtitle {
  margin: 2px 0 0; font-size: var(--text-sm); color: var(--text-muted);
}

.arvi-badge {
  display: inline-flex; align-items: center; font-family: var(--font-sans);
  font-weight: var(--weight-semibold); line-height: 1; white-space: nowrap;
  border-radius: var(--radius-pill);
}
.arvi-badge .arvi-badge-dot { border-radius: 50%; flex-shrink: 0; }

.arvi-meter-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.arvi-meter-track {
  position: relative; height: 8px; background: var(--surface-sunken);
  border: 1px solid var(--border-hairline); border-radius: var(--radius-pill);
}
.arvi-meter-fill {
  position: absolute; left: 0; top: 0; bottom: 0; border-radius: var(--radius-pill);
  transition: width var(--duration-base) var(--ease-standard);
}
.arvi-meter-threshold {
  position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--gray-500);
  border-radius: 2px; transform: translateX(-1px);
}
.arvi-meter-caption { display: block; margin-top: 6px; font-size: var(--text-micro); color: var(--text-faint); }

.arvi-warning-banner {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 12px 14px; background: var(--action-subtle-bg);
  border: 1px solid var(--blue-100); border-radius: var(--radius-md);
}
.arvi-warning-banner.compact { align-items: center; padding: 8px 12px; }
.arvi-warning-banner span {
  font-size: var(--text-sm); line-height: var(--leading-snug);
  color: var(--blue-900); font-weight: var(--weight-medium);
}
.arvi-warning-banner.compact span { font-size: var(--text-xs); }

.arvi-dist-row { display: flex; align-items: center; gap: 12px; }
.arvi-dist-label { display: flex; align-items: center; gap: 7px; width: 170px; flex-shrink: 0; }
.arvi-dist-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.arvi-dist-label span { font-size: var(--text-sm); color: var(--text-body); font-family: var(--font-mono); }
.arvi-dist-track {
  flex: 1; height: 8px; background: var(--surface-sunken); border-radius: var(--radius-pill);
  overflow: hidden; border: 1px solid var(--border-hairline);
}
.arvi-dist-fill { height: 100%; background: var(--action); border-radius: var(--radius-pill); }
.arvi-dist-count {
  width: 28px; text-align: right; font-size: var(--text-sm); font-weight: var(--weight-semibold);
  color: var(--text-strong); font-variant-numeric: tabular-nums;
}

.arvi-table-wrap { width: 100%; overflow-x: auto; }
table.arvi-table { width: 100%; border-collapse: collapse; font-family: var(--font-sans); }
table.arvi-table th {
  text-align: left; padding: 10px 14px; font-size: var(--text-micro);
  font-weight: var(--weight-semibold); color: var(--text-muted);
  letter-spacing: var(--tracking-eyebrow); text-transform: uppercase;
  border-bottom: 1px solid var(--border-strong); white-space: nowrap;
}
table.arvi-table th.arvi-num, table.arvi-table td.arvi-num { text-align: right; }
table.arvi-table td {
  padding: 10px 14px; font-size: var(--text-sm); color: var(--text-body);
  border-bottom: 1px solid var(--border-hairline); white-space: nowrap;
}
table.arvi-table td.arvi-num { font-variant-numeric: tabular-nums; }
table.arvi-table tr.arvi-row-alt { background: var(--surface-sunken); }
table.arvi-table td.arvi-empty { text-align: center; color: var(--text-muted); padding: 20px 14px; }
"""


def inject_theme() -> None:
    """Inject the ARVI-RX theme CSS. Call once, right after set_page_config."""
    st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)


def sidebar_brand(brand: str = "ARVI-RX") -> None:
    """Render the RX chip + wordmark at the top of the sidebar."""
    st.sidebar.markdown(
        f"""
        <div class="arvi-sidebar-brand">
          <span class="arvi-chip">RX</span>
          <span class="arvi-wordmark">{html.escape(brand)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning_banner(text: str = WARNING_TEXT, compact: bool = False) -> None:
    """Render the mandatory educational disclaimer banner."""
    compact_cls = " compact" if compact else ""
    st.markdown(
        f"""
        <div class="arvi-warning-banner{compact_cls}" role="note">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" style="flex-shrink:0;margin-top:{0 if compact else 1}px">
            <circle cx="12" cy="12" r="9" stroke="var(--blue-600)" stroke-width="1.8"/>
            <path d="M12 8v5" stroke="var(--blue-600)" stroke-width="1.8" stroke-linecap="round"/>
            <circle cx="12" cy="16.2" r="1" fill="var(--blue-600)"/>
          </svg>
          <span>{html.escape(text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_badge_html(predicted_class: str, size: str = "md", label: str | None = None) -> str:
    """Return an inline HTML span for the reserved-color result badge."""
    cfg = RESULT_BADGE_CONFIG.get(predicted_class, RESULT_BADGE_CONFIG["uncertain"])
    s = _BADGE_SIZES.get(size, _BADGE_SIZES["md"])
    shown_label = html.escape(label or cfg["label"])
    return (
        f'<span class="arvi-badge" style="gap:{s["gap"]}px;padding:{s["pad"]};'
        f'font-size:{s["font"]};color:{cfg["fg"]};background:{cfg["bg"]};'
        f'border:1px solid {cfg["border"]};">'
        f'<span class="arvi-badge-dot" style="width:{s["dot"]}px;height:{s["dot"]}px;background:{cfg["dot"]}"></span>'
        f"{shown_label}</span>"
    )


def card_header_html(title: str, subtitle: str | None = None) -> str:
    """Return a Card-style header block (title + optional muted subtitle)."""
    subtitle_html = (
        f'<p class="arvi-card-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    )
    return (
        f'<div class="arvi-card-header"><h3 class="arvi-card-title">{html.escape(title)}</h3>'
        f"{subtitle_html}</div>"
    )


def confidence_meter_html(value: float, threshold: float = 0.60, label: str = "Confiance") -> str:
    """Return the confidence bar with the 0.60 decision threshold marked."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    pct = max(0.0, min(1.0, value)) * 100
    below = value < threshold
    fill_color = "var(--gray-400)" if below else "var(--action)"
    below_note = " · en dessous → incertain" if below else ""
    return (
        '<div class="arvi-meter">'
        f'<div class="arvi-meter-row"><span class="arvi-eyebrow" style="margin:0">{html.escape(label)}</span>'
        f'<span style="font-size:var(--text-base);font-weight:var(--weight-semibold);'
        f'color:var(--text-strong);font-variant-numeric:tabular-nums">{value:.2f}</span></div>'
        '<div class="arvi-meter-track">'
        f'<div class="arvi-meter-fill" style="width:{pct:.1f}%;background:{fill_color}"></div>'
        f'<div class="arvi-meter-threshold" style="left:{threshold * 100:.1f}%" title="Seuil {threshold:.2f}"></div>'
        "</div>"
        f'<span class="arvi-meter-caption">Seuil de décision {threshold:.2f}{below_note}</span>'
        "</div>"
    )


def dist_bar_html(label: str, predicted_class: str, count: int, max_count: int) -> str:
    """Return one row of the class-distribution bars (dashboard)."""
    cfg = RESULT_BADGE_CONFIG.get(predicted_class, RESULT_BADGE_CONFIG["uncertain"])
    pct = (count / max_count * 100) if max_count else 0
    return (
        '<div class="arvi-dist-row">'
        f'<div class="arvi-dist-label"><span class="arvi-dist-dot" style="background:{cfg["dot"]}"></span>'
        f"<span>{html.escape(label)}</span></div>"
        f'<div class="arvi-dist-track"><div class="arvi-dist-fill" style="width:{pct:.1f}%"></div></div>'
        f'<div class="arvi-dist-count">{count}</div>'
        "</div>"
    )


def data_table(
    rows: Iterable[dict[str, Any]],
    headers: dict[str, str],
    badge_key: str | None = None,
    numeric_keys: Iterable[str] = (),
    empty_text: str = "Aucune donnée.",
) -> None:
    """Render a compact, zebra-striped history table matching the DataTable component.

    `headers` maps row dict keys (in display order) to their column labels.
    `badge_key` (if given) renders that column's value as a ResultBadge pill.
    """
    rows = list(rows)
    numeric_keys = set(numeric_keys)

    header_cells = "".join(
        f'<th class="{"arvi-num" if key in numeric_keys else ""}">{html.escape(label)}</th>'
        for key, label in headers.items()
    )

    if not rows:
        body = f'<tr><td class="arvi-empty" colspan="{len(headers)}">{html.escape(empty_text)}</td></tr>'
    else:
        body_rows = []
        for i, row in enumerate(rows):
            cells = []
            for key in headers:
                value = row.get(key, "")
                if key == badge_key:
                    cell_html = result_badge_html(str(value), size="sm")
                else:
                    cell_html = html.escape("" if value is None else str(value))
                cls = "arvi-num" if key in numeric_keys else ""
                cells.append(f'<td class="{cls}">{cell_html}</td>')
            row_cls = "arvi-row-alt" if i % 2 else ""
            body_rows.append(f'<tr class="{row_cls}">{"".join(cells)}</tr>')
        body = "".join(body_rows)

    st.markdown(
        f'<div class="arvi-table-wrap"><table class="arvi-table">'
        f"<thead><tr>{header_cells}</tr></thead><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )
