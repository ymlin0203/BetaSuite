"""Runtime typography refinement for BetaSuite.

This lightweight patch keeps the analytical app logic unchanged while refining
how the expanded BetaSuite name is rendered in Streamlit Markdown blocks.
"""

from __future__ import annotations


def _brand_title(class_name: str) -> str:
    return f'''
<h1 class="brand-lockup {class_name}" aria-label="Biological Ecosystem Trajectory and Association Suite">
    <span class="brand-line">
        <span class="brand-cap">B</span><span class="brand-rest">iological</span>
        <span class="brand-cap">E</span><span class="brand-rest">cosystem</span>
    </span>
    <span class="brand-line">
        <span class="brand-cap">T</span><span class="brand-rest">rajectory</span>
        <span class="brand-connector">and</span>
        <span class="brand-cap">A</span><span class="brand-rest">ssociation</span>
        <span class="brand-cap">S</span><span class="brand-rest">uite</span>
    </span>
</h1>
'''


_BRAND_CSS = '''
            .brand-lockup {
                position: relative;
                z-index: 2;
                display: flex;
                flex-direction: column;
                gap: 0.06em;
                max-width: 900px;
                margin: 0;
                color: var(--apple-text);
                font-size: clamp(44px, 6vw, 76px);
                line-height: 0.9;
                letter-spacing: -0.055em;
                font-weight: 850;
            }

            .brand-line {
                display: block;
                white-space: normal;
            }

            .brand-cap {
                font-size: 1em;
                font-weight: 900;
                letter-spacing: -0.055em;
            }

            .brand-rest {
                font-size: 0.54em;
                font-weight: 760;
                letter-spacing: -0.035em;
                margin-right: 0.18em;
                opacity: 0.92;
                vertical-align: baseline;
            }

            .brand-connector {
                font-size: 0.42em;
                font-weight: 680;
                letter-spacing: -0.02em;
                color: var(--apple-muted);
                margin: 0 0.18em 0 0.04em;
                vertical-align: baseline;
            }

            .workspace-hero .brand-lockup {
                font-size: clamp(23px, 2.6vw, 32px);
                line-height: 0.95;
                gap: 0.03em;
                letter-spacing: -0.045em;
            }

            .workspace-hero .brand-rest {
                font-size: 0.58em;
            }

            .workspace-hero .brand-connector {
                font-size: 0.46em;
            }
'''


def _patch_streamlit_markdown() -> None:
    try:
        import streamlit as st
    except Exception:
        return

    if getattr(st.markdown, "_betasuite_brand_patch", False):
        return

    original_markdown = st.markdown

    def patched_markdown(body, *args, **kwargs):
        if isinstance(body, str):
            if ".landing-hero p {" in body and ".brand-lockup" not in body:
                body = body.replace("            .landing-hero p {", _BRAND_CSS + "\n            .landing-hero p {")

            body = body.replace(
                "<h1>Biological Ecosystem<br>Trajectory and Association Suite</h1>",
                _brand_title("landing-brand"),
            )
            body = body.replace(
                "<h1>Biological Ecosystem Trajectory and Association Suite</h1>",
                _brand_title("workspace-brand"),
            )

        return original_markdown(body, *args, **kwargs)

    patched_markdown._betasuite_brand_patch = True
    st.markdown = patched_markdown


_patch_streamlit_markdown()
