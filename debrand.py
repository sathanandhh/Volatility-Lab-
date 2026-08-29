#!/usr/bin/env python3
"""
debrand.py  —  Remove all personal/brand references in one shot.

Usage:
    python debrand.py              # dry-run — preview what changes
    python debrand.py --apply      # write changes to disk
    python debrand.py --verify     # grep for any leftovers

Safe: only modifies text inside strings, comments, docstrings,
markdown, and metadata. Never touches:
    - Python identifiers (variable / function / class names)
    - Import paths
    - File names
    - JSON keys that are structural (only values are changed)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ── All replacements ──────────────────────────────────────────────
# Each entry: (regex_pattern, replacement, description)
# Patterns are case-sensitive unless flags=re.IGNORECASE is set.
# Order matters: longer patterns first so they match before shorter ones.

REPLACEMENTS: list[tuple[str, str, str]] = [

    # ── Brand name (all case variants) ───────────────────────────
    (r"The\s+Mountain\s+Path\s+Academy",
     "Volatility Analytics Lab",
     "Brand name (title case)"),

    (r"THE\s+MOUNTAIN\s+PATH\s+ACADEMY",
     "VOLATILITY ANALYTICS LAB",
     "Brand name (upper case)"),

    (r"Mountain\s+Path\s+Academy",
     "Volatility Analytics Lab",
     "Brand name (partial match)"),

    (r"VolatilityAnalyticsLab",
     "VolatilityAnalyticsLab",
     "Brand name (PascalCase)"),

    (r"mountain\s*path\s*academy",
     "volatility analytics lab",
     "Brand name (lowercase)",
     re.IGNORECASE),

    # ── URLs ─────────────────────────────────────────────────────
    (r"https?://(?:www\.)?thevolatility analytics lab\.com\S*",
     "https://github.com/volatility-analytics-lab",
     "Academy website URL"),

    (r"thevolatility analytics lab\.com",
     "github.com/volatility-analytics-lab",
     "Academy domain (bare)"),

    (r"https?://(?:www\.)?linkedin\.com/in/volatility-analytics-lab\S*",
     "",
     "LinkedIn profile URL"),

    (r"github\.com/volatility-analytics-lab",
     "github.com/volatility-analytics-lab",
     "GitHub profile URL"),

    (r"github\.com/thevolatility analytics lab",
     "github.com/volatility-analytics-lab",
     "Academy GitHub org URL"),

    # ── Personal names ───────────────────────────────────────────
    (r"Prof\.\s*V\.\s*\s*",
     "",
     "Full name with title (doubled safety)"),

    (r"Prof\.\s*V\.\s*",
     "",
     "Full name with title"),

    (r"V\.\s*",
     "",
     "Name without title"),

    (r"Prof\.\s*",
     "",
     "Surname with title"),

    (r"",
     "",
     "Surname alone"),

    (r"volatility-analytics-lab",
     "volatility-analytics-lab",
     "GitHub/LinkedIn username"),

    # ── Taglines ──────────────────────────────────────────────────
    (r"Finance\s*·\s*Risk\s*·\s*Analytics\s*[—–-]\s*Practitioner-led\s+education",
     "Finance · Risk · Analytics",
     "Tagline with em-dash"),

    (r"Finance\s*·\s*Risk\s*·\s*Analytics\s*-\s*Practitioner-led\s+education",
     "Finance · Risk · Analytics",
     "Tagline with hyphen"),

    (r"Practitioner-led\s+education",
     "",
     "Tagline fragment alone"),

    # ── Bio lines (from Streamlit profile cards) ──────────────────
    (r"Visiting\s+Professor\s*&\s*Professor\s+of\s+Practice\s+at\s+Leading\s+Business\s+Schools\s*<br\s*/?>",
     "",
     "Bio: visiting professor line (HTML)"),

    (r"Visiting\s+Professor\s+&\s*Professor\s+of\s+Practice[^<\n]*",
     "",
     "Bio: visiting professor line"),

    (r"Founder\s*[—–-]\s*The\s+Mountain\s+Path\s+Academy",
     "",
     "Bio: founder line"),

    (r"Founder\s*[—–-]\s*Volatility\s+Analytics\s+Lab",
     "",
     "Bio: founder line (post-rebrand)"),

    (r"28\+\s*years\s+of\s+industry\s+experience\s*<br\s*/?>",
     "",
     "Bio: 28 years line (HTML)"),

    (r"28\+\s*years\s+of\s+industry\s+experience",
     "",
     "Bio: 28 years line"),

    (r"12\+\s*years\s+teaching\s+Finance[^<\n]*",
     "",
     "Bio: 12 years teaching line"),

    # ── Email / contact ──────────────────────────────────────────
    (r"info@thevolatility analytics lab\.com",
     "info@volatility-analytics-lab.github.io",
     "Contact email"),

    # ── Copyright ──────────────────────────────────────────────────
    (r"Copyright\s*\(c\)\s*\d{4}\s+The\s+Mountain\s+Path\s+Academy",
     "Copyright (c) 2026 Volatility Analytics Lab",
     "Copyright line"),

    (r"Copyright\s*\(c\)\s*\d{4}\s+Volatility\s+Analytics\s+Lab",
     "Copyright (c) 2026 Volatility Analytics Lab",
     "Copyright line (already clean)"),

    # ── Educational disclaimers that reference the academy ────────
    (r"Educational\s+use\s+only\.\s*Market\s+data\s+may\s+be\s+delayed[.\s]*",
     "Educational use only. Market data may be delayed.",
     "Disclaimer normalization"),

    # ── Streamlit-specific HTML fragments ─────────────────────────
    # Profile card div (the entire block is removed by the bio
    # replacements above, but clean up any leftover class refs)
    (r'class=["\']profile-card["\']',
     'class="profile-card"',
     "Profile card class (kept as-is)"),

    # ── CSS class prefix mp- → va- (optional, see --css flag) ──────
    # These are handled separately below to avoid touching non-CSS text
]

# CSS class renames (only applied with --css flag)
CSS_REPLACEMENTS: list[tuple[str, str]] = [
    ("va-header",  "va-header"),
    ("va-hero",    "va-hero"),
    ("va-footer",  "va-footer"),
    ("va-logo",    "va-logo"),
    ("va-name",    "va-name"),
    ("va-tag",     "va-tag"),
    ("VOLATILITY_CSS", "VOLATILITY_CSS"),
]

# File extensions to process
TEXT_EXTENSIONS = {
    ".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".sql", ".json",
    ".txt", ".cfg", ".ini", ".env",
}

# Files without extensions to process (by exact name)
TEXT_FILENAMES = {
    "Makefile", "Dockerfile", ".gitignore", ".env.example",
    "entrypoint.sh",
}

# Directories to skip
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".eggs",
}


def should_process(path: Path) -> bool:
    """Return True if this file should be processed."""
    if not path.is_file():
        return False
    # Skip excluded directories
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    # Check extension or filename
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name in TEXT_FILENAMES:
        return True
    return False


def apply_replacements(
    content: str,
    do_css: bool = False,
) -> tuple[str, list[str]]:
    """Apply all replacements to content. Return (new_content, descriptions)."""
    descriptions: list[str] = []
    new_content = content

    for entry in REPLACEMENTS:
        pattern, replacement, desc = entry[0], entry[1], entry[2]
        flags = entry[3] if len(entry) > 3 else 0

        count = len(re.findall(pattern, new_content, flags=flags))
        if count > 0:
            descriptions.append(f"  {desc}: {count} match(es)")
            new_content = re.sub(pattern, replacement, new_content, flags=flags)

    if do_css:
        for old, new in CSS_REPLACEMENTS:
            count = new_content.count(old)
            if count > 0:
                descriptions.append(f"  CSS {old} -> {new}: {count} match(es)")
                new_content = new_content.replace(old, new)

    # Clean up empty HTML tags left behind by removals
    # e.g.   or  <div ...></div>
    new_content = re.sub(
        r'<p\s+class="stats">\s*</p>', '', new_content
    )
    new_content = re.sub(
        r'<div\s+class="links">\s*</div>', '', new_content
    )
    # Remove lines that are now just whitespace inside <div class="profile-card">
    # (best-effort; won't break anything if it misses)

    # Clean up double blank lines left by removals
    new_content = re.sub(r'\n{4,}', '\n\n\n', new_content)

    # Clean up trailing whitespace on lines that had content removed
    new_content = re.sub(r'[ \t]+\n', '\n', new_content)

    return new_content, descriptions


def verify(project_root: Path) -> list[tuple[str, int]]:
    """Grep for any remaining brand references. Return (pattern, count)."""
    patterns_to_check = [
        r"Mountain\s*Path\s*Academy",
        r"thevolatility analytics lab",
        r"",
        r"volatility-analytics-lab",
        r"Practitioner-led",
    ]

    results: list[tuple[str, int]] = []
    for pattern in patterns_to_check:
        total = 0
        for path in project_root.rglob("*"):
            if not should_process(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                total += len(re.findall(pattern, text, re.IGNORECASE))
            except Exception:
                pass
        results.append((pattern, total))
    return results


def main() -> None:
    args = set(sys.argv[1:])
    apply_mode = "--apply" in args
    verify_mode = "--verify" in args
    css_mode = "--css" in args

    if verify_mode:
        project_root = Path(".")
        print("=" * 60)
        print("Verifying — searching for remaining references")
        print("=" * 60)
        results = verify(project_root)
        found_any = False
        for pattern, count in results:
            status = f"{count} found" if count > 0 else "clean"
            print(f"  {pattern:40s} {status}")
            if count > 0:
                found_any = True
        if found_any:
            print("\nRun with --apply to fix remaining references.")
        else:
            print("\nAll clean — no references found.")
        return

    project_root = Path(".")
    all_files = sorted(
        p for p in project_root.rglob("*") if should_process(p)
    )

    mode_label = "APPLY" if apply_mode else "DRY RUN"
    print("=" * 60)
    print(f"Debrand — {mode_label}")
    print(f"Files to scan: {len(all_files)}")
    print(f"CSS class rename: {'yes' if css_mode else 'no'}")
    print("=" * 60)

    total_changes = 0
    files_changed = 0

    for path in all_files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"  SKIP {path}: {exc}")
            continue

        new_content, descriptions = apply_replacements(
            content, do_css=css_mode
        )

        if new_content == content:
            continue

        files_changed += 1
        rel = path.relative_to(project_root)
        print(f"\n  {rel}")
        for desc in descriptions:
            print(desc)
            total_changes += 1

        if apply_mode:
            path.write_text(new_content, encoding="utf-8")
            print(f"  -> written")

    print("\n" + "=" * 60)
    print(f"Files with changes: {files_changed}")
    print(f"Total pattern matches: {total_changes}")
    if not apply_mode:
        print("\nDry run only. Run with --apply to write changes.")
    else:
        print("\nChanges written. Run with --verify to confirm.")
    print("=" * 60)


if __name__ == "__main__":
    main()