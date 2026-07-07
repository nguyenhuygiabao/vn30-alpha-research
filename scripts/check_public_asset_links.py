from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, unquote


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

ASSET_TAGS = {
    "img": ["src"],
    "iframe": ["src"],
    "script": ["src"],
    "link": ["href"],
    "a": ["href"],
}


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for attr in ASSET_TAGS.get(tag, []):
            value = attrs.get(attr)
            if value:
                self.links.append(value)


def should_check(value):
    value = value.strip()
    if not value:
        return False

    lower = value.lower()

    if lower.startswith(("http://", "https://", "mailto:", "javascript:", "data:", "#")):
        return False

    return lower.endswith((".html", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".css", ".js"))


def main():
    missing = []

    for html_path in REPORTS.rglob("*.html"):
        parser = AssetParser()
        parser.feed(html_path.read_text(encoding="utf-8", errors="replace"))

        for raw_link in parser.links:
            clean_link = raw_link.split("#", 1)[0].split("?", 1)[0]
            if not should_check(clean_link):
                continue

            parsed = urlparse(clean_link)
            rel = unquote(parsed.path)
            target = (html_path.parent / rel).resolve()

            if not target.exists():
                missing.append((html_path.relative_to(ROOT), raw_link, target.relative_to(ROOT) if ROOT in target.parents else target))

    if missing:
        print("Missing public assets found:")
        for page, raw_link, target in missing:
            print(f"  page:   {page}")
            print(f"  link:   {raw_link}")
            print(f"  target: {target}")
            print()
        raise SystemExit(1)

    print("Public asset link check passed. No missing local HTML/image assets found.")


if __name__ == "__main__":
    main()
