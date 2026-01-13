"""
EPUB generation from markdown content.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import markdown

logger = logging.getLogger(__name__)


@dataclass
class EPUBMetadata:
    """Metadata for EPUB file."""

    title: str
    author: str = "Unknown"
    language: str = "en"
    identifier: str = ""
    publisher: str = ""
    description: str = ""
    date: str = ""

    def __post_init__(self) -> None:
        if not self.identifier:
            self.identifier = f"urn:uuid:{uuid.uuid4()}"
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d")


class EPUBBuilder:
    """Builds EPUB files from markdown content."""

    def __init__(self, metadata: EPUBMetadata) -> None:
        """Initialize EPUB builder.

        Args:
            metadata: Book metadata
        """
        self.metadata = metadata
        self.md_converter = markdown.Markdown(
            extensions=['tables', 'fenced_code', 'toc']
        )

    def build(
        self,
        markdown_content: str,
        output_path: Path,
        split_chapters: bool = True,
    ) -> Path:
        """Build EPUB from markdown content.

        Args:
            markdown_content: Full markdown text
            output_path: Where to save the EPUB
            split_chapters: Whether to split into chapters

        Returns:
            Path to created EPUB file
        """
        import zipfile

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Split content into chapters if requested
        if split_chapters:
            chapters = self._split_into_chapters(markdown_content)
        else:
            chapters = [("Content", markdown_content)]

        # Create EPUB structure
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as epub:
            # Mimetype must be first and uncompressed
            epub.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)

            # META-INF/container.xml
            epub.writestr('META-INF/container.xml', self._container_xml())

            # OEBPS/content.opf
            chapter_ids = [f"chapter{i}" for i in range(len(chapters))]
            epub.writestr('OEBPS/content.opf', self._content_opf(chapter_ids))

            # OEBPS/toc.ncx
            epub.writestr('OEBPS/toc.ncx', self._toc_ncx(chapters))

            # OEBPS/nav.xhtml (EPUB3 navigation)
            epub.writestr('OEBPS/nav.xhtml', self._nav_xhtml(chapters))

            # OEBPS/stylesheet.css
            epub.writestr('OEBPS/stylesheet.css', self._stylesheet())

            # Chapter XHTML files
            for i, (title, content) in enumerate(chapters):
                html_content = self._markdown_to_xhtml(content, title)
                epub.writestr(f'OEBPS/chapter{i}.xhtml', html_content)

        logger.info(f"Created EPUB with {len(chapters)} chapters: {output_path}")
        return output_path

    def _split_into_chapters(self, content: str) -> list[tuple[str, str]]:
        """Split markdown content into chapters.

        Looks for:
        - # Header (H1)
        - ## Chapter X
        - Chapter X markers

        Args:
            content: Full markdown text

        Returns:
            List of (title, content) tuples
        """
        # Pattern for chapter/section markers
        chapter_pattern = re.compile(
            r'^(#{1,2}\s+.+|Chapter\s+\d+.*|CHAPTER\s+\d+.*|Part\s+\d+.*)$',
            re.MULTILINE | re.IGNORECASE
        )

        # Find all chapter markers
        matches = list(chapter_pattern.finditer(content))

        if not matches:
            # No chapters found, return as single chapter
            return [("Content", content)]

        chapters = []

        for i, match in enumerate(matches):
            title = match.group(1).strip().lstrip('#').strip()

            # Get content from this marker to the next (or end)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            chapter_content = content[start:end].strip()

            # Skip very short "chapters" (likely just headers)
            if len(chapter_content) > 100:
                chapters.append((title, chapter_content))

        # If we didn't get any substantial chapters, return everything
        if not chapters:
            return [("Content", content)]

        # Handle content before first chapter marker
        if matches and matches[0].start() > 100:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                chapters.insert(0, ("Introduction", preamble))

        return chapters

    def _markdown_to_xhtml(self, markdown_text: str, title: str) -> str:
        """Convert markdown to valid XHTML.

        Args:
            markdown_text: Markdown content
            title: Chapter title

        Returns:
            XHTML string
        """
        # Reset converter state
        self.md_converter.reset()

        # Convert markdown to HTML
        html_body = self.md_converter.convert(markdown_text)

        # Wrap in XHTML document
        xhtml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{self.metadata.language}">
<head>
    <meta charset="UTF-8"/>
    <title>{self._escape_xml(title)}</title>
    <link rel="stylesheet" type="text/css" href="stylesheet.css"/>
</head>
<body>
{html_body}
</body>
</html>'''

        return xhtml

    def _container_xml(self) -> str:
        """Generate META-INF/container.xml."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''

    def _content_opf(self, chapter_ids: list[str]) -> str:
        """Generate OEBPS/content.opf (package document)."""
        manifest_items = ['<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>']
        manifest_items.append('<item id="css" href="stylesheet.css" media-type="text/css"/>')

        spine_items = []

        for chap_id in chapter_ids:
            manifest_items.append(
                f'<item id="{chap_id}" href="{chap_id}.xhtml" media-type="application/xhtml+xml"/>'
            )
            spine_items.append(f'<itemref idref="{chap_id}"/>')

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="BookId">{self._escape_xml(self.metadata.identifier)}</dc:identifier>
        <dc:title>{self._escape_xml(self.metadata.title)}</dc:title>
        <dc:creator>{self._escape_xml(self.metadata.author)}</dc:creator>
        <dc:language>{self.metadata.language}</dc:language>
        <dc:date>{self.metadata.date}</dc:date>
        <meta property="dcterms:modified">{datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")}</meta>
    </metadata>
    <manifest>
        {chr(10).join(manifest_items)}
    </manifest>
    <spine>
        {chr(10).join(spine_items)}
    </spine>
</package>'''

    def _toc_ncx(self, chapters: list[tuple[str, str]]) -> str:
        """Generate OEBPS/toc.ncx (for EPUB2 compatibility)."""
        nav_points = []
        for i, (title, _) in enumerate(chapters):
            nav_points.append(f'''
        <navPoint id="navpoint{i}" playOrder="{i+1}">
            <navLabel><text>{self._escape_xml(title)}</text></navLabel>
            <content src="chapter{i}.xhtml"/>
        </navPoint>''')

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{self._escape_xml(self.metadata.identifier)}"/>
        <meta name="dtb:depth" content="1"/>
        <meta name="dtb:totalPageCount" content="0"/>
        <meta name="dtb:maxPageNumber" content="0"/>
    </head>
    <docTitle><text>{self._escape_xml(self.metadata.title)}</text></docTitle>
    <navMap>
        {''.join(nav_points)}
    </navMap>
</ncx>'''

    def _nav_xhtml(self, chapters: list[tuple[str, str]]) -> str:
        """Generate OEBPS/nav.xhtml (EPUB3 navigation)."""
        nav_items = []
        for i, (title, _) in enumerate(chapters):
            nav_items.append(f'<li><a href="chapter{i}.xhtml">{self._escape_xml(title)}</a></li>')

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{self.metadata.language}">
<head>
    <meta charset="UTF-8"/>
    <title>Table of Contents</title>
    <link rel="stylesheet" type="text/css" href="stylesheet.css"/>
</head>
<body>
    <nav epub:type="toc">
        <h1>Table of Contents</h1>
        <ol>
            {chr(10).join(nav_items)}
        </ol>
    </nav>
</body>
</html>'''

    def _stylesheet(self) -> str:
        """Generate default stylesheet."""
        return '''body {
    font-family: Georgia, serif;
    line-height: 1.6;
    margin: 1em;
    text-align: justify;
}

h1, h2, h3, h4, h5, h6 {
    font-family: Helvetica, Arial, sans-serif;
    line-height: 1.3;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

h1 { font-size: 1.8em; }
h2 { font-size: 1.5em; }
h3 { font-size: 1.3em; }

p {
    margin: 0.5em 0;
    text-indent: 1.5em;
}

p:first-of-type {
    text-indent: 0;
}

blockquote {
    margin: 1em 2em;
    font-style: italic;
    border-left: 3px solid #ccc;
    padding-left: 1em;
}

code {
    font-family: "Courier New", monospace;
    font-size: 0.9em;
    background: #f4f4f4;
    padding: 0.1em 0.3em;
}

pre {
    font-family: "Courier New", monospace;
    font-size: 0.85em;
    background: #f4f4f4;
    padding: 1em;
    overflow-x: auto;
    white-space: pre-wrap;
}

table {
    border-collapse: collapse;
    margin: 1em 0;
    width: 100%;
}

th, td {
    border: 1px solid #ccc;
    padding: 0.5em;
    text-align: left;
}

th {
    background: #f0f0f0;
}
'''

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
