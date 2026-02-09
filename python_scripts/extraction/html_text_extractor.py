import json
import trafilatura
from sentence_extractor import extract_sentences

html = """
<html>
    <head>
        <title>Sample Page</title>
        <style>.hidden{display:none}</style>
        <script>console.log("noise")</script>
    </head>
    <body>
        <nav>Home | About | Contact</nav>
        <article>
            <h1>Extracting Text from HTML</h1>
            <p>This is the first sentence.</p>
            <p>This pipeline removes boilerplate and splits sentences correctly.</p>
            <p>It works well for scraped content!</p>
        </article>
        <footer>© 2026 Example Corp</footer>
    </body>
</html>
"""

def extract_html_text(html: str) -> str:
    return trafilatura.extract(html)

if __name__ == "__main__":
    html_text = extract_html_text(html)
    print(f"\nHTML Text:\n{html_text}")

    html_sentences = extract_sentences(html_text, valid_only=True)
    print(f"Sentences ({len(html_sentences)})\n{json.dumps(html_sentences, indent=2)}")
