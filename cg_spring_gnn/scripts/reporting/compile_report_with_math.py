import os
import sys

# Ensure project root (cg_spring_gnn) is on sys.path and is current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import re
import markdown
import subprocess

md_path = r"C:\Users\sriva\.gemini\antigravity-ide\brain\029fbcbc-bb91-4a30-9c3b-37696e486a0c\full_project_report_and_study_guide.md"
html_path = r"C:\Users\sriva\.gemini\antigravity-ide\brain\029fbcbc-bb91-4a30-9c3b-37696e486a0c\scratch\report.html"
pdf_path = r"c:\Users\sriva\OneDrive\Desktop\sem7\ee798\full_project_report_and_study_guide.pdf"

print("Reading master markdown...")
with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Clean up any double backslashes in LaTeX equations (e.g. \\text -> \text)
text = text.replace(r"\\text", r"\text")
text = text.replace(r"\\mathbf", r"\mathbf")
text = text.replace(r"\\mathbb", r"\mathbb")
text = text.replace(r"\\mathcal", r"\mathcal")
text = text.replace(r"\\frac", r"\frac")
text = text.replace(r"\\sum", r"\sum")
text = text.replace(r"\\Delta", r"\Delta")
text = text.replace(r"\\mu", r"\mu")
text = text.replace(r"\\sigma", r"\sigma")
text = text.replace(r"\\approx", r"\approx")
text = text.replace(r"\\le", r"\le")
text = text.replace(r"\\ge", r"\ge")
text = text.replace(r"\\log", r"\log")
text = text.replace(r"\\parallel", r"\parallel")
text = text.replace(r"\\left", r"\left")
text = text.replace(r"\\right", r"\right")
text = text.replace(r"\\hat", r"\hat")
text = text.replace(r"\\bar", r"\bar")

# Save cleaned markdown
with open(md_path, "w", encoding="utf-8") as f:
    f.write(text)

# 2. Protect LaTeX math expressions from markdown parser mangling
math_blocks = []

def protect_display(m):
    idx = len(math_blocks)
    math_blocks.append(m.group(0))
    return f"<!--MATH_DISP_{idx}_BLOCK-->"

def protect_inline(m):
    idx = len(math_blocks)
    math_blocks.append(m.group(0))
    return f"<!--MATH_INL_{idx}_BLOCK-->"

# Protect display math $$...$$
protected_text = re.sub(r"\$\$(.+?)\$\$", protect_display, text, flags=re.DOTALL)

# Protect inline math $...$ (avoid currency or empty matches)
protected_text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", protect_inline, protected_text)

print(f"Protected {len(math_blocks)} math expressions from markdown parser.")

# 3. Format local image paths for Chrome
protected_text = protected_text.replace(
    r"C:/Users/sriva/.gemini/antigravity-ide/brain/029fbcbc-bb91-4a30-9c3b-37696e486a0c/",
    "file:///C:/Users/sriva/.gemini/antigravity-ide/brain/029fbcbc-bb91-4a30-9c3b-37696e486a0c/"
)

# 4. Render Markdown to HTML
html_body = markdown.markdown(protected_text, extensions=['tables', 'fenced_code', 'toc'])

# 5. Restore LaTeX expressions
for idx, block in enumerate(math_blocks):
    html_body = html_body.replace(f"&lt;!--MATH_DISP_{idx}_BLOCK--&gt;", block)
    html_body = html_body.replace(f"<!--MATH_DISP_{idx}_BLOCK-->", block)
    html_body = html_body.replace(f"&lt;!--MATH_INL_{idx}_BLOCK--&gt;", block)
    html_body = html_body.replace(f"<!--MATH_INL_{idx}_BLOCK-->", block)

print("Restored pristine LaTeX expressions into HTML.")

# 6. Build Complete High-Fidelity HTML with KaTeX + MathJax Styling
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MARTINI 3 CG Force Field GNN - Technical Report</title>

<!-- KaTeX CSS & JS for Lightning-Fast Crisp Vector Typography -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

<script>
window.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "\\[", right: "\\]", display: true}},
            {{left: "$", right: "$", display: false}},
            {{left: "\\(", right: "\\)", display: false}}
        ],
        throwOnError: false,
        output: "htmlAndMathml",
        strict: false
    }});
}});
</script>

<style>
@page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13.5px;
    line-height: 1.65;
    color: #1e293b;
    max-width: 960px;
    margin: 0 auto;
    padding: 20px;
    background-color: #ffffff;
}}

h1, h2, h3, h4, h5 {{
    color: #0f172a;
    font-weight: 700;
    line-height: 1.3;
    page-break-after: avoid;
}}

h1 {{
    font-size: 2.1em;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 0.3em;
    margin-top: 10px;
    margin-bottom: 20px;
    color: #1e3a8a;
}}

h2 {{
    font-size: 1.5em;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 0.25em;
    margin-top: 32px;
    margin-bottom: 16px;
    color: #1e40af;
}}

h3 {{
    font-size: 1.25em;
    margin-top: 24px;
    margin-bottom: 12px;
    color: #0f766e;
}}

h4 {{
    font-size: 1.1em;
    margin-top: 18px;
    margin-bottom: 8px;
    color: #334155;
}}

p {{
    margin: 0.8em 0;
}}

/* Crisp Mathematics Styling */
.katex {{
    font-size: 1.08em !important;
}}

.katex-display {{
    margin: 1.4em 0 !important;
    padding: 0.8em 1.2em !important;
    background: #f8fafc;
    border-radius: 6px;
    border-left: 4px solid #3b82f6;
    overflow-x: auto;
    page-break-inside: avoid;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin: 18px 0;
    font-size: 12.5px;
    page-break-inside: avoid;
}}

table th, table td {{
    border: 1px solid #cbd5e1;
    padding: 8px 12px;
    text-align: left;
}}

table th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
    border-bottom: 2px solid #94a3b8;
}}

table tr:nth-child(even) {{
    background-color: #f8fafc;
}}

code {{
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    font-size: 88%;
    padding: 0.15em 0.35em;
    font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    color: #b91c1c;
}}

pre {{
    background-color: #0f172a;
    color: #f8fafc;
    border-radius: 6px;
    font-size: 12px;
    line-height: 1.5;
    overflow-x: auto;
    padding: 14px 18px;
    margin: 16px 0;
    page-break-inside: avoid;
}}

pre code {{
    background: transparent;
    border: none;
    padding: 0;
    color: #e2e8f0;
}}

img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    margin: 16px auto;
    display: block;
    page-break-inside: avoid;
}}

blockquote {{
    border-left: 4px solid #3b82f6;
    background-color: #eff6ff;
    color: #1e3a8a;
    padding: 10px 18px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
}}

hr {{
    height: 2px;
    background: #e2e8f0;
    border: none;
    margin: 30px 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"HTML written to {html_path} ({len(full_html)} chars)")

# 7. Convert HTML to PDF using Chrome Headless with Virtual Time Budget
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
cmd = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    "--virtual-time-budget=10000",
    f"--print-to-pdf={pdf_path}",
    "--no-pdf-header-footer",
    html_path
]

print("Executing Chrome headless PDF conversion...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Chrome Return Code:", res.returncode)

if os.path.exists(pdf_path):
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"SUCCESS: Generated {pdf_path} ({size_mb:.2f} MB)")
else:
    print("FAILED to generate PDF:", res.stderr)
