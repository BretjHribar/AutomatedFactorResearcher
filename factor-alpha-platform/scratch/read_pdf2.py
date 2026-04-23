import fitz
import sys

doc = fitz.open(r'references/ssrn-4388526.pdf')
for i in range(20, min(50, len(doc))):
    text = doc[i].get_text()
    sys.stdout.buffer.write(f"\n=== PAGE {i+1} ===\n".encode('utf-8'))
    sys.stdout.buffer.write(text.encode('utf-8', errors='replace'))
    sys.stdout.buffer.write(b'\n')
