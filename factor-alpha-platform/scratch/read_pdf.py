import fitz
import sys

doc = fitz.open(r'references/ssrn-4388526.pdf')
print(f"Total pages: {len(doc)}")
for i in range(min(20, len(doc))):
    text = doc[i].get_text()
    sys.stdout.buffer.write(f"\n=== PAGE {i+1} ===\n".encode('utf-8'))
    sys.stdout.buffer.write(text.encode('utf-8', errors='replace'))
    sys.stdout.buffer.write(b'\n')
