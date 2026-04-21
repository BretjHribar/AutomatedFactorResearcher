import fitz
import sys

doc = fitz.open('references/quantitative-portfolio-management-the-art-and-science-of-statistical-arbitrage.pdf')

# Extract Chapter 5 (Trading Costs & Market Elasticity) and Chapter 6 (Portfolio Capacity)
# Based on TOC: Chapter 5 starts page 172 (PDF p209), Chapter 6 starts ~page 190
# Focus on: 5.2 Impact, 5.2.2 Linear impact model, 5.4 Inelasticity, 6.6 Portfolio capacity

target_pages = list(range(208, 245))  # pages 209-245 (0-indexed: 208-244)

with open('_isichenko_capacity.txt', 'w', encoding='utf-8') as f:
    for i in target_pages:
        if i < len(doc):
            page = doc[i]
            text = page.get_text()
            f.write(f"\n{'='*80}\nPDF PAGE {i+1}\n{'='*80}\n")
            f.write(text)

print(f"Extracted {len(target_pages)} pages to _isichenko_capacity.txt")
