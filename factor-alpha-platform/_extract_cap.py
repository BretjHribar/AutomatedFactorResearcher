import fitz

doc = fitz.open('references/quantitative-portfolio-management-the-art-and-science-of-statistical-arbitrage.pdf')

with open('_capacity_section.txt', 'w', encoding='utf-8') as f:
    for i in range(238, 252):
        if i < len(doc):
            text = doc[i].get_text()
            f.write(f'\n{"="*80}\nPDF PAGE {i+1}\n{"="*80}\n')
            f.write(text)

print("Done - extracted pages 239-252")
