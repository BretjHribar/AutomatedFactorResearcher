import fitz

doc = fitz.open('references/quantitative-portfolio-management-the-art-and-science-of-statistical-arbitrage.pdf')

# From TOC: 6.6 Portfolio capacity is on page 205 of the book
# Book page 205 = PDF page ~236 (31 page offset for front matter)
# Let me search for it
for i in range(230, 260):
    if i < len(doc):
        text = doc[i].get_text()
        if 'capacity' in text.lower() or '6.6' in text:
            with open(f'_cap_page_{i+1}.txt', 'w', encoding='utf-8') as f:
                f.write(text)
            print(f'Page {i+1}: Found capacity/6.6 mention')
            # print first 200 chars
            print(f'  {text[:200].strip()}')
