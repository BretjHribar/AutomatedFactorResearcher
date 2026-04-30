"""
Build GICS-based classifications from FMP profile data.

Maps FMP sector/industry names to official GICS codes:
  - Sector (2-digit): 10-60
  - Industry Group (4-digit): 1010-6010
  - Industry (6-digit): 101010-601010
  - Sub-Industry (8-digit): 10101010-60101040

The FMP industry names map closely to GICS sub-industries.
"""
import json
import os

# ─── GICS Sector Mapping (FMP sector name → GICS sector code + name) ───
FMP_SECTOR_TO_GICS = {
    "Technology":             (45, "Information Technology"),
    "Healthcare":             (35, "Health Care"),
    "Financial Services":     (40, "Financials"),
    "Consumer Cyclical":      (25, "Consumer Discretionary"),
    "Consumer Defensive":     (30, "Consumer Staples"),
    "Industrials":            (20, "Industrials"),
    "Energy":                 (10, "Energy"),
    "Basic Materials":        (15, "Materials"),
    "Real Estate":            (60, "Real Estate"),
    "Communication Services": (50, "Communication Services"),
    "Utilities":              (55, "Utilities"),
}

# ─── FMP Industry → GICS Sub-Industry Mapping ───
# Format: "FMP Industry Name": (GICS_sub_industry_code, "GICS Sub-Industry Name",
#                                GICS_industry_code, "GICS Industry Name",
#                                GICS_industry_group_code, "GICS Industry Group Name")
FMP_INDUSTRY_TO_GICS = {
    # ══════════════════════════════════════════════════════════════
    # ENERGY (10)
    # ══════════════════════════════════════════════════════════════
    "Oil & Gas Integrated":              (10102010, "Integrated Oil & Gas",       101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),
    "Oil & Gas Exploration & Production":(10102020, "Oil & Gas Exploration & Production", 101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),
    "Oil & Gas Refining & Marketing":    (10102030, "Oil & Gas Refining & Marketing", 101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),
    "Oil & Gas Midstream":               (10102040, "Oil & Gas Storage & Transportation", 101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),
    "Oil & Gas Equipment & Services":    (10101020, "Oil & Gas Equipment & Services", 101010, "Energy Equipment & Services", 1010, "Energy"),
    "Oil & Gas Drilling":                (10101010, "Oil & Gas Drilling",          101010, "Energy Equipment & Services", 1010, "Energy"),
    "Oil & Gas Energy":                  (10102020, "Oil & Gas Exploration & Production", 101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),
    "Coal":                              (10102050, "Coal & Consumable Fuels",     101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),
    "Solar":                             (10102060, "Renewable Energy",            101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),
    "Uranium":                           (10102050, "Coal & Consumable Fuels",     101020, "Oil, Gas & Consumable Fuels", 1010, "Energy"),

    # ══════════════════════════════════════════════════════════════
    # MATERIALS (15)
    # ══════════════════════════════════════════════════════════════
    "Chemicals":                         (15101010, "Commodity Chemicals",         151010, "Chemicals", 1510, "Materials"),
    "Chemicals - Specialty":             (15101050, "Specialty Chemicals",         151010, "Chemicals", 1510, "Materials"),
    "Agricultural Inputs":               (15101030, "Fertilizers & Agricultural Chemicals", 151010, "Chemicals", 1510, "Materials"),
    "Construction Materials":            (15102010, "Construction Materials",      151020, "Construction Materials", 1510, "Materials"),
    "Gold":                              (15104030, "Gold",                        151040, "Metals & Mining", 1510, "Materials"),
    "Steel":                             (15104050, "Steel",                       151040, "Metals & Mining", 1510, "Materials"),
    "Aluminum":                          (15104010, "Aluminum",                    151040, "Metals & Mining", 1510, "Materials"),
    "Copper":                            (15104020, "Copper",                      151040, "Metals & Mining", 1510, "Materials"),
    "Other Precious Metals":             (15104040, "Precious Metals & Minerals",  151040, "Metals & Mining", 1510, "Materials"),
    "Industrial Materials":              (15104025, "Diversified Metals & Mining",  151040, "Metals & Mining", 1510, "Materials"),
    "Paper, Lumber & Forest Products":   (15105020, "Paper Products",              151050, "Paper & Forest Products", 1510, "Materials"),

    # ══════════════════════════════════════════════════════════════
    # INDUSTRIALS (20)
    # ══════════════════════════════════════════════════════════════
    "Aerospace & Defense":               (20101010, "Aerospace & Defense",         201010, "Aerospace & Defense", 2010, "Capital Goods"),
    "Construction":                      (20103010, "Construction & Engineering",  201030, "Construction & Engineering", 2010, "Capital Goods"),
    "Engineering & Construction":        (20103010, "Construction & Engineering",  201030, "Construction & Engineering", 2010, "Capital Goods"),
    "Electrical Equipment & Parts":      (20104010, "Electrical Components & Equipment", 201040, "Electrical Equipment", 2010, "Capital Goods"),
    "Conglomerates":                     (20105010, "Industrial Conglomerates",    201050, "Industrial Conglomerates", 2010, "Capital Goods"),
    "Industrial - Machinery":            (20106010, "Construction Machinery & Heavy Transportation Equipment", 201060, "Machinery", 2010, "Capital Goods"),
    "Agricultural - Machinery":          (20106020, "Agricultural & Farm Machinery", 201060, "Machinery", 2010, "Capital Goods"),
    "Manufacturing - Tools & Accessories":(20106015, "Industrial Machinery & Supplies & Components", 201060, "Machinery", 2010, "Capital Goods"),
    "Manufacturing - Metal Fabrication": (20106015, "Industrial Machinery & Supplies & Components", 201060, "Machinery", 2010, "Capital Goods"),
    "Manufacturing - Textiles":          (20106015, "Industrial Machinery & Supplies & Components", 201060, "Machinery", 2010, "Capital Goods"),
    "Business Equipment & Supplies":     (20107010, "Trading Companies & Distributors", 201070, "Trading Companies & Distributors", 2010, "Capital Goods"),
    "Specialty Business Services":       (20201070, "Diversified Support Services", 202010, "Commercial Services & Supplies", 2020, "Commercial & Professional Services"),
    "Waste Management":                  (20201050, "Environmental & Facilities Services", 202010, "Commercial Services & Supplies", 2020, "Commercial & Professional Services"),
    "Security & Protection Services":    (20201060, "Security & Alarm Services",   202010, "Commercial Services & Supplies", 2020, "Commercial & Professional Services"),
    "Consulting Services":               (20202020, "Research & Consulting Services", 202020, "Professional Services", 2020, "Commercial & Professional Services"),
    "Staffing & Employment Services":    (20202010, "Human Resource & Employment Services", 202020, "Professional Services", 2020, "Commercial & Professional Services"),
    "Airlines, Airports & Air Services": (20301010, "Air Freight & Logistics",     203010, "Air Freight & Logistics", 2030, "Transportation"),
    "Integrated Freight & Logistics":    (20301010, "Air Freight & Logistics",     203010, "Air Freight & Logistics", 2030, "Transportation"),
    "Marine Shipping":                   (20303010, "Marine Transportation",        203030, "Marine Transportation", 2030, "Transportation"),
    "Railroads":                         (20304010, "Rail Transportation",          203040, "Road & Rail", 2030, "Transportation"),
    "Trucking":                          (20304020, "Trucking",                     203040, "Road & Rail", 2030, "Transportation"),
    "Industrial - Distribution":         (20107010, "Trading Companies & Distributors", 201070, "Trading Companies & Distributors", 2010, "Capital Goods"),
    "Industrial - Infrastructure Operations": (20105010, "Industrial Conglomerates", 201050, "Industrial Conglomerates", 2010, "Capital Goods"),
    "Industrial - Pollution & Treatment Controls": (20201050, "Environmental & Facilities Services", 202010, "Commercial Services & Supplies", 2020, "Commercial & Professional Services"),
    "Rental & Leasing Services":         (20201070, "Diversified Support Services", 202010, "Commercial Services & Supplies", 2020, "Commercial & Professional Services"),
    "Packaging & Containers":            (15103020, "Paper Packaging",              151030, "Containers & Packaging", 1510, "Materials"),
    "Auto - Parts":                      (25101020, "Auto Parts & Equipment",       251010, "Automobiles & Components", 2510, "Automobiles & Components"),

    # ══════════════════════════════════════════════════════════════
    # CONSUMER DISCRETIONARY (25)
    # ══════════════════════════════════════════════════════════════
    "Auto - Manufacturers":              (25101010, "Automobile Manufacturers",     251010, "Automobiles", 2510, "Automobiles & Components"),
    "Auto - Dealerships":                (25101020, "Auto Parts & Equipment",       251010, "Automobiles", 2510, "Automobiles & Components"),
    "Auto - Recreational Vehicles":      (25101010, "Automobile Manufacturers",     251010, "Automobiles", 2510, "Automobiles & Components"),
    "Furnishings, Fixtures & Appliances":(25201030, "Home Furnishings",             252010, "Household Durables", 2520, "Consumer Durables & Apparel"),
    "Residential Construction":          (25201010, "Homebuilding",                 252010, "Household Durables", 2520, "Consumer Durables & Apparel"),
    "Leisure":                           (25202010, "Leisure Products",             252020, "Leisure Products", 2520, "Consumer Durables & Apparel"),
    "Apparel - Manufacturers":           (25203010, "Apparel, Accessories & Luxury Goods", 252030, "Textiles, Apparel & Luxury Goods", 2520, "Consumer Durables & Apparel"),
    "Apparel - Footwear & Accessories":  (25203020, "Footwear",                    252030, "Textiles, Apparel & Luxury Goods", 2520, "Consumer Durables & Apparel"),
    "Luxury Goods":                      (25203010, "Apparel, Accessories & Luxury Goods", 252030, "Textiles, Apparel & Luxury Goods", 2520, "Consumer Durables & Apparel"),
    "Travel Lodging":                    (25301010, "Casinos & Gaming",             253010, "Hotels, Restaurants & Leisure", 2530, "Consumer Services"),
    "Travel Services":                   (25301040, "Hotels, Resorts & Cruise Lines", 253010, "Hotels, Restaurants & Leisure", 2530, "Consumer Services"),
    "Gambling, Resorts & Casinos":       (25301010, "Casinos & Gaming",             253010, "Hotels, Restaurants & Leisure", 2530, "Consumer Services"),
    "Restaurants":                       (25301040, "Restaurants",                   253010, "Hotels, Restaurants & Leisure", 2530, "Consumer Services"),
    "Apparel - Retail":                  (25504010, "Apparel Retail",               255040, "Specialty Retail", 2550, "Retailing"),
    "Specialty Retail":                  (25504060, "Specialty Stores",             255040, "Specialty Retail", 2550, "Retailing"),
    "Home Improvement":                  (25504030, "Home Improvement Retail",     255040, "Specialty Retail", 2550, "Retailing"),
    "Department Stores":                 (25503020, "Department Stores",            255030, "Multiline Retail", 2550, "Retailing"),
    "Discount Stores":                   (25503010, "General Merchandise Stores",   255030, "Multiline Retail", 2550, "Retailing"),
    "Personal Products & Services":      (30302010, "Personal Care Products",       303020, "Personal Care Products", 3030, "Household & Personal Products"),
    "Electronic Gaming & Multimedia":    (25401025, "Interactive Home Entertainment", 254010, "Media", 2540, "Media & Entertainment"),

    # ══════════════════════════════════════════════════════════════
    # CONSUMER STAPLES (30)
    # ══════════════════════════════════════════════════════════════
    "Packaged Foods":                    (30202030, "Packaged Foods & Meats",       302020, "Food Products", 3020, "Food, Beverage & Tobacco"),
    "Food Distribution":                 (30202030, "Packaged Foods & Meats",       302020, "Food Products", 3020, "Food, Beverage & Tobacco"),
    "Food Confectioners":                (30202030, "Packaged Foods & Meats",       302020, "Food Products", 3020, "Food, Beverage & Tobacco"),
    "Agricultural Farm Products":        (30202010, "Agricultural Products & Services", 302020, "Food Products", 3020, "Food, Beverage & Tobacco"),
    "Beverages - Non-Alcoholic":         (30201010, "Soft Drinks & Non-alcoholic Beverages", 302010, "Beverages", 3020, "Food, Beverage & Tobacco"),
    "Beverages - Alcoholic":             (30201020, "Brewers",                      302010, "Beverages", 3020, "Food, Beverage & Tobacco"),
    "Beverages - Wineries & Distilleries":(30201030, "Distillers & Vintners",       302010, "Beverages", 3020, "Food, Beverage & Tobacco"),
    "Tobacco":                           (30203010, "Tobacco",                      302030, "Tobacco", 3020, "Food, Beverage & Tobacco"),
    "Household & Personal Products":     (30302010, "Personal Care Products",       303020, "Personal Care Products", 3030, "Household & Personal Products"),
    "Grocery Stores":                    (30101040, "Food Retail",                  301010, "Consumer Staples Distribution & Retail", 3010, "Consumer Staples Distribution & Retail"),
    "Education & Training Services":     (25301050, "Education Services",           253010, "Hotels, Restaurants & Leisure", 2530, "Consumer Services"),

    # ══════════════════════════════════════════════════════════════
    # HEALTH CARE (35)
    # ══════════════════════════════════════════════════════════════
    "Medical - Devices":                 (35101010, "Health Care Equipment",        351010, "Health Care Equipment & Supplies", 3510, "Health Care Equipment & Services"),
    "Medical - Instruments & Supplies":  (35101020, "Health Care Supplies",         351010, "Health Care Equipment & Supplies", 3510, "Health Care Equipment & Services"),
    "Medical - Equipment & Services":    (35101010, "Health Care Equipment",        351010, "Health Care Equipment & Supplies", 3510, "Health Care Equipment & Services"),
    "Medical - Distribution":            (35102020, "Health Care Distributors",     351020, "Health Care Providers & Services", 3510, "Health Care Equipment & Services"),
    "Medical - Care Facilities":         (35102010, "Health Care Facilities",       351020, "Health Care Providers & Services", 3510, "Health Care Equipment & Services"),
    "Medical - Healthcare Plans":        (35102030, "Managed Health Care",          351020, "Health Care Providers & Services", 3510, "Health Care Equipment & Services"),
    "Medical - Healthcare Information Services": (35103010, "Health Care Technology", 351030, "Health Care Technology", 3510, "Health Care Equipment & Services"),
    "Medical - Diagnostics & Research":  (35201010, "Life Sciences Tools & Services", 352010, "Life Sciences Tools & Services", 3520, "Pharmaceuticals, Biotechnology & Life Sciences"),
    "Biotechnology":                     (35201010, "Biotechnology",                352010, "Biotechnology", 3520, "Pharmaceuticals, Biotechnology & Life Sciences"),
    "Drug Manufacturers - General":      (35202010, "Pharmaceuticals",              352020, "Pharmaceuticals", 3520, "Pharmaceuticals, Biotechnology & Life Sciences"),
    "Drug Manufacturers - Specialty & Generic": (35202010, "Pharmaceuticals",       352020, "Pharmaceuticals", 3520, "Pharmaceuticals, Biotechnology & Life Sciences"),
    "Medical - Pharmaceuticals":         (35202010, "Pharmaceuticals",              352020, "Pharmaceuticals", 3520, "Pharmaceuticals, Biotechnology & Life Sciences"),

    # ══════════════════════════════════════════════════════════════
    # FINANCIALS (40)
    # ══════════════════════════════════════════════════════════════
    "Banks - Diversified":               (40101010, "Diversified Banks",            401010, "Banks", 4010, "Banks"),
    "Banks - Regional":                  (40101015, "Regional Banks",               401010, "Banks", 4010, "Banks"),
    "Financial - Capital Markets":       (40203020, "Investment Banking & Brokerage", 402030, "Capital Markets", 4020, "Diversified Financials"),
    "Financial - Data & Stock Exchanges":(40203030, "Financial Exchanges & Data",   402030, "Capital Markets", 4020, "Diversified Financials"),
    "Financial - Diversified":           (40201020, "Other Diversified Financial Services", 402010, "Diversified Financial Services", 4020, "Diversified Financials"),
    "Financial - Credit Services":       (40202010, "Consumer Finance",             402020, "Consumer Finance", 4020, "Diversified Financials"),
    "Financial - Mortgages":             (40204010, "Mortgage Finance",             402040, "Mortgage Real Estate Investment Trusts", 4020, "Diversified Financials"),
    "Financial - Conglomerates":         (40201030, "Multi-Sector Holdings",        402010, "Diversified Financial Services", 4020, "Diversified Financials"),
    "Asset Management":                  (40203010, "Asset Management & Custody Banks", 402030, "Capital Markets", 4020, "Diversified Financials"),
    "Asset Management - Global":         (40203010, "Asset Management & Custody Banks", 402030, "Capital Markets", 4020, "Diversified Financials"),
    "Asset Management - Income":         (40203010, "Asset Management & Custody Banks", 402030, "Capital Markets", 4020, "Diversified Financials"),
    "Investment - Banking & Investment Services": (40203020, "Investment Banking & Brokerage", 402030, "Capital Markets", 4020, "Diversified Financials"),
    "Insurance - Diversified":           (40301040, "Multi-line Insurance",         403010, "Insurance", 4030, "Insurance"),
    "Insurance - Life":                  (40301020, "Life & Health Insurance",      403010, "Insurance", 4030, "Insurance"),
    "Insurance - Property & Casualty":   (40301030, "Property & Casualty Insurance", 403010, "Insurance", 4030, "Insurance"),
    "Insurance - Specialty":             (40301050, "Insurance Brokers",            403010, "Insurance", 4030, "Insurance"),
    "Insurance - Brokers":               (40301050, "Insurance Brokers",            403010, "Insurance", 4030, "Insurance"),
    "Insurance - Reinsurance":           (40301060, "Reinsurance",                  403010, "Insurance", 4030, "Insurance"),

    # ══════════════════════════════════════════════════════════════
    # INFORMATION TECHNOLOGY (45)
    # ══════════════════════════════════════════════════════════════
    "Software - Application":            (45103010, "Application Software",         451030, "Software", 4510, "Software & Services"),
    "Software - Infrastructure":         (45103020, "Systems Software",             451030, "Software", 4510, "Software & Services"),
    "Software - Services":               (45103010, "Application Software",         451030, "Software", 4510, "Software & Services"),
    "Information Technology Services":   (45102010, "IT Consulting & Other Services", 451020, "IT Services", 4510, "Software & Services"),
    "Communication Equipment":           (45201020, "Communications Equipment",     452010, "Communications Equipment", 4520, "Technology Hardware & Equipment"),
    "Computer Hardware":                 (45202030, "Technology Hardware, Storage & Peripherals", 452020, "Technology Hardware, Storage & Peripherals", 4520, "Technology Hardware & Equipment"),
    "Consumer Electronics":              (45202030, "Technology Hardware, Storage & Peripherals", 452020, "Technology Hardware, Storage & Peripherals", 4520, "Technology Hardware & Equipment"),
    "Hardware, Equipment & Parts":       (45203010, "Electronic Equipment & Instruments", 452030, "Electronic Equipment, Instruments & Components", 4520, "Technology Hardware & Equipment"),
    "Technology Distributors":           (45203030, "Technology Distributors",       452030, "Electronic Equipment, Instruments & Components", 4520, "Technology Hardware & Equipment"),
    "Semiconductors":                    (45301020, "Semiconductors",               453010, "Semiconductors & Semiconductor Equipment", 4530, "Semiconductors & Semiconductor Equipment"),

    # ══════════════════════════════════════════════════════════════
    # COMMUNICATION SERVICES (50)
    # ══════════════════════════════════════════════════════════════
    "Telecommunications Services":       (50101020, "Integrated Telecommunication Services", 501010, "Diversified Telecommunication Services", 5010, "Telecommunication Services"),
    "Internet Content & Information":    (50203010, "Interactive Media & Services", 502030, "Interactive Media & Services", 5020, "Media & Entertainment"),
    "Entertainment":                     (50202010, "Movies & Entertainment",       502020, "Entertainment", 5020, "Media & Entertainment"),
    "Broadcasting":                      (50201010, "Cable & Satellite",            502010, "Media", 5020, "Media & Entertainment"),
    "Publishing":                        (50201040, "Publishing",                   502010, "Media", 5020, "Media & Entertainment"),
    "Advertising Agencies":              (50201020, "Advertising",                  502010, "Media", 5020, "Media & Entertainment"),

    # ══════════════════════════════════════════════════════════════
    # UTILITIES (55)
    # ══════════════════════════════════════════════════════════════
    "Regulated Electric":                (55101010, "Electric Utilities",           551010, "Electric Utilities", 5510, "Utilities"),
    "Regulated Gas":                     (55102010, "Gas Utilities",                551020, "Gas Utilities", 5510, "Utilities"),
    "Diversified Utilities":             (55103010, "Multi-Utilities",              551030, "Multi-Utilities", 5510, "Utilities"),
    "General Utilities":                 (55103010, "Multi-Utilities",              551030, "Multi-Utilities", 5510, "Utilities"),
    "Regulated Water":                   (55104010, "Water Utilities",              551040, "Water Utilities", 5510, "Utilities"),
    "Independent Power Producers":       (55105010, "Independent Power Producers & Energy Traders", 551050, "Independent Power and Renewable Electricity Producers", 5510, "Utilities"),
    "Renewable Utilities":               (55105020, "Renewable Electricity",        551050, "Independent Power and Renewable Electricity Producers", 5510, "Utilities"),

    # ══════════════════════════════════════════════════════════════
    # REAL ESTATE (60)
    # ══════════════════════════════════════════════════════════════
    "Real Estate - Services":            (60102010, "Real Estate Operating Companies", 601020, "Real Estate Management & Development", 6010, "Equity Real Estate Investment Trusts"),
    "Real Estate - Development":         (60102020, "Real Estate Development",       601020, "Real Estate Management & Development", 6010, "Equity Real Estate Investment Trusts"),
    "Real Estate - Diversified":         (60102010, "Real Estate Operating Companies", 601020, "Real Estate Management & Development", 6010, "Equity Real Estate Investment Trusts"),
}


def build_gics_classifications(cache_dir: str = "data/fmp_cache"):
    """Rebuild classifications.json with GICS codes from FMP profile data."""
    
    cls_path = os.path.join(cache_dir, "classifications.json")
    with open(cls_path) as f:
        old_cls = json.load(f)

    new_cls = {}
    unmapped = set()
    mapped_count = 0

    for sym, info in old_cls.items():
        fmp_sector = info.get("fmp_sector") or info.get("sector") or ""
        fmp_industry = info.get("fmp_industry") or info.get("industry") or ""

        # Map FMP sector to GICS
        gics_sector = FMP_SECTOR_TO_GICS.get(fmp_sector, (99, "Unknown"))
        gics_sector_code, gics_sector_name = gics_sector

        # Map FMP industry to GICS sub-industry
        gics_sub = FMP_INDUSTRY_TO_GICS.get(fmp_industry)

        if gics_sub:
            sub_code, sub_name, ind_code, ind_name, ig_code, ig_name = gics_sub
            mapped_count += 1
        else:
            # Fallback: use FMP industry name as-is
            sub_code = gics_sector_code * 1000000 + 999999
            sub_name = fmp_industry or "Unknown"
            ind_code = gics_sector_code * 10000 + 9999
            ind_name = fmp_industry or "Unknown"
            ig_code = gics_sector_code * 100 + 99
            ig_name = fmp_sector or "Unknown"
            if fmp_industry:
                unmapped.add(f"{fmp_sector} | {fmp_industry}")

        new_cls[sym] = {
            # GICS hierarchy (what WQ BRAIN uses)
            "sector": str(gics_sector_code),
            "sector_name": gics_sector_name,
            "industry_group": str(ig_code),
            "industry_group_name": ig_name,
            "industry": str(ind_code),
            "industry_name": ind_name,
            "subindustry": str(sub_code),
            "subindustry_name": sub_name,
            # Keep FMP originals
            "fmp_sector": fmp_sector,
            "fmp_industry": fmp_industry,
        }

    # Save updated classifications
    with open(cls_path, "w") as f:
        json.dump(new_cls, f, indent=2)

    # Stats
    sectors = set(v["sector"] for v in new_cls.values())
    igs = set(v["industry_group"] for v in new_cls.values())
    inds = set(v["industry"] for v in new_cls.values())
    subs = set(v["subindustry"] for v in new_cls.values())

    print(f"GICS classifications built for {len(new_cls)} tickers")
    print(f"  Mapped: {mapped_count}/{len(new_cls)} ({mapped_count/len(new_cls)*100:.1f}%)")
    print(f"  GICS hierarchy:")
    print(f"    Sectors:          {len(sectors)}")
    print(f"    Industry Groups:  {len(igs)}")
    print(f"    Industries:       {len(inds)}")
    print(f"    Sub-Industries:   {len(subs)}")

    if unmapped:
        print(f"\n  Unmapped ({len(unmapped)}):")
        for u in sorted(unmapped):
            print(f"    {u}")

    # Print sector breakdown
    sector_counts = {}
    for v in new_cls.values():
        sn = v["sector_name"]
        sector_counts[sn] = sector_counts.get(sn, 0) + 1
    print(f"\n  Sector breakdown:")
    for sn in sorted(sector_counts, key=sector_counts.get, reverse=True):
        print(f"    {sn:<40} {sector_counts[sn]:>5}")

    return new_cls


if __name__ == "__main__":
    build_gics_classifications()
