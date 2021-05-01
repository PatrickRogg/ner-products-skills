def is_product_page(page):
    return 'produkt' in page['url'].lower() or 'product' in page['url'].lower() or (
            'title' in page and ('product' in page['title'].lower() or 'produkt' in page['title'].lower()))
