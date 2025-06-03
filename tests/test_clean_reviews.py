import pandas as pd
from src.clean_reviews import clean_text, anonymize_author, parse_date

def test_clean_text():
    assert clean_text('<b>Bonjour!</b> Livraison rapide.') == 'Bonjour! Livraison rapide.'
    assert clean_text('Produit abîmé!!!') == 'Produit abîmé!'
    assert clean_text('   Plusieurs    espaces   ') == 'Plusieurs espaces'

def test_anonymize_author():
    hash1 = anonymize_author('Alice')
    hash2 = anonymize_author('Alice')
    assert hash1 == hash2
    assert hash1 != anonymize_author('Bob')

def test_parse_date():
    assert parse_date('2024-06-03') == '2024-06-03'
    assert parse_date('03/06/2024') == '2024-06-03'
    assert parse_date('invalid') == ''
