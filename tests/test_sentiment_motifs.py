import pandas as pd
from src.sentiment_motifs import detect_motif

def test_detect_motif_livraison():
    assert detect_motif("J'ai eu un retard de livraison.") == 'livraison'
    assert detect_motif("Colis non reçu.") == 'livraison'

def test_detect_motif_produit():
    assert detect_motif("Produit abîmé à la réception.") == 'produit'
    assert detect_motif("Article cassé.") == 'produit'

def test_detect_motif_service():
    assert detect_motif("Service client injoignable.") == 'service client'
    assert detect_motif("Impossible de contacter le support.") == 'service client'

def test_detect_motif_remboursement():
    assert detect_motif("Demande de remboursement non traitée.") == 'remboursement'
    assert detect_motif("Retour accepté.") == 'remboursement'

def test_detect_motif_autre():
    assert detect_motif("Très satisfait de mon achat.") == 'autre'
