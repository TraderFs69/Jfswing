
# 📈 Coach Swing Scanner (Python)

Ce script scanne le S&P 500 pour détecter les titres avec un signal d'achat basé sur :
- UT Bot (ATR-based trail)
- MACD (5,13,4) haussier
- Stochastique (8,5,3) positif sous 50
- ADX > 20
- OBV au-dessus de sa moyenne 20 périodes

## ✅ Fichiers
- coach_swing.py : script principal
- requirements.txt : dépendances à installer

## 🚀 Exécution locale
1. Installer les dépendances :
   pip install -r requirements.txt
2. Lancer le script :
   python coach_swing.py

Le script génère un fichier coach_swing_signals.csv avec les tickers détectés.
