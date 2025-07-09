import re
import difflib
import os
from dotenv import load_dotenv
from sheets_utils import get_sheets_service

print("Loaded faq_utils.py")

load_dotenv()
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

def get_faqs():
    print("[LOG] Accessing FAQ sheet...")
    service = get_sheets_service()
    sheet = service.spreadsheets()
    try:
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range='FAQ!A2:C').execute()
        values = result.get('values', [])
        faqs = []
        for row in values:
            if len(row) >= 2:
                faqs.append({
                    'question': row[0].strip(),
                    'answer': row[1].strip(),
                    'lang': row[2].strip().lower() if len(row) > 2 else 'en'
                })
        return faqs
    except Exception as e:
        print(f"[ERROR] Failed to fetch FAQs: {e}")
        return []

def get_sheet_faqs(sheet_name):
    print(f"[LOG] Accessing {sheet_name} sheet...")
    service = get_sheets_service()
    sheet = service.spreadsheets()
    try:
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=f'{sheet_name}!A2:C').execute()
        values = result.get('values', [])
        faqs = []
        for row in values:
            if len(row) >= 2:
                faqs.append({
                    'question': row[0].strip(),
                    'answer': row[1].strip(),
                    'lang': row[2].strip().lower() if len(row) > 2 else 'en',
                    'source': sheet_name
                })
        return faqs
    except Exception as e:
        print(f"[ERROR] Failed to fetch {sheet_name} FAQs: {e}")
        return []

def match_faq(query, faqs, lang, normalization_dict=None):
    queries_to_try = [query]
    if normalization_dict:
        queries_to_try.append(normalize_query(query, normalization_dict))
    best_score = 0
    best_faq = None
    best_lang = None
    lang_priority = [lang]
    if lang != 'en':
        lang_priority.append('en')
    if lang != 'fil':
        lang_priority.append('fil')
    for try_lang in lang_priority:
        for try_query in queries_to_try:
            query_l = try_query.lower().strip()
            for faq in faqs:
                faq_q = faq['question'].lower().strip()
                ratio = difflib.SequenceMatcher(None, query_l, faq_q).ratio()
                faq_words = set(faq_q.split())
                query_words = set(query_l.split())
                word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                if (query_l in faq_q or faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == try_lang:
                    if ratio > best_score or word_overlap > best_score:
                        best_score = max(ratio, word_overlap)
                        best_faq = faq
                        best_lang = try_lang
    if best_score > 0.55:
        return best_faq
    return None

def load_normalization_dict():
    print("[LOG] Loading normalization dictionary from Language sheet...")
    service = get_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range='Language!A2:D'
    ).execute()
    values = result.get('values', [])
    normalization = {}
    for row in values:
        if len(row) >= 2:
            original = row[0].strip().lower()
            normalized = row[1].strip().lower()
            normalization[original] = normalized
    return normalization

def normalize_query(query, normalization_dict):
    words = query.lower().split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)
