from flask import Flask, request
from pymessenger.bot import Bot
from dotenv import load_dotenv
from langdetect import detect
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime, timedelta
import os
from openai import AzureOpenAI
import requests
import difflib
import random
import re
import time
from sheets_utils import get_weekly_room_schedule, get_vacant_rooms
import base64

# Decode the environment variable
creds_base64 = os.getenv("GOOGLE_CREDS_BASE64")

if creds_base64:
    creds_json = base64.b64decode(creds_base64).decode('utf-8')
    with open("credentials.json", "w") as f:
        f.write(creds_json)
else:
    raise Exception("GOOGLE_CREDS_BASE64 environment variable is not set.")

# Load environment variables
load_dotenv()
_DEPT_SHEET_MAP = None
_DEPT_SHEET_MAP_TIME = 0
_DEPT_SHEET_MAP_TTL = 60  # seconds
DEPT_MAIN_SHEET_ID = os.getenv('DEPT_MAIN_SHEET_ID')  # Set this in your .env
DEPT_MAIN_SHEET_TAB = os.getenv('DEPT_MAIN_SHEET_TAB', 'Departments')
user_department_context = {}  # {sender_id: department}
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'credentials.json'

app = Flask(__name__)
bot = Bot(PAGE_ACCESS_TOKEN)

PIXEL_PERSONALITY = {
    'en': "You are PIXEL, a polite and helpful student assistant at the Computer Engineering Department of the Polytechnic University of the Philippines who is also aware of the latest trends in technology. You are familiar with department procedures, requirements, and academic policies. You help students in a fun and friendly manner but with a hint of professionalism. You are also aware of the latest trends in filipino pop-culture and respond like a trendy young adult. You also refer to the users as 'iskolar' from time to time. If a question is out of scope, politely say so.",
    'fil': "Ikaw si PIXEL, isang magalang at matulunging student assistant ng Computer Engineering Department ng Polytechnic University of the Philippines na may kaalaman sa mga pinakabagong uso sa teknolohiya. Pamilyar ka sa mga proseso, requirements, at patakaran ng departamento. Ikaw ay friendly at masaya na makipagtulong sa mga estudyante pero ikaw ay may pagka-propesyonal din. Ikaw ay may kaalaman din sa mga pinakabagong uso sa pop-culture ng mga Pilipino at sumasagot tulad ng isang trendy na filipino young adult. Tinatawag mo rin ang mga users na 'iskolar' paminsan-minsan. Kung ang tanong ay wala sa iyong saklaw, sabihin ito nang magalang."
}

# GitHub Model API setup
# GITHUB_ENDPOINT = "https://models.github.ai/inference"
# GITHUB_MODEL = "openai/gpt-4.1"

github_client = AzureOpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-15-preview"
)

def ask_github_gpt(messages, model=AZURE_OPENAI_DEPLOYMENT):
    try:
        response = github_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Print the full exception for debugging
        print(f"[ERROR][ask_github_gpt] Exception: {e}")
        # Try to detect rate limit error (429)
        if hasattr(e, 'status_code') and e.status_code == 429:
            return "Sorry, the chatbot is currently busy (rate limit reached). Please try again in a few moments."
        # If the error has a response attribute, print more details
        if hasattr(e, 'response'):
            print(f"[ERROR][ask_github_gpt] Response: {e.response}")
            try:
                print(f"[ERROR][ask_github_gpt] Response text: {e.response.text}")
            except Exception:
                pass
        return None

def get_dept_sheet_map():
    global _DEPT_SHEET_MAP, _DEPT_SHEET_MAP_TIME
    now = time.time()
    if _DEPT_SHEET_MAP is None or now - _DEPT_SHEET_MAP_TIME > _DEPT_SHEET_MAP_TTL:
        try:
            service = get_sheets_service()
            sheet = service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=DEPT_MAIN_SHEET_ID,
                range=f'{DEPT_MAIN_SHEET_TAB}!A2:B'
            ).execute()
            values = result.get('values', [])
            dept_map = {}
            for row in values:
                if len(row) >= 2:
                    dept = row[0].strip().lower()
                    sheet_id = row[1].strip()
                    dept_map[dept] = sheet_id
            _DEPT_SHEET_MAP = dept_map
            _DEPT_SHEET_MAP_TIME = now
        except Exception as e:
            print(f"[ERROR] Could not fetch department sheet map: {e}")
            _DEPT_SHEET_MAP = {}
    return _DEPT_SHEET_MAP


def get_department_keywords_map():
    """
    Returns a dict mapping each department keyword/abbreviation/synonym (from Language!A, category 'department')
    to the canonical department name (from Language!B, which should match the Departments tab).
    Example: {'cpe': 'computer engineering', 'com eng': 'computer engineering', ...}
    """
    service = get_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=DEPT_MAIN_SHEET_ID,
        range='Language!A2:D'
    ).execute()
    values = result.get('values', [])
    dept_keywords_map = {}
    for row in values:
        if len(row) >= 3:
            keyword = row[0].strip().lower()
            canonical = row[1].strip().lower()
            category = row[2].strip().lower()
            if category == 'engineering':
                dept_keywords_map[keyword] = canonical
    return dept_keywords_map



def get_available_departments():
    return list(get_dept_sheet_map().keys())


def detect_department(text):
    """
    Improved: Detects the department in the given text using all department keywords/abbreviations/synonyms.
    - Matches whole words first (not just substrings)
    - Prefers longer/more specific keywords
    - Uses fuzzy match only as last resort
    - Adds debug output
    Returns the canonical department name if found, else None.
    """
    dept_keywords_map = get_department_keywords_map()
    text_l = text.lower()
    # Sort keywords by length (desc) to prefer more specific/longer matches
    sorted_keywords = sorted(dept_keywords_map.keys(), key=lambda k: -len(k))
    # Whole word match first
    for keyword in sorted_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_l):
            print(f"[DEBUG][detect_department] Whole word match: '{keyword}' -> '{dept_keywords_map[keyword]}'")
            return dept_keywords_map[keyword]
    # Substring match (if no whole word match)
    for keyword in sorted_keywords:
        if keyword in text_l:
            print(f"[DEBUG][detect_department] Substring match: '{keyword}' -> '{dept_keywords_map[keyword]}'")
            return dept_keywords_map[keyword]
    # Fuzzy match as last resort
    matches = difflib.get_close_matches(text_l, dept_keywords_map.keys(), n=1, cutoff=0.8)
    if matches:
        print(f"[DEBUG][detect_department] Fuzzy match: '{matches[0]}' -> '{dept_keywords_map[matches[0]]}'")
        return dept_keywords_map[matches[0]]
    print(f"[DEBUG][detect_department] No department match for: {text}")
    return None

def get_user_spreadsheet_id(sender_id):
    dept_map = get_dept_sheet_map()
    dept = user_department_context.get(sender_id, None)
    if dept and dept in dept_map:
        return dept_map[dept]
    if 'cpe' in dept_map:
        return dept_map['cpe']
    elif dept_map:
        return list(dept_map.values())[0]
    else:
        return SPREADSHEET_ID  # fallback

def get_sheets_service_for_user(sender_id):
    spreadsheet_id = get_user_spreadsheet_id(sender_id)
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('sheets', 'v4', credentials=creds), spreadsheet_id

def get_faqs_for_user(sender_id, tab='FAQ'):
    service, spreadsheet_id = get_sheets_service_for_user(sender_id)
    print(f"[DEBUG] Fetching from spreadsheet: {spreadsheet_id}, tab: {tab}")  # <-- Add this line
    sheet = service.spreadsheets()
    try:
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=f'{tab}!A2:C').execute()
        values = result.get('values', [])
        faqs = []
        for row in values:
            if len(row) >= 2:
                faqs.append({
                    'question': row[0].strip(),
                    'answer': row[1].strip(),
                    'lang': row[2].strip().lower() if len(row) > 2 else 'en',
                    'source': tab
                })
        return faqs
    except Exception as e:
        print(f"[ERROR] Failed to fetch FAQs from tab '{tab}': {e}")
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
        print(f"[DEBUG] Parsed {sheet_name} FAQs: {faqs}")
        return faqs
    except Exception as e:
        print(f"[ERROR] Failed to fetch {sheet_name} FAQs: {e}")
        return []

def get_combined_faqs(sender_id):
    # Fetch from master sheet
    master_service = get_sheets_service()  # Uses SPREADSHEET_ID (master)
    master_faqs = []
    for tab in ['FAQ', 'PIXEL']:
        master_faqs += get_sheet_faqs(tab)  # You may need to adjust this to use master_service

    # Fetch from department sheet
    dept_service, dept_spreadsheet_id = get_sheets_service_for_user(sender_id)
    dept_faqs = []
    for tab in ['Department', 'Laboratory']:
        try:
            dept_faqs += get_faqs_for_user(sender_id, tab)
        except Exception as e:
            print(f"[DEBUG][get_combined_faqs] Could not fetch dept tab '{tab}': {e}")

    return master_faqs + dept_faqs

def log_unanswered(query, lang):
    service = get_sheets_service()
    sheet = service.spreadsheets()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = {'values': [[query, lang, now]]}
    sheet.values().append(
        spreadsheetId=DEPT_MAIN_SHEET_ID,
        range='Unanswered!A:C',
        valueInputOption='USER_ENTERED',
        body=body
    ).execute()

def log_unanswered_for_user(sender_id, query, lang):
    """
    Logs unanswered queries to the correct department's Unanswered tab.
    Falls back to master sheet if department is unknown or sheet does not exist.
    Adds debug output and fuzzy match for department name.
    """
    dept_map = get_dept_sheet_map()
    # Try to detect department for this query
    dept = detect_department(query)
    if not dept:
        dept = user_department_context.get(sender_id, None)
    dept_key = None
    if dept:
        dept_lc = dept.strip().lower()
        if dept_lc in dept_map:
            dept_key = dept_lc
        else:
            # Fuzzy match fallback
            import difflib
            matches = difflib.get_close_matches(dept_lc, dept_map.keys(), n=1, cutoff=0.7)
            if matches:
                dept_key = matches[0]
    spreadsheet_id = None
    if dept_key:
        spreadsheet_id = dept_map[dept_key]
        print(f"[DEBUG][log_unanswered_for_user] Logging to department sheet: '{dept_key}' (ID: {spreadsheet_id}) for query: {query}")
    else:
        spreadsheet_id = DEPT_MAIN_SHEET_ID  # fallback to master
        print(f"[DEBUG][log_unanswered_for_user] Department not found, logging to master sheet for query: {query}")

    service = get_sheets_service()
    sheet = service.spreadsheets()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = {'values': [[query, lang, now]]}
    sheet.values().append(
        spreadsheetId=spreadsheet_id,
        range='Unanswered!A:C',
        valueInputOption='USER_ENTERED',
        body=body
    ).execute()
    
def detect_language(text):
    try:
        lang = detect(text)
        return 'fil' if lang in ['tl', 'fil'] else 'en'
    except:
        return 'en'

def get_language_keywords(categories=None):
    print("[LOG] Accessing Language sheet...")
    service = get_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range='Language!A2:D'
    ).execute()
    values = result.get('values', [])
    keywords = set()
    if categories:
        categories = set(c.lower() for c in categories)
    for row in values:
        if len(row) >= 4:
            original = row[0].strip().lower()
            category = row[2].strip().lower()
            if not categories or category in categories:
                keywords.add(original)
    return keywords

def get_greeting_keywords():
    return get_language_keywords(categories=['greeting'])

def get_thanks_keywords():
    return get_language_keywords(categories=['thanks'])

def fuzzy_match(query, choices, threshold=0.7):
    # Return the best match above threshold, or None
    matches = difflib.get_close_matches(query.lower(), [c.lower() for c in choices], n=1, cutoff=threshold)
    return matches[0] if matches else None

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
    # Replace all words/phrases in the query with their normalized equivalents
    words = query.lower().split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def match_faq(query, faqs, lang, normalization_dict=None):
    # Try both normalized and original queries for better matching
    print(f"[DEBUG][match_faq] Incoming query: {repr(query)}, lang: {lang}")
    queries_to_try = [query]
    if normalization_dict:
        queries_to_try.append(normalize_query(query, normalization_dict))
    best_score = 0
    best_faq = None
    for try_query in queries_to_try:
        query_l = try_query.lower().strip()
        print(f"[DEBUG][match_faq] Trying query variant: {repr(query_l)}")
        for faq in faqs:
            faq_q = faq['question'].lower().strip()
            # Normalize the FAQ question as well
            if normalization_dict:
                faq_q_norm = normalize_query(faq_q, normalization_dict)
            else:
                faq_q_norm = faq_q
            # Compare both original and normalized forms
            for compare_faq_q in [faq_q, faq_q_norm]:
                ratio = difflib.SequenceMatcher(None, query_l, compare_faq_q).ratio()
                faq_words = set(compare_faq_q.split())
                query_words = set(query_l.split())
                word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                print(f"[DEBUG][match_faq] Comparing to FAQ: {repr(compare_faq_q)} | ratio={ratio:.3f} | word_overlap={word_overlap:.3f} | lang={faq['lang']}")
                if (query_l in compare_faq_q or compare_faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == lang:
                    print(f"[DEBUG][match_faq] New best match: {faq['question']} (score={ratio:.3f}, overlap={word_overlap:.3f})")
                    best_score = ratio
                    best_faq = faq
    # If not found and lang is not 'en', try English
    if not best_faq and lang != 'en':
        print(f"[DEBUG][match_faq] No match in lang={lang}, trying 'en'")
        for try_query in queries_to_try:
            query_l = try_query.lower().strip()
            for faq in faqs:
                faq_q = faq['question'].lower().strip()
                if normalization_dict:
                    faq_q_norm = normalize_query(faq_q, normalization_dict)
                else:
                    faq_q_norm = faq_q
                for compare_faq_q in [faq_q, faq_q_norm]:
                    ratio = difflib.SequenceMatcher(None, query_l, compare_faq_q).ratio()
                    faq_words = set(compare_faq_q.split())
                    query_words = set(query_l.split())
                    word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                    print(f"[DEBUG][match_faq] (en) Comparing to FAQ: {repr(compare_faq_q)} | ratio={ratio:.3f} | word_overlap={word_overlap:.3f} | lang={faq['lang']}")
                    if (query_l in compare_faq_q or compare_faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == 'en':
                        print(f"[DEBUG][match_faq] (en) New best match: {faq['question']} (score={ratio:.3f}, overlap={word_overlap:.3f})")
                        best_score = ratio
                        best_faq = faq
    # If not found and lang is not 'fil', try Filipino
    if not best_faq and lang != 'fil':
        print(f"[DEBUG][match_faq] No match in lang={lang}, trying 'fil'")
        for try_query in queries_to_try:
            query_l = try_query.lower().strip()
            for faq in faqs:
                faq_q = faq['question'].lower().strip()
                if normalization_dict:
                    faq_q_norm = normalize_query(faq_q, normalization_dict)
                else:
                    faq_q_norm = faq_q
                for compare_faq_q in [faq_q, faq_q_norm]:
                    ratio = difflib.SequenceMatcher(None, query_l, compare_faq_q).ratio()
                    faq_words = set(compare_faq_q.split())
                    query_words = set(query_l.split())
                    word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                    print(f"[DEBUG][match_faq] (fil) Comparing to FAQ: {repr(compare_faq_q)} | ratio={ratio:.3f} | word_overlap={word_overlap:.3f} | lang={faq['lang']}")
                    if (query_l in compare_faq_q or compare_faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == 'fil':
                        print(f"[DEBUG][match_faq] (fil) New best match: {faq['question']} (score={ratio:.3f}, overlap={word_overlap:.3f})")
                        best_score = ratio
                        best_faq = faq
    # Lower threshold slightly for better matching
    print(f"[DEBUG][match_faq] Final best_score: {best_score:.3f}, best_faq: {best_faq['question'] if best_faq else None}")
    if best_score > 0.55:
        return best_faq
    return None

def get_sheets_service():
    print("[LOG] Accessing Google Sheets service...")
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('sheets', 'v4', credentials=creds)

# def get_faqs():
#     print("[LOG] Accessing FAQ sheet...")
#     service = get_sheets_service()
#     sheet = service.spreadsheets()
#     try:
#         result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range='FAQ!A2:C').execute()
#         values = result.get('values', [])
#         print(f"[DEBUG] Raw FAQ values from sheet: {values}")
#         faqs = []
#         for row in values:
#             if len(row) >= 2:
#                 faqs.append({
#                     'question': row[0].strip(),
#                     'answer': row[1].strip(),
#                     'lang': row[2].strip().lower() if len(row) > 2 else 'en'
#                 })
#         print(f"[DEBUG] Parsed FAQs: {faqs}")
#         return faqs
#     except Exception as e:
#         print(f"[ERROR] Failed to fetch FAQs: {e}")
#         return []

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
        print(f"[DEBUG] Parsed {sheet_name} FAQs: {faqs}")
        return faqs
    except Exception as e:
        print(f"[ERROR] Failed to fetch {sheet_name} FAQs: {e}")
        return []

# def log_unanswered(query, lang):
#     service = get_sheets_service()
#     sheet = service.spreadsheets()
#     now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     body = {'values': [[query, lang, now]]}
#     sheet.values().append(
#         spreadsheetId=SPREADSHEET_ID,
#         range='Unanswered!A:C',
#         valueInputOption='USER_ENTERED',
#         body=body
#     ).execute()

def detect_language(text):
    try:
        lang = detect(text)
        return 'fil' if lang in ['tl', 'fil'] else 'en'
    except:
        return 'en'

def get_language_keywords(categories=None):
    print("[LOG] Accessing Language sheet...")
    service = get_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range='Language!A2:D'
    ).execute()
    values = result.get('values', [])
    keywords = set()
    if categories:
        categories = set(c.lower() for c in categories)
    for row in values:
        if len(row) >= 4:
            original = row[0].strip().lower()
            category = row[2].strip().lower()
            if not categories or category in categories:
                keywords.add(original)
    return keywords

def get_greeting_keywords():
    return get_language_keywords(categories=['greeting'])

def get_thanks_keywords():
    return get_language_keywords(categories=['thanks'])

def fuzzy_match(query, choices, threshold=0.7):
    # Return the best match above threshold, or None
    matches = difflib.get_close_matches(query.lower(), [c.lower() for c in choices], n=1, cutoff=threshold)
    return matches[0] if matches else None

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
    # Replace all words/phrases in the query with their normalized equivalents
    words = query.lower().split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def match_faq(query, faqs, lang, normalization_dict=None):
    # Try both normalized and original queries for better matching
    print(f"[DEBUG][match_faq] Incoming query: {repr(query)}, lang: {lang}")
    queries_to_try = [query]
    if normalization_dict:
        queries_to_try.append(normalize_query(query, normalization_dict))
    best_score = 0
    best_faq = None
    for try_query in queries_to_try:
        query_l = try_query.lower().strip()
        print(f"[DEBUG][match_faq] Trying query variant: {repr(query_l)}")
        for faq in faqs:
            faq_q = faq['question'].lower().strip()
            # Normalize the FAQ question as well
            if normalization_dict:
                faq_q_norm = normalize_query(faq_q, normalization_dict)
            else:
                faq_q_norm = faq_q
            # Compare both original and normalized forms
            for compare_faq_q in [faq_q, faq_q_norm]:
                ratio = difflib.SequenceMatcher(None, query_l, compare_faq_q).ratio()
                faq_words = set(compare_faq_q.split())
                query_words = set(query_l.split())
                word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                print(f"[DEBUG][match_faq] Comparing to FAQ: {repr(compare_faq_q)} | ratio={ratio:.3f} | word_overlap={word_overlap:.3f} | lang={faq['lang']}")
                if (query_l in compare_faq_q or compare_faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == lang:
                    print(f"[DEBUG][match_faq] New best match: {faq['question']} (score={ratio:.3f}, overlap={word_overlap:.3f})")
                    best_score = ratio
                    best_faq = faq
    # If not found and lang is not 'en', try English
    if not best_faq and lang != 'en':
        print(f"[DEBUG][match_faq] No match in lang={lang}, trying 'en'")
        for try_query in queries_to_try:
            query_l = try_query.lower().strip()
            for faq in faqs:
                faq_q = faq['question'].lower().strip()
                if normalization_dict:
                    faq_q_norm = normalize_query(faq_q, normalization_dict)
                else:
                    faq_q_norm = faq_q
                for compare_faq_q in [faq_q, faq_q_norm]:
                    ratio = difflib.SequenceMatcher(None, query_l, compare_faq_q).ratio()
                    faq_words = set(compare_faq_q.split())
                    query_words = set(query_l.split())
                    word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                    print(f"[DEBUG][match_faq] (en) Comparing to FAQ: {repr(compare_faq_q)} | ratio={ratio:.3f} | word_overlap={word_overlap:.3f} | lang={faq['lang']}")
                    if (query_l in compare_faq_q or compare_faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == 'en':
                        print(f"[DEBUG][match_faq] (en) New best match: {faq['question']} (score={ratio:.3f}, overlap={word_overlap:.3f})")
                        best_score = ratio
                        best_faq = faq
    # If not found and lang is not 'fil', try Filipino
    if not best_faq and lang != 'fil':
        print(f"[DEBUG][match_faq] No match in lang={lang}, trying 'fil'")
        for try_query in queries_to_try:
            query_l = try_query.lower().strip()
            for faq in faqs:
                faq_q = faq['question'].lower().strip()
                if normalization_dict:
                    faq_q_norm = normalize_query(faq_q, normalization_dict)
                else:
                    faq_q_norm = faq_q
                for compare_faq_q in [faq_q, faq_q_norm]:
                    ratio = difflib.SequenceMatcher(None, query_l, compare_faq_q).ratio()
                    faq_words = set(compare_faq_q.split())
                    query_words = set(query_l.split())
                    word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                    print(f"[DEBUG][match_faq] (fil) Comparing to FAQ: {repr(compare_faq_q)} | ratio={ratio:.3f} | word_overlap={word_overlap:.3f} | lang={faq['lang']}")
                    if (query_l in compare_faq_q or compare_faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == 'fil':
                        print(f"[DEBUG][match_faq] (fil) New best match: {faq['question']} (score={ratio:.3f}, overlap={word_overlap:.3f})")
                        best_score = ratio
                        best_faq = faq
    # Lower threshold slightly for better matching
    print(f"[DEBUG][match_faq] Final best_score: {best_score:.3f}, best_faq: {best_faq['question'] if best_faq else None}")
    if best_score > 0.55:
        return best_faq
    return None

def check_github_model_rate_limit():
    url = AZURE_OPENAI_ENDPOINT
    headers = {"Authorization": f"Bearer {AZURE_OPENAI_KEY}"}
    data = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": [{"role": "user", "content": "Hello!"}]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        print("[DEBUG] Response status code:", response.status_code)
        print("[DEBUG] Response headers:", response.headers)
        print("[DEBUG] Response body:", response.text)
        print("[Rate Limit] X-RateLimit-Limit:", response.headers.get("X-RateLimit-Limit"))
        print("[Rate Limit] X-RateLimit-Remaining:", response.headers.get("X-RateLimit-Remaining"))
        print("[Rate Limit] X-RateLimit-Reset:", response.headers.get("X-RateLimit-Reset"))
        return {
            "limit": response.headers.get("X-RateLimit-Limit"),
            "remaining": response.headers.get("X-RateLimit-Remaining"),
            "reset": response.headers.get("X-RateLimit-Reset")
        }
    except Exception as e:
        print(f"[ERROR] Could not check rate limit: {e}")
        return None

def clean_faq_response(text):
    import re
    # Remove asterisks
    #text = text.replace('*', '')
    # Only add blank lines before numbered steps at the start of a line (not after numbers in text)
    # e.g., 1. Step one\n2. Step two, but not "Room 323" or "2.00"
    text = re.sub(r'(?m)^(\d+)\.(?!\d)', r'\n\1.', text)
    # Remove extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove blank lines after 'Room' or similar patterns
    text = re.sub(r'(Room)\s*\n\s*(\d+)', r'\1 \2', text, flags=re.IGNORECASE)
    # Remove blank lines after numbers like 2.00
    text = re.sub(r'(\d)\s*\n\s*(\d{2,})', r'\1\2', text)
    return text.strip()

def extract_main_context(question):
    # Heuristic: extract main noun/phrase after 'how to', 'how can I', 'paano', etc.
    import re
    # Try to extract after 'how to', 'how can i', 'paano', etc.
    patterns = [
        r'how (?:can i|do i|to)\s+(.*?)\??$',
        r'paano(?:\s+ba)?(?:\s+ang|\s+mag|\s+ma|\s+makaka)?\s*(.*?)\??$',
        r'what is (.*?)\??$',
        r'ano(?:\s+nga)?(?:\s+ba)?(?:\s+ang)?\s*(.*?)\??$'
    ]
    for pat in patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m and m.group(1):
            return m.group(1).strip()
    # Fallback: use first 3 words
    return ' '.join(question.split()[:3])

def get_personalized_faq_response(faq_question, faq_answer, lang, user_query=None, context=None, skip_greeting=False):
    # Avoid greeting if already greeted recently or if answer already contains a greeting
    import re
    answer = faq_answer
    if skip_greeting or (context and context.get('greeted')):
        # Remove greeting from answer if present (simple heuristic)
        answer = re.sub(r'^(hi|hello|heyyy|kumusta|kamusta|magandang \w+|good \w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po)[!,. ]*', '', answer, flags=re.IGNORECASE)
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = (
        f"The user asked: '{user_query}'. The answer is: '{answer}'. "
        f"Reply as PIXEL, making your response natural, helpful, and contextually accurate. Avoid repeating the answer. Add a blank line between steps or paragraphs. Only greet the user if they haven't been greeted recently."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        response = clean_faq_response(response)
        return response
    except Exception as e:
        print(f"[ERROR] Could not personalize FAQ response: {e}")
        return answer

def get_dynamic_greeting(lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "Greet the user as PIXEL in a trendy, friendly way. Use the user's language."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception as e:
        print(f"[ERROR] Could not generate dynamic greeting: {e}")
        return "Hello, iskolar! 👋 How can I help you today?"

def get_dynamic_thanks(lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "The user thanked you. Respond as PIXEL with a friendly, trendy 'you're welcome' or 'no problem, do you have any more questions?' message, in the user's language."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception as e:
        return "No problem! Do you have any more questions?"

def get_dynamic_closing(lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "Say goodbye or closing statement as PIXEL in a trendy, friendly way. Use the user's language."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception as e:
        print(f"[ERROR] Could not generate dynamic closing: {e}")
        return "You're welcome! If you have more questions, just message me anytime!"

# --- Room Query Parsing and Handling (Enhanced for Time Ranges) ---
def parse_room_query(message):
    """Extract day and time or time range from a room vacancy query."""
    message = message.lower()
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
    day_patterns = {
        'today': datetime.now().strftime('%A').lower(),
        'tomorrow': (datetime.now() + timedelta(days=1)).strftime('%A').lower(),
        'monday': 'monday',
        'tuesday': 'tuesday',
        'wednesday': 'wednesday',
        'thursday': 'thursday',
        'friday': 'friday',
        'saturday': 'saturday',
        'lunes': 'monday',
        'martes': 'tuesday',
        'miyerkules': 'wednesday',
        'huwebes': 'thursday',
        'biyernes': 'friday',
        'sabado': 'saturday'
    }
    # Extract day
    day = None
    for day_term, day_value in day_patterns.items():
        if day_term in message:
            day = day_value
            break
    # Extract time range (e.g. 1:00 pm to 3:00 pm, 1-3 pm, 13:00-15:00)
    time_range_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|\-|–|—)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?'
    match = re.search(time_range_pattern, message)
    if match:
        shour = int(match.group(1))
        sminute = match.group(2) if match.group(2) else '00'
        speriod = match.group(3)
        ehour = int(match.group(4))
        eminute = match.group(5) if match.group(5) else '00'
        eperiod = match.group(6)
        # Normalize periods
        if not speriod and eperiod:
            speriod = eperiod
        if not eperiod and speriod:
            eperiod = speriod
        # Convert to 24-hour
        if speriod:
            if speriod.lower() == 'pm' and shour < 12:
                shour += 12
            elif speriod.lower() == 'am' and shour == 12:
                shour = 0
        if eperiod:
            if eperiod.lower() == 'pm' and ehour < 12:
                ehour += 12
            elif eperiod.lower() == 'am' and ehour == 12:
                ehour = 0
        start_time = f"{shour:02d}:{sminute}"
        end_time = f"{ehour:02d}:{eminute}"
        return day, start_time, end_time
    # Fallback: single time
    time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)'
    match = re.search(time_pattern, message)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3)
        if period:
            period = period.lower()
        if period == 'pm' and hour < 12:
            hour += 12
        elif period == 'am' and hour == 12:
            hour = 0
        time_str = f"{hour:02d}:{minute:02d}"
        return day, time_str, None
    return day, None, None

def normalize_time_string(t):
    """Converts time strings like '8:00 AM', '08:00', '8:00', '8 AM' to '08:00' 24-hour format."""
    import re
    t = t.strip().lower().replace('.', ':')
    match = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', t)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    period = match.group(3)
    if period == 'pm' and hour < 12:
        hour += 12
    elif period == 'am' and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"

def is_room_column(col):
    # Accepts columns like 'room', 'cea', 'cea300', 'cea301', etc.
    col = col.strip().lower()
    return (
        col.startswith('room') or
        col.startswith('cea') or
        (col.isupper() and any(c.isdigit() for c in col))
    )

def get_vacant_rooms(day, time):
    schedule = get_weekly_room_schedule_normalized()
    day = day.strip().lower()
    if day not in schedule:
        return []
    vacant = []
    norm_time = normalize_time_string(time)
    for row in schedule[day]:
        sheet_time = normalize_time_string(row.get('TIME', ''))
        if sheet_time == norm_time:
            for k, v in row.items():
                if is_room_column(k):
                    if not v.strip():
                        vacant.append(k)
            break
    return vacant

def get_vacant_rooms_for_range(day, start_time, end_time):
    print(f"[DEBUG][ENTRY] get_vacant_rooms_for_range called with day={repr(day)}, start_time={repr(start_time)}, end_time={repr(end_time)}")
    """
    Returns a list of rooms that are vacant for the entire time range on the given day.
    Handles merged/unmerged cell logic (blank means same as above, not vacant).
    Adds debug output to all days, and prints day_lookup value and type.
    """
    schedule = get_weekly_room_schedule_normalized()
    day_lookup = day.strip().lower()
    print(f"[DEBUG] get_vacant_rooms_for_range: day='{day}' (lookup='{day_lookup}', type={type(day_lookup)}) available keys={list(schedule.keys())}")
    if day_lookup not in schedule:
        print(f"[DEBUG] Day '{day_lookup}' not found in schedule keys: {list(schedule.keys())}")
        return []
    rows = schedule[day_lookup]
    if not rows:
        print(f"[DEBUG] No rows for day '{day_lookup}'")
        return []
    # Dynamically detect the time column key (prefer 'TIME', else first non-room column)
    first_row_keys = list(rows[0].keys())
    time_col_key = None
    for k in first_row_keys:
        if k.strip().lower() == 'time':
            time_col_key = k
            break
    if not time_col_key:
        for k in first_row_keys:
            if not is_room_column(k) and k.strip() != '':
                time_col_key = k
                break
    print(f"[DEBUG] Detected time column key: '{time_col_key}'")
    room_keys = [k for k in first_row_keys if is_room_column(k)]
    # Print the keys and raw/normalized time values for each row (for all days)
    for idx, row in enumerate(rows):
        raw_time = row.get(time_col_key, '') if time_col_key else ''
        norm_time = normalize_time_string(raw_time)
        print(f"[DEBUG][{day_lookup.upper()}] Row {idx} keys: {list(row.keys())}")
        print(f"[DEBUG][{day_lookup.upper()}] Row {idx} raw time: '{raw_time}', normalized: '{norm_time}'")
    # Find all time slots in the schedule for the day
    time_slots = [normalize_time_string(row.get(time_col_key, '')) if time_col_key else None for row in rows]
    print(f"[DEBUG] Time slots for {day_lookup}: {time_slots}")
    def time_to_minutes(t):
        h, m = map(int, t.split(':'))
        return h * 60 + m
    smin = time_to_minutes(normalize_time_string(start_time))
    emin = time_to_minutes(normalize_time_string(end_time))
    # Find indices for the requested range
    start_idx = None
    end_idx = None
    for idx, t in enumerate(time_slots):
        if t:
            tmin = time_to_minutes(t)
            if start_idx is None and tmin >= smin:
                start_idx = idx
            if tmin <= emin:
                end_idx = idx
    print(f"[DEBUG] Requested range: {start_time}-{end_time} ({smin}-{emin} min), start_idx={start_idx}, end_idx={end_idx}")
    if start_idx is None or end_idx is None or end_idx < start_idx:
        print(f"[DEBUG] Invalid indices: start_idx={start_idx}, end_idx={end_idx}")
        return []
    vacant_rooms = []
    for room in room_keys:
        vacant = True
        for i in range(start_idx, end_idx + 1):
            val = rows[i].get(room, '').strip()
            debug_val = val  # Save for debug
            # If blank, treat as 'same as above' (not vacant), so check previous rows
            if not val:
                for j in range(i-1, -1, -1):
                    prev_val = rows[j].get(room, '').strip()
                    if prev_val:
                        val = prev_val
                        break
            print(f"[DEBUG][{day_lookup.upper()}] Room {room} slot {i} value: '{debug_val}' (after lookup: '{val}')")
            if val:
                vacant = False
                break
        if vacant:
            vacant_rooms.append(room)
    print(f"[DEBUG] Vacant rooms for {day_lookup} {start_time}-{end_time}: {vacant_rooms}")
    return vacant_rooms

def handle_room_query(message_text, lang):
    day, start_time, end_time = parse_room_query(message_text)
    print(f"[DEBUG][handle_room_query] Parsed: day={day}, start_time={start_time}, end_time={end_time}")
    if not day:
        day = datetime.now().strftime('%A').lower()
    day = day.strip().lower()
    if start_time and end_time:
        vacant_rooms = get_vacant_rooms_for_range(day, start_time, end_time)
        if vacant_rooms:
            return f"The following rooms are free on {day.title()} from {start_time} to {end_time}: {', '.join(vacant_rooms)}."
        else:
            return f"Sorry, no rooms are free on {day.title()} from {start_time} to {end_time}."
    elif start_time:
        vacant_rooms = get_vacant_rooms(day, start_time)
        if vacant_rooms:
            return f"The following rooms are free on {day.title()} at {start_time}: {', '.join(vacant_rooms)}."
        else:
            return f"Sorry, no rooms are free on {day.title()} at {start_time}."
    else:
        schedule = get_weekly_room_schedule_normalized()
        if day in schedule:
            vacant_rooms = set()
            for row in schedule[day]:
                for k, v in row.items():
                    if is_room_column(k) and not v.strip():
                        vacant_rooms.add(k)
            if vacant_rooms:
                return f"Rooms that are free at some time on {day.title()}: {', '.join(sorted(vacant_rooms))}."
            else:
                return f"Sorry, I couldn't find any free rooms for {day.title()}."
        else:
            return f"Sorry, I couldn't find the schedule for {day.title()}."

def get_weekly_room_schedule_normalized():
    """Fetches and parses the schedule, normalizing all day keys to lowercase and stripping whitespace."""
    schedule = get_weekly_room_schedule()
    normalized = {}
    for k, v in schedule.items():
        norm_key = k.strip().lower()
        normalized[norm_key] = v
    # Debug print: show all day keys and a sample row for each
    print("[DEBUG] Parsed schedule days:", list(normalized.keys()))
    for day, rows in normalized.items():
        print(f"[DEBUG] {day}: {rows[:1]}")
    return normalized

# --- Caching for FAQ and normalization dict ---
_FAQ_CACHE = None
_FAQ_CACHE_TIME = 0
_FAQ_CACHE_TTL = 30  # seconds (reduced for faster Google Sheets update reflection during testing)

_NORMALIZATION_CACHE = None
_NORMALIZATION_CACHE_TIME = 0
_NORMALIZATION_CACHE_TTL = 30  # seconds (reduced for faster Google Sheets update reflection during testing)

def get_faq_sheet_names():
    service = get_sheets_service()
    try:
        sheets_metadata = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        # Normalize and trim all sheet names
        sheet_names = [s['properties']['title'].strip() for s in sheets_metadata.get('sheets', [])]
        # Exclude only non-FAQ sheets (case-insensitive, trimmed)
        exclude = ['schedule', 'language', 'unanswered']
        faq_sheets = [name for name in sheet_names if name.strip().lower() not in exclude]
        print("[DEBUG] FAQ sheets to fetch (normalized):", faq_sheets)
        return faq_sheets
    except Exception as e:
        print(f"[ERROR] Could not fetch sheet names: {e}")
        # Fallback to default
        return ['FAQ', 'Laboratory', 'PIXEL', 'Department']

# def get_faqs_cached():
#     global _FAQ_CACHE, _FAQ_CACHE_TIME
#     now = time.time()
#     if _FAQ_CACHE is None or now - _FAQ_CACHE_TIME > _FAQ_CACHE_TTL:
#         faqs = []
#         for sheet_name in get_faq_sheet_names():
#             sheet_name_trimmed = sheet_name.strip()
#             if sheet_name_trimmed.lower() == 'faq':
#                 faqs += get_faqs_for_user(sender_id)
#             else:
#                 faqs += get_sheet_faqs(sheet_name_trimmed)
#         print(f"[DEBUG] Total FAQs fetched: {len(faqs)}")
#         _FAQ_CACHE = faqs
#         _FAQ_CACHE_TIME = now
#     return _FAQ_CACHE

def load_normalization_dict_cached():
    global _NORMALIZATION_CACHE, _NORMALIZATION_CACHE_TIME
    now = time.time()
    if _NORMALIZATION_CACHE is None or now - _NORMALIZATION_CACHE_TIME > _NORMALIZATION_CACHE_TTL:
        _NORMALIZATION_CACHE = load_normalization_dict()
        _NORMALIZATION_CACHE_TIME = now
    return _NORMALIZATION_CACHE

# Add a set to track processed message IDs to prevent duplicate replies
deduped_message_ids = set()

# Add a simple in-memory context for greeting tracking (per session/user)
user_greeting_context = {}
# Add per-user context for clarification and recent messages
user_query_context = {}

# --- Helper: Detect follow-up requests for more info/resources ---
def is_followup_resource_request(text):
    followup_patterns = [
        r"more info", r"more information", r"can you give.*resource", r"can you provide.*resource", r"can you send.*resource", r"can you share.*resource",
        r"give me.*resource", r"provide.*resource", r"send.*resource", r"share.*resource", r"can i have.*resource", r"can i get.*resource", r"can i see.*resource",
        r"can i access.*resource", r"can you give.*reference", r"can you provide.*reference", r"can you send.*reference", r"can you share.*reference",
        r"give me.*reference", r"provide.*reference", r"send.*reference", r"share.*reference", r"can i have.*reference", r"can i get.*reference", r"can i see.*reference",
        r"can i access.*reference", r"more details", r"can you elaborate", r"can you explain more", r"can you expand", r"can you go deeper", r"can you give more"
    ]
    for pat in followup_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def detect_greeting(text):
    greeting_keywords = get_greeting_keywords()
    return any(k in text.lower() for k in greeting_keywords)

def detect_assignment_homework(text):
    import re
    # Patterns for assignment/homework detection
    patterns = [
        r'(multiple choice|choose the correct answer|checkbox|true or false|enumerate|match the following|fill in the blank|essay|short answer|assignment|homework|quiz|test|exam)',
        r'\b[a-dA-D]\b[\).]',  # e.g., a) b) c)
        r'\btrue\b|\bfalse\b',
        r'\bwhich of the following\b',
        r'\bcorrect answer\b',
        r'\bselect all that apply\b',
        r'\banswer the following\b',
        r'\bexplain your answer\b',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def extract_questions(text):
    import re
    # Split on question marks, but also split on newlines and sentences that look like questions
    # Accept lines that start with question words or verbs, even without a question mark
    question_starters = r'(how|what|when|where|who|why|can|is|are|do|does|did|will|would|should|could|may|paano|ano|saan|kailan|bakit|pwede|puwede|magkano|ilan|sinong|sino|pwedeng|paano|mag|may|meron|possible|possible ba|tell me|give me|explain|define|describe|list|provide|show|help|need|want|require|could you|would you|can you|please|gusto|kailangan|ano ang|ano ba|ano po|ano nga|ano kaya|ano yung|ano iyong|ano ito|ano iyan|ano yan|ano ito|ano po ba|ano po ang|ano po ito|ano po iyan|ano po yan|ano po ito|ano po ba ito|ano po ba iyan|ano po ba yan|ano po ba ito|ano po ba iyan|ano po ba yan)'
    # Split on question marks, newlines, or sentences that look like questions
    parts = re.split(r'(?<=[?])\s+|\n+', text)
    questions = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # If it ends with a question mark, or starts with a question word, or is short and imperative
        if p.endswith('?') or re.match(rf'^{question_starters}\b', p, re.IGNORECASE):
            questions.append(p)
        # Also treat as question if it's a short sentence (<=10 words) and contains a verb
        elif len(p.split()) <= 10 and re.search(r'\b(is|are|do|does|did|will|would|should|could|may|can|have|has|had|need|want|require|help|explain|define|describe|list|provide|show|tell|give|explain|pwede|pwedeng|gusto|kailangan)\b', p, re.IGNORECASE):
            questions.append(p)
    # If nothing matched, fallback to the whole text
    if not questions:
        questions = [text.strip()]
    return questions

def get_dynamic_greeting_again(lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "The user greeted you again. Respond as PIXEL with a friendly, trendy 'hello again' message, in the user's language."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception as e:
        return "Hello again! How can I help you?"

def get_assignment_guidance_response(user_query, lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = f"The user asked: '{user_query}'. It looks like a homework or assignment question. As PIXEL, politely explain that you can't answer assignments directly, but you can guide them about the topic and encourage them to answer on their own."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception as e:
        return "Sorry, I can't answer assignments directly, but I can help explain the topic!"

def get_cpe_keywords():
    # Only include department, laboratory, and academic keywords
    service = get_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range='Language!A2:D'
    ).execute()
    values = result.get('values', [])
    cpe_keywords = set()
    for row in values:
        if len(row) >= 3:
            category = row[2].strip().lower()
            if category in ['department', 'laboratory', 'academic']:
                cpe_keywords.add(row[0].strip().lower())
    return cpe_keywords

def is_cpe_related(query, cpe_keywords):
    q = query.lower()
    return any(kw in q for kw in cpe_keywords)

def contains_link_request(text):
    import re
    # Look for common link request phrases
    link_patterns = [
        r'link', r'website', r'url', r'webpage', r'can you give.*link', r'can you provide.*link', r'can you send.*link', r'can you share.*link', r'give me.*link', r'provide.*link', r'send.*link', r'share.*link', r'where can i find.*link', r'where can i get.*link', r'where do i get.*link', r'where do i find.*link', r'can i have.*link', r'can i get.*link', r'can i see.*link', r'can i access.*link', r'can you give.*website', r'can you provide.*website', r'can you send.*website', r'can you share.*website', r'give me.*website', r'provide.*website', r'send.*website', r'share.*website', r'where can i find.*website', r'where can i get.*website', r'where do i get.*website', r'where do i find.*website', r'can i have.*website', r'can i get.*website', r'can i see.*website', r'can i access.*website',
    ]
    for pat in link_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

# --- Helper: Detect illegal/unsafe queries ---
def is_illegal_query(text):
    illegal_patterns = [
        r"dark web", r"how to hack", r"how to kill", r"how to murder", r"how to make a bomb", r"how to use drugs", r"buy drugs", r"illegal drugs", r"child porn", r"terrorist", r"terrorism", r"commit crime", r"steal", r"rob", r"rape", r"assault", r"suicide", r"self-harm", r"harm others", r"violence", r"explosives", r"weapon", r"guns", r"shoot", r"stab", r"poison", r"overdose", r"illegal activity", r"crime", r"criminal activity"
    ]
    for pat in illegal_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

# --- Helper: Detect greeting/closing with intent to ask ("I have a question") ---
def is_greeting_with_intent_to_ask(text):
    greeting_keywords = get_greeting_keywords()
    has_greeting = any(k in text.lower() for k in greeting_keywords)
    intent_patterns = [r"i have (a|some|few)? questions?", r"may (tanong|katanungan)", r"i want to ask", r"can i ask", r"mag tatanong"]
    for pat in intent_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return has_greeting
    return False

# --- Helper: Improved closing detection ---
def detect_closing(text):
    closing_keywords = get_language_keywords(categories=['closing'])
    return any(k in text.lower() for k in closing_keywords)

# --- Advanced Intent/Time/Room Detection Helpers (from Azure) ---
def get_current_time_info():
    now = datetime.now()
    hour = now.hour
    if hour < 12:
        part_of_day = 'morning'
    elif hour < 18:
        part_of_day = 'afternoon'
    else:
        part_of_day = 'evening'
    return {
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M'),
        'part_of_day': part_of_day
    }

def is_ask_day_query(text):
    patterns = [
        r'what day is it( today)?',
        r'anong araw( na)?( ngayon)?',
        r'ano ang araw( ngayon)?',
        r'what day today',
        r'what is today',
        r'what day is today',
        r'what day are we',
        r'what day',
        r'anong araw',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def is_ask_time_query(text):
    patterns = [
        r'what time is it',
        r'anong oras( na)?',
        r'ano ang oras',
        r'what time now',
        r'what is the time',
        r'what time',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def is_ask_room_today_query(text):
    patterns = [
        r'(what|which) room.*(available|vacant|free).*today',
        r'anong room.*(bakante|malaya|walang laman|walang tao).*ngayon',
        r'(room|kwarto).*available.*today',
        r'(room|kwarto).*bakante.*ngayon',
        r'(room|kwarto).*free.*today',
        r'(room|kwarto).*vacant.*today',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def is_room_query(text):
    # Heuristic: looks for room, vacant, free, available, bakante, malaya, etc. and a time or time range
    import re
    patterns = [
        r'(room|kwarto|cea)[\w\s]*?(vacant|free|available|bakante|malaya|walang laman|walang tao)',
        r'(anong|which|what)[\w\s]*?(room|kwarto|cea)[\w\s]*?(vacant|free|available|bakante|malaya|walang laman|walang tao)',
        r'(room|kwarto|cea)[\w\s]*?(at|ng|on)?[\w\s]*?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
        r'(room|kwarto|cea)[\w\s]*?(from|to|between|hanggang|mula|sa|ng)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

@app.route('/', methods=['GET'])
def verify():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Token verification failed", 403
        return str(request.args["hub.challenge"]), 200 #delete string if needed
    return "The chatbot is working!", 200

@app.route('/', methods=['POST'])
def webhook():
    data = request.get_json()
    if data["object"] == "page":
        normalization_dict = load_normalization_dict_cached()
        cpe_keywords = get_cpe_keywords()
        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:
                message_id = messaging_event.get("message", {}).get("mid")
                if message_id and message_id in deduped_message_ids:
                    continue
                if message_id:
                    deduped_message_ids.add(message_id)
                sender_id = messaging_event["sender"]["id"]
                if messaging_event.get("message"):
                    message = messaging_event["message"]
                    message_text = message.get("text")
                    if not message_text:
                        continue
                    lang = detect_language(message_text)
                    faqs = get_combined_faqs(sender_id)  # Use combined master+department FAQ logic
                    dept = detect_department(message_text)
                    if dept:
                        user_department_context[sender_id] = dept
                    
                    if "what departments" in message_text.lower():
                        depts = get_available_departments()
                        bot.send_text_message(sender_id, "Available departments: " + ', '.join(dept.upper() for dept in depts))
                        continue
                    # Context tracking for greetings
                    if sender_id not in user_greeting_context:
                        user_greeting_context[sender_id] = {'greeted': False}
                    context = user_greeting_context[sender_id]
                    # Context tracking for clarifications
                    if sender_id not in user_query_context:
                        user_query_context[sender_id] = {'pending_clarification': None, 'last_question': None, 'last_message': None}
                    query_context = user_query_context[sender_id]
                    is_greeting = detect_greeting(message_text)
                    questions = extract_questions(message_text)
                    is_assignment = detect_assignment_homework(message_text)
                    thanks_keywords = get_thanks_keywords()
                    closing_keywords = get_language_keywords(categories=['closing'])
                    has_thanks = any(k in message_text.lower() for k in thanks_keywords)
                    has_closing = any(k in message_text.lower() for k in closing_keywords)

                    # --- ADVANCED INTENT/TIME/ROOM DETECTION (from Azure) ---
                    if is_ask_day_query(message_text):
                        time_info = get_current_time_info()
                        day = datetime.now().strftime('%A')
                        bot.send_text_message(sender_id, f"Today is {day} ({time_info['date']}).")
                        continue
                    if is_ask_time_query(message_text):
                        time_info = get_current_time_info()
                        bot.send_text_message(sender_id, f"The current time is {time_info['time']} on {time_info['date']}.")
                        continue
                    if is_ask_room_today_query(message_text):
                        schedule = get_weekly_room_schedule()
                        day = datetime.now().strftime('%A').lower()
                        day_sched = schedule.get(day)
                        if not day_sched:
                            bot.send_text_message(sender_id, f"Sorry, I couldn't find the schedule for today.")
                            continue
                        # Use robust room column detection
                        room_keys = [k for k in day_sched[0].keys() if is_room_column(k)]
                        vacant_rooms = []
                        for row in day_sched:
                            for room in room_keys:
                                if not row.get(room, '').strip() and room not in vacant_rooms:
                                    vacant_rooms.append(room)
                        if vacant_rooms:
                            bot.send_text_message(sender_id, f"The following rooms are vacant at some time today: {', '.join(vacant_rooms)}. If you would like to reserve a room, you may coordinate with our student assistants, iskolar!")
                        else:
                            bot.send_text_message(sender_id, "Sorry, I couldn't find any vacant rooms for today.")
                        continue
                    # --- END ADVANCED INTENT/TIME/ROOM DETECTION ---

                    # --- ADVANCED ROOM/TIME RANGE QUERY HANDLING ---
                    if is_room_query(message_text):
                        bot.send_text_message(sender_id, handle_room_query(message_text, lang))
                        continue
                    # --- END ADVANCED ROOM/TIME RANGE QUERY HANDLING ---

                    # --- Context-aware clarification logic ---
                    # If there is a pending clarification, treat this message as the clarification
                    if query_context['pending_clarification']:
                        unclear_term = query_context['pending_clarification']
                        clarified_question = query_context['last_question']
                        # Combine the unclear question and the user's clarification
                        system_prompt = PIXEL_PERSONALITY[lang]
                        user_prompt = (
                            f"The user previously asked: '{clarified_question}', but the term '{unclear_term}' was unclear. "
                            f"The user clarified: '{message_text}'. As PIXEL, provide a helpful answer using this clarification."
                        )
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        try:
                            gpt_response = ask_github_gpt(messages)
                            bot.send_text_message(sender_id, gpt_response)
                            # Track last topic/answer for follow-up
                            query_context['last_topic'] = clarified_question
                            query_context['last_answer'] = gpt_response
                        except Exception as gpt_e:
                            fallback = ("Sorry, I encountered an error. Please try again." if lang == 'en' else "Paumanhin, may naganap na error. Pakisubukan muli.")
                            bot.send_text_message(sender_id, fallback)
                        # Clear clarification context after use
                        query_context['pending_clarification'] = None
                        query_context['last_question'] = None
                        query_context['last_message'] = message_text
                        continue

                    # --- Context-aware follow-up for more info/resources ---
                    if is_followup_resource_request(message_text):
                        last_topic = query_context.get('last_topic')
                        last_answer = query_context.get('last_answer')
                        if last_topic:
                            system_prompt = PIXEL_PERSONALITY[lang]
                            user_prompt = (
                                f"The user previously asked about: '{last_topic}'. You replied: '{last_answer}'. Now the user wants more info or resources. "
                                f"As PIXEL, provide more specific information, resources, or references about the topic. If you know official department or academic resources, include them."
                            )
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ]
                            try:
                                gpt_response = ask_github_gpt(messages)
                                bot.send_text_message(sender_id, gpt_response)
                                # Update last answer
                                query_context['last_answer'] = gpt_response
                            except Exception as gpt_e:
                                fallback = ("Sorry, I encountered an error. Please try again." if lang == 'en' else "Paumanhin, may naganap na error. Pakisubukan muli.")
                                bot.send_text_message(sender_id, fallback)
                        else:
                            fallback = ("Sorry, I need more context. Please specify the topic you want resources for." if lang == 'en' else "Paumanhin, kailangan ko ng mas malinaw na paksa. Pakispecify kung anong topic ang gusto mong resources.")
                            bot.send_text_message(sender_id, fallback)
                        continue

                    # --- Illegal/unsafe query detection (highest priority) ---
                    if is_illegal_query(message_text):
                        system_prompt = PIXEL_PERSONALITY[lang]
                        user_prompt = (
                            "The user asked an illegal, unsafe, or inappropriate question: '"
                            + message_text + "'. As PIXEL, politely but firmly refuse to answer, discourage such actions, and remind the user that you cannot help with these topics. Do not provide any instructions or encouragement."
                        )
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        try:
                            gpt_response = ask_github_gpt(messages)
                            bot.send_text_message(sender_id, gpt_response)
                        except Exception as gpt_e:
                            fallback = ("Sorry, I can't help with that." if lang == 'en' else "Paumanhin, hindi ako pwedeng tumulong diyan.")
                            bot.send_text_message(sender_id, fallback)
                        continue

                    # --- Greeting with intent to ask, but no actual question ---
                    if is_greeting_with_intent_to_ask(message_text) and not questions:
                        greeting = get_dynamic_greeting(lang)
                        bot.send_text_message(sender_id, greeting + (" Go ahead and ask your questions!" if lang == 'en' else " Sige, itanong mo lang!"))
                        context['greeted'] = True
                        continue

                    # --- Greeting/Closing only (no questions) ---
                    if is_greeting and not questions:
                        if context['greeted']:
                            greeting = get_dynamic_greeting_again(lang)
                        else:
                            greeting = get_dynamic_greeting(lang)
                            context['greeted'] = True
                        bot.send_text_message(sender_id, greeting)
                        continue
                    if detect_closing(message_text) and not questions:
                        bot.send_text_message(sender_id, get_dynamic_closing(lang))
                        continue

                    # PRIORITY 1: Homework/assignment detection (block all other answers if present)
                    if is_assignment:
                        bot.send_text_message(sender_id, get_assignment_guidance_response(message_text, lang))
                        continue

                    # PRIORITY 2: Thanks/closing dynamic responses (only if not combined with a question)
                    if (has_thanks or has_closing) and not questions:
                        if has_thanks and not has_closing:
                            bot.send_text_message(sender_id, get_dynamic_thanks(lang))
                        elif has_closing and not has_thanks:
                            bot.send_text_message(sender_id, get_dynamic_closing(lang))
                        elif has_thanks and has_closing:
                            bot.send_text_message(sender_id, get_dynamic_thanks(lang))
                        continue

                    # PRIORITY 3: Greeting only (no questions)
                    if is_greeting and not questions:
                        if context['greeted']:
                            greeting = get_dynamic_greeting_again(lang)
                        else:
                            greeting = get_dynamic_greeting(lang)
                            context['greeted'] = True
                        bot.send_text_message(sender_id, greeting)
                        continue

                    # PRIORITY 4: If greeting + question(s), answer question(s) only (no greeting reply)
                    if questions:
                        answers = []
                        faq_matches = []
                        non_faq_questions = []
                        for q in questions:
                            print(f"[DEBUG][webhook] Checking question: {repr(q)}")
                            q_norm = normalize_query(q, normalization_dict)
                            print(f"[DEBUG][webhook] Normalized question: {repr(q_norm)}")
                            print(f"[DEBUG][webhook] FAQ cache size: {len(faqs)}")
                            faq_match = match_faq(q_norm, faqs, lang, normalization_dict=normalization_dict)
                            if faq_match:
                                print(f"[DEBUG][webhook] FAQ match found: {faq_match['question']} (sheet: {faq_match.get('source', 'unknown')})")
                                faq_matches.append((q, faq_match))
                            else:
                                print(f"[DEBUG][webhook] No FAQ match found for: {repr(q)}")
                                non_faq_questions.append(q)
                        # If any FAQ matches, combine answers contextually
                        if faq_matches:
                            combined = []
                            for idx, (q, faq) in enumerate(faq_matches):
                                skip_greeting = idx > 0
                                if idx == 0:
                                    answer = get_personalized_faq_response(faq['question'], faq['answer'], lang, user_query=q, context=context, skip_greeting=skip_greeting)
                                else:
                                    answer = get_personalized_faq_response(faq['question'], faq['answer'], lang, user_query=q, context=context, skip_greeting=True)
                                    answer = re.sub(r'^(hi|hello|heyyy|kumusta|kamusta|magandang \\w+|good \\w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*', '', answer, flags=re.IGNORECASE)
                                    answer = f"As for your next question, {answer.strip()}"
                                combined.append(answer.strip())
                            answers.append("\n\n".join(combined))
                        # For non-FAQ questions, classify as CPE or general
                        for q in non_faq_questions:
                            if contains_link_request(q):
                                answers.append("Sorry, I can't send links unless they're part of the official FAQ. If you need a specific link, please check the FAQ or ask the department directly.")
                                continue
                            if is_cpe_related(q, cpe_keywords):
                                log_unanswered(q, lang)
                                system_prompt = PIXEL_PERSONALITY[lang]
                                user_prompt = f"The user asked: '{q}'. This is not in the FAQ. As PIXEL, provide a helpful answer, but first say that the exact answer isn't in the FAQ and will be forwarded to the department."
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                                try:
                                    gpt_response = ask_github_gpt(messages)
                                    answers.append(gpt_response)
                                    # Track last topic/answer for follow-up
                                    query_context['last_topic'] = q
                                    query_context['last_answer'] = gpt_response
                                except Exception as gpt_e:
                                    fallback = ("Sorry, I encountered an error. Please try again." if lang == 'en' else "Paumanhin, may naganap na error. Pakisubukan muli.")
                                    answers.append(fallback)
                            else:
                                # General query, let GPT-4.1 answer directly (no preamble)
                                system_prompt = PIXEL_PERSONALITY[lang]
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": q}
                                ]
                                try:
                                    gpt_response = ask_github_gpt(messages)
                                    answers.append(gpt_response)
                                    # Track last topic/answer for follow-up
                                    query_context['last_topic'] = q
                                    query_context['last_answer'] = gpt_response
                                except Exception as gpt_e:
                                    fallback = ("Sorry, I encountered an error. Please try again." if lang == 'en' else "Paumanhin, may naganap na error. Pakisubukan muli.")
                                    answers.append(fallback)
                        # Mark as greeted if user included a greeting
                        if is_greeting:
                            context['greeted'] = True
                        else:
                            context['greeted'] = False
                        # One message, one reply: join all answers
                        bot.send_text_message(sender_id, "\n\n".join(answers))
                        check_github_model_rate_limit()
                        continue

                    # PRIORITY 5: Fallback (not a greeting, not a question, not thanks/closing, not assignment)
                    log_unanswered(message_text, lang)
                    fallback = ("Sorry, I couldn't understand your message. Could you please rephrase or ask about CpE topics?" if lang == 'en' else "Paumanhin, hindi ko naintindihan ang iyong mensahe. Maaari mo bang ulitin o magtanong tungkol sa CpE?")
                    bot.send_text_message(sender_id, fallback)
                    check_github_model_rate_limit()
    return "ok", 200

if __name__ == '__main__':
    app.run(debug=True)
