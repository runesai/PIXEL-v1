from sheets_utils import get_sheets_service
import os
from dotenv import load_dotenv

print("Loaded keyword_utils.py")

load_dotenv()
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

def get_language_keywords(categories=None):
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

def get_acad_dept_lab_keywords():
    service = get_sheets_service()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range='Language!A2:D'
    ).execute()
    values = result.get('values', [])
    keywords = set()
    for row in values:
        if len(row) >= 3:
            category = row[2].strip().lower()
            if category in ['academic', 'department', 'lab', 'laboratory']:
                keywords.add(row[0].strip().lower())
    return keywords

def get_cpe_keywords():
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
            if category in ['department', 'academic', 'lab', 'process', 'politeness', 'pronoun', 'particle', 'common', 'question']:
                cpe_keywords.add(row[0].strip().lower())
    return cpe_keywords

def is_cpe_related(query, cpe_keywords):
    q = query.lower()
    return any(kw in q for kw in cpe_keywords)
