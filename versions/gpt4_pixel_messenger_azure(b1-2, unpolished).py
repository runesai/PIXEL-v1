from flask import Flask, request
from pymessenger.bot import Bot
from dotenv import load_dotenv
from langdetect import detect
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime
import os
from openai import AzureOpenAI
import requests
import difflib
import random
import re
import time

# Load environment variables
load_dotenv()
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'credentials.json'

app = Flask(__name__)
bot = Bot(PAGE_ACCESS_TOKEN)

PIXEL_PERSONALITY = {
    'en': "You are PIXEL, a polite and helpful student assistant at the Computer Engineering Department of the Polytechnic University of the Philippines who is also aware of the latest trends in technology. You are familiar with department procedures, requirements, and academic policies. You help students in a fun and friendly manner but with a hint of professionalism. You are also aware of the latest trends in filipino pop-culture and respond like a trendy young adult. You also refer to the users as 'iskolar' from time to time. If a question is out of scope, politely say so.",
    'fil': "Ikaw si PIXEL, isang magalang at matulunging student assistant ng Computer Engineering Department ng Polytechnic University of the Philippines na may kaalaman sa mga pinakabagong uso sa teknolohiya. Pamilyar ka sa mga proseso, requirements, at patakaran ng departamento. Ikaw ay friendly at masaya na makipagtulong sa mga estudyante pero ikaw ay may pagka-propesyonal din. Ikaw ay may kaalaman din sa mga pinakabagong uso sa pop-culture ng mga Pilipino at sumasagot tulad ng isang trendy na filipino young adult. Tinatawag mo rin ang mga users na 'iskolar' paminsan-minsan. Kung ang tanong ay wala sa iyong saklaw, sabihin ito nang magalang."
}

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

def get_sheets_service():
    print("[LOG] Accessing Google Sheets service...")
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('sheets', 'v4', credentials=creds)

def get_faqs():
    print("[LOG] Accessing FAQ sheet...")
    service = get_sheets_service()
    sheet = service.spreadsheets()
    try:
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range='FAQ!A2:C').execute()
        values = result.get('values', [])
        print(f"[DEBUG] Raw FAQ values from sheet: {values}")
        faqs = []
        for row in values:
            if len(row) >= 2:
                faqs.append({
                    'question': row[0].strip(),
                    'answer': row[1].strip(),
                    'lang': row[2].strip().lower() if len(row) > 2 else 'en'
                })
        print(f"[DEBUG] Parsed FAQs: {faqs}")
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
        print(f"[DEBUG] Parsed {sheet_name} FAQs: {faqs}")
        return faqs
    except Exception as e:
        print(f"[ERROR] Failed to fetch {sheet_name} FAQs: {e}")
        return []

def log_unanswered(query, lang):
    service = get_sheets_service()
    sheet = service.spreadsheets()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = {'values': [[query, lang, now]]}
    sheet.values().append(
        spreadsheetId=SPREADSHEET_ID,
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
    queries_to_try = [query]
    if normalization_dict:
        queries_to_try.append(normalize_query(query, normalization_dict))
    best_score = 0
    best_faq = None
    best_lang = None
    # Try user language, then English, then Filipino
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
    # Lower threshold slightly for better matching
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
        answer = re.sub(r'^(hi|hello|heyyy|kumusta|kamusta|magandang \w+|good \w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*', '', answer, flags=re.IGNORECASE)
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

def get_dynamic_greeting(lang, time_info=None):
    system_prompt = PIXEL_PERSONALITY[lang]
    if time_info:
        user_prompt = f"Greet the user as PIXEL in a trendy, friendly way. Use the user's language. It is currently {time_info['part_of_day']} ({time_info['time']}) on {time_info['date']}. Reference the time of day in your greeting if appropriate."
    else:
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

def get_dynamic_closing(lang, time_info=None):
    system_prompt = PIXEL_PERSONALITY[lang]
    if time_info:
        user_prompt = f"Say goodbye or closing statement as PIXEL in a trendy, friendly way. Use the user's language. It is currently {time_info['part_of_day']} ({time_info['time']}) on {time_info['date']}. Reference the time of day in your closing if appropriate."
    else:
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

# --- Caching for FAQ and normalization dict ---
_FAQ_CACHE = None
_FAQ_CACHE_TIME = 0
_FAQ_CACHE_TTL = 30  # seconds (reduced for faster Google Sheets update reflection during testing)

_NORMALIZATION_CACHE = None
_NORMALIZATION_CACHE_TIME = 0
_NORMALIZATION_CACHE_TTL = 30  # seconds (reduced for faster Google Sheets update reflection during testing)

def get_faqs_cached():
    global _FAQ_CACHE, _FAQ_CACHE_TIME
    now = time.time()
    if _FAQ_CACHE is None or now - _FAQ_CACHE_TIME > _FAQ_CACHE_TTL:
        faqs_main = get_faqs()
        faqs_lab = get_sheet_faqs('Laboratory')
        faqs_pixel = get_sheet_faqs('PIXEL')
        faqs_dept = get_sheet_faqs('Department')
        _FAQ_CACHE = faqs_main + faqs_lab + faqs_pixel + faqs_dept
        _FAQ_CACHE_TIME = now
    return _FAQ_CACHE

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

# --- Helper: Improved greeting detection (allow extra words, e.g., "good morning, pixel!") ---
def detect_greeting(text):
    greeting_keywords = get_greeting_keywords()
    text_l = text.lower().strip()
    # Match if greeting keyword is at the start, or if the message contains a greeting phrase (even with extra words)
    for k in greeting_keywords:
        if text_l.startswith(k) or re.search(rf'\b{k}\b', text_l):
            return True
    # Also match common time-based greetings with extra words (e.g., "good morning, pixel!")
    if re.match(r'^(good|magandang) (morning|afternoon|evening|day|gabi|umaga|tanghali|hapon|gabi)', text_l):
        return True
    return False

# --- Helper: Get current time and date for PIXEL's awareness ---
def get_current_time_info():
    from datetime import datetime
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

# --- Helper: Improved intent-to-ask detection (more variations) ---
def is_greeting_with_intent_to_ask(text):
    greeting_keywords = get_greeting_keywords()
    has_greeting = any(k in text.lower() for k in greeting_keywords)
    intent_patterns = [
        r"i have (a|some|few)? questions?",
        r"may (tanong|katanungan)",
        r"i want to ask",
        r"can i ask( a)? questions?",
        r"can i ask( a)? question[?]?",
        r"mag tatanong",
        r"i have a question",
        r"i have some questions",
        r"i have a few questions",
        r"pwede po magtanong",
        r"can i ask something",
        r"may i ask",
        r"can i ask you something",
        r"can i ask for help",
        r"i have a follow[- ]?up question[.!?]?",
        r"i have another question[.!?]?",
        r"i have another question though[.!?]?",
        r"i have a question though[.!?]?",
        r"i have a follow[- ]?up[.!?]?"
    ]
    for pat in intent_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return has_greeting or detect_greeting(text)
    return False

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
    question_starters = r'(how|what|when|where|who|why|can|is|are|do|does|did|will|would|should|could|may|paano|ano|saan|kailan|bakit|pwede|puwede|magkano|ilan|sinong|sino|pwedeng|paano|mag|may|meron|possible|possible ba|tell me|give me|explain|define|describe|list|provide|show|help|need|want|require|can you|could you|would you|can you|please|gusto|kailangan|ano ang|ano ba|ano po|ano nga|ano kaya|ano yung|ano iyong|ano ito|ano iyan|ano yan|ano ito|ano po ba|ano po ang|ano po ito|ano po iyan|ano po yan|ano po ito|ano po ba ito|ano po ba iyan|ano po ba yan|ano po ba ito|ano po ba iyan|ano po ba yan)'
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

def get_dynamic_greeting_again(lang, time_info=None):
    system_prompt = PIXEL_PERSONALITY[lang]
    if time_info:
        user_prompt = f"The user greeted you again. Respond as PIXEL with a friendly, trendy 'hello again' message, in the user's language. It is currently {time_info['part_of_day']} ({time_info['time']}) on {time_info['date']}. Reference the time of day if appropriate."
    else:
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
    # Extract department/lab/academic keywords from language sheet
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
    intent_patterns = [r"i have (a|some|few)? ?questions?( though)?[.!?]?",
                       r"i have (a|some|few)? ?question( though)?[.!?]?",
                       r"may (tanong|katanungan)",
                       r"i want to ask",
                       r"can i ask( a)? questions?[.!?]?",
                       r"can i ask( a)? question[.!?]?",
                       r"mag tatanong",
                       r"i have some questions",
                       r"i have a few questions",
                       r"pwede po magtanong",
                       r"pwede po magpatulong",
                       r"can i ask something",
                       r"may i ask",
                       r"can i ask you something",
                       r"can i ask for help",
                       r"i have a follow[- ]?up question[.!?]?",
                       r"i have another question[.!?]?",
                       r"i have another question though[.!?]?",
                       r"i have a question though[.!?]?",
                       r"i have a follow[- ]?up[.!?]?"]
    for pat in intent_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return has_greeting
    return False

# --- Missing helper function: strip_greeting ---
import re

def strip_greeting(text):
    # Remove common greetings at the start of a string (PIXEL style)
    return re.sub(
        r'^(hi|hello|heyyy|kumusta|kamusta|magandang \w+|good \w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*',
        '', text, flags=re.IGNORECASE
    ).strip()

def get_dynamic_not_in_faq_preamble(lang, with_greeting=True):
    system_prompt = PIXEL_PERSONALITY[lang]
    if with_greeting:
        user_prompt = "A user asked a CpE/academic/department/lab question that is not in the official FAQ. As PIXEL, generate a friendly, trendy preamble with a greeting, telling the user you'll forward the question and provide helpful info."
    else:
        user_prompt = "A user asked a CpE/academic/department/lab question that is not in the official FAQ. As PIXEL, generate a friendly, trendy preamble WITHOUT a greeting, telling the user you'll forward the question and provide helpful info."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        return ("Hi iskolar! The exact answer to your question is not in the official FAQ. I'll forward this to the CpE Department/Laboratory. Here's some helpful info:" if with_greeting else "This is not in the official FAQ. I'll forward this to the CpE Department/Laboratory. Here's some helpful info:")

def get_transition_phrase(idx, question):
    # Returns a transition phrase for subsequent answers
    base_phrases = [
        "As for your next question:",
        "To answer your next query:",
        f"To answer your question '{question.strip()}':",
        "Moving on to your next question:",
        "Here's what I can say about your next question:"
    ]
    # Cycle through or pick based on idx
    return base_phrases[(idx-2) % len(base_phrases)]

def get_transition_phrase_lang(idx, question, lang):
    if lang == 'fil':
        base_phrases = [
            "Para sa sumunod mo na tanong:",
            f"Para sa tanong mong '{question.strip()}':",
            "Narito ang sagot sa sunod mong tanong:",
            "Para naman sa sunod mong tanong:"
        ]
    else:
        base_phrases = [
            "As for your next question:",
            f"To answer your question '{question.strip()}':",
            "Moving on to your next question:",
            "Here's what I can say about your next question:"
        ]
    return base_phrases[(idx-2) % len(base_phrases)]

# --- Helper: Dynamic fallback message ---
def get_dynamic_fallback_message(lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "You could not understand the user's question or instruction. As PIXEL, reply in a friendly, trendy way: 'Sorry, I didn't catch your question. Could you please ask your question?' Never use a closing statement."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        return ("Sorry, I didn't catch your question. Could you please ask your question?")

# --- Helper: Detect imperative/instruction queries ---
def is_instruction_query(text, lang):
    patterns_en = [
        r'^(make|create|list|give|show|write|generate|compose|draw|explain|summarize|describe|tell|provide|enumerate|can you list|can you make|can you give|can you show|can you write|can you generate|can you compose|can you draw|can you explain|can you summarize|can you describe|can you tell|can you provide|can you enumerate|give me instructions|give instructions|how do i|how to|instructions for|steps for|steps to|guide to|guide for)',
    ]
    patterns_fil = [
        r'^(gawin|gumawa|ilista|ibigay|ipakita|isulat|gawin mo|pwede mo bang gawin|pwede mo bang ilista|pwede mo bang ibigay|pwede mo bang ipakita|pwede mo bang isulat|paki-|magbigay|maglista|magpakita|magsulat|magcompose|maggenerate|magdraw|magpaliwanag|magsummarize|magdescribe|magsabi|magprovide|mag-enumerate|paano gumawa|paano ilista|paano ibigay|paano ipakita|paano isulat|paano magbigay|paano maglista|paano magpakita|paano magsulat|paano magcompose|paano maggenerate|paano magdraw|paano magpaliwanag|paano magsummarize|paano magdescribe|paano magsabi|paano magprovide|paano mag-enumerate|paano gumawa ng|paano ilista ang|paano ibigay ang|paano ipakita ang|paano isulat ang|paano magbigay ng|paano maglista ng|paano magpakita ng|paano magsulat ng|paano magcompose ng|paano maggenerate ng|paano magdraw ng|paano magpaliwanag ng|paano magsummarize ng|paano magdescribe ng|paano magsabi ng|paano magprovide ng|paano mag-enumerate ng|paano gumawa ng|paano ilista ng|paano ibigay ng|paano ipakita ng|paano isulat ng|paano magbigay ng|paano maglista ng|paano magpakita ng|paano magsulat ng|paano magcompose ng|paano maggenerate ng|paano magdraw ng|paano magpaliwanag ng|paano magsummarize ng|paano magdescribe ng|paano magsabi ng|paano magprovide ng|paano mag-enumerate ng)',
    ]
    patterns = patterns_en if lang == 'en' else patterns_en + patterns_fil
    for pat in patterns:
        if re.search(pat, text.strip().lower()):
            return True
    return False

@app.route('/', methods=['GET'])
def verify():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Token verification failed", 403
        return request.args["hub.challenge"], 200
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
                    faqs = get_faqs_cached()
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

                    # --- Illegal/unsafe query detection (highest priority, dynamic PIXEL response) ---
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
                    # Remove intent-to-ask phrases from questions
                    filtered_questions = [q for q in questions if not is_intent_to_ask_phrase(q)]
                    # If nothing left, try stripping intent-to-ask phrase from the message and extract again
                    if not filtered_questions:
                        cleaned_message = strip_intent_to_ask_phrase(message_text)
                        if cleaned_message and not is_intent_to_ask_phrase(cleaned_message):
                            filtered_questions = extract_questions(cleaned_message)
                            filtered_questions = [q for q in filtered_questions if q.strip()]
                    only_intent_to_ask = (
                        not filtered_questions or
                        (len(filtered_questions) == 1 and is_intent_to_ask_phrase(filtered_questions[0]))
                    )
                    if is_greeting_with_intent_to_ask(message_text) and only_intent_to_ask:
                        # Use dynamic intent-to-ask prompt, never fallback or closing
                        bot.send_text_message(sender_id, get_dynamic_intent_to_ask_prompt(lang))
                        context['greeted'] = True
                        continue

                    # --- Greeting/Closing only (no real questions) ---
                    if is_greeting and (not filtered_questions or (len(filtered_questions) == 1 and filtered_questions[0].strip().lower() == message_text.strip().lower())):
                        time_info = get_current_time_info()
                        if context['greeted']:
                            greeting = get_dynamic_greeting_again(lang, time_info=time_info)
                        else:
                            greeting = get_dynamic_greeting(lang, time_info=time_info)
                            context['greeted'] = True
                        bot.send_text_message(sender_id, greeting)
                        continue
                    if detect_closing(message_text) and (not filtered_questions or (len(filtered_questions) == 1 and filtered_questions[0].strip().lower() == message_text.strip().lower())):
                        time_info = get_current_time_info()
                        # Never reply with a closing as fallback to intent-to-ask
                        if only_intent_to_ask:
                            fallback = ("Sorry, I didn't catch your question. Could you please ask your question?" if lang == 'en' else "Paumanhin, hindi ko nakuha ang iyong tanong. Maaari mo bang itanong ulit?")
                            bot.send_text_message(sender_id, fallback)
                        else:
                            bot.send_text_message(sender_id, get_dynamic_closing(lang, time_info=time_info))
                        continue

                    # PRIORITY 4: If greeting + question(s), answer question(s) only (no greeting reply)
                    if filtered_questions:
                        answers = []
                        faq_matches = []
                        non_faq_questions = []
                        user_instruction = extract_user_instruction(message_text)
                        listed_questions = extract_listed_questions(message_text)
                        if listed_questions and user_instruction:
                            filtered_questions = listed_questions
                        # --- Get academic/department/lab keywords for logging/preamble ---
                        acad_dept_lab_keywords = get_acad_dept_lab_keywords()
                        answer_texts = []
                        answer_questions = []
                        preamble_needed = False
                        preamble_inserted = False
                        for q in filtered_questions:
                            q_norm = normalize_query(q, normalization_dict)
                            faq_match = match_faq(q_norm, faqs, lang, normalization_dict=normalization_dict)
                            if faq_match:
                                answer = get_personalized_faq_response(faq_match['question'], faq_match['answer'], lang, user_query=q, context=context, skip_greeting=(len(answer_texts) > 0))
                                # For all but the first answer, add a transition phrase and never a greeting
                                if len(answer_texts) > 0:
                                    transition = get_transition_phrase(len(answer_texts)+1, q, lang)
                                    answer = strip_greeting(answer)
                                    answer = f"{transition} {answer}"
                                answer_texts.append(answer)
                                answer_questions.append(q)
                            else:
                                # Check if query contains any academic/department/lab keyword (EN or FIL)
                                q_l = q.lower()
                                is_acad_dept_lab = any(kw in q_l for kw in acad_dept_lab_keywords)
                                answer_questions.append(q)
                                if is_acad_dept_lab:
                                    preamble_needed = True
                                    answer_texts.append(None)  # Mark for preamble logic
                                else:
                                    answer_texts.append(None)
                        # --- Now generate answers for non-FAQ questions ---
                        for idx, (q, a) in enumerate(zip(answer_questions, answer_texts)):
                            if a is not None:
                                continue  # Already answered as FAQ
                            q_l = q.lower()
                            is_acad_dept_lab = any(kw in q_l for kw in acad_dept_lab_keywords)
                            q_type = classify_cpe_query(q, cpe_keywords)
                            # Instruction/imperative detection
                            if is_instruction_query(q, lang):
                                system_prompt = PIXEL_PERSONALITY[lang]
                                user_prompt = (
                                    f"The user gave this instruction or query: '{q}'. As PIXEL, explain in a friendly, trendy way that you aren't exactly made for non-CpE/tech queries, but still try to help and provide a relevant, helpful answer or suggestion. If you can't answer, generate a dynamic fallback in PIXEL's voice based on: 'Sorry, I didn't catch your question. Could you please ask your question?' Never use a closing statement."
                                )
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                                try:
                                    gpt_response = ask_github_gpt(messages)
                                    gpt_response = cpe_shortcut(gpt_response)
                                    answer = gpt_response
                                except Exception:
                                    answer = get_dynamic_fallback_message(lang)
                                if idx > 0:
                                    transition = get_transition_phrase(idx+1, q, lang)
                                    answer = strip_greeting(answer)
                                    answer = f"{transition} {answer}"
                                answer_texts[idx] = answer
                                continue
                            if contains_link_request(q):
                                answer = "Sorry, I can't send links unless they're part of the official FAQ. If you need a specific link, please check the FAQ or ask the department directly."
                                if idx > 0:
                                    transition = get_transition_phrase(idx+1, q, lang)
                                    answer = f"{transition} {answer}"
                                answer_texts[idx] = answer
                                continue
                            if is_acad_dept_lab:
                                log_unanswered(q, lang)
                                with_greeting = (idx == 0)
                                preamble = get_dynamic_not_in_faq_preamble(lang, with_greeting=with_greeting)
                                # Dynamic answer after preamble, instruct not to mention names unless in FAQ
                                system_prompt = PIXEL_PERSONALITY[lang]
                                user_prompt = f"The user asked: '{q}'. This is not in the official FAQ. As PIXEL, provide a helpful suggestion or info. Do NOT mention any specific person's name unless it is in the official FAQ. Do not repeat the preamble or greeting. Use 'CpE' as a shortcut if needed."
                                if user_instruction:
                                    user_prompt += f" The user also requested: '{user_instruction}'."
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                                try:
                                    gpt_response = ask_github_gpt(messages)
                                    gpt_response = cpe_shortcut(gpt_response)
                                    answer = preamble + "\n\n" + gpt_response
                                except Exception:
                                    answer = preamble + "\n\n" + get_dynamic_fallback_message(lang)
                                if idx > 0:
                                    transition = get_transition_phrase(idx+1, q, lang)
                                    answer = strip_greeting(answer)
                                    answer = f"{transition} {answer}"
                                answer_texts[idx] = answer
                            elif q_type == 'general':
                                # General/non-CpE/tech query, no preamble, no logging
                                # Always try to answer, but mention PIXEL's scope
                                system_prompt = PIXEL_PERSONALITY[lang]
                                user_prompt = (
                                    f"The user asked: '{q}'. This is not related to CpE or technology. As PIXEL, explain in a friendly, trendy way that you aren't exactly made for non-CpE/tech queries, but still try to help and provide a relevant, helpful answer or suggestion. If you can't answer, generate a dynamic fallback in PIXEL's voice based on: 'Sorry, I didn't catch your question. Could you please ask your question?' Never use a closing statement."
                                )
                                if user_instruction:
                                    user_prompt += f" The user also requested: '{user_instruction}'."
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                                try:
                                    gpt_response = ask_github_gpt(messages)
                                    gpt_response = cpe_shortcut(gpt_response)
                                    answer = gpt_response
                                except Exception:
                                    answer = get_dynamic_fallback_message(lang)
                                if idx > 0:
                                    transition = get_transition_phrase(idx+1, q, lang)
                                    answer = strip_greeting(answer)
                                    answer = f"{transition} {answer}"
                                answer_texts[idx] = answer
                                query_context['last_topic'] = q
                                query_context['last_answer'] = answer
                            else:
                                system_prompt = PIXEL_PERSONALITY[lang]
                                user_prompt = f"The user asked: '{q}'. As PIXEL, provide a helpful answer or suggestion. Do not use any preamble or greeting. Use 'CpE' as a shortcut if needed."
                                if user_instruction:
                                    user_prompt += f" The user also requested: '{user_instruction}'."
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                                try:
                                    gpt_response = ask_github_gpt(messages)
                                    gpt_response = cpe_shortcut(gpt_response)
                                    answer = gpt_response
                                except Exception:
                                    answer = get_dynamic_fallback_message(lang)
                                if idx > 0:
                                    transition = get_transition_phrase(idx+1, q, lang)
                                    answer = strip_greeting(answer)
                                    answer = f"{transition} {answer}"
                                answer_texts[idx] = answer
                                query_context['last_topic'] = q
                                query_context['last_answer'] = answer
                        # --- Compose the final message ---
                        final_message = ""
                        if listed_questions and user_instruction:
                            final_message += format_list_answers(filtered_questions, answer_texts, instruction=user_instruction)
                        else:
                            final_message += "\n\n".join(answer_texts)
                        bot.send_text_message(sender_id, final_message.strip())
                        check_github_model_rate_limit()
                        continue
                    # PRIORITY 5: Fallback (not a greeting, not a question, not thanks/closing, not assignment)
                    log_unanswered(message_text, lang)
                    fallback = get_dynamic_fallback_message(lang)
                    bot.send_text_message(sender_id, fallback)
                    check_github_model_rate_limit()
    return "ok", 200

def detect_closing(text):
    closing_keywords = get_language_keywords(categories=['closing'])
    text_l = text.lower()
    return any(k in text_l for k in closing_keywords)

def get_dynamic_intent_to_ask_prompt(lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "The user said an intent-to-ask phrase (like 'I have a question' or 'I have some more questions'). Respond as PIXEL in a trendy, friendly way, prompting the user to ask their actual question. Do not answer any question, just encourage them to ask."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        return "Alright, iskolar! What question do you have in mind?"

# --- Helper: Detect if user gave instructions for answering a list of questions ---
def extract_user_instruction(text):
    # Look for common instruction patterns (e.g., 'answer each', 'summarize', 'in Filipino', etc.)
    instruction_patterns = [
        r'(answer each|answer all|answer every|please answer each|please answer all|please answer every)',
        r'(summarize|summarise|in summary|give a summary)',
        r'(in filipino|in english|sagutin sa filipino|sagutin sa ingles)',
        r'(briefly|short answer|detailed answer|one sentence each|one paragraph each)'
    ]
    for pat in instruction_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None

# --- Helper: Split numbered/bulleted lists into questions ---
def extract_listed_questions(text):
    # Split on numbered or bulleted list patterns
    lines = text.split('\n')
    questions = []
    for line in lines:
        line = line.strip()
        if re.match(r'^(\d+\.|[-*•])\s+', line):
            q = re.sub(r'^(\d+\.|[-*•])\s+', '', line)
            if q:
                questions.append(q)
    # If not enough, fallback to extract_questions
    if len(questions) < 2:
        return None
    return questions

# --- Helper: Classify query as CpE Department, Laboratory, or General ---
def classify_cpe_query(query, cpe_keywords):
    # You can expand this with more precise keywords if needed
    lab_keywords = {'lab', 'laboratory', 'equipment', 'apparatus', 'experiment', 'lab room', 'lab schedule', 'lab fee'}
    dept_keywords = {'department', 'office', 'chair', 'coordinator', 'faculty', 'adviser', 'advising', 'enrollment', 'clearance', 'forms', 'requirements', 'grades', 'section', 'block', 'subject', 'curriculum', 'cpe', 'computer engineering'}
    q = query.lower()
    if any(kw in q for kw in lab_keywords):
        return 'lab'
    if any(kw in q for kw in dept_keywords) or is_cpe_related(q, cpe_keywords):
        return 'dept'
    return 'general'

# --- Helper: Replace 'Computer Engineering' with 'CpE' shortcut ---
def cpe_shortcut(text):
    return re.sub(r'computer engineering', 'CpE', text, flags=re.IGNORECASE)

# --- Helper: Format answers for lists with instructions ---
def format_list_answers(questions, answers, instruction=None):
    formatted = []
    for idx, (q, a) in enumerate(zip(questions, answers), 1):
        prefix = f"{idx}. {q.strip()}\n"
        # For all but the first answer, add a transition phrase and never a greeting
        if idx > 1:
            transition = get_transition_phrase(idx+1, q)
            a = strip_greeting(a.strip())
            a = f"{transition} {a}"
        formatted.append(f"{prefix}{a.strip()}")
    return "\n\n".join(formatted)

def get_transition_phrase(idx, question):
    # Returns a transition phrase for subsequent answers
    base_phrases = [
        "As for your next question:",
        "To answer your next query:",
        f"To answer your question '{question.strip()}':",
        "Moving on to your next question:",
        "Here's what I can say about your next question:"
    ]
    # Cycle through or pick based on idx
    return base_phrases[(idx-2) % len(base_phrases)]

def get_transition_phrase_lang(idx, question, lang):
    if lang == 'fil':
        base_phrases = [
            "Para sa sumunod mo na tanong:",
            f"Para sa tanong mong '{question.strip()}':",
            "Narito ang sagot sa sunod mong tanong:",
            "Para naman sa sunod mong tanong:"
        ]
    else:
        base_phrases = [
            "As for your next question:",
            f"To answer your question '{question.strip()}':",
            "Moving on to your next question:",
            "Here's what I can say about your next question:"
        ]
    return base_phrases[(idx-2) % len(base_phrases)]

# --- Missing helper function: strip_greeting ---
import re

def strip_greeting(text):
    # Remove common greetings at the start of a string (PIXEL style)
    return re.sub(
        r'^(hi|hello|heyyy|kumusta|kamusta|magandang \w+|good \w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*',
        '', text, flags=re.IGNORECASE
    ).strip()

def get_dynamic_not_in_faq_preamble(lang, with_greeting=True):
    system_prompt = PIXEL_PERSONALITY[lang]
    if with_greeting:
        user_prompt = "A user asked a CpE/academic/department/lab question that is not in the official FAQ. As PIXEL, generate a friendly, trendy preamble with a greeting, telling the user you'll forward the question and provide helpful info."
    else:
        user_prompt = "A user asked a CpE/academic/department/lab question that is not in the official FAQ. As PIXEL, generate a friendly, trendy preamble WITHOUT a greeting, telling the user you'll forward the question and provide helpful info."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        return ("Hi iskolar! The exact answer to your question is not in the official FAQ. I'll forward this to the CpE Department/Laboratory. Here's some helpful info:" if with_greeting else "This is not in the official FAQ. I'll forward this to the CpE Department/Laboratory. Here's some helpful info:")

def get_acad_dept_lab_keywords():
    # This function should return a set of academic/department/lab keywords from your language sheet or config
    # For now, return a sample set. Replace with your actual logic if needed.
    return set([
        'department', 'academic', 'lab', 'laboratory', 'equipment', 'apparatus', 'experiment', 'lab room', 'lab schedule', 'lab fee',
        'office', 'chair', 'coordinator', 'faculty', 'adviser', 'advising', 'enrollment', 'clearance', 'forms', 'requirements', 'grades', 'section', 'block', 'subject', 'curriculum', 'cpe', 'computer engineering',
        # Add Filipino equivalents if needed
        'departamento', 'akademiko', 'laboratoryo', 'kagamitan', 'eksperimento', 'silid-laboratoryo', 'iskedyul ng lab', 'bayad sa lab',
        'opisina', 'tagapangulo', 'tagapag-ugnay', 'guro', 'tagapayo', 'pagpapayo', 'pagpapatala', 'clearance', 'mga form', 'mga requirement', 'mga grado', 'seksyon', 'bloke', 'paksa', 'kurikulum'
    ])

def is_intent_to_ask_phrase(text):
    intent_patterns = [
        r"i have (another|a|some|few)? ?questions?( though)?[.!?]?",
        r"i have (another|a|some|few)? ?question( though)?[.!?]?",
        r"may (tanong|katanungan)",
        r"i want to ask",
        r"can i ask( a)? questions?[.!?]?",
        r"can i ask( a)? question[.!?]?",
        r"mag tatanong",
        r"i have some questions",
        r"i have a few questions",
        r"pwede po magtanong",
        r"pwede po magpatulong",
        r"can i ask something",
        r"may i ask",
        r"can i ask you something",
        r"can i ask for help",
        r"i have a follow[- ]?up question[.!?]?",
        r"i have another question[.!?]?",
        r"i have another question though[.!?]?",
        r"i have a question though[.!?]?",
        r"i have a follow[- ]?up[.!?]?"
    ]
    for pat in intent_patterns:
        if re.fullmatch(pat, text.strip(), re.IGNORECASE):
            return True
    return False

def strip_intent_to_ask_phrase(text):
    # Remove intent-to-ask phrase from the start of the message
    patterns = [
        r"^(i have (another|a|some|few)? ?questions?( though)?[.!?]?)(:|,| )?",
        r"^(i have (another|a|some|few)? ?question( though)?[.!?]?)(:|,| )?",
        r"^(may (tanong|katanungan))(:|,| )?",
        r"^(i want to ask)(:|,| )?",
        r"^(can i ask( a)? questions?[.!?]?)(:|,| )?",
        r"^(can i ask( a)? question[.!?]?)(:|,| )?",
        r"^(mag tatanong)(:|,| )?",
        r"^(i have some questions)(:|,| )?",
        r"^(i have a few questions)(:|,| )?",
        r"^(pwede po magtanong)(:|,| )?",
        r"^(pwede po magpatulong)(:|,| )?",
        r"^(can i ask something)(:|,| )?",
        r"^(may i ask)(:|,| )?",
        r"^(can i ask you something)(:|,| )?",
        r"^(can i ask for help)(:|,| )?",
        r"^(i have a follow[- ]?up question[.!?]?)(:|,| )?",
        r"^(i have another question[.!?]?)(:|,| )?",
        r"^(i have another question though[.!?]?)(:|,| )?",
        r"^(i have a question though[.!?]?)(:|,| )?",
        r"^(i have a follow[- ]?up[.!?]?)(:|,| )?"
    ]
    t = text.strip()
    for pat in patterns:
        t_new = re.sub(pat, '', t, flags=re.IGNORECASE)
        if t_new != t:
            t = t_new.strip()
    return t

def get_acad_dept_lab_keywords():
    # This function should return a set of academic/department/lab keywords from your language sheet or config
    # For now, return a sample set. Replace with your actual logic if needed.
    return set([
        'department', 'academic', 'lab', 'laboratory', 'equipment', 'apparatus', 'experiment', 'lab room', 'lab schedule', 'lab fee',
        'office', 'chair', 'coordinator', 'faculty', 'adviser', 'advising', 'enrollment', 'clearance', 'forms', 'requirements', 'grades', 'section', 'block', 'subject', 'curriculum', 'cpe', 'computer engineering',
        # Add Filipino equivalents if needed
        'departamento', 'akademiko', 'laboratoryo', 'kagamitan', 'eksperimento', 'silid-laboratoryo', 'iskedyul ng lab', 'bayad sa lab',
        'opisina', 'tagapangulo', 'tagapag-ugnay', 'guro', 'tagapayo', 'pagpapayo', 'pagpapatala', 'clearance', 'mga form', 'mga requirement', 'mga grado', 'seksyon', 'bloke', 'paksa', 'kurikulum'
    ])

# --- Missing helper function: strip_greeting ---
import re

def strip_greeting(text):
    # Remove common greetings at the start of a string (PIXEL style)
    return re.sub(
        r'^(hi|hello|heyyy|kumusta|kamusta|magandang \w+|good \w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*',
        '', text, flags=re.IGNORECASE
    ).strip()

def get_dynamic_not_in_faq_preamble(lang, with_greeting=True):
    system_prompt = PIXEL_PERSONALITY[lang]
    if with_greeting:
        user_prompt = "A user asked a CpE/academic/department/lab question that is not in the official FAQ. As PIXEL, generate a friendly, trendy preamble with a greeting, telling the user you'll forward the question and provide helpful info."
    else:
        user_prompt = "A user asked a CpE/academic/department/lab question that is not in the official FAQ. As PIXEL, generate a friendly, trendy preamble WITHOUT a greeting, telling the user you'll forward the question and provide helpful info."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        return ("Hi iskolar! The exact answer to your question is not in the official FAQ. I'll forward this to the CpE Department/Laboratory. Here's some helpful info:" if with_greeting else "This is not in the official FAQ. I'll forward this to the CpE Department/Laboratory. Here's some helpful info:")

def get_transition_phrase(idx, question):
    # Returns a transition phrase for subsequent answers
    base_phrases = [
        "As for your next question:",
        "To answer your next query:",
        f"To answer your question '{question.strip()}':",
        "Moving on to your next question:",
        "Here's what I can say about your next question:"
    ]
    # Cycle through or pick based on idx
    return base_phrases[(idx-2) % len(base_phrases)]

def get_transition_phrase_lang(idx, question, lang):
    if lang == 'fil':
        base_phrases = [
            "Para sa sumunod mo na tanong:",
            f"Para sa tanong mong '{question.strip()}':",
            "Narito ang sagot sa sunod mong tanong:",
            "Para naman sa sunod mong tanong:"
        ]
    else:
        base_phrases = [
            "As for your next question:",
            f"To answer your question '{question.strip()}':",
            "Moving on to your next question:",
            "Here's what I can say about your next question:"
        ]
    return base_phrases[(idx-2) % len(base_phrases)]

# --- Helper: Dynamic fallback message ---
def get_dynamic_fallback_message(lang):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "You could not understand the user's question or instruction. As PIXEL, reply in a friendly, trendy way: 'Sorry, I didn't catch your question. Could you please ask your question?' Never use a closing statement."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        return ("Sorry, I didn't catch your question. Could you please ask your question?")

# --- Helper: Detect imperative/instruction queries ---
def is_instruction_query(text, lang):
    patterns_en = [
        r'^(make|create|list|give|show|write|generate|compose|draw|explain|summarize|describe|tell|provide|enumerate|can you list|can you make|can you give|can you show|can you write|can you generate|can you compose|can you draw|can you explain|can you summarize|can you describe|can you tell|can you provide|can you enumerate|give me instructions|give instructions|how do i|how to|instructions for|steps for|steps to|guide to|guide for)',
    ]
    patterns_fil = [
        r'^(gawin|gumawa|ilista|ibigay|ipakita|isulat|gawin mo|pwede mo bang gawin|pwede mo bang ilista|pwede mo bang ibigay|pwede mo bang ipakita|pwede mo bang isulat|paki-|magbigay|maglista|magpakita|magsulat|magcompose|maggenerate|magdraw|magpaliwanag|magsummarize|magdescribe|magsabi|magprovide|mag-enumerate|paano gumawa|paano ilista|paano ibigay|paano ipakita|paano isulat|paano magbigay|paano maglista|paano magpakita|paano magsulat|paano magcompose|paano maggenerate|paano magdraw|paano magpaliwanag|paano magsummarize|paano magdescribe|paano magsabi|paano magprovide|paano mag-enumerate|paano gumawa ng|paano ilista ang|paano ibigay ang|paano ipakita ang|paano isulat ang|paano magbigay ng|paano maglista ng|paano magpakita ng|paano magsulat ng|paano magcompose ng|paano maggenerate ng|paano magdraw ng|paano magpasa)',
    ]

if __name__ == '__main__':
    app.run(debug=True)