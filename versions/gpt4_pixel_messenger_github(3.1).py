from flask import Flask, request
from pymessenger.bot import Bot
from dotenv import load_dotenv
from langdetect import detect
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime
import os
from openai import OpenAI
import requests
import difflib
import random
import re
import time

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
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
GITHUB_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODEL = "openai/gpt-4.1"

github_client = OpenAI(
    base_url=GITHUB_ENDPOINT,
    api_key=GITHUB_TOKEN,
)

def ask_github_gpt(messages, model=GITHUB_MODEL):
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
    for try_query in queries_to_try:
        query_l = try_query.lower().strip()
        for faq in faqs:
            faq_q = faq['question'].lower().strip()
            # Use fuzzy matching and also check if all main words in FAQ are present in the query
            ratio = difflib.SequenceMatcher(None, query_l, faq_q).ratio()
            faq_words = set(faq_q.split())
            query_words = set(query_l.split())
            word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
            if (query_l in faq_q or faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == lang:
                best_score = max(ratio, word_overlap)
                best_faq = faq
    # If not found and lang is not 'en', try English
    if not best_faq and lang != 'en':
        for try_query in queries_to_try:
            query_l = try_query.lower().strip()
            for faq in faqs:
                faq_q = faq['question'].lower().strip()
                ratio = difflib.SequenceMatcher(None, query_l, faq_q).ratio()
                faq_words = set(faq_q.split())
                query_words = set(query_l.split())
                word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                if (query_l in faq_q or faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == 'en':
                    best_score = max(ratio, word_overlap)
                    best_faq = faq
    # If not found and lang is not 'fil', try Filipino
    if not best_faq and lang != 'fil':
        for try_query in queries_to_try:
            query_l = try_query.lower().strip()
            for faq in faqs:
                faq_q = faq['question'].lower().strip()
                ratio = difflib.SequenceMatcher(None, query_l, faq_q).ratio()
                faq_words = set(faq_q.split())
                query_words = set(query_l.split())
                word_overlap = len(faq_words & query_words) / max(1, len(faq_words))
                if (query_l in faq_q or faq_q in query_l or ratio > best_score or word_overlap > 0.6) and faq['lang'] == 'fil':
                    best_score = max(ratio, word_overlap)
                    best_faq = faq
    # Lower threshold slightly for better matching
    if best_score > 0.55:
        return best_faq
    return None

def check_github_model_rate_limit():
    url = "https://models.github.ai/inference"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    data = {
        "model": GITHUB_MODEL,
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

# --- Caching for FAQ and normalization dict ---
_FAQ_CACHE = None
_FAQ_CACHE_TIME = 0
_FAQ_CACHE_TTL = 300  # seconds (5 minutes)

_NORMALIZATION_CACHE = None
_NORMALIZATION_CACHE_TIME = 0
_NORMALIZATION_CACHE_TTL = 300  # seconds (5 minutes)

def get_faqs_cached():
    global _FAQ_CACHE, _FAQ_CACHE_TIME
    now = time.time()
    if _FAQ_CACHE is None or now - _FAQ_CACHE_TIME > _FAQ_CACHE_TTL:
        _FAQ_CACHE = get_faqs()
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
                    is_greeting = detect_greeting(message_text)
                    questions = extract_questions(message_text)
                    is_assignment = detect_assignment_homework(message_text)
                    thanks_keywords = get_thanks_keywords()
                    closing_keywords = get_language_keywords(categories=['closing'])
                    has_thanks = any(k in message_text.lower() for k in thanks_keywords)
                    has_closing = any(k in message_text.lower() for k in closing_keywords)
                    # 1. Homework/assignment detection (block all other answers if present)
                    if is_assignment:
                        bot.send_text_message(sender_id, get_assignment_guidance_response(message_text, lang))
                        continue
                    # 2. Thanks/closing dynamic responses
                    if has_thanks and not has_closing:
                        bot.send_text_message(sender_id, get_dynamic_thanks(lang))
                        continue
                    if has_closing and not has_thanks:
                        bot.send_text_message(sender_id, get_dynamic_closing(lang))
                        continue
                    if has_thanks and has_closing:
                        bot.send_text_message(sender_id, get_dynamic_thanks(lang))
                        continue
                    # 3. Greeting only
                    if is_greeting and not questions:
                        if context['greeted']:
                            greeting = get_dynamic_greeting_again(lang)
                        else:
                            greeting = get_dynamic_greeting(lang)
                            context['greeted'] = True
                        bot.send_text_message(sender_id, greeting)
                        continue
                    # 4. If greeting + question(s), answer question(s) only
                    if questions:
                        answers = []
                        faq_matches = []
                        non_faq_questions = []
                        for q in questions:
                            q_norm = normalize_query(q, normalization_dict)
                            faq_match = match_faq(q_norm, faqs, lang, normalization_dict=normalization_dict)
                            if faq_match:
                                faq_matches.append((q, faq_match))
                            else:
                                non_faq_questions.append(q)
                        # If any FAQ matches, combine answers contextually
                        if faq_matches:
                            combined = []
                            for idx, (q, faq) in enumerate(faq_matches):
                                skip_greeting = idx > 0
                                # For the second and subsequent questions, use a transition
                                if idx == 0:
                                    answer = get_personalized_faq_response(faq['question'], faq['answer'], lang, user_query=q, context=context, skip_greeting=skip_greeting)
                                else:
                                    answer = get_personalized_faq_response(faq['question'], faq['answer'], lang, user_query=q, context=context, skip_greeting=True)
                                    # Remove greeting and add transition
                                    answer = re.sub(r'^(hi|hello|heyyy|kumusta|kamusta|magandang \\w+|good \\w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*', '', answer, flags=re.IGNORECASE)
                                    answer = f"As for your next question, {answer.strip()}"
                                combined.append(answer.strip())
                            answers.append("\n\n".join(combined))
                        # For non-FAQ questions, classify as CPE or general
                        for q in non_faq_questions:
                            # If the user is asking for a link but it's not in the FAQ, reply with a special message
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
                    # 5. If not a greeting and not a question, fallback
                    log_unanswered(message_text, lang)
                    fallback = ("Sorry, I couldn't understand your message. Could you please rephrase or ask about CpE topics?" if lang == 'en' else "Paumanhin, hindi ko naintindihan ang iyong mensahe. Maaari mo bang ulitin o magtanong tungkol sa CpE?")
                    bot.send_text_message(sender_id, fallback)
                    check_github_model_rate_limit()
    return "ok", 200

if __name__ == '__main__':
    app.run(debug=True)
