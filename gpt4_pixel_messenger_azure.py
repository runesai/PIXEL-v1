from flask import Flask, request
from pymessenger.bot import Bot
from dotenv import load_dotenv
from langdetect import detect
from googleapiclient.discovery import build
from google.oauth2 import service_account
from datetime import datetime
from openai import AzureOpenAI
from string_utils import strip_greeting, strip_intent_to_ask_phrase
from intent_utils import (
    is_intent_to_ask_phrase, is_greeting_with_intent_to_ask, detect_greeting, detect_closing,
    extract_questions, extract_user_instruction, extract_listed_questions, detect_assignment_homework,
    is_illegal_query, is_followup_resource_request, strip_intent_to_ask_phrase, is_schedule_query
)
from keyword_utils import (
    get_acad_dept_lab_keywords, get_greeting_keywords, get_thanks_keywords, get_language_keywords,
    get_cpe_keywords, is_cpe_related
)
from faq_utils import get_faqs, get_sheet_faqs, match_faq, load_normalization_dict, normalize_query
from sheets_utils import get_sheets_service, log_unanswered, is_room_occupied
from response_utils import (
    get_transition_phrase, format_list_answers, get_dynamic_fallback
)
import os
import difflib
import requests
import time
import re

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

def detect_language(text):
    try:
        lang = detect(text)
        return 'fil' if lang in ['tl', 'fil'] else 'en'
    except:
        return 'en'
    
def fuzzy_match(query, choices, threshold=0.7):
    # Return the best match above threshold, or None
    matches = difflib.get_close_matches(query.lower(), [c.lower() for c in choices], n=1, cutoff=threshold)
    return matches[0] if matches else None

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

def contains_link_request(text):
    # Look for common link request phrases
    link_patterns = [
        r'link', r'website', r'url', r'webpage', r'can you give.*link', r'can you provide.*link', r'can you send.*link', r'can you share.*link', r'give me.*link', r'provide.*link', r'send.*link', r'share.*link', r'where can i find.*link', r'where can i get.*link', r'where do i get.*link', r'where do i find.*link', r'can i have.*link', r'can i get.*link', r'can i see.*link', r'can i access.*link', r'can you give.*website', r'can you provide.*website', r'can you send.*website', r'can you share.*website', r'give me.*website', r'provide.*website', r'send.*website', r'share.*website', r'where can i find.*website', r'where can i get.*website', r'where do i get.*website', r'where do i find.*website', r'can i have.*website', r'can i get.*website', r'can i see.*website', r'can i access.*website',
    ]
    for pat in link_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def is_ask_day_query(text):
    # Detects queries like "what day is it today?", "anong araw ngayon?", etc.
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
    # Detects queries like "what time is it?", "anong oras na?", etc.
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
    # Detects queries like "what room is available today?", "anong room ang bakante ngayon?"
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
        faqs = get_faqs_cached()
        official_names = extract_official_names_from_faq(faqs)
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
                    # --- NEW: Handle day/time/room-today queries ---
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
                        from sheets_utils import get_weekly_room_schedule
                        schedule = get_weekly_room_schedule()
                        day = datetime.now().strftime('%A').lower()
                        day_sched = schedule.get(day)
                        if not day_sched:
                            bot.send_text_message(sender_id, f"Sorry, I couldn't find the schedule for today.")
                            continue
                        room_keys = [k for k in day_sched[0].keys() if k.lower().startswith('room') or k.upper().startswith('CEA')]
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
                    print("Extracted questions:", questions)  # DEBUG
                    is_assignment = detect_assignment_homework(message_text)
                    print("is_assignment:", is_assignment)  # DEBUG
                    thanks_keywords = get_thanks_keywords()
                    closing_keywords = get_language_keywords(categories=['closing'])
                    has_thanks = any(k in message_text.lower() for k in thanks_keywords)
                    has_closing = any(k in message_text.lower() for k in closing_keywords)

                    # --- Assignment/homework detection (handle before FAQ/general logic) ---
                    if is_assignment:
                        bot.send_text_message(sender_id, get_assignment_guidance_response(message_text, lang))
                        continue

                    # --- Schedule query detection (handle before FAQ/general logic) ---
                    # DEBUG: Show intent detection for schedule query
                    print(f"[DEBUG] is_schedule_query('{message_text}') = {is_schedule_query(message_text)}")
                    if is_schedule_query(message_text):
                        print(f"[DEBUG] Entered schedule query block. Raw message: {message_text}")
                        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
                        found_day = next((d for d in days if d in message_text.lower()), None)
                        if not found_day and re.search(r'\btoday\b', message_text, re.IGNORECASE):
                            found_day = datetime.now().strftime('%A').lower()
                        # Improved room regex: match 'CEA300', 'Room 300', 'CEA 300', etc.
                        room_match = re.search(r'\b(?:room\s*|cea\s*)?(\d{3,})\b', message_text, re.IGNORECASE)
                        found_room = None
                        if room_match:
                            # Try to reconstruct the room name as it appears in the schedule (e.g., 'CEA300')
                            found_room = f"CEA{room_match.group(1)}"
                        # Improved time regex: match '12:00 PM to 2:00 PM', '12:00 PM - 2:00 PM', '12:00 PM', etc.
                        time_match = re.search(r'(\d{1,2}:\d{2}\s*[ap]m\s*(?:to|-|–)\s*\d{1,2}:\d{2}\s*[ap]m|\d{1,2}:\d{2}\s*[ap]m|\d{1,2}\s*[ap]m)', message_text, re.IGNORECASE)
                        found_time = time_match.group(0) if time_match else None
                        print(f"[DEBUG] found_day: {found_day}, found_time: {found_time}, found_room: {found_room}")
                        # If user asks for vacant/available/borrowable room (no room specified, but day and time are present)
                        vacant_keywords = r'(vacant|available|free|borrow|open|bakante|malaya|walang laman|walang tao|pwedeng hiramin|pwede bang hiramin)'
                        if found_day and not found_room and found_time and re.search(vacant_keywords, message_text, re.IGNORECASE):
                            from sheets_utils import get_weekly_room_schedule
                            schedule = get_weekly_room_schedule()
                            day_sched = schedule.get(found_day)
                            if not day_sched:
                                bot.send_text_message(sender_id, f"Sorry, I couldn't find the schedule for {found_day.title()}.")
                                continue
                            def parse_time(tstr):
                                tstr = tstr.strip().lower().replace('.', '')
                                tstr = tstr.replace('am', ' am').replace('pm', ' pm').replace('  ', ' ')
                                try:
                                    return datetime.strptime(tstr, '%I:%M %p')
                                except ValueError:
                                    try:
                                        return datetime.strptime(tstr, '%I %p')
                                    except ValueError:
                                        return None
                            def get_range_from_str(timestr):
                                timestr = timestr.replace('–', '-').replace('to', '-').replace('--', '-').replace('  ', ' ').replace(' - ', '-').strip()
                                parts = [p.strip() for p in timestr.split('-')]
                                if len(parts) == 2:
                                    t1 = parse_time(parts[0])
                                    t2 = parse_time(parts[1])
                                    return (t1, t2)
                                elif len(parts) == 1:
                                    t1 = parse_time(parts[0])
                                    return (t1, t1)
                                return (None, None)
                            user_start, user_end = get_range_from_str(found_time)
                            if not user_start or not user_end:
                                bot.send_text_message(sender_id, f"Sorry, I couldn't understand the time '{found_time}'. Please use a format like '10:00 AM to 1:00 PM'.")
                                continue
                            room_keys = [k for k in day_sched[0].keys() if k.lower().startswith('room') or k.upper().startswith('CEA')]
                            # For single time: find the slot containing the user's time
                            if user_start == user_end:
                                containing_slot = None
                                for row in day_sched:
                                    slot_time_str = row.get('Time', '') or row.get('TIME', '')
                                    slot_parts = [p.strip() for p in slot_time_str.replace('–', '-').replace('to', '-').replace('--', '-').replace('  ', ' ').replace(' - ', '-').split('-')]
                                    if len(slot_parts) == 2:
                                        slot_start = parse_time(slot_parts[0])
                                        slot_end = parse_time(slot_parts[1])
                                    elif len(slot_parts) == 1:
                                        slot_start = parse_time(slot_parts[0])
                                        slot_end = slot_start
                                    else:
                                        continue
                                    if slot_start and slot_end and slot_start <= user_start < slot_end:
                                        containing_slot = row
                                        break
                                if containing_slot:
                                    vacant_rooms = [room for room in room_keys if not containing_slot.get(room, '').strip()]
                                    if vacant_rooms:
                                        slot_time_str = containing_slot.get('Time', '') or containing_slot.get('TIME', '')
                                        bot.send_text_message(sender_id, f"The following rooms are vacant at {slot_time_str} on {found_day.title()}: {', '.join(vacant_rooms)}. If you would like to reserve a room, you may coordinate with our student assistants, iskolar!")
                                    else:
                                        bot.send_text_message(sender_id, f"Sorry, there are no vacant rooms at {found_time} on {found_day.title()}.")
                                else:
                                    bot.send_text_message(sender_id, f"Sorry, I couldn't find a schedule slot containing {found_time} on {found_day.title()}.")
                                continue
                            # For time range: aggregate all slots that overlap with the user's range
                            slot_times = []
                            for row in day_sched:
                                slot_time_str = row.get('Time', '') or row.get('TIME', '')
                                slot_parts = [p.strip() for p in slot_time_str.replace('–', '-').replace('to', '-').replace('--', '-').replace('  ', ' ').replace(' - ', '-').split('-')]
                                if len(slot_parts) == 2:
                                    slot_start = parse_time(slot_parts[0])
                                    slot_end = parse_time(slot_parts[1])
                                elif len(slot_parts) == 1:
                                    slot_start = parse_time(slot_parts[0])
                                    slot_end = slot_start
                                else:
                                    continue
                                # Overlap with user range
                                if slot_start and slot_end and not (slot_end <= user_start or slot_start >= user_end):
                                    slot_times.append((slot_time_str, row))
                            if not slot_times:
                                bot.send_text_message(sender_id, f"Sorry, there are no vacant rooms at {found_time} on {found_day.title()}.")
                                continue
                            room_vacancy = {room: [] for room in room_keys}
                            for slot_time_str, row in slot_times:
                                for room in room_keys:
                                    if not row.get(room, '').strip():
                                        room_vacancy[room].append(slot_time_str)
                            fully_vacant = []
                            partially_vacant = {}
                            total_slots = len(slot_times)
                            for room, free_slots in room_vacancy.items():
                                if len(free_slots) == total_slots and total_slots > 0:
                                    fully_vacant.append(room)
                                elif free_slots:
                                    partially_vacant[room] = free_slots
                            msg = ""
                            if fully_vacant:
                                msg += f"The following rooms are fully vacant at {found_time} on {found_day.title()}:\n" + ', '.join(fully_vacant) + "\n\n"
                            if partially_vacant:
                                msg += "The following rooms are only partially vacant during your requested time range:\n"
                                for room, times in partially_vacant.items():
                                    msg += f"{room}: free at {', '.join(times)}\n"
                            if not msg:
                                msg = f"Sorry, there are no vacant rooms at {found_time} on {found_day.title()}."
                            bot.send_text_message(sender_id, msg.strip())
                            continue
                        # Default: check specific room occupancy
                        if found_day and found_room and found_time:
                            from sheets_utils import get_weekly_room_schedule
                            schedule = get_weekly_room_schedule()
                            day_sched = schedule.get(found_day)
                            slots_in_range = []
                            # --- Time range parsing helpers ---
                            def parse_time(tstr):
                                tstr = tstr.strip().lower().replace('.', '')
                                tstr = tstr.replace('am', ' am').replace('pm', ' pm').replace('  ', ' ')
                                try:
                                    return datetime.strptime(tstr, '%I:%M %p')
                                except ValueError:
                                    try:
                                        return datetime.strptime(tstr, '%I %p')
                                    except ValueError:
                                        return None
                            def get_range_from_str(timestr):
                                # Accepts '12:00 PM to 2:00 PM', '12pm-2pm', etc.
                                timestr = timestr.replace('–', '-').replace('to', '-').replace('--', '-').replace('  ', ' ').replace(' - ', '-').strip()
                                parts = [p.strip() for p in timestr.split('-')]
                                if len(parts) == 2:
                                    t1 = parse_time(parts[0])
                                    t2 = parse_time(parts[1])
                                    return (t1, t2)
                                elif len(parts) == 1:
                                    t1 = parse_time(parts[0])
                                    return (t1, t1)
                                return (None, None)
                            user_start, user_end = get_range_from_str(found_time)
                            if not user_start or not user_end:
                                bot.send_text_message(sender_id, f"Sorry, I couldn't understand the time '{found_time}'. Please use a format like '12:00 PM to 2:00 PM'.")
                                continue
                            # Find all slots that overlap with the user's range
                            for row in day_sched or []:
                                slot_time_str = row.get('Time', '') or row.get('TIME', '')
                                slot_parts = [p.strip() for p in slot_time_str.replace('–', '-').replace('to', '-').replace('--', '-').replace('  ', ' ').replace(' - ', '-').split('-')]
                                if len(slot_parts) == 2:
                                    slot_start = parse_time(slot_parts[0])
                                    slot_end = parse_time(slot_parts[1])
                                elif len(slot_parts) == 1:
                                    slot_start = parse_time(slot_parts[0])
                                    slot_end = slot_start
                                else:
                                    continue
                                # Check for overlap
                                if slot_start and slot_end and not (slot_end <= user_start or slot_start >= user_end):
                                    slots_in_range.append(row)
                            # Always determine matched_room_col, even if no slots_in_range
                            room_keys = list(day_sched[0].keys()) if day_sched else []
                            def normalize_room_name(s):
                                return re.sub(r'[^a-zA-Z0-9]', '', s).lower()
                            found_room_norm = normalize_room_name(found_room)
                            matched_room_col = None
                            for k in room_keys:
                                if normalize_room_name(k) == found_room_norm:
                                    matched_room_col = k
                                    break
                            if not matched_room_col:
                                for k in room_keys:
                                    if found_room_norm in normalize_room_name(k):
                                        matched_room_col = k
                                        break
                            if not matched_room_col:
                                bot.send_text_message(sender_id, f"Sorry, I couldn't find the room '{found_room}' in the schedule for {found_day.title()}.")
                                continue
                            # Determine if user asked for a range or a single time
                            is_time_range = False
                            user_time_parts = [p.strip() for p in found_time.replace('–', '-').replace('to', '-').replace('--', '-').split('-')]
                            if len(user_time_parts) == 2:
                                is_time_range = True
                            if not slots_in_range:
                                if not is_time_range:
                                    # For single time, try to find a slot containing the user's time
                                    user_time = parse_time(user_time_parts[0])
                                    found_slot = None
                                    for row in day_sched or []:
                                        slot_time_str = row.get('Time', '') or row.get('TIME', '')
                                        slot_parts = [p.strip() for p in slot_time_str.replace('–', '-').replace('to', '-').replace('--', '-').replace('  ', ' ').replace(' - ', '-').split('-')]
                                        if len(slot_parts) == 2:
                                            slot_start = parse_time(slot_parts[0])
                                            slot_end = parse_time(slot_parts[1])
                                        elif len(slot_parts) == 1:
                                            slot_start = parse_time(slot_parts[0])
                                            slot_end = slot_start
                                        else:
                                            continue
                                        if slot_start and slot_end and user_time and slot_start <= user_time < slot_end:
                                            found_slot = row
                                            break
                                    if found_slot:
                                        occupant = found_slot.get(matched_room_col, '').strip() if matched_room_col else ''
                                        # Find the full merged time range for this occupant
                                        idx = day_sched.index(found_slot)
                                        # Search upwards
                                        start_idx = idx
                                        while start_idx > 0:
                                            prev_row = day_sched[start_idx - 1]
                                            prev_val = prev_row.get(matched_room_col, '').strip() if matched_room_col else ''
                                            if prev_val == occupant and prev_val:
                                                start_idx -= 1
                                            else:
                                                break
                                        # Search downwards
                                        end_idx = idx
                                        while end_idx + 1 < len(day_sched):
                                            next_row = day_sched[end_idx + 1]
                                            next_val = next_row.get(matched_room_col, '').strip() if matched_room_col else ''
                                            if next_val == occupant and next_val:
                                                end_idx += 1
                                            else:
                                                break
                                        # Get the time labels for the merged range
                                        start_time_str = day_sched[start_idx].get('Time', '') or day_sched[start_idx].get('TIME', '')
                                        end_time_str = day_sched[end_idx].get('Time', '') or day_sched[end_idx].get('TIME', '')
                                        # Parse and format the full time range
                                        def parse_time(tstr):
                                            tstr = tstr.strip().lower().replace('.', '')
                                            tstr = tstr.replace('am', ' am').replace('pm', ' pm').replace('  ', ' ')
                                            try:
                                                return datetime.strptime(tstr, '%I:%M %p')
                                            except ValueError:
                                                try:
                                                    return datetime.strptime(tstr, '%I %p')
                                                except ValueError:
                                                    return None
                                        start_time = parse_time(start_time_str.split('-')[0]) if start_time_str else None
                                        # Use the end of the last slot as the end time
                                        end_time = parse_time(end_time_str.split('-')[-1]) if end_time_str else None
                                        if start_time and end_time:
                                            full_time_range = f"{start_time.strftime('%I:%M%p').lstrip('0')} - {end_time.strftime('%I:%M%p').lstrip('0')}"
                                        else:
                                            full_time_range = f"{start_time_str} - {end_time_str}" if start_time_str and end_time_str else start_time_str or end_time_str or ''
                                        if occupant:
                                            details = occupant.replace('\n', ', ')
                                            bot.send_text_message(sender_id, f"{found_room} is occupied at the following times on {found_day.title()} during {found_time}:\n\n{full_time_range}, {details}")
                                        else:
                                            bot.send_text_message(sender_id, f"{found_room} is free at {found_time} on {found_day.title()}.")
                                    else:
                                        bot.send_text_message(sender_id, f"Sorry, I couldn't find any schedule slot containing {found_time} on {found_day.title()}.")
                            else:
                                # Aggregate occupancy for the room across all slots
                                occupied_details = []
                                unique_occupants = set()
                                for slot in slots_in_range:
                                    occupant = slot.get(matched_room_col, '').strip()
                                    slot_time = slot.get('Time', '') or slot.get('TIME', '')
                                    if occupant:
                                        details = occupant.replace('\n', ', ')
                                        occupied_details.append((slot_time, details))
                                        unique_occupants.add(details)
                                if occupied_details:
                                    if is_time_range:
                                        # For time ranges, print unique occupant(s) only once
                                        if len(unique_occupants) == 1:
                                            occ = list(unique_occupants)[0]
                                            bot.send_text_message(sender_id, f"{found_room} is occupied at the following times on {found_day.title()} during {found_time}:\n\n{occ}")
                                        else:
                                            occs = '\n'.join(unique_occupants)
                                            bot.send_text_message(sender_id, f"{found_room} is occupied at the following times on {found_day.title()} during {found_time}:\n\n{occs}")
                                    else:
                                        # For a single time, find the slot whose range contains the user's time
                                        user_time = parse_time(user_time_parts[0])
                                        found_slot = None
                                        for row in day_sched or []:
                                            slot_time_str = row.get('Time', '') or row.get('TIME', '')
                                            slot_parts = [p.strip() for p in slot_time_str.replace('–', '-').replace('to', '-').replace('--', '-').replace('  ', ' ').replace(' - ', '-').split('-')]
                                            if len(slot_parts) == 2:
                                                slot_start = parse_time(slot_parts[0])
                                                slot_end = parse_time(slot_parts[1])
                                            elif len(slot_parts) == 1:
                                                slot_start = parse_time(slot_parts[0])
                                                slot_end = slot_start
                                            else:
                                                continue
                                            if slot_start and slot_end and user_time and slot_start <= user_time < slot_end:
                                                found_slot = row
                                                break
                                        if found_slot:
                                            occupant = found_slot.get(matched_room_col, '').strip()
                                            slot_time = found_slot.get('Time', '') or found_slot.get('TIME', '')
                                            if occupant:
                                                details = occupant.replace('\n', ', ')
                                                bot.send_text_message(sender_id, f"{found_room} is occupied at the following times on {found_day.title()} during {found_time}:\n\n{slot_time}, {details}")
                                            else:
                                                bot.send_text_message(sender_id, f"{found_room} is free at {found_time} on {found_day.title()}.")
                                        else:
                                            bot.send_text_message(sender_id, f"Sorry, I couldn't find any schedule slot containing {found_time} on {found_day.title()}.")
                                else:
                                    if is_time_range:
                                        bot.send_text_message(sender_id, f"{found_room} is free for the entire period {found_time} on {found_day.title()}.")
                                    else:
                                        bot.send_text_message(sender_id, f"{found_room} is free at {found_time} on {found_day.title()}.")
                        continue

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
                    # Only filter out true intent-to-ask phrases (e.g., "I have a question"),
                    # but treat imperative/instructional queries (e.g., "can you make me a poem") as valid questions.
                    filtered_questions = []
                    for q in questions:
                        if is_intent_to_ask_phrase(q):
                            print(f"[DEBUG] Filtering out intent-to-ask phrase: '{q}'")
                        else:
                            filtered_questions.append(q)
                    print("Filtered questions:", filtered_questions)  # DEBUG
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
                    # If only_intent_to_ask is True, always reply with the dynamic intent-to-ask prompt (never greeting or closing)
                    if only_intent_to_ask:
                        bot.send_text_message(sender_id, get_dynamic_intent_to_ask_prompt(lang))
                        context['greeted'] = True
                        continue
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
                            # Always normalize the query using the normalization dictionary (handles Filipino/English variations)
                            q_norm = normalize_query(q, normalization_dict)
                            faq_match = match_faq(q_norm, faqs, lang, normalization_dict=normalization_dict)
                            if faq_match:
                                # Use the full user message for context in the personalized response
                                answer = get_personalized_faq_response(faq_match['question'], faq_match['answer'], lang, user_query=message_text, context=context, skip_greeting=(len(answer_texts) > 0))
                                answer_texts.append(answer)
                                answer_questions.append(q)
                            else:
                                # Check if query contains any academic/department/lab keyword
                                q_l = q.lower()
                                is_acad_dept_lab = any(kw in q_l for kw in acad_dept_lab_keywords)
                                answer_questions.append(q)
                                if is_acad_dept_lab:
                                    # Always log unanswered CpE/academic/lab queries
                                    log_unanswered(q, lang)
                                    preamble_needed = True
                                    answer_texts.append(None)  # Mark for preamble logic
                                else:
                                    answer_texts.append(None)
                        # --- Now generate answers for non-FAQ questions ---
                        for idx, (q, a) in enumerate(zip(answer_questions, answer_texts)):
                            if a is not None:
                                # For all but the first answer, add a transition phrase and never a greeting
                                if idx > 0:
                                    transition = get_transition_phrase(idx+1, q, lang)
                                    a = strip_greeting(a.strip())
                                    a = f"{transition} {a}"
                                answer_texts[idx] = a
                                continue  # Already answered as FAQ
                            q_l = q.lower()
                            is_acad_dept_lab = any(kw in q_l for kw in acad_dept_lab_keywords)
                            q_type = classify_cpe_query(q, cpe_keywords)
                            if contains_link_request(q):
                                answer_texts[idx] = "Sorry, I can't send links unless they're part of the official FAQ. If you need a specific link, please check the FAQ or ask the department directly."
                                continue
                            if is_acad_dept_lab:
                                # Always log unanswered CpE/academic/lab queries (already done above, but safe to repeat)
                                log_unanswered(q, lang)
                                with_greeting = (idx == 0)
                                preamble = get_dynamic_not_in_faq_preamble(lang, with_greeting=with_greeting)
                                # Use the full user message for context in the GPT prompt
                                system_prompt = PIXEL_PERSONALITY[lang]
                                user_prompt = f"The user asked: '{message_text}'. This is not in the official FAQ. As PIXEL, provide a helpful suggestion or info. Do NOT mention any specific person's name unless it is in the official FAQ. Do not repeat the preamble or greeting. Use 'CpE' as a shortcut if needed."
                                if user_instruction:
                                    user_prompt += f" The user also requested: '{user_instruction}'."
                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                                try:
                                    gpt_response = ask_github_gpt(messages)
                                    gpt_response = cpe_shortcut(gpt_response)
                                    # Remove any specific names if not in FAQ (allow only FAQ-listed names)
                                    # import re
                                    # def redact_unofficial_names(text, allowed_names):
                                    #     # Find all capitalized name-like phrases
                                    #     name_pattern = re.compile(r'(Dr\.?|Engr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?|Sir|Ma\'am)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+[A-Z][a-z]+)*')
                                    #     sentences = re.split(r'(?<=[.!?])\s+', text)
                                    #     filtered = []
                                    #     for sent in sentences:
                                    #         found = False
                                    #         for match in name_pattern.findall(sent):
                                    #             if match.strip() and match.strip() not in allowed_names:
                                    #                 found = True
                                    #                 break
                                    #         if found:
                                    #             # Replace with a general fallback
                                    #             filtered.append("You may contact the department office for the latest information.")
                                    #         else:
                                    #             filtered.append(sent)
                                    #     return ' '.join(filtered)
                                    # gpt_response = redact_unofficial_names(gpt_response, official_names)
                                    # Add newline after preamble before answer
                                    answer = preamble + "\n\n" + gpt_response
                                    # For all but the first answer, add a transition phrase
                                    if idx > 0:
                                        transition = get_transition_phrase(idx+1, q, lang)
                                        answer = f"{transition} {answer}"
                                    answer_texts[idx] = answer
                                    query_context['last_topic'] = q
                                    query_context['last_answer'] = gpt_response
                                except Exception as gpt_e:
                                    fallback = get_dynamic_fallback(lang)
                                    answer = preamble + "\n\n" + fallback
                                    if idx > 0:
                                        transition = get_transition_phrase(idx+1, q, lang)
                                        answer = f"{transition} {answer}"
                                    answer_texts[idx] = answer
                            elif q_type == 'general':
                                # General/non-CpE/tech query, no preamble, no logging
                                # Always try to answer, but mention PIXEL's scope
                                system_prompt = PIXEL_PERSONALITY[lang]
                                # Use the full user message for context
                                user_prompt = (
                                    f"The user said: '{message_text}'. This is not related to CpE or technology. As PIXEL, explain in a friendly, trendy way that you aren't exactly made for non-CpE/tech queries, but still try to help and provide a relevant, helpful answer or suggestion. If you can't answer, generate a dynamic fallback in PIXEL's voice based on: 'Sorry, I couldn't understand your message.' Never use a greeting or closing in the fallback."
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
                                    if idx > 0:
                                        transition = get_transition_phrase(idx+1, q, lang)
                                        answer = strip_greeting(answer)
                                        answer = f"{transition} {answer}"
                                    answer_texts[idx] = answer
                                    query_context['last_topic'] = q
                                    query_context['last_answer'] = gpt_response
                                except Exception as gpt_e:
                                    # Dynamic fallback, still in PIXEL's voice, no greeting/closing
                                    fallback = get_dynamic_fallback(lang)
                                    answer = fallback
                                    if idx > 0:
                                        transition = get_transition_phrase(idx+1, q, lang)
                                        answer = f"{transition} {answer}"
                                    answer_texts[idx] = answer
                            else:
                                system_prompt = PIXEL_PERSONALITY[lang]
                                user_prompt = f"The user said: '{message_text}'. As PIXEL, provide a helpful answer or suggestion. Do not use any preamble or greeting. Use 'CpE' as a shortcut if needed."
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
                                    if idx > 0:
                                        transition = get_transition_phrase(idx+1, q, lang)
                                        answer = f"{transition} {answer}"
                                    answer_texts[idx] = answer
                                    query_context['last_topic'] = q
                                    query_context['last_answer'] = gpt_response
                                except Exception as gpt_e:
                                    fallback = get_dynamic_fallback(lang)
                                    answer = fallback
                                    if idx > 0:
                                        transition = get_transition_phrase(idx+1, q, lang)
                                        answer = f"{transition} {answer}"
                                    answer_texts[idx] = answer
                        # --- Compose the final message ---
                        final_message = ""
                        if listed_questions and user_instruction:
                            final_message += format_list_answers(filtered_questions, answer_texts, instruction=user_instruction, lang=lang)
                        else:
                            final_message += "\n\n".join(answer_texts)
                        bot.send_text_message(sender_id, final_message.strip())
                        check_github_model_rate_limit()
                        continue
                    # PRIORITY 5: Fallback (not a greeting, not a question, not thanks/closing, not assignment)
                    log_unanswered(message_text, lang)
                    fallback = ("Sorry, I couldn't understand your message. Could you please rephrase or ask about CpE topics?" if lang == 'en' else "Paumanhin, hindi ko naintindihan ang iyong mensahe. Maaari mo bang ulitin o magtanong tungkol sa CpE?")
                    bot.send_text_message(sender_id, fallback)
                    check_github_model_rate_limit()
    return "ok", 200

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

# --- Helper: Classify query as CpE Department, Laboratory, or General ---
def classify_cpe_query(query, cpe_keywords):
    # You can expand this with more precise keywords if needed
    lab_keywords = {'lab', 'laboratory', 'equipment', 'apparatus', 'experiment', 'lab room', 'lab schedule', 'lab fee'}
    dept_keywords = {'department', 'office', 'chair', 'coordinator', 'faculty', 'adviser', 'advising', 'enrollment', 'clearance', 'forms', 'requirements', 'grades', 'section', 'block', 'subject', 'curriculum', 'CpE', 'computer engineering'}
    q = query.lower()
    if any(kw in q for kw in lab_keywords):
        return 'lab'
    if any(kw in q for kw in dept_keywords) or is_cpe_related(q, cpe_keywords):
        return 'dept'
    return 'general'

# --- Helper: Replace 'Computer Engineering' with 'CpE' shortcut ---
def cpe_shortcut(text):
    return re.sub(r'computer engineering', 'CpE', text, flags=re.IGNORECASE)


# --- Helper: Generate dynamic 'not in the FAQ' preamble (with/without greeting) ---
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

# --- Helper: Generate dynamic response for non-CpE/tech instructions ---
def get_dynamic_non_cpe_instruction_response(lang, instruction):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = (
        f"The user gave this instruction or query: '{instruction}'. As PIXEL, explain in a friendly, trendy way that you aren't exactly made for non-CpE/tech queries, but still try to help and provide a relevant, helpful answer or suggestion. If you can't answer, generate a dynamic fallback in PIXEL's voice based on: 'Sorry, I couldn't understand your message.' Never use a greeting or closing in the fallback."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        # Dynamic fallback, still in PIXEL's voice, no greeting/closing
        return ("Sorry, I couldn't understand your message. If you have any CpE or tech questions, I'm here to help!")


# --- Helper: Extract all official names from FAQ ---
def extract_official_names_from_faq(faqs):
    import re
    names = set()
    name_pattern = re.compile(r'(Dr\.?|Engr\.?|Mr\.?|Ms\.?|Mrs\.?|Prof\.?|Sir|Ma\'am)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+[A-Z][a-z]+)*')
    for faq in faqs:
        for field in ['question', 'answer']:
            text = faq.get(field, '')
            for match in name_pattern.findall(text):
                if match:
                    names.add(match.strip())
    return names

if __name__ == '__main__':
    # DEBUG: Print parsed schedule for all days
    from sheets_utils import get_weekly_room_schedule
    schedule = get_weekly_room_schedule()
    for day in schedule:
        print(f"\n[DEBUG] --- {day.upper()} SCHEDULE ---")
        for row in schedule[day]:
            print(row)
        print(f"[DEBUG] --- END {day.upper()} SCHEDULE ---")
    app.run(debug=True)