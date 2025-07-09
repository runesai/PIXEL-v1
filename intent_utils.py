import re
from keyword_utils import get_greeting_keywords, get_language_keywords

print("Loaded intent_utils.py")

def is_intent_to_ask_phrase(text):
    # Only match true intent-to-ask phrases, not instructional/imperative queries
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
    # Do NOT match 'can you ...', 'could you ...', 'will you ...', 'would you ...', etc.
    # Only match if the phrase is about the user asking a question, not requesting an action
    for pat in intent_patterns:
        if re.fullmatch(pat, text.strip(), re.IGNORECASE):
            return True
    return False

def strip_intent_to_ask_phrase(text):
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
            return has_greeting
    return False

def detect_greeting(text):
    greeting_keywords = get_greeting_keywords()
    text_l = text.lower().strip()
    for k in greeting_keywords:
        if text_l.startswith(k) or re.search(rf'\b{k}\b', text_l):
            return True
    if re.match(r'^(good|magandang) (morning|afternoon|evening|day|gabi|umaga|tanghali|hapon|gabi)', text_l):
        return True
    return False

def detect_closing(text):
    closing_keywords = get_language_keywords(categories=['closing'])
    text_l = text.lower()
    return any(k in text_l for k in closing_keywords)

def extract_questions(text):
    # Enhanced: also match imperative/instructional/command queries
    question_starters = r'(how|what|when|where|who|why|can|is|are|do|does|did|will|would|should|could|may|paano|ano|saan|kailan|bakit|pwede|puwede|magkano|ilan|sinong|sino|pwedeng|paano|mag|may|meron|possible|possible ba|tell me|give me|explain|define|describe|list|provide|show|help|need|want|require|can you|could you|would you|please|gusto|kailangan|make|create|build|write|generate|compose|prepare|draw|design|construct|make me|can you list|give me instructions on|give instructions on|show me how to|walk me through|demonstrate|illustrate|draft|outline|summarize|enumerate)'
    imperative_patterns = [
        r'^(make|create|build|write|generate|compose|prepare|draw|design|construct)\b.*',
        r'^(make me|can you list|give me instructions on|give instructions on|show me how to|walk me through|demonstrate|illustrate|draft|outline|summarize|enumerate)\b.*',
        r'^(please )?(list|provide|show|help|explain|define|describe|give|summarize|enumerate)\b.*',
    ]
    # Split only on question marks or newlines, not on every possible starter
    parts = re.split(r'(?<=[?])\s+|\n+', text)
    questions = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Always keep the full phrase for imperative/instructional queries
        if p.endswith('?') or re.match(rf'^{question_starters}\b', p, re.IGNORECASE):
            questions.append(p)
        elif any(re.match(pat, p, re.IGNORECASE) for pat in imperative_patterns):
            questions.append(p)
        # Only treat as a question if the full phrase is short and contains a question verb
        elif len(p.split()) <= 10 and re.search(r'\b(is|are|do|does|did|will|would|should|could|may|can|have|has|had|need|want|require|help|explain|define|describe|list|provide|show|tell|give|explain|pwede|pwedeng|gusto|kailangan)\b', p, re.IGNORECASE):
            questions.append(p)
    # If nothing matched, just return the full text
    if not questions:
        questions = [text.strip()]
    return questions

def extract_user_instruction(text):
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

def extract_listed_questions(text):
    lines = text.split('\n')
    questions = []
    for line in lines:
        line = line.strip()
        if re.match(r'^(\d+\.|[-*•])\s+', line):
            q = re.sub(r'^(\d+\.|[-*•])\s+', '', line)
            if q:
                questions.append(q)
    if len(questions) < 2:
        return None
    return questions

def detect_assignment_homework(text):
    patterns = [
        r'(multiple choice|choose the correct answer|checkbox|true or false|enumerate|match the following|fill in the blank|essay|short answer|assignment|homework|quiz|test|exam)',
        r'\b[a-dA-D]\b[\).]',
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

def is_illegal_query(text):
    illegal_patterns = [
        r"dark web", r"how to hack", r"how to kill", r"how to murder", r"how to make a bomb", r"how to use drugs", r"buy drugs", r"illegal drugs", r"child porn", r"terrorist", r"terrorism", r"commit crime", r"steal", r"rob", r"rape", r"assault", r"suicide", r"self-harm", r"harm others", r"violence", r"explosives", r"weapon", r"guns", r"shoot", r"stab", r"poison", r"overdose", r"illegal activity", r"crime", r"criminal activity"
    ]
    for pat in illegal_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

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

def is_schedule_query(text):
    """
    Returns True if the text is likely asking about a room schedule, availability, or occupancy.
    """
    schedule_patterns = [
        # --- Added more robust patterns for natural queries ---
        r"who is using [a-z]*\d+",  # e.g. who is using cea300
        r"who's in [a-z]*\d+",
        r"who is in [a-z]*\d+",
        r"is [a-z]*\d+ (free|vacant|available|occupied)",
        r"what room is (vacant|available|free|open)",
        r"what room is occupied",
        r"which room is (vacant|available|free|open)",
        r"which room is occupied",
        r"who's using [a-z]*\d+",
        r"who uses [a-z]*\d+",
        r"who occupies [a-z]*\d+",
        r"who is assigned to [a-z]*\d+",
        r"who has class in [a-z]*\d+",
        r"who is scheduled in [a-z]*\d+",
        r"who is scheduled at [a-z]*\d+",
        r"who is present in [a-z]*\d+",
        r"who is present at [a-z]*\d+",
        r"who's present in [a-z]*\d+",
        r"who's present at [a-z]*\d+",
        r"who's scheduled in [a-z]*\d+",
        r"who's scheduled at [a-z]*\d+",
    ]
    text_l = text.lower()
    for pat in schedule_patterns:
        if re.search(pat, text_l):
            return True
    return False
