import re

print("Loaded response_utils.py")

def strip_greeting(text):
    return re.sub(
        r'^(hi|hello|heyyy|kumusta|kamusta|magandang \\w+|good \\w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*',
        '', text, flags=re.IGNORECASE
    ).strip()

def get_transition_phrase(idx, question, lang=None):
    # Returns a transition phrase for subsequent answers, language-aware
    if lang == 'fil':
        base_phrases = [
            "Para sa sumunod mo na tanong:",
            "Para sa iyong kasunod na tanong:",
            f"Para sa tanong mong '{question.strip()}':",
            "Narito ang sagot sa iyong kasunod na tanong:",
            "Ito ang masasabi ko sa iyong kasunod na tanong:"
        ]
    else:
        base_phrases = [
            "As for your next question:",
            "To answer your next query:",
            f"To answer your question '{question.strip()}':",
            "Moving on to your next question:",
            "Here's what I can say about your next question:"
        ]
    return base_phrases[(idx-2) % len(base_phrases)]

def format_list_answers(questions, answers, instruction=None, lang=None):
    formatted = []
    for idx, (q, a) in enumerate(zip(questions, answers), 1):
        prefix = f"{idx}. {q.strip()}\n"
        # For all but the first answer, add a transition phrase and never a greeting
        if idx > 1:
            transition = get_transition_phrase(idx, q, lang)
            a = strip_greeting(a.strip())
            a = f"{transition} {a}"
        formatted.append(f"{prefix}{a.strip()}")
    return "\n\n".join(formatted)

def get_dynamic_fallback(lang, ask_github_gpt, PIXEL_PERSONALITY):
    system_prompt = PIXEL_PERSONALITY[lang]
    user_prompt = "You couldn't answer the user's question, no matter how hard you tried. As PIXEL, generate a friendly, trendy fallback message based on: 'Sorry, I couldn't understand your message.' Do not use a greeting or closing."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = ask_github_gpt(messages)
        return response.strip()
    except Exception:
        return "Sorry, I couldn't understand your message."
