import re

print("Loaded string_utils.py")

def strip_greeting(text):
    # Remove common greetings at the start of a string (PIXEL style)
    return re.sub(
        r'^(hi|hello|heyyy|kumusta|kamusta|magandang \\w+|good \\w+|hello, iskolar!|hi, iskolar!|heyyy, iskolar!|hello po|hi po|sure, iskolar!)[!,. ]*',
        '', text, flags=re.IGNORECASE
    ).strip()

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
