import os
from flask import Flask, request
from pymessenger.bot import Bot
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect
import google.auth
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
import json
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from functools import lru_cache
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Global configuration
CONFIG = {
    'faq_cache_timeout': 60,
    'keyword_threshold': 0.3,
    'base_threshold': 0.4,
    'language_bonus': 0.1,
    'sheet_ranges': {
        'faq': 'FAQ!A2:C',
        'language': 'Language!A2:D',
        'test_cases': 'TestCases!A2:E',
        'patterns': 'Patterns!A2:F'  # Updated to include context columns
    }
}

# Global state
STATE = {
    'models': {},
    'faq_cache': {},
    'language_variations': {
        'fil': {},
        'en': {},
        'categories': defaultdict(set)
    },
    'patterns': defaultdict(lambda: defaultdict(list)),
    'last_cache_refresh': 0,
    'conversations': defaultdict(lambda: {
        'last_intent': None,
        'last_response': None,
        'message_count': 0,
        'last_interaction': None,
        'greeting_sent': False
    })
}

# Language codes for NLLB model
LANG_CODES = {
    'en': 'eng_Latn',
    'fil': 'fil_Latn',
    'tl': 'fil_Latn'  # Adding Tagalog code
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Facebook Messenger credentials
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
if not PAGE_ACCESS_TOKEN or not VERIFY_TOKEN or not SPREADSHEET_ID:
    logger.warning("Missing credentials in environment variables")

app = Flask(__name__)

# Global variables to store models and embeddings
global_models = {}
faq_cache = {}
language_variations = {
    'fil': {},  # Will store Filipino word mappings
    'en': {},   # Will store English word mappings
    'categories': defaultdict(set)  # Will store words by category
}

# Chatbot personality and context
CHATBOT_PERSONALITY = {
    'en': "I am PIXEL, a polite and helpful student assistant at the CpE Department. I'm very familiar with department procedures, requirements, and academic policies. I aim to help students with their concerns in a friendly and professional manner.",
    'fil': "Ako si PIXEL, ang magalang at matulunging student assistant ng CpE Department. Pamilyar ako sa mga proseso, requirements, at patakaran ng departamento. Layunin kong tulungan ang mga estudyante sa kanilang mga katanungan nang may respeto at propesyonalismo."
}

# Add these to your global variables
FAQ_PATTERNS = defaultdict(list)
KEYWORD_VECTORIZER = None
KEYWORD_MATRIX = None
KEYWORD_THRESHOLD = 0.3  # Adjust this threshold as needed
FAQ_CACHE_TIMEOUT = 60  # Refresh cache every 60 seconds
last_cache_refresh = 0  # Track when cache was last refreshed

def initialize_models():
    """Initialize all models and patterns once at startup"""
    try:
        global global_models
        
        # Facebook Messenger credentials
        PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
        VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
        if not PAGE_ACCESS_TOKEN or not VERIFY_TOKEN:
            logger.warning("Missing Facebook credentials in environment variables")
        global_models['bot'] = Bot(PAGE_ACCESS_TOKEN)

        # Initialize translation model
        logger.info("Loading models and tokenizers...")
        model_name = "facebook/nllb-200-distilled-600M"
        global_models['tokenizer'] = AutoTokenizer.from_pretrained(model_name)
        global_models['model'] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Initialize sentence transformer
        global_models['sentence_model'] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Move model to CPU if CUDA is not available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        global_models['model'] = global_models['model'].to(device)
        
        # Initialize patterns and cache
        refresh_patterns()
        refresh_faq_cache()
        
        logger.info("Models and patterns initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

def get_google_sheets_service():
    """Initialize and return Google Sheets service"""
    try:
        logger.info("Attempting to initialize Google Sheets service...")
        # Load credentials from the service account file
        credentials = service_account.Credentials.from_service_account_file(
            'credentials.json',
            scopes=[
                'https://www.googleapis.com/auth/spreadsheets',  # Full access to sheets
            ]
        )
        logger.info("Credentials loaded successfully")
        
        # Build the service
        service = build('sheets', 'v4', credentials=credentials)
        logger.info("Google Sheets service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Error creating Google Sheets service: {str(e)}")
        return None

def extract_keywords(text):
    """Extract important keywords from text"""
    # Common words to ignore
    stop_words = {'the', 'and', 'or', 'in', 'at', 'to', 'for', 'a', 'of', 'is', 'are', 'was', 'were'}
    
    # Clean text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split into words
    words = text.split()
    # Remove stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return set(keywords)

def learn_patterns_from_faq():
    """Learn patterns and keywords from FAQ data"""
    global FAQ_PATTERNS, KEYWORD_VECTORIZER, KEYWORD_MATRIX
    
    try:
        if not faq_cache.get('qa_pairs'):
            refresh_faq_cache()
            
        if not faq_cache.get('qa_pairs'):
            logger.warning("No FAQ data available for pattern learning")
            return
            
        # Get all questions from FAQ
        questions = [row[0] for row in faq_cache['qa_pairs']]
        
        # Create TF-IDF vectorizer
        KEYWORD_VECTORIZER = TfidfVectorizer(
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            max_features=1000,
            stop_words='english'
        )
        
        # Fit and transform questions to get keyword importance
        KEYWORD_MATRIX = KEYWORD_VECTORIZER.fit_transform(questions)
        
        # Extract patterns for each question
        FAQ_PATTERNS.clear()
        for idx, question in enumerate(questions):
            # Get English and Filipino versions
            en_keywords = extract_keywords(question)
            
            # Create patterns for variations
            patterns = []
            
            # Add exact phrase patterns
            patterns.append(re.escape(question.lower()))
            
            # Add keyword-based patterns
            if len(en_keywords) > 1:
                # Create pattern with keywords in different orders
                keyword_pattern = r'.*'.join(map(re.escape, en_keywords))
                patterns.append(keyword_pattern)
            
            # Add common variations for Filipino
            fil_patterns = []
            for pattern in patterns:
                # Add common Filipino question starters
                fil_patterns.extend([
                    f"paano {pattern}",
                    f"pano {pattern}",
                    f"papaano {pattern}",
                    f"ano {pattern}",
                    f"saan {pattern}",
                    f"kailan {pattern}"
                ])
                
                # Add politeness markers
                fil_patterns.extend([
                    f"{pattern} po",
                    f"po {pattern}",
                    f"{pattern} po ba"
                ])
            
            patterns.extend(fil_patterns)
            
            # Store unique patterns
            FAQ_PATTERNS[idx] = list(set(patterns))
            
        logger.info(f"Learned {sum(len(patterns) for patterns in FAQ_PATTERNS.values())} patterns from {len(questions)} FAQs")
        
    except Exception as e:
        logger.error(f"Error learning FAQ patterns: {str(e)}")

def find_best_pattern_match(query, language):
    """Find the best matching FAQ using learned patterns"""
    try:
        best_score = 0
        best_idx = None
        
        # Clean the query
        cleaned_query = clean_and_normalize_text(query, language)
        query_lower = cleaned_query.lower()
        
        # Check each FAQ's patterns
        for idx, patterns in FAQ_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Calculate match score based on pattern length
                    score = len(pattern) / len(query_lower)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
        
        return best_idx, best_score
        
    except Exception as e:
        logger.error(f"Error matching patterns: {str(e)}")
        return None, 0

def find_keyword_matches(query):
    """Find matches based on keyword similarity"""
    try:
        if KEYWORD_VECTORIZER is None or KEYWORD_MATRIX is None:
            return None, 0
            
        # Transform query to TF-IDF space
        query_vector = KEYWORD_VECTORIZER.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, KEYWORD_MATRIX)[0]
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return best_idx, float(best_score)
        
    except Exception as e:
        logger.error(f"Error matching keywords: {str(e)}")
        return None, 0

def should_refresh_cache():
    """Check if cache needs refreshing"""
    global last_cache_refresh
    current_time = int(datetime.now().timestamp())
    return (current_time - last_cache_refresh) > CONFIG['faq_cache_timeout']

def refresh_all_sheets():
    """Refresh all sheet data"""
    try:
        logger.info("Refreshing all sheets...")
        # Refresh FAQ and Language variations
        refresh_faq_cache()
        # Refresh patterns
        refresh_patterns()
        logger.info("All sheets refreshed successfully")
    except Exception as e:
        logger.error(f"Error refreshing sheets: {str(e)}")

def refresh_language_variations():
    """Load language variations from Google Sheets with category-based weights"""
    global language_variations
    try:
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service")
            return
            
        # Get language variations data
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=CONFIG['sheet_ranges']['language']  # Using existing 'Language!A2:D'
        ).execute()
        
        values = result.get('values', [])
        if values:
            # Reset language variations
            language_variations = {
                'fil': {},
                'en': {},
                'categories': defaultdict(set),
                'category_weights': {
                    # Define weights based on category importance
                    'academic': 3.0,  # High weight for academic terms (gpa, units, subjects)
                    'politeness': 3.0,      # High weight for politeness markers (po, ho)
                    'question': 3.0,     # High weight for question words (ano, paano, saan)
                    'action': 2.5,        # Medium weight for action-related terms (submit, enroll, register)
                    'greeting': 2.5,      # Medium weight for greetings (hello, hi, good morning)
                    'transfer': 2.5,  # Medium weight for transfer-related terms (shifting, transferring)
                    'time': 2.5,          # Medium weight for time-related terms (kailan, oras)
                    'pronoun': 2.0,      # Good weight for pronouns (ako, ikaw, siya)
                    'status': 2.0,        # Good weight for status-related terms (status, update)
                    'common': 2.0,       # Good weight for common words (mga, ng, sa)
                    'particle': 2.0,     # Good weight for particles (ng, sa, ang)
                    'conjunction': 1.5,  # Medium weight for conjunctions
                    'default': 1.0       # Default weight for uncategorized words
                }
            }
            
            # Process each row
            for row in values:
                if len(row) >= 4:  # Ensure we have all columns
                    original = row[0].lower().strip()
                    normalized = row[1].lower().strip()
                    category = row[2].lower().strip()
                    language = row[3].lower().strip()
                    
                    # Store in appropriate language dictionary
                    if language in language_variations:
                        language_variations[language][original] = {
                            'normalized': normalized,
                            'category': category,
                            # Get weight based on category, default to 1.0 if category not in weights
                            'weight': language_variations['category_weights'].get(
                                category, 
                                language_variations['category_weights']['default']
                            )
                        }
                        
                    # Store in categories
                    if category:
                        language_variations['categories'][category].add(normalized)
                        
            logger.info(f"Loaded {len(values)} language variations with category-based weights")
        else:
            logger.warning("No language variations data found in sheet")
            
    except Exception as e:
        logger.error(f"Error loading language variations: {str(e)}")

def refresh_faq_cache():
    """Refresh the FAQ embeddings cache and extract keywords"""
    global faq_cache, last_cache_refresh
    try:
        logger.info("Refreshing FAQ cache...")
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service")
            return
            
        # Refresh language variations first
        refresh_language_variations()
            
        # Get FAQ data
        sheet = service.spreadsheets()
        logger.info(f"Fetching FAQ data from sheet {SPREADSHEET_ID}")
        
        # Verify FAQ sheet exists
        try:
            result = sheet.values().get(
                spreadsheetId=SPREADSHEET_ID,
                range='FAQ!A2:C'
            ).execute()
        except Exception as e:
            logger.error(f"Error accessing FAQ sheet: {str(e)}")
            return
        
        values = result.get('values', [])
        if values:
            logger.info(f"Retrieved {len(values)} FAQ entries")
            questions = [row[0] for row in values if len(row) > 0]
            if questions:
                # Store embeddings for semantic search
                logger.info("Generating embeddings for questions...")
                faq_cache['embeddings'] = global_models['sentence_model'].encode(questions)
                faq_cache['qa_pairs'] = values
                
                # Extract and store keywords for each question
                faq_cache['keywords'] = []
                for q in questions:
                    # Get keywords from both original and normalized text
                    original_keywords = extract_keywords(q)
                    normalized_en = clean_and_normalize_text(q, 'en')
                    normalized_fil = clean_and_normalize_text(q, 'fil')
                    normalized_keywords = extract_keywords(normalized_en) | extract_keywords(normalized_fil)
                    
                    # Combine all keywords
                    all_keywords = original_keywords | normalized_keywords
                    faq_cache['keywords'].append(all_keywords)
                
                # Update last refresh time
                last_cache_refresh = int(datetime.now().timestamp())
                logger.info(f"FAQ cache refreshed successfully with {len(values)} entries")
            else:
                logger.warning("No questions found in FAQ sheet")
        else:
            logger.warning("No data found in FAQ sheet")
            
    except Exception as e:
        logger.error(f"Error refreshing FAQ cache: {str(e)}")

def calculate_keyword_similarity(query_keywords, faq_keywords):
    """Calculate similarity based on keyword overlap"""
    if not query_keywords or not faq_keywords:
        return 0
    
    intersection = len(query_keywords & faq_keywords)
    union = len(query_keywords | faq_keywords)
    
    return intersection / union if union > 0 else 0

def clean_and_normalize_text(text, language):
    """Clean and normalize text using variations from Google Sheets"""
    try:
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Split into words
        words = text.split()
        normalized_words = []
        
        # Apply language-specific normalizations from sheets
        variations = language_variations.get(language, {})
        for word in words:
            # Try to find a normalized form
            normalized = variations.get(word, word)
            normalized_words.append(normalized)
        
        normalized_text = ' '.join(normalized_words)
        logger.debug(f"Normalized '{text}' to '{normalized_text}'")
        return normalized_text
        
    except Exception as e:
        logger.error(f"Error in text normalization: {str(e)}")
        return text  # Return original text if normalization fails

def calculate_similarity_score(query, faq_text, language):
    """Calculate overall similarity score between query and FAQ text"""
    # Clean and normalize both texts
    clean_query = clean_and_normalize_text(query, language)
    clean_faq = clean_and_normalize_text(faq_text, language)
    
    # Get keywords
    query_keywords = extract_keywords(clean_query)
    faq_keywords = extract_keywords(clean_faq)
    
    # Calculate keyword similarity
    keyword_similarity = calculate_keyword_similarity(query_keywords, faq_keywords)
    
    # Calculate semantic similarity
    query_embedding = global_models['sentence_model'].encode([clean_query])
    faq_embedding = global_models['sentence_model'].encode([clean_faq])
    semantic_similarity = float(cosine_similarity(query_embedding, faq_embedding)[0][0])
    
    # Combine scores (70% keyword, 30% semantic)
    combined_score = (0.7 * keyword_similarity) + (0.3 * semantic_similarity)
    
    return combined_score, clean_query, keyword_similarity, semantic_similarity

def get_time_context():
    """Get time-based context for responses"""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'night'

def get_pattern_response(text, language, sender_id=None):
    """Get context-aware response for matching pattern"""
    text_lower = text.lower()
      # Get or initialize conversation context
    context = STATE['conversations'][sender_id] if sender_id else {
        'last_intent': None,
        'last_response': None,
        'message_count': 0,
        'last_interaction': None,
        'greeting_sent': False
    }
    
    current_time = datetime.now()
      # Update conversation context if sender_id is provided
    if sender_id:
        if context['last_interaction']:
            time_diff = (current_time - context['last_interaction']).total_seconds()
            # Reset context if last interaction was more than 30 minutes ago
            if time_diff > 1800:  # 30 minutes
                context.update({
                    'last_intent': None,
                    'last_response': None,
                    'message_count': 0,
                    'greeting_sent': False
                })
        
        # Update context
        context['last_interaction'] = current_time
        context['message_count'] += 1
    
    # Get time context
    time_of_day = get_time_context()
    
    # Check each pattern type
    for pattern_type, languages in STATE['patterns'].items():
        # Get patterns for the current language
        patterns = languages.get(language, [])
        
        # Try each pattern
        for pattern_data in patterns:
            if re.search(pattern_data['pattern'], text_lower):
                # Check if pattern has context requirements
                if pattern_data['context_type'] and pattern_data['context_value']:
                    # Handle different context types
                    if pattern_data['context_type'] == 'time_of_day':
                        if pattern_data['context_value'] != time_of_day:
                            continue
                    elif pattern_data['context_type'] == 'message_count':
                        count_op = pattern_data['context_value'][0]  # Get operator (> or <)
                        count_val = int(pattern_data['context_value'][1:])  # Get value
                        if count_op == '>':
                            if not context['message_count'] > count_val:
                                continue
                        elif count_op == '<':
                            if not context['message_count'] < count_val:
                                continue
                    elif pattern_data['context_type'] == 'last_intent':
                        if pattern_data['context_value'] != context['last_intent']:
                            continue
                    elif pattern_data['context_type'] == 'greeting_status':
                        if pattern_data['context_value'].lower() == 'true':
                            if not context['greeting_sent']:
                                continue
                        else:
                            if context['greeting_sent']:
                                continue
                
                # If we get here, all context requirements are met
                response = pattern_data['response']
                
                # Update context if sender_id is provided
                if sender_id:
                    context['last_intent'] = pattern_type
                    context['last_response'] = response
                    if pattern_type == 'greeting':
                        context['greeting_sent'] = True
                
                return response
    
    return None

def search_in_faq(query, language, sender_id=None):
    """Search for an answer using enhanced matching"""
    try:
        logger.info(f"Processing query: '{query}' in language: {language}")
        
        # Check for common chat patterns first
        logger.info("Checking for pattern matches...")
        pattern_response = get_pattern_response(query, language, sender_id)
        if pattern_response:
            logger.info("Found matching pattern response")
            return pattern_response
        
        # If we get here, update context to show we're handling an FAQ
        if sender_id:
            STATE['conversations'][sender_id]['last_intent'] = 'faq'
            logger.info(f"Updated conversation context for {sender_id} - last_intent: faq")
        
        # Check if FAQ cache needs refresh
        if not faq_cache.get('qa_pairs'):
            logger.info("FAQ cache empty, refreshing...")
            refresh_faq_cache()
            if not faq_cache.get('qa_pairs'):
                logger.warning("Still no FAQ data after refresh, logging unanswered query")
                log_unanswered_query(query)
                return None
        
        logger.info(f"Searching through {len(faq_cache['qa_pairs'])} FAQ entries")
        
        # Find best match using combined similarity
        best_score = 0
        best_answer = None
        
        for qa_pair in faq_cache['qa_pairs']:
            try:
                # Check if we have enough columns for the question and answer
                if len(qa_pair) < 2:
                    logger.warning(f"Skipping invalid FAQ entry (needs question and answer): {qa_pair}")
                    # Return a message informing that the answer is not yet available
                    response = "I apologize, but the answer to your query is not yet available in my database. This has been logged for review." if language == 'en' else "Paumanhin, pero ang sagot sa iyong katanungan ay hindi pa available sa aking database. Ito ay naitala na para sa pagsusuri."
                    return response
                
                # Calculate similarity score
                score, clean_query, keyword_sim, semantic_sim = calculate_similarity_score(query, qa_pair[0], language)
                logger.debug(f"FAQ: '{qa_pair[0]}', Score: {score:.2f} (keyword: {keyword_sim:.2f}, semantic: {semantic_sim:.2f})")
                
                if score > best_score:
                    best_score = score
                    best_answer = qa_pair[1]  # Always get English answer
                    
            except Exception as e:
                logger.warning(f"Error processing FAQ entry {qa_pair}: {str(e)}")
                continue
        
        # Apply threshold with language bonus
        threshold = 0.4 - (0.1 if language == 'fil' else 0)
        logger.info(f"Best match score: {best_score:.2f} (threshold: {threshold})")
        
        if best_score > threshold and best_answer:
            logger.info("Found matching FAQ response")            # If query is in Filipino, translate the answer
            
             # Clean up the answer format
            answer = best_answer.strip()
            # Ensure proper line breaks
            answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)
            # Remove excessive spaces
            answer = re.sub(r' +', ' ', answer)
            
            if language == 'fil':
                logger.info("Translating answer to Filipino...")

                translated_answer = translate_text(answer, 'en')
                return translated_answer
            return answer
            #     try:
            #         ## Force device to CPU if needed
            #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #         global_models['model'].to(device)
                    
            #         ## Set up the translation with proper language codes
            #         tokenizer = global_models['tokenizer']
            #         model = global_models['model']
                    
            #         ## Add source language token and prepare the text
            #         inputs = tokenizer(f"eng_Latn {best_answer}", return_tensors="pt", padding=True).to(device)
                    
            #         ## Generate translation with Filipino as target
            #         translated = model.generate(
            #             **inputs,
            #             forced_bos_token_id=tokenizer.convert_tokens_to_ids("fil_Latn"),
            #             max_length=256,  # Increased max length for longer responses
            #             num_return_sequences=1,
            #             temperature=0.8,  # Slightly increased temperature for more natural translations
            #             top_p=0.95,
            #             do_sample=True    # Enable sampling for more natural outputs
            #         )
                    
            #         ## Decode the translation
            #         translated_answer = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    
            #         if translated_answer:
            #             return translated_answer
            #         else:
            #             ##If translation is empty, try fallback method
            #             return translate_text(best_answer, 'en')
                        
            #     except Exception as e:
            #         logger.error(f"Translation failed: {str(e)}")
            #         ## If translation fails, try to use translate_text as fallback
            #         try:
            #             return translate_text(best_answer, 'en')
            #         except:
            #             ## If all translation attempts fail, return with a note
            #             return f"{best_answer}\n\n(Paumanhin, nagkaroon ng problema sa pagsasalin.)"
            # return best_answer
            
        logger.info("No matching FAQ found, logging unanswered query")
        log_unanswered_query(query)
        return None
        
    except Exception as e:
        logger.error(f"Error searching FAQ: {str(e)}")
        return None

def translate_text(text, source_lang):
    try:
        tokenizer = global_models['tokenizer']
        model = global_models['model']
        
        # Split text into paragraphs
        paragraphs = text.split('\n')
        translated_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():  # Only translate non-empty paragraphs
                src_lang = LANG_CODES[source_lang]
                tgt_lang = LANG_CODES['fil' if source_lang == 'en' else 'en']
                
                # Add language tokens to input
                inputs = tokenizer(f"{src_lang} {paragraph}", return_tensors="pt", padding=True)
                
                # Generate translation
                translated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=256,  # Increased for longer texts
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                translated_paragraphs.append(translated_text)
        
        ## Rejoin paragraphs with proper spacing
        return '\n\n'.join(translated_paragraphs)
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text

def generate_ai_response(text, language):
    """Generate a contextual response using the translation model"""
    try:
        tokenizer = global_models['tokenizer']
        model = global_models['model']
        src_lang = LANG_CODES[language]
        
        # Add conversation context
        if language == 'fil':
            context = "Bilang isang magalang at matulunging chatbot, ang aking sagot ay: "
        else:
            context = "As a helpful and friendly chatbot, my response is: "
            
        enhanced_text = f"{src_lang} {context}{text}"
        
        # Prepare the input
        inputs = tokenizer(enhanced_text, return_tensors="pt", padding=True)
        
        # Generate response in the same language
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(src_lang),
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def verify():
    """Handle the webhook verification from Facebook"""
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    return "Hello world", 200

@app.route('/', methods=['POST'])
def webhook():
    """Handle incoming messages"""
    try:
        data = request.get_json()
        if data["object"] != "page":
            return "ok", 200
        
        # Track processed messages
        processed_messages = set()
            
        # Process each message
        for entry in data["entry"]:
            for event in entry["messaging"]:
                if not event.get("message"):
                    continue

                message_id = event.get("message", {}).get("mid")
                if message_id in processed_messages:
                    continue

                processed_messages.add(message_id)    
                sender_id = event["sender"]["id"]
                message_text = event["message"].get("text", "")
                
                if not message_text:
                    continue
                
                # Process message
                try:
                    # Refresh all sheets if needed
                    if should_refresh_cache():
                        refresh_all_sheets()
                    
                    # Get answer with context
                    language = detect_language_enhanced(message_text)
                    answer = search_in_faq(message_text, language, sender_id)
                    
                    if answer:

                         global_models['bot'].send_text_message(sender_id, answer)
                        # Remove personality if present
                        # answer = answer.split("\n\n")[0]
                    else:
                        # Generate contextual fallback response
                        context = STATE['conversations'][sender_id]
                        if context['message_count'] > 1:
                            answer = ("Paumanhin, hindi ko pa rin maintindihan. Pwede mo bang i-rephrase?" 
                                    if language == 'fil' else 
                                    "I'm still having trouble understanding. Could you rephrase that?")
                        else:
                            answer = ("Paumanhin, hindi ko maintindihan ang iyong mensahe. Pwede mo bang i-rephrase?" 
                                    if language == 'fil' else 
                                    "I'm sorry, I don't understand your message. Could you please rephrase it?")
                    
                    global_models['bot'].send_text_message(sender_id, answer)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    fallback = "Sorry, there was a technical error. Please try again later."
                    global_models['bot'].send_text_message(sender_id, fallback)
        
        return "ok", 200
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return "error", 500

def load_test_cases():
    """Load test cases from Google Sheets"""
    try:
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service")
            return None
            
        # Get test cases data
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='TestCases!A2:E'  # Get all test cases
        ).execute()
        
        values = result.get('values', [])
        if not values:
            logger.warning("No test cases found in sheet")
            return None
            
        # Organize test cases by category
        test_cases = defaultdict(dict)
        for row in values:
            if len(row) >= 3:  # Minimum required columns: Category, Query Type, Query
                category = row[0]
                query_type = row[1]
                query = row[2]
                expected_lang = row[3] if len(row) > 3 else None
                expected_type = row[4] if len(row) > 4 else None
                
                test_cases[category][query_type] = {
                    'query': query,
                    'expected_language': expected_lang,
                    'expected_type': expected_type
                }
        
        return test_cases
        
    except Exception as e:
        logger.error(f"Error loading test cases: {str(e)}")
        return None

def run_test_case(query, expected_lang=None, expected_type=None):
    """Run a single test case and return results"""
    results = {
        'query': query,
        'detected_language': None,
        'answer': None,
        'answer_type': None,
        'passed': False,
        'errors': []
    }
    
    try:
        # Test language detection
        results['detected_language'] = detect_language_enhanced(query)
        if expected_lang and results['detected_language'] != expected_lang:
            results['errors'].append(f"Language mismatch: expected {expected_lang}, got {results['detected_language']}")
        
        # Test answer generation
        answer = search_in_faq(query, results['detected_language'])
        results['answer'] = answer
        
        if answer:
            # Check if this is a pattern response
            is_pattern = False
            for pattern_type, languages in STATE['patterns'].items():
                for lang_patterns in languages.values():
                    if any(answer == p['response'] for p in lang_patterns):
                        is_pattern = True
                        break
                if is_pattern:
                    break
            
            results['answer_type'] = 'Pattern' if is_pattern else 'FAQ'
            
            if expected_type and results['answer_type'] != expected_type:
                results['errors'].append(f"Answer type mismatch: expected {expected_type}, got {results['answer_type']}")
        else:
            results['answer_type'] = 'None'
            if expected_type != 'None':
                results['errors'].append("Expected an answer but got none")
        
        # Check if test passed
        results['passed'] = len(results['errors']) == 0
        
    except Exception as e:
        results['errors'].append(f"Test error: {str(e)}")
    
    return results

def test_faq_queries():
    """Test the FAQ system with test cases from Google Sheets"""
    print("\nTesting FAQ System")
    print("=" * 80)
    
    # Load test cases
    test_cases = load_test_cases()
    if not test_cases:
        print("❌ No test cases available")
        return
    
    # Track statistics
    stats = {
        'total': 0,
        'passed': 0,
        'failed': 0
    }
    
    # Run tests by category
    for category, queries in test_cases.items():
        print(f"\n📋 Testing Category: {category}")
        print("-" * 40)
        
        for query_type, test_data in queries.items():
            stats['total'] += 1
            print(f"\n🔍 Test: {query_type}")
            
            # Run test
            results = run_test_case(
                test_data['query'],
                test_data['expected_language'],
                test_data['expected_type']
            )
            
            # Print results
            print(f"Query: '{results['query']}'")
            print(f"Language: {results['detected_language']}")
            if results['answer']:
                print(f"Answer: {results['answer'][:100]}..." if len(results['answer']) > 100 else results['answer'])
                print(f"Answer Type: {results['answer_type']}")
            
            if results['passed']:
                print("✅ Test passed")
                stats['passed'] += 1
            else:
                print("❌ Test failed:")
                for error in results['errors']:
                    print(f"   - {error}")
                stats['failed'] += 1
            
            print("-" * 20)
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"Total Tests: {stats['total']}")
    print(f"Passed: {stats['passed']} ({(stats['passed']/stats['total']*100):.1f}%)")
    print(f"Failed: {stats['failed']} ({(stats['failed']/stats['total']*100):.1f}%)")
    print("=" * 80)

def detect_language_enhanced(text):
    """Enhanced language detection using category-based weights"""
    try:
        # Initialize scores
        scores = {
            'fil': 0,
            'en': 0
        }
        
        # Clean and lowercase the text
        text = text.lower().strip()
        words = text.split()
        
        # Check each word
        for word in words:
            # Check Filipino words
            if word in language_variations.get('fil', {}):
                word_data = language_variations['fil'][word]
                scores['fil'] += word_data['weight']
                
            # Check English words
            if word in language_variations.get('en', {}):
                word_data = language_variations['en'][word]
                scores['en'] += word_data['weight']
            
            # Additional context scoring
            for category in language_variations['categories']:
                if word in language_variations['categories'][category]:
                    # Add small bonus for Filipino-specific categories
                    if category in ['academic', 'politeness', 'question', 'action', 'greeting', 'transfer', 'time', 'pronoun', 'status', 'common', 'particle',  'conjunction', 'default']:
                        scores['fil'] += 0.3
        
        # Add contextual analysis
        text_lower = text.lower()
        # Check for Filipino question patterns
        if any(pattern in text_lower for pattern in ['ano', 'paano', 'saan', 'sino', 'kailan']):
            scores['fil'] += 1.5
        # Check for politeness markers at the end
        if text_lower.endswith('po') or text_lower.endswith('po?'):
            scores['fil'] += 2.0
            
        logger.debug(f"Language scores - Filipino: {scores['fil']}, English: {scores['en']}")
        
        # Return language with highest score, default to English if tied
        return 'fil' if scores['fil'] > scores['en'] else 'en'
        
    except Exception as e:
        logger.error(f"Language detection error: {str(e)}")
        return 'en'  # Default to English on error

def calculate_similarity(query, faq_text, language):
    """Calculate similarity between query and FAQ text"""
    try:
        # Clean and normalize texts
        clean_query = clean_and_normalize_text(query, language)
        clean_faq = clean_and_normalize_text(faq_text, language)
        
        # Calculate keyword similarity
        query_keywords = extract_keywords(clean_query)
        faq_keywords = extract_keywords(clean_faq)
        keyword_sim = len(query_keywords & faq_keywords) / len(query_keywords | faq_keywords) if query_keywords else 0
        
        # Calculate semantic similarity
        embeddings = global_models['sentence_model'].encode([clean_query, clean_faq])
        semantic_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        
        # Return weighted combination
        return (0.7 * keyword_sim) + (0.3 * semantic_sim)
        
    except Exception as e:
        logger.error(f"Similarity calculation error: {str(e)}")
        return 0

def log_unanswered_query(query):
    """Log unanswered queries to Google Sheets for analysis"""
    try:
        logger.info(f"Attempting to log unanswered query: {query}")
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service for logging")
            return
            
        # Prepare the log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = [[timestamp, query]]
        
        # Append to the Unanswered Queries sheet
        logger.info("Appending to Unanswered sheet...")
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range='Unanswered!A:B',  # Assumes sheet named "Unanswered" with columns for timestamp and query
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body={'values': log_entry}
        ).execute()
        
        logger.info(f"Successfully logged unanswered query: {query}")
        
    except Exception as e:
        logger.error(f"Error logging unanswered query: {str(e)}")

def refresh_patterns():
    """Load chat patterns from Google Sheets"""
    try:
        logger.info("Refreshing patterns from Google Sheets...")
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service")
            return
            
        # Get patterns data
        sheet = service.spreadsheets()
        logger.info(f"Fetching patterns from range: {CONFIG['sheet_ranges']['patterns']}")
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=CONFIG['sheet_ranges']['patterns']
        ).execute()
        
        values = result.get('values', [])
        if values:
            # Reset patterns
            STATE['patterns'].clear()
            
            # Process each pattern
            pattern_count = 0
            for row in values:
                if len(row) >= 4:  # Minimum required columns
                    pattern_type = row[0].lower().strip()
                    pattern = row[1].strip()
                    language = row[2].lower().strip()
                    response = row[3].strip()
                    
                    # Get context information if available
                    context_type = row[4].lower().strip() if len(row) > 4 else None
                    context_value = row[5].strip() if len(row) > 5 else None
                    
                    # Store by type and language
                    if language not in STATE['patterns'][pattern_type]:
                        STATE['patterns'][pattern_type][language] = []
                    
                    STATE['patterns'][pattern_type][language].append({
                        'pattern': pattern,
                        'response': response,
                        'context_type': context_type,
                        'context_value': context_value
                    })
                    pattern_count += 1
            
            logger.info(f"Successfully loaded {pattern_count} patterns across {len(STATE['patterns'])} types")
            # Log pattern types and languages for debugging
            for pattern_type, languages in STATE['patterns'].items():
                logger.info(f"Pattern type '{pattern_type}' has patterns for languages: {list(languages.keys())}")
        else:
            logger.warning("No patterns found in sheet")
            
    except Exception as e:
        logger.error(f"Error loading patterns: {str(e)}")

def setup_patterns_sheet():
    """Set up the Patterns sheet structure with context columns if they don't exist"""
    try:
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service")
            return
            
        # Get current sheet data
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='Patterns!A1:F1'  # Get header row
        ).execute()
        
        values = result.get('values', [])
        expected_headers = ["Pattern Type", "Pattern", "Language", "Response", "Context Type", "Context Value"]
        
        if not values:
            # Sheet is empty, add headers
            update_request = {
                'values': [expected_headers]
            }
            sheet.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range='Patterns!A1',
                valueInputOption='RAW',
                body=update_request
            ).execute()
            logger.info("Added headers to Patterns sheet")
        else:
            current_headers = values[0]
            if len(current_headers) < len(expected_headers):
                # Need to add missing context columns
                update_request = {
                    'values': [expected_headers]
                }
                sheet.values().update(
                    spreadsheetId=SPREADSHEET_ID,
                    range='Patterns!A1',
                    valueInputOption='RAW',
                    body=update_request
                ).execute()
                logger.info("Updated Patterns sheet headers to include context columns")
        
        logger.info("Patterns sheet structure is ready for contextual patterns")
        
    except Exception as e:
        logger.error(f"Error setting up Patterns sheet: {str(e)}")

# Example usage:
if __name__ == "__main__":
    try:
        logger.info("Starting Flask server...")
        
        # Set up patterns sheet structure if SETUP_PATTERNS environment variable is set
        if os.getenv('SETUP_PATTERNS') == 'true':
            setup_patterns_sheet()
        
        initialize_models()  # Initialize models at startup
        
        # Run tests if in test mode
        if os.getenv('TEST_MODE') == 'true':
            test_faq_queries()
        
        # Start server
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise
