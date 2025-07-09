import os
import re
import sys
import logging
import torch
import numpy as np
import difflib
from flask import Flask, request
from pymessenger.bot import Bot
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect
import difflib
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
import signal
import openai

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

# Global state management
STATE = {
    'conversations': {},  # Store conversation state by user ID
    'patterns': {
        'greeting': {'en': [], 'fil': []},
        'farewell': {'en': [], 'fil': []},
        'thanks': {'en': [], 'fil': []},
        'confusion': {'en': [], 'fil': []},
        'smalltalk': {'en': [], 'fil': []}
    },
    'last_refresh': None
}

# Language code mappings for NLLB model
LANG_CODES = {
    'en': 'eng_Latn',
    'fil': 'fil_Latn',
    'tl': 'fil_Latn',
    'ceb': 'fil_Latn'
}

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pixel_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("Starting PIXEL chatbot...")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
logger.debug("Environment variables loaded")

# Verify tokens
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

if not PAGE_ACCESS_TOKEN or not VERIFY_TOKEN or not SPREADSHEET_ID:
    logger.error("Missing required environment variables")
    logger.debug(f"PAGE_ACCESS_TOKEN exists: {bool(PAGE_ACCESS_TOKEN)}")
    logger.debug(f"VERIFY_TOKEN exists: {bool(VERIFY_TOKEN)}")
    logger.debug(f"SPREADSHEET_ID exists: {bool(SPREADSHEET_ID)}")
    sys.exit(1)

logger.info("Environment variables verified")

# Initialize Flask app
app = Flask(__name__)
bot = Bot(PAGE_ACCESS_TOKEN)
logger.info("Flask app and bot initialized")

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
        logger.info("Starting model initialization sequence...")
        
        # Step 1: Initialize sentence transformer
        try:
            logger.info("Loading sentence transformer model...")
            global_models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {str(e)}")
            return False
        
        # Step 2: Initialize translation model
        try:
            logger.info("Loading translation model...")
            model_name = "facebook/nllb-200-distilled-600M"
            # Set device before loading model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Load tokenizer without downloading again
            global_models['tokenizer'] = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
            logger.info("Tokenizer loaded")
            
            # Load model without downloading again
            global_models['model'] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                local_files_only=True
            ).to(device)
            logger.info("Translation model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load translation model: {str(e)}")
            return False
            
        # Step 3: Test language detection
        try:
            logger.info("Testing language detection...")
            test_text = "Hello world"
            detect(test_text)
            logger.info("Language detection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize language detection: {str(e)}")
            return False
        
        # Step 4: Initialize patterns and cache
        try:
            logger.info("Initializing patterns...")
            refresh_patterns()
            refresh_faq_cache()
            logger.info("Patterns and cache initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize patterns or cache: {str(e)}")
            # Continue anyway since this is not critical
        
        # Final check
        required_models = ['sentence_model', 'tokenizer', 'model']
        for model in required_models:
            if model not in global_models:
                logger.error(f"Missing required model: {model}")
                return False
        
        logger.info("All models initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Fatal error during model initialization: {str(e)}")
        return False

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
        
        # Create TF-IDF vectorizer with more lenient parameters
        KEYWORD_VECTORIZER = TfidfVectorizer(
            ngram_range=(1, 3),  # Use unigrams, bigrams and trigrams
            max_features=2000,
            stop_words=['the', 'and', 'or', 'in', 'at', 'to', 'for', 'a', 'of', 'is', 'are', 'was', 'were'],
            analyzer='char_wb',  # Character n-grams for better typo tolerance
            min_df=1,
            max_df=0.9
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
            
            # Add fuzzy keyword-based patterns for typo tolerance
            if len(en_keywords) > 1:
                # Create pattern with keywords in different orders, allowing for fuzzy matches
                keyword_pattern = r'.*?'.join(f'({w}{{1,}})' for w in map(re.escape, en_keywords))
                patterns.append(keyword_pattern)
            
            # Add common variations for Filipino using language_variations
            fil_patterns = []
            for pattern in patterns:
                # Get question words from language variations
                question_words = set()
                for category in ['question', 'common']:
                    for word in language_variations['categories'].get(category, []):
                        if word in language_variations['fil']:
                            question_words.add(word)
                
                # Add patterns with question words
                for word in question_words:
                    variations = []
                    # Add the normalized form
                    variations.append(word)
                    # Find all variations that normalize to this word
                    for original, normalized in language_variations['fil'].items():
                        if normalized == word:
                            variations.append(original)
                            
                    for variation in variations:
                        fil_patterns.extend([
                            f"{variation}\\s+{pattern}",
                            f"{variation}\\s+po\\s+{pattern}",
                            f"{variation}\\s+{pattern}\\s+po"
                        ])
                
                # Add politeness markers and particles from language variations
                particles = language_variations['categories'].get('particle', set())
                for particle in particles:
                    fil_patterns.extend([
                        f"{pattern}\\s+{particle}",
                        f"{particle}\\s+{pattern}",
                        f"{pattern}\\s+{particle}\\s+ba",
                        f"{particle}\\s+ba\\s+{pattern}"
                    ])
            
            patterns.extend(fil_patterns)
            
            # Store unique patterns
            FAQ_PATTERNS[idx] = list(set(patterns))
            
        logger.info(f"Learned {sum(len(patterns) for patterns in FAQ_PATTERNS.values())} patterns from {len(questions)} FAQs")
        
    except Exception as e:
        logger.error(f"Error learning FAQ patterns: {str(e)}")

def find_best_pattern_match(query, language):
    """Find the best matching FAQ using learned patterns with improved fuzzy matching"""
    try:
        best_score = 0
        best_idx = None
        
        # Clean and normalize the query
        cleaned_query = clean_and_normalize_text(query, language)
        query_lower = cleaned_query.lower()
        
        # Check each FAQ's patterns with fuzzy matching
        for idx, patterns in FAQ_PATTERNS.items():
            for pattern in patterns:
                # Try exact pattern match first
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score = len(pattern) / len(query_lower)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        continue
                
                # If no exact match, try fuzzy matching
                try:
                    # Remove special regex characters for leven distance calc
                    clean_pattern = re.sub(r'[\\.*+?^${}()|[\]]+', '', pattern)
                    if len(clean_pattern) > 0:
                        # Calculate Levenshtein distance ratio
                        ratio = difflib.SequenceMatcher(None, clean_pattern, query_lower).ratio()
                        if ratio > 0.8 and ratio > best_score:  # 80% similarity threshold
                            best_score = ratio
                            best_idx = idx
                except Exception as e:
                    logger.debug(f"Error in fuzzy matching: {str(e)}")
                    continue
        
        # If no good match found, try keyword matching
        if best_score < 0.4:  # Threshold for acceptable pattern match
            k_idx, k_score = find_keyword_matches(query_lower)
            if k_score > 0.5:  # Higher threshold for keyword matches
                best_idx = k_idx
                best_score = k_score
        
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
    """Load language variations from Google Sheets"""
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
            range=CONFIG['sheet_ranges']['language']
        ).execute()
        
        values = result.get('values', [])
        if values:
            # Reset language variations
            language_variations = {
                'fil': {},
                'en': {},
                'categories': defaultdict(set)
            }
            
            # Process each row (skipping header)
            for row in values:
                if len(row) >= 4:  # Ensure we have all columns
                    original = row[0].lower().strip()
                    normalized = row[1].lower().strip()
                    category = row[2].lower().strip()
                    language = row[3].lower().strip()
                    
                    # Store in appropriate language dictionary
                    if language in language_variations:
                        language_variations[language][original] = normalized
                        
                    # Store in categories
                    if category:
                        language_variations['categories'][category].add(normalized)
                        
            logger.info(f"Loaded {len(values)} language variations")
        else:
            logger.warning("No language variations data found in sheet")
            
    except Exception as e:
        logger.error(f"Error loading language variations: {str(e)}")

def refresh_faq_cache():
    """Refresh the FAQ embeddings cache and properly handle multi-line answers"""
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
        
        # Get FAQ data with formatting
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='FAQ!A2:C',
            valueRenderOption='UNFORMATTED_VALUE'  # Get raw values
        ).execute()
        
        values = result.get('values', [])
        if values:
            logger.info(f"Retrieved {len(values)} FAQ entries")
            # Process and clean the FAQ entries
            cleaned_values = []
            for row in values:
                if len(row) >= 2:  # Must have at least question and answer
                    question = row[0].strip() if isinstance(row[0], str) else str(row[0])
                    # Handle multi-line answers by properly joining lines
                    answer = row[1]
                    if isinstance(answer, str):
                        # Normalize line endings and remove extra whitespace
                        answer = re.sub(r'\s*\n\s*', '\n', answer.strip())
                        # Ensure consistent spacing between bullet points/numbers
                        answer = re.sub(r'(\d+\.|•|\*)\s*', r'\1 ', answer)
                    else:
                        answer = str(answer)
                    
                    cleaned_values.append([question, answer])
            
            if cleaned_values:
                # Store embeddings for semantic search
                questions = [row[0] for row in cleaned_values]
                logger.info("Generating embeddings for questions...")
                faq_cache['embeddings'] = global_models['sentence_model'].encode(questions)
                faq_cache['qa_pairs'] = cleaned_values
                
                # Update cache timestamp
                last_cache_refresh = int(datetime.now().timestamp())
                
                # Learn patterns from the new FAQ data
                learn_patterns_from_faq()
                
                logger.info("FAQ cache refreshed successfully")
            else:
                logger.warning("No valid FAQ entries found")
        else:
            logger.warning("No FAQ data found in sheet")
            
    except Exception as e:
        logger.error(f"Error refreshing FAQ cache: {str(e)}")
        return None
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
    try:
        text_lower = text.lower()
        logger.info(f"Checking pattern response for: '{text_lower}' in {language}")
        
        # Get or initialize conversation context
        if sender_id:
            if sender_id not in STATE['conversations']:
                STATE['conversations'][sender_id] = {
                    'last_intent': None,
                    'last_response': None,
                    'message_count': 0,
                    'last_interaction': None,
                    'greeting_sent': False
                }
            context = STATE['conversations'][sender_id]
        else:
            context = {
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
    except Exception as e:
        logger.error(f"Error in get_pattern_response: {str(e)}")
        return None

def search_in_faq(query, language, sender_id=None):
    """Search for an answer using enhanced matching"""
    try:
        logger.info(f"Processing query: '{query}' in language: {language}")
        
        # Initialize conversation state if needed
        if sender_id and sender_id not in STATE['conversations']:
            STATE['conversations'][sender_id] = {
                'last_intent': None,
                'last_response': None,
                'message_count': 0,
                'last_interaction': None,
                'greeting_sent': False
            }
        
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
            if language == 'fil':
                logger.info("Translating answer to Filipino...")
                try:
                    # Force device to CPU if needed
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    global_models['model'].to(device)
                    
                    # Set up the translation with proper language codes
                    tokenizer = global_models['tokenizer']
                    model = global_models['model']
                    
                    # Add source language token and prepare the text
                    inputs = tokenizer(f"eng_Latn {best_answer}", return_tensors="pt", padding=True).to(device)
                    
                    # Generate translation with Filipino as target
                    translated = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids("fil_Latn"),
                        max_length=256,  # Increased max length for longer responses
                        num_return_sequences=1,
                        temperature=0.8,  # Slightly increased temperature for more natural translations
                        top_p=0.95,
                        do_sample=True    # Enable sampling for more natural outputs
                    )
                    
                    # Decode the translation
                    translated_answer = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    
                    if translated_answer:
                        return translated_answer
                    else:
                        # If translation is empty, try fallback method
                        return translate_text(best_answer, 'en')
                        
                except Exception as e:
                    logger.error(f"Translation failed: {str(e)}")
                    # If translation fails, try to use translate_text as fallback
                    try:
                        return translate_text(best_answer, 'en')
                    except:
                        # If all translation attempts fail, return with a note
                        return f"{best_answer}\n\n(Paumanhin, nagkaroon ng problema sa pagsasalin.)"
            return best_answer
            
        logger.info("No matching FAQ found, logging unanswered query")
        log_unanswered_query(query)
        return None
        
    except Exception as e:
        logger.error(f"Error searching FAQ: {str(e)}")
        return None

def translate_text(text, target_language):
    """Translate text using the NLLB model"""
    try:
        if not text:
            return text
            
        # Get the source language
        try:
            src_lang = detect(text)
            if src_lang == target_language:
                return text
        except:
            src_lang = 'en'  # Default to English if detection fails
            
        # Map to NLLB language codes
        src_code = LANG_CODES.get(src_lang, 'eng_Latn')
        target_code = LANG_CODES.get(target_language, 'fil_Latn')
        
        # Tokenize and translate
        inputs = global_models['tokenizer'](text, return_tensors="pt")
        translated = global_models['model'].generate(
            **inputs,
            forced_bos_token_id=global_models['tokenizer'].lang_code_to_id[target_code],
            max_length=512,
            num_beams=5,
            temperature=0.8
        )
        
        # Decode the translation
        translated_text = global_models['tokenizer'].batch_decode(translated, skip_special_tokens=True)[0]
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def generate_ai_response(text, language):
    """Generate a contextual response using the translation model"""
    try:
        tokenizer = global_models['tokenizer']
        model = global_models['model']
        src_lang = LANG_CODES[language]
    except Exception as e:
        logger.error(f"Error initializing translation models: {str(e)}")
        return None
    
    try:
        
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
            logger.warning("Failed verification - token mismatch")
            return "Token verification failed", 403
        logger.info("Webhook verified successfully")
        return request.args["hub.challenge"], 200
    return "Hello world", 200

@app.route('/', methods=['POST'])
def webhook():
    """Handle incoming messages"""
    try:
        data = request.get_json()
        logger.debug(f"Received webhook data: {data}")
        
        if data["object"] == "page":
            for entry in data["entry"]:
                for messaging_event in entry["messaging"]:
                    sender_id = messaging_event["sender"]["id"]
                    
                    if messaging_event.get("message"):
                        message = messaging_event["message"]
                        message_text = message.get("text")
                        
                        if not message_text:
                            continue
                            
                        logger.info(f"Received message from {sender_id}: {message_text}")
                        
                        try:
                            # Language detection with better error handling
                            try:
                                detected_lang = detect(message_text)
                                # Map similar languages to our supported ones
                                detected_lang = 'fil' if detected_lang in ['tl', 'ceb', 'fil'] else 'en'
                                logger.info(f"Detected language: {detected_lang}")
                            except Exception as lang_error:
                                logger.warning(f"Language detection failed: {lang_error}, defaulting to English")
                                detected_lang = 'en'
                            
                            # Check pattern matches first (greetings, thanks, etc)
                            response = get_pattern_response(message_text, detected_lang, sender_id)
                            if response:
                                logger.info(f"Found pattern response: {response}")
                                bot.send_text_message(sender_id, response)
                                continue
                            
                            # Try FAQ search if no pattern match
                            response = search_in_faq(message_text, detected_lang, sender_id)
                            if response:
                                logger.info(f"Found FAQ response: {response}")
                                bot.send_text_message(sender_id, response)
                                continue
                            
                            # If we get here, no response was found
                            # Check if it's a greeting/thanks before logging as unanswered
                            if not any(word in message_text.lower() 
                                for word in ['hello', 'hi', 'thanks', 'thank', 'kumusta', 'salamat']):
                                log_unanswered_query(message_text)
                                # Use appropriate fallback message
                                response = ("Paumanhin, hindi ko maintindihan ang iyong katanungan. "
                                    "Maaari mo bang ipaliwanag ito sa ibang paraan?") if detected_lang == 'fil' else (
                                    "I apologize, but I don't understand your question. Could you please rephrase it?")
                                bot.send_text_message(sender_id, response)
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                            # Send appropriate error message
                            error_msg = ("Sorry, I encountered an error. Please try again." if detected_lang == 'en' 
                                else "Paumanhin, may naganap na error. Pakisubukan muli.")
                            bot.send_text_message(sender_id, error_msg)
                            

        return "ok", 200
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        return str(e), 500

def load_test_cases():
    """Load test cases from Google Sheets"""
    try:
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service")
            return []
            
        # Get test cases data
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=CONFIG['sheet_ranges']['test_cases']
        ).execute()
        
        values = result.get('values', [])
        if not values:
            logger.warning("No test cases found")
            return []
            
        test_cases = []
        for row in values:
            if len(row) >= 3:  # Minimum: query, expected_lang, expected_type
                test_case = {
                    'query': row[0],
                    'expected_lang': row[1],
                    'expected_type': row[2],
                    'expected_response': row[3] if len(row) > 3 else None,
                    'notes': row[4] if len(row) > 4 else None
                }
                test_cases.append(test_case)
                
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
        
    except Exception as e:
        logger.error(f"Error loading test cases: {str(e)}")
        return []

def run_test_case(query, expected_lang=None, expected_type=None):
    """Run a single test case and return results"""
    try:
        # Detect language
        detected_lang = detect(query)
        detected_lang = 'fil' if detected_lang in ['tl', 'ceb', 'fil'] else 'en'
        
        # Get response
        response = search_in_faq(query, detected_lang)
        
        # Determine response type
        response_type = 'unknown'
        if response:
            if re.search(r'(sorry|apologize|couldn\'t find|paumanhin|hindi mahanap)', response, re.IGNORECASE):
                response_type = 'not_found'
            else:
                response_type = 'faq'
                
        results = {
            'query': query,
            'detected_lang': detected_lang,
            'response_type': response_type,
            'response': response,
            'passed_lang': not expected_lang or detected_lang == expected_lang,
            'passed_type': not expected_type or response_type == expected_type,
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error running test case: {str(e)}")
        return None

def run_all_tests():
    """Run all test cases and generate a report"""
    try:
        test_cases = load_test_cases()
        if not test_cases:
            logger.error("No test cases available")
            return
            
        results = []
        for test in test_cases:
            result = run_test_case(
                test['query'],
                test.get('expected_lang'),
                test.get('expected_type')
            )
            if result:
                results.append(result)
                
        # Calculate statistics
        total = len(results)
        passed_lang = sum(1 for r in results if r['passed_lang'])
        passed_type = sum(1 for r in results if r['passed_type'])
        passed_both = sum(1 for r in results if r['passed_lang'] and r['passed_type'])
        
        # Log results
        logger.info(f"""
Test Results:
Total tests: {total}
Language detection accuracy: {(passed_lang/total)*100:.1f}%
Response type accuracy: {(passed_type/total)*100:.1f}%
Overall accuracy: {(passed_both/total)*100:.1f}%
        """)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return None

def refresh_patterns():
    """Load and refresh conversation patterns from Google Sheets"""
    try:
        logger.info("Refreshing conversation patterns...")
        service = get_google_sheets_service()
        if not service:
            logger.error("Could not initialize Google Sheets service")
            return

        # Get patterns data with proper format
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range='Patterns!A2:F',  # Get all pattern fields
            valueRenderOption='FORMULA'  # Get formulas as formulas
        ).execute()
        
        # Log pattern loading attempt
        logger.info("Attempting to load patterns from Google Sheets...")

        values = result.get('values', [])
        if not values:
            logger.warning("No patterns found in sheet")
            return

        # Reset patterns in STATE
        for pattern_type in STATE['patterns']:
            STATE['patterns'][pattern_type] = {'en': [], 'fil': []}

        # Process each pattern row
        for row in values:
            if len(row) >= 6:  # pattern, response, type, language, context_type, context_value
                pattern_data = {
                    'pattern': row[0].strip().lower(),
                    'response': row[1].strip(),
                    'type': row[2].strip().lower(),
                    'language': row[3].strip().lower(),
                    'context_type': row[4].strip().lower() if row[4].strip() else None,
                    'context_value': row[5].strip() if len(row) > 5 and row[5].strip() else None
                }

                # Add to appropriate category and language
                if pattern_data['type'] in STATE['patterns'] and pattern_data['language'] in ['en', 'fil']:
                    STATE['patterns'][pattern_data['type']][pattern_data['language']].append(pattern_data)

        STATE['last_refresh'] = datetime.now()
        logger.info("Patterns refreshed successfully")

    except Exception as e:
        logger.error(f"Error refreshing patterns: {str(e)}")

def log_unanswered_query(query):
    """Log queries that couldn't be answered for later review"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to local file
        log_entry = f"{timestamp}\t{query}\n"
        with open('unanswered_queries.log', 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        # Log to Google Sheets
        service = get_google_sheets_service()
        if service:
            sheet = service.spreadsheets()
            values = [[timestamp, query]]
            body = {'values': values}
            
            try:
                sheet.values().append(
                    spreadsheetId=SPREADSHEET_ID,
                    range='Unanswered!A:B',  # Specify the exact sheet and range
                    valueInputOption='RAW',
                    insertDataOption='INSERT_ROWS',
                    body=body
                ).execute()
                logger.info(f"Logged unanswered query to Google Sheets: {query}")
            except Exception as sheet_error:
                logger.error(f"Failed to log to Google Sheets: {str(sheet_error)}")
        
        logger.info(f"Logged unanswered query locally: {query}")
        
    except Exception as e:
        logger.error(f"Error logging unanswered query: {str(e)}")

# Helper function to check if patterns need refresh
def should_refresh_patterns():
    """Check if patterns need to be refreshed"""
    if STATE['last_refresh'] is None:
        return True
        
    time_since_refresh = (datetime.now() - STATE['last_refresh']).total_seconds()
    return time_since_refresh > CONFIG['faq_cache_timeout']

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal, cleaning up...")
    # Add any cleanup code here
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        # Initialize models first
        logger.info("Starting model initialization...")
        if not initialize_models():
            logger.error("Failed to initialize models")
            sys.exit(1)
        
        # Start the Flask server
        logger.info("Starting Flask server...")
        # Use host='0.0.0.0' to make the server externally visible
        # Set threaded=True to handle multiple requests
        app.run(
            host='0.0.0.0',  # Allow external connections
            port=5000,  # Standard port
            debug=False,  # Disable debug mode in production
            use_reloader=False,  # Prevent auto-reloader
            threaded=True  # Enable multi-threading
        )
    except Exception as e:
        logger.error(f"Fatal error during startup: {str(e)}")
        sys.exit(1)
