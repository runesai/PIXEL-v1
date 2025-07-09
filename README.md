# Facebook Messenger Chatbot with FAQ Database

This is a Facebook Messenger chatbot that can communicate in both English and Filipino. It uses a GPT 4.1 model for generating responses and Google Sheets as a FAQ database.

## Features

- Bilingual support (English and Filipino)
- Automatic language detection
- Google Sheets integration for FAQ management
- Fallback to AI-generated responses when FAQ answer is not available
- Automatic logging of unanswered questions for admin review

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Facebook Setup**
   - Create a Facebook Page and Facebook App
   - Get your Page Access Token from Facebook Developer Console
   - Set up a webhook for your Facebook App

3. **Google Sheets Setup**
   - Create a new Google Sheet with two sheets named "FAQ" and "Unanswered"
   - FAQ sheet structure:
     - Column A: Questions
     - Column B: English Answers
     - Column C: Filipino Answers
   - Unanswered sheet structure:
     - Column A: Questions
     - Column B: Timestamp
   - Create a Google Cloud Project
   - Enable Google Sheets API
   - Create a service account and download credentials
   - Share your Google Sheet with the service account email

4. **Environment Setup**
   - Create a `.env` file with the following variables:
     ```
     PAGE_ACCESS_TOKEN=your_facebook_page_access_token
     VERIFY_TOKEN=your_webhook_verify_token
     SPREADSHEET_ID=your_google_spreadsheet_id
     ```
   - Place your Google Cloud service account credentials in `credentials.json`

5. **Running the Bot**
   ```bash
   python main.py
   ```

## Usage

1. Send a message to your Facebook Page in either English or Filipino
2. The bot will:
   - Detect the language
   - Search for an answer in the FAQ database
   - If found, respond with the appropriate language version
   - If not found:
     - Add the question to the Unanswered sheet
     - Generate an AI response as a fallback
     - Inform the user that the question has been logged

## Admin Tasks

1. Regularly check the "Unanswered" sheet for new questions
2. Add appropriate answers to the FAQ sheet in both languages
3. The bot will automatically use the new answers for future similar questions

## Note

Make sure to keep your credentials secure and never commit them to version control. The `.env` file and `credentials.json` should be added to your `.gitignore`. 
