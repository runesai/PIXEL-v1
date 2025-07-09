from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime
import os

print("Loaded sheets_utils.py")

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'credentials.json'
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')

def get_sheets_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('sheets', 'v4', credentials=creds)

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

def get_weekly_room_schedule():
    """
    Fetches and robustly parses the 'Schedule' sheet, handling multiple day tables (Monday-Saturday), merged/unmerged cells, and day headers in any column or row (even with blank/merged header rows).
    Returns a dict: { 'monday': [ { 'Time': ..., 'Room 1': ..., ... }, ... ], ... }
    """
    print("[DEBUG] Accessing 'Schedule' sheet via get_weekly_room_schedule()...")
    service = get_sheets_service()
    sheet = service.spreadsheets().get(
        spreadsheetId=SPREADSHEET_ID,
        ranges=["Schedule"],
        includeGridData=True
    ).execute()
    grid = sheet['sheets'][0]['data'][0]['rowData']
    days = [d.lower() for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]]
    schedule = {}
    current_day = None
    headers = []
    i = 0
    while i < len(grid):
        row = grid[i]
        values = [cell.get('formattedValue', '') if 'formattedValue' in cell else '' for cell in row.get('values', [])]
        # Scan every cell for a day header (case-insensitive)
        found_day = None
        for idx, v in enumerate(values):
            v_clean = v.strip().lower()
            if v_clean in days:
                found_day = v_clean
                break
        if found_day:
            current_day = found_day
            schedule[current_day] = []
            # Find the next non-empty row as header
            j = i + 1
            while j < len(grid):
                next_row = grid[j]
                next_values = [cell.get('formattedValue', '') if 'formattedValue' in cell else '' for cell in next_row.get('values', [])]
                if any(val.strip() for val in next_values):
                    headers = next_values
                    break
                j += 1
            last_values = {h: '' for h in headers}
            i = j  # Move to header row
            i += 1  # Move to first data row
            continue
        # Data row: collect until next day header or end
        if current_day and headers:
            # Skip blank rows
            if not any(v.strip() for v in values):
                i += 1
                continue
            row_dict = {}
            for idx, h in enumerate(headers):
                val = values[idx] if idx < len(values) and values[idx] else ''
                if val:
                    row_dict[h] = val
                    last_values[h] = val
                else:
                    prev_val = last_values[h] if h in last_values else ''
                    row_dict[h] = prev_val
                    last_values[h] = prev_val
            schedule[current_day].append(row_dict)
        i += 1
    # Print a sample for debug
    for d in schedule:
        print(f"[DEBUG] {d.title()} sample:", schedule[d][:2])
        break
    return schedule

# Helper to get vacant rooms at a given day and time

def get_vacant_rooms(day, time):
    """
    Returns a list of vacant rooms for a given day and time.
    """
    schedule = get_weekly_room_schedule()
    day = day.lower()
    if day not in schedule:
        return []
    vacant = []
    for row in schedule[day]:
        if row.get('TIME', '').strip() == time.strip():
            for k, v in row.items():
                if k.lower().startswith('room') or k.upper().startswith('CEA'):
                    if not v.strip():
                        vacant.append(k)
            break
    return vacant

def is_room_occupied(day, room, time):
    """
    Checks if a room is occupied at a given day and time.
    day: string (e.g., 'monday')
    room: string (e.g., 'Room 1')
    time: string (exact match to the 'Time' column, e.g., '1:00 PM - 2:00 PM')
    Returns the occupant's name/info if occupied, or '' if free.
    """
    schedule = get_weekly_room_schedule()
    day = day.lower()
    if day not in schedule:
        return None  # Invalid day
    for row in schedule[day]:
        if row.get('TIME', '').strip() == time.strip():
            return row.get(room, '').strip()
    return None  # Time not found
