import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Google Sheets Setup ---
SCOPE = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'credentials.json'  # Make sure this file exists in your workspace
SHEET_NAME = 'ChatbotData'  # Change to your actual sheet name

class GoogleSheetsManager:
    def __init__(self):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPE)
        self.client = gspread.authorize(creds)
        self.sheet = self.client.open(SHEET_NAME).sheet1

    def get_all_data(self):
        return self.sheet.get_all_records()

    def add_row(self, question, answer, keywords):
        self.sheet.append_row([question, answer, keywords])

    def update_row(self, row_idx, question, answer, keywords):
        self.sheet.update(f'A{row_idx}:C{row_idx}', [[question, answer, keywords]])

    def delete_row(self, row_idx):
        self.sheet.delete_rows(row_idx)

class ChatbotSheetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Chatbot Google Sheets Manager')
        self.manager = GoogleSheetsManager()
        self.create_widgets()
        self.refresh_data()

    def create_widgets(self):
        self.tree = ttk.Treeview(self.root, columns=('Question', 'Answer', 'Keywords'), show='headings')
        self.tree.heading('Question', text='Question')
        self.tree.heading('Answer', text='Answer')
        self.tree.heading('Keywords', text='Keywords')
        self.tree.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text='Add', command=self.add_entry).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text='Edit', command=self.edit_entry).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text='Delete', command=self.delete_entry).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text='Refresh', command=self.refresh_data).pack(side=tk.LEFT, padx=5, pady=5)

    def refresh_data(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        data = self.manager.get_all_data()
        for idx, row in enumerate(data, start=2):  # Google Sheets rows start at 2 for data
            self.tree.insert('', 'end', iid=idx, values=(row['Question'], row['Answer'], row['Keywords']))

    def add_entry(self):
        q = simpledialog.askstring('Add', 'Question:')
        a = simpledialog.askstring('Add', 'Answer:')
        k = simpledialog.askstring('Add', 'Keywords (comma-separated):')
        if q and a and k:
            self.manager.add_row(q, a, k)
            self.refresh_data()

    def edit_entry(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning('Edit', 'No entry selected.')
            return
        idx = int(selected[0])
        values = self.tree.item(selected[0], 'values')
        q = simpledialog.askstring('Edit', 'Question:', initialvalue=values[0])
        a = simpledialog.askstring('Edit', 'Answer:', initialvalue=values[1])
        k = simpledialog.askstring('Edit', 'Keywords (comma-separated):', initialvalue=values[2])
        if q and a and k:
            self.manager.update_row(idx, q, a, k)
            self.refresh_data()

    def delete_entry(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning('Delete', 'No entry selected.')
            return
        idx = int(selected[0])
        if messagebox.askyesno('Delete', 'Are you sure you want to delete this entry?'):
            self.manager.delete_row(idx)
            self.refresh_data()

if __name__ == '__main__':
    root = tk.Tk()
    app = ChatbotSheetGUI(root)
    root.mainloop()
