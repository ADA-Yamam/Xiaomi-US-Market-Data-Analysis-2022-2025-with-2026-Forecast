# smart_input_loader.py
import pandas as pd
from tkinter import Tk, filedialog

def load_csv_or_sql():
    """
    Smart input loader:
    - Ask user to select a local CSV file
    - Or paste a URL to CSV
    - (Optional: add SQL later)
    Returns a pandas DataFrame
    """
    print("Choose input method:")
    print("1. Select local CSV file")
    print("2. Paste CSV URL")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Open file dialog
        Tk().withdraw()  # hide root window
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            raise ValueError("No file selected.")
        print(f"Loading CSV from: {file_path}")
        df = pd.read_csv(file_path)

    elif choice == "2":
        url = input("Paste CSV URL: ").strip()
        if not url:
            raise ValueError("No URL provided.")
        print(f"Loading CSV from URL: {url}")
        df = pd.read_csv(url)

    else:
        raise ValueError("Invalid choice. Enter 1 or 2.")

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    df = load_csv_or_sql()
    print(df.head())
