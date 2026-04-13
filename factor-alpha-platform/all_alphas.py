import sqlite3
import os

DB_PATH = "data/alphas_5m.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, expression FROM alphas ORDER BY id").fetchall()
    conn.close()
    
    with open("all_alphas.txt", "w") as f:
        for r in rows:
            f.write(f"Alpha #{r[0]}:\n{r[1]}\n\n")
            print(f"Alpha #{r[0]}: Written")

if __name__ == "__main__":
    main()
