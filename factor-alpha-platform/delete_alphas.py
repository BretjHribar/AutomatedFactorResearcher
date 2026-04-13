import sqlite3
import os

DB_PATH = "data/alphas_5m.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM evaluations WHERE alpha_id IN (7, 10)")
        print("Deleted #7 and #10 from evaluations.")
        conn.execute("DELETE FROM alphas WHERE id IN (7, 10)")
        print("Deleted #7 and #10 from alphas.")
        conn.commit()
        print("Deletion successful.")
    except Exception as e:
        print(f"Error during deletion: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
