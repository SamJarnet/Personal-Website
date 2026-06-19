import sqlite3
import bcrypt

class WebLoginManager:
    def __init__(self):
        self.conn = sqlite3.connect("logins.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT NOT NULL,
                password TEXT NOT NULL)''')
        self.conn.commit()

    def signup(self, username, password):
        if not username or not password:
            return {"status": "error", "msg": "Enter all data."}
            
        self.cursor.execute("SELECT username FROM users WHERE username=?", [username])
        if self.cursor.fetchone() is not None:
            return {"status": "error", "msg": "Username already exists"}
            
        encoded_password = password.encode("utf-8")
        hashed_password = bcrypt.hashpw(encoded_password, bcrypt.gensalt())
        self.cursor.execute("INSERT INTO users VALUES (?, ?)", [username, hashed_password])
        self.conn.commit()
        return {"status": "success", "msg": "Account has been created."}

    def login(self, username, password):
        if not username or not password:
            return {"status": "error", "msg": "Enter all data."}
            
        self.cursor.execute("SELECT password FROM users WHERE username=?", [username]) 
        result = self.cursor.fetchone()
        
        if result and bcrypt.checkpw(password.encode("utf-8"), result[0]):
            return {"status": "success", "msg": "Login successful", "user": username}
        else:
            return {"status": "error", "msg": "Invalid username or password."}