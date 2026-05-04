from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

passwords = {
    "operador": "operador123",
    "admin": "admin123"
}

for usuario, password in passwords.items():
    password_hash = pwd_context.hash(password)
    print(f"{usuario}: {password_hash}")