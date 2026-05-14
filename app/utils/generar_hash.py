from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

password = input("Introduce la contraseña: ")

password_hash = pwd_context.hash(password)

print("\nHash generado:\n")
print(password_hash)