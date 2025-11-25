from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv("./config/.env")

from routes import  base
app=FastAPI()
app.include_router(base.base_router)

def main():
    print("Hello from document-chat!")


if __name__ == "__main__":
    main()
