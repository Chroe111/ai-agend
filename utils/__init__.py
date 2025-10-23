def load_settings():
    from dotenv import load_dotenv
    from .settings import Settings
    load_dotenv()
    return Settings()

settings = load_settings()
