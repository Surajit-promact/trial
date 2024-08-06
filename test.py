import os
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.llms.gemini import Gemini

load_dotenv()
google_api = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api)

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)


"""
models/gemini-1.0-pro-latest
models/gemini-1.0-pro
models/gemini-pro
models/gemini-1.0-pro-001
models/gemini-1.0-pro-vision-latest
models/gemini-pro-vision
models/gemini-1.5-pro-latest
models/gemini-1.5-pro-001
models/gemini-1.5-pro
models/gemini-1.5-pro-exp-0801
models/gemini-1.5-flash-latest
models/gemini-1.5-flash-001
models/gemini-1.5-flash
"""