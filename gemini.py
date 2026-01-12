import google.generativeai as genai
import os

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("hello")
    print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")
