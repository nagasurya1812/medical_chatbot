import google.generativeai as genai
genai.configure(api_key="AIzaSyAJ9MI88QR6wbbABsDokTUQXeMU9oNtOh4")  
models = genai.list_models()
print([m.name for m in models])
