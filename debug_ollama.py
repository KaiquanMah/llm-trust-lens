import ollama

try:
    client = ollama.Client(host="http://localhost:11434")
    result = client.list()
    print("Type of result:", type(result))
    print("\nAttributes:")
    for model in result.models:
        print(f"Model attributes:", vars(model))
except Exception as e:
    print(f"Error: {e}")
