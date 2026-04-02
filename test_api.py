import requests

BASE = "http://localhost:8000"

# Health check
r = requests.get(f"{BASE}/health")
print(f"Health: {r.json()}")

# Chat turn 1
r = requests.post(f"{BASE}/chat", json={
    "session_id": "api-test-001",
    "message": "Hi, I am Sakshi. I prefer Python and I am building a RAG system."
})
print(f"\nTurn 1: {r.json()['response']}")
print(f"Memory: {r.json()['memory_metadata']}")

# Chat turn 2
r = requests.post(f"{BASE}/chat", json={
    "session_id": "api-test-001",
    "message": "What language should I use for my project?"
})
print(f"\nTurn 2: {r.json()['response']}")

# Inspect memory
r = requests.get(f"{BASE}/memory/api-test-001")
state = r.json()
print(f"\nMemory state: {state['turn_count']} turns, "
      f"{state['long_term_count']} long-term entries")

# Auto-generated docs — open this in your browser
print(f"\nAPI docs: {BASE}/docs")
