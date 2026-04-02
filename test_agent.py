from memoryos.agent import MemoryAgent
from memoryos.config import AppConfig, MemoryConfig

config = AppConfig(
    memory=MemoryConfig(
        short_term_window=10,
        summarisation_threshold=6,
        turns_to_summarise=4,
        long_term_top_k=3,
    )
)

agent = MemoryAgent(config, session_id="demo-session-001")

conversations = [
    "Hi, I am Sakshi. I am building a RAG system in Python.",
    "I prefer concise technical explanations.",
    "I am using ChromaDB for local development.",
    "What retrieval strategy would you recommend for my RAG system?",
]

for message in conversations:
    print(f"\nUser: {message}")
    result = agent.chat(message)
    print(f"Assistant: {result['response']}")
    print(f"Memory: lt={result['memory']['long_term_size']} "
          f"st={result['memory']['short_term_size']} "
          f"episodes={result['memory']['episode_count']}")

print(f"\nFull memory state:")
import json
print(json.dumps(agent.get_memory_state(), indent=2))
