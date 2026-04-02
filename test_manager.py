from memoryos.memory.manager import MemoryManager
from memoryos.config import AppConfig, MemoryConfig

# Small thresholds for testing
import dataclasses
config = AppConfig(
    memory=MemoryConfig(
        short_term_window=10,
        summarisation_threshold=4,
        turns_to_summarise=3,
        long_term_top_k=3,
    )
)

manager = MemoryManager(config, session_id="test-manager-001")

# Simulate a conversation
turns = [
    ("user", "I am building a RAG system in Python"),
    ("assistant", "Great choice, what retrieval strategy are you using?"),
    ("user", "Hybrid BM25 and dense retrieval with ChromaDB"),
    ("assistant", "Smart approach, that covers both lexical and semantic gaps"),
    ("user", "I prefer concise technical explanations"),
    ("assistant", "Noted, I will keep things brief"),
]

for role, content in turns:
    result = manager.add_message(role, content)
    print(f"[{role}] stored_lt={result['added_to_long_term']} "
          f"episode={result['episode_created'] is not None}")

print(f"\nManager state: {manager}")

# Check assembled context
messages = manager.get_messages_for_llm("what retrieval strategy should I use?")
print(f"\nSystem prompt:\n{messages[0]['content']}")
print(f"\nTotal messages for LLM: {len(messages)}")
