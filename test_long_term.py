from memoryos.memory.long_term import LongTermMemory
from memoryos.config import MemoryConfig, ModelConfig, StorageConfig

mem = LongTermMemory(
    memory_config=MemoryConfig(),
    model_config=ModelConfig(),
    storage_config=StorageConfig(chroma_persist_dir="./test_chroma"),
    session_id="test-session-001"
)

# Store some memories
mem.add("user", "I prefer concise explanations over long ones", turn_index=0)
mem.add("user", "I am building a RAG system in Python", turn_index=1)
mem.add("user", "My name is Sakshi", turn_index=2)
mem.add("assistant", "Got it, I will keep things brief", turn_index=3)

print(f"Stored {mem.count()} memories")

# Retrieve by semantic similarity
results = mem.retrieve("what is the user building?")
print(f"\nQuery: 'what is the user building?'")
for r in results:
    print(f"  [{r.relevance_score}] {r.content}")

# Check prompt format
print(f"\nPrompt block:\n{mem.to_prompt_format('preferred explanation style')}")
