from memoryos.memory.short_term import ShortTermMemory
from memoryos.memory.episodic import EpisodicMemory
from memoryos.config import MemoryConfig, ModelConfig

# Small threshold so we can trigger summarisation easily
mem_cfg = MemoryConfig(
    short_term_window=10,
    summarisation_threshold=4,
    turns_to_summarise=3
)
model_cfg = ModelConfig()

short_term = ShortTermMemory(mem_cfg)
episodic = EpisodicMemory(mem_cfg, model_cfg)

# Add turns until we hit the threshold
short_term.add("user", "I am building a RAG system in Python")
short_term.add("assistant", "Great, what kind of documents are you indexing?")
short_term.add("user", "YouTube transcripts, I want to search across channels")
short_term.add("user", "I prefer using ChromaDB over Pinecone for local dev")

print(f"Short-term has {len(short_term)} messages")
print(f"Should summarise: {short_term.should_summarise()}")

# Trigger summarisation
episode = episodic.maybe_summarise(short_term)

if episode:
    print(f"\nEpisode created: {episode.id}")
    print(f"Turn range: {episode.turn_range}")
    print(f"Summary:\n{episode.summary}")
    print(f"\nShort-term after eviction: {len(short_term)} messages")
    print(f"\nPrompt format:\n{episodic.to_prompt_format()}")
else:
    print("No summarisation triggered")
