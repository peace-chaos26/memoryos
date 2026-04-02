from memoryos.eval import MemoryEvaluator
from memoryos.config import AppConfig, MemoryConfig

config = AppConfig(
    memory=MemoryConfig(
        short_term_window=10,
        long_term_top_k=3,
        summarisation_threshold=6,
        turns_to_summarise=4,
    )
)

evaluator = MemoryEvaluator(config)
print("Running full evaluation...\n")
result = evaluator.run_full_eval()

print(f"\n{'='*40}")
print("EVALUATION RESULTS")
print('='*40)
print(result.summary())
