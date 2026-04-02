from dataclasses import dataclass, field
from openai import OpenAI
from memoryos.agent import MemoryAgent
from memoryos.memory.episodic import Episode
from memoryos.memory.short_term import Message
from memoryos.config import AppConfig


@dataclass
class EvalResult:
    """Results from one full evaluation run."""
    memory_hit_rate: float = 0.0
    faithfulness_scores: list[float] = field(default_factory=list)
    memory_lift_scores: list[dict] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        if not self.faithfulness_scores:
            return 0.0
        return round(sum(self.faithfulness_scores) / len(self.faithfulness_scores), 3)

    def summary(self) -> str:
        lift_scores = [s["with_memory"] - s["without_memory"]
                      for s in self.memory_lift_scores]
        avg_lift = round(sum(lift_scores) / len(lift_scores), 3) if lift_scores else 0.0
        return (
            f"Memory Hit Rate:     {self.memory_hit_rate:.1%}\n"
            f"Avg Faithfulness:    {self.avg_faithfulness:.3f}\n"
            f"Avg Memory Lift:     {avg_lift:+.3f}\n"
            f"Episodes Evaluated:  {len(self.faithfulness_scores)}\n"
            f"Lift Tests Run:      {len(self.memory_lift_scores)}"
        )


class MemoryEvaluator:
    """
    Evaluation harness for MemoryOS.

    Measures three dimensions:
    1. Memory hit rate — does retrieval surface the right memory?
    2. Faithfulness — are episodic summaries accurate?
    3. Memory lift — does memory improve response quality?
    """

    FAITHFULNESS_PROMPT = """You are evaluating whether a summary is faithful 
to its source conversation turns.

Source turns:
{source_turns}

Generated summary:
{summary}

Score the summary from 0.0 to 1.0:
- 1.0: Every statement in the summary is explicitly supported by the source
- 0.5: Most statements are supported, minor additions present  
- 0.0: Summary contains significant information not in the source

Respond with only a number between 0.0 and 1.0."""

    QUALITY_PROMPT = """Rate this response from 0.0 to 1.0 for helpfulness 
and contextual accuracy given the conversation.

User asked: {query}
Response: {response}
Known context: {context}

Score only. Respond with a single number between 0.0 and 1.0."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = OpenAI()

    def _llm_score(self, prompt: str) -> float:
        """Ask LLM to score something. Returns float 0.0-1.0."""
        response = self._client.chat.completions.create(
            model=self.config.model.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,   # deterministic scoring
        )
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.0

    def eval_memory_hit_rate(
        self,
        test_cases: list[dict]
    ) -> float:
        """
        Measure retrieval accuracy.

        Each test case:
        {
            "stored_fact": "I prefer Python over Java",
            "query": "what programming language does the user prefer?",
            "expected_keyword": "python"
        }
        """
        hits = 0
        for case in test_cases:
            # Create a fresh agent and store the fact
            agent = MemoryAgent(self.config, session_id=f"eval-hit-{hits}")
            agent.memory.long_term.add("user", case["stored_fact"], turn_index=0)

            # Retrieve and check if expected keyword appears
            memories = agent.memory.long_term.retrieve(case["query"])
            retrieved_text = " ".join(m.content.lower() for m in memories)

            if case["expected_keyword"].lower() in retrieved_text:
                hits += 1

        return hits / len(test_cases) if test_cases else 0.0

    def eval_faithfulness(
        self,
        episode: Episode,
        source_turns: list[Message]
    ) -> float:
        """
        Score how faithful an episodic summary is to its source turns.
        Uses LLM-as-judge pattern.
        """
        source_text = "\n".join(
            f"{t.role}: {t.content}" for t in source_turns
        )
        prompt = self.FAITHFULNESS_PROMPT.format(
            source_turns=source_text,
            summary=episode.summary
        )
        return self._llm_score(prompt)

    def eval_memory_lift(
        self,
        query: str,
        known_context: str,
        agent_with_memory: MemoryAgent,
        agent_without_memory: MemoryAgent,
    ) -> dict:
        """
        Compare response quality with vs without memory.
        Returns scores for both and the delta.
        """
        result_with = agent_with_memory.chat(query)
        result_without = agent_without_memory.chat(query)

        score_with = self._llm_score(self.QUALITY_PROMPT.format(
            query=query,
            response=result_with["response"],
            context=known_context,
        ))
        score_without = self._llm_score(self.QUALITY_PROMPT.format(
            query=query,
            response=result_without["response"],
            context=known_context,
        ))

        return {
            "query": query,
            "with_memory": score_with,
            "without_memory": score_without,
            "lift": round(score_with - score_without, 3),
            "response_with": result_with["response"],
            "response_without": result_without["response"],
        }

    def run_full_eval(self) -> EvalResult:
        """
        Run all three evaluations with built-in test cases.
        Returns an EvalResult with summary statistics.
        """
        result = EvalResult()

        # 1. Memory hit rate test cases
        hit_cases = [
            {
                "stored_fact": "I prefer Python over Java for backend work",
                "query": "what programming language does the user prefer?",
                "expected_keyword": "python"
            },
            {
                "stored_fact": "I am building a RAG system with ChromaDB",
                "query": "what database is the user using?",
                "expected_keyword": "chromadb"
            },
            {
                "stored_fact": "My name is Sakshi and I work at ZS Associates",
                "query": "what is the user's name?",
                "expected_keyword": "sakshi"
            },
        ]
        result.memory_hit_rate = self.eval_memory_hit_rate(hit_cases)
        print(f"Hit rate: {result.memory_hit_rate:.1%}")

        # 2. Faithfulness — run a conversation to generate an episode
        print("Generating episode for faithfulness eval...")
        from memoryos.config import MemoryConfig
        import dataclasses
        small_config = dataclasses.replace(
            self.config,
            memory=dataclasses.replace(
                self.config.memory,
                summarisation_threshold=3,
                turns_to_summarise=3,
            )
        )
        agent = MemoryAgent(small_config, session_id="eval-faith-001")
        agent.chat("I prefer concise explanations")
        agent.chat("I am building a RAG system in Python")
        agent.chat("I use ChromaDB for local vector storage")

        episodes = agent.memory.episodic.get_all_episodes()
        if episodes:
            source_turns = agent.memory.short_term.get_all()
            score = self.eval_faithfulness(episodes[0], source_turns)
            result.faithfulness_scores.append(score)
            print(f"Faithfulness: {score:.3f}")

        # 3. Memory lift
        print("Running memory lift eval...")
        agent_mem = MemoryAgent(self.config, session_id="eval-lift-mem")
        agent_mem.chat("I prefer Python and I am building a RAG system")
        agent_mem.chat("I like concise technical answers")

        agent_no_mem = MemoryAgent(self.config, session_id="eval-lift-nomem")

        lift_result = self.eval_memory_lift(
            query="What language should I use for my RAG system?",
            known_context="User prefers Python, is building a RAG system, likes concise answers",
            agent_with_memory=agent_mem,
            agent_without_memory=agent_no_mem,
        )
        result.memory_lift_scores.append(lift_result)
        print(f"Memory lift: {lift_result['lift']:+.3f}")

        return result