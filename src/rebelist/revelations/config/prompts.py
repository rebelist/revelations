from textwrap import dedent

from rebelist.revelations.domain import ChatAdapterPort
from rebelist.revelations.domain.models import PromptConfig
from rebelist.revelations.domain.services import AnswerEvaluatorPort

##### Chat Prompts #####

chat_system_template = dedent("""
You are a helpful expert on company documentation. Provide clear, direct answers.

Rules:
1. Answer directly - focus on the current question
2. For docs: Base answers ONLY on provided Context. Use Markdown. Cite sources.
3. For conversation: Use history naturally. Greetings/small talk → answer directly.
4. If unknown: Say "I don't have that information in the available documentation."
5. Be concise - skip filler phrases. Start with the answer immediately.
6. Always cite sources at the end:
   References:
   - URL (plain text, not markdown link)
""").strip()

chat_user_template = dedent(f"""
--- Context in Markdown Format ---
{{{ChatAdapterPort.HUMAN_TEMPLATE_CONTEXT_KEY}}}
--- Question ---
{{{ChatAdapterPort.HUMAN_TEMPLATE_INPUT_KEY}}}
""").strip()

chat_prompt_config = PromptConfig(system_template=chat_system_template, human_template=chat_user_template)


##### Benchmark Prompts #####

benchmark_system_template = """
You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to
the reference answer. Only give 5/5 scores for perfect answers.
"""

benchmark_user_template = dedent(f"""
Question:
{{{AnswerEvaluatorPort.HUMAN_TEMPLATE_QUESTION_KEY}}}

Generated Answer:
{{{AnswerEvaluatorPort.HUMAN_TEMPLATE_ANSWER_KEY}}}

Reference Answer:
{{{AnswerEvaluatorPort.HUMAN_TEMPLATE_REFERENCE_KEY}}}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question,
   covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?
4. Reasoning: Concise feedback on the answer quality,

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension.
If the answer is wrong, then the accuracy score must be 1.
""").strip()

benchmark_prompt_config = PromptConfig(
    system_template=benchmark_system_template, human_template=benchmark_user_template
)

__all__ = ['chat_prompt_config', 'benchmark_prompt_config']
