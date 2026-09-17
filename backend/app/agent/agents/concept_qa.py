"""Concept QA Agent - 概念与原理问答"""
from app.agent.agents.base import PromptedAgent
from app.prompt.templates import CONCEPT_QA_SYSTEM_PROMPT


class ConceptQAAgent(PromptedAgent):
    system_prompt = CONCEPT_QA_SYSTEM_PROMPT
    retry_hint = (
        "上一次回答过于简略或缺少依据。请补充关键术语的中英文对照，"
        "解释清楚“为什么”和“如何起作用”，并为每条结论标注来源文件名。"
    )
