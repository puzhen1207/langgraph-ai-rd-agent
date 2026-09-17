"""Academic Writing Agent - 学术写作与润色"""
from app.agent.agents.base import PromptedAgent
from app.prompt.templates import ACADEMIC_WRITING_SYSTEM_PROMPT


class AcademicWritingAgent(PromptedAgent):
    system_prompt = ACADEMIC_WRITING_SYSTEM_PROMPT
    retry_hint = (
        "上一次回答没有区分原文与修改稿。请先给出完整的修改稿，"
        "再用简短条目说明主要修改点与理由，并确保没有增删原文中的事实、"
        "数据或限定条件。"
    )
