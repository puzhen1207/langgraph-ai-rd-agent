"""Literature Review Agent - 研究现状与综述"""
from app.agent.agents.base import PromptedAgent
from app.prompt.templates import LITERATURE_REVIEW_SYSTEM_PROMPT


class LiteratureReviewAgent(PromptedAgent):
    system_prompt = LITERATURE_REVIEW_SYSTEM_PROMPT
    retry_hint = (
        "上一次回答缺少综述结构。请按“整体脉络 / 主要路线 / 共识与分歧 / "
        "尚未解决的问题 / 知识库覆盖情况”重新组织，每条路线标注来源文件名，"
        "并在来源结论冲突时同时呈现。"
    )
