"""Method Compare Agent - 方法对比与选型"""
from app.agent.agents.base import PromptedAgent
from app.prompt.templates import METHOD_COMPARE_SYSTEM_PROMPT


class MethodCompareAgent(PromptedAgent):
    system_prompt = METHOD_COMPARE_SYSTEM_PROMPT
    retry_hint = (
        "上一次回答缺少对比结构。请用 Markdown 表格给出对比总览（核心思路 / "
        "适用场景 / 数据与算力要求 / 主要局限），逐项标注依据，"
        "并给出有条件的选型建议而非无条件结论。"
    )
