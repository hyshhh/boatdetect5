"""Agent 模块 — 仅保留 AgentResult 数据类（Agent 路线已移除）"""

from __future__ import annotations


class AgentResult:
    """结构化识别结果，供 pipeline 使用。"""

    def __init__(
        self,
        hull_number: str = "",
        description: str = "",
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        answer: str = "",
    ):
        self.hull_number = hull_number
        self.description = description
        self.match_type = match_type        # "exact" | "semantic" | "none"
        self.semantic_match_ids = semantic_match_ids or []
        self.answer = answer
