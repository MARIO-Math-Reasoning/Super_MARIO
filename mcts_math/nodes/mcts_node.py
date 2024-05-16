"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Type

from pydantic import BaseModel, PrivateAttr, field_validator

from .base_node import BaseNode


class MCTSNode(BaseNode):

    prior: float = 1.0
    c_puct: float = 1.5

    __visit_count: int = PrivateAttr(default=0)
    __value_sum: float = PrivateAttr(default=0)

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0
        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update_visit_count(self, count: int) -> None:
        self.__visit_count = count

    def update(self, value: float) -> None:
        # init value
        if self.value == -100:
            self.value = value
        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value: float, start_node: Type[BaseNode]) -> None:
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)

    def puct(self) -> float:
        q_value = self.q_value() if self.visit_count() > 0 else 0
        u_value = self.c_puct * self.prior * np.sqrt(self.parent.visit_count()) / (1 + self.visit_count())
        return q_value + u_value
        
