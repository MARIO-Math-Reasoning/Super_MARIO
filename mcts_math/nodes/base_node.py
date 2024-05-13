"""
author: lovecambi
email: interfk@gmail.com
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Type

from pydantic import BaseModel, PrivateAttr, field_validator


class BaseNode(BaseModel):

    state: Dict[str, str] = {"text": "", "extra_info": ""}
    additional_state_keys: List[str] = []
    parent: Optional[Any] = None
    children: List[Any] = []
    depth: int = 0
    is_terminal: bool = False
    reward: Optional[float] = None
    value: Optional[float] = -100

    tag: str = "0"
    consecutive_errors: int = 0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        for key in self.additional_state_keys:
            self.state[key] = ""

    def has_children(self) -> bool:
        return self.children != []

    def is_root(self) -> bool:
        return self.parent is None
