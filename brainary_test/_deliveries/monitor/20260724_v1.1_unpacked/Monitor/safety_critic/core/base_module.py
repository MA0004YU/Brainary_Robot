from abc import ABC, abstractmethod

from .context import SafetyContext


class BaseModule(ABC):
    """WP4 Safety Brain 所有模块的抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def process(self, ctx: SafetyContext) -> SafetyContext:
        ...

    def should_skip(self, ctx: SafetyContext) -> bool:
        return False
