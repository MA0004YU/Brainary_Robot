from typing import Any, Dict, List, Optional


class CapabilityRegistry:
    def __init__(self):
        self._skills: List[Dict[str, Any]] = []
        self._mcps: List[Dict[str, Any]] = []
        self._tools: List[Dict[str, Any]] = []

    # -------- skills --------
    def register_skill(
        self,
        name: str,
        description: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._skills.append(
            {
                "name": name,
                "description": description,
                "inputs": inputs or [],
                "outputs": outputs or [],
                "enabled": enabled,
                "metadata": metadata or {},
            }
        )

    def list_skills(self) -> List[Dict[str, Any]]:
        return self._skills

    # -------- mcp services --------
    def register_mcp(
        self,
        name: str,
        description: str,
        endpoint: str = "",
        capabilities: Optional[List[str]] = None,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._mcps.append(
            {
                "name": name,
                "description": description,
                "endpoint": endpoint,
                "capabilities": capabilities or [],
                "enabled": enabled,
                "metadata": metadata or {},
            }
        )

    def list_mcps(self) -> List[Dict[str, Any]]:
        return self._mcps

    # -------- tools --------
    def register_tool(
        self,
        name: str,
        description: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tools.append(
            {
                "name": name,
                "description": description,
                "inputs": inputs or [],
                "outputs": outputs or [],
                "enabled": enabled,
                "metadata": metadata or {},
            }
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._tools

    # -------- unified snapshot --------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "skills": self.list_skills(),
            "mcps": self.list_mcps(),
            "tools": self.list_tools(),
        }