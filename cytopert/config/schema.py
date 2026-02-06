"""Configuration schema for CytoPert."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None


class ProvidersConfig(BaseModel):
    """Configuration for LLM providers."""

    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)


class AgentDefaults(BaseModel):
    """Default agent configuration."""

    workspace: str = "~/.cytopert/workspace"
    model: str = "anthropic/claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.3
    max_tool_iterations: int = 20


class AgentsConfig(BaseModel):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class DataConfig(BaseModel):
    """Data and census configuration."""

    census_version: str | None = None
    local_h5ad_path: str | None = None


class ScenarioConfig(BaseModel):
    """Per-scenario workflow configuration (e.g. nfatc1_mammary)."""

    tissue_filter: list[str] = Field(default_factory=list)
    perturbation_genes: list[str] = Field(default_factory=list)
    state_groups: list[str] = Field(default_factory=list)


class Config(BaseSettings):
    """Root configuration for CytoPert."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    workflow: dict[str, dict] = Field(default_factory=dict)  # scenario name -> config

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser().resolve()

    def get_api_key(self) -> str | None:
        """Get API key in priority order."""
        return (
            self.providers.openrouter.api_key
            or self.providers.deepseek.api_key
            or self.providers.anthropic.api_key
            or self.providers.openai.api_key
            or self.providers.vllm.api_key
            or None
        )

    def get_api_base(self) -> str | None:
        """Get API base URL if using OpenRouter or vLLM."""
        if self.providers.openrouter.api_key:
            return self.providers.openrouter.api_base or "https://openrouter.ai/api/v1"
        if self.providers.openai.api_base:
            return self.providers.openai.api_base
        if self.providers.vllm.api_base:
            return self.providers.vllm.api_base
        return None

    model_config = {"env_prefix": "CYTOPERT_", "env_nested_delimiter": "__"}
