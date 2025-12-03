"""Prompt template loading and rendering utilities."""

import re
from pathlib import Path
from typing import Any

import chevron
import yaml


class PromptTemplate:
    """A prompt template with YAML frontmatter and Handlebars body."""

    def __init__(
        self,
        name: str,
        metadata: dict[str, Any],
        template: str,
    ):
        self.name = name
        self.metadata = metadata
        self.template = template

    @property
    def description(self) -> str:
        return self.metadata.get("description", "")

    @property
    def model(self) -> str | None:
        return self.metadata.get("model")

    @property
    def reasoning_effort(self) -> str | None:
        return self.metadata.get("reasoning_effort")

    @property
    def output_schema(self) -> dict | None:
        return self.metadata.get("output_schema")

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables."""
        return chevron.render(self.template, kwargs)


class PromptLoader:
    """Loads and caches prompt templates from a directory."""

    def __init__(self, prompts_dir: str | Path):
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, PromptTemplate] = {}

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)

        if match:
            frontmatter_str, body = match.groups()
            try:
                frontmatter = yaml.safe_load(frontmatter_str) or {}
            except yaml.YAMLError:
                frontmatter = {}
            return frontmatter, body.strip()

        return {}, content.strip()

    def load(self, name: str) -> PromptTemplate:
        """Load a prompt template by name (without .md extension)."""
        if name in self._cache:
            return self._cache[name]

        file_path = self.prompts_dir / f"{name}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        metadata, template = self._parse_frontmatter(content)

        prompt = PromptTemplate(name=name, metadata=metadata, template=template)
        self._cache[name] = prompt
        return prompt

    def render(self, name: str, **kwargs: Any) -> str:
        """Load and render a prompt template in one call."""
        return self.load(name).render(**kwargs)

    def list_prompts(self) -> list[str]:
        """List all available prompt template names."""
        if not self.prompts_dir.exists():
            return []
        return [p.stem for p in self.prompts_dir.glob("*.md")]


def get_prompt_loader(prompts_dir: str | Path = "prompts") -> PromptLoader:
    """Get a prompt loader instance."""
    return PromptLoader(prompts_dir)

