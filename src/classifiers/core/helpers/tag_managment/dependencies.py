"""Helpers for tag dependecy management."""
from typing import Set


def _bfs(key: str, edges: dict) -> Set[str]:
    """breadth first search through a dict of edges"""
    if key not in edges:
        return set()

    collected = set()
    queue = [key]

    while queue:
        k = queue.pop(0)
        collected.add(k)
        queue.extend(edges.get(k, set()) - collected)

    collected.remove(key)
    return collected


class DependencyGraph:
    """Helper class to hold and manage tag dependencies."""

    def __init__(self):
        """New DependencyGraph."""
        self.dependencies = {}
        self.dependents = {}

    def add_dependency(self, tag: str, depends_on: str):
        """Adds new dependency.

        Internally, this is stored both as "A has dependency B" and "B has dependent A".

        Args:
            tag: dependent
            depends_on: dependency
        """
        self.dependencies.setdefault(tag, set()).add(depends_on)
        self.dependents.setdefault(depends_on, set()).add(tag)

    def remove_dependency(self, tag: str, depends_on: str):
        """Removes a previously added dependency.

        Args:
            tag: dependent
            depends_on: dependency
        """
        self.dependencies[tag].remove(depends_on)
        if not self.dependencies[tag]:
            self.dependencies.pop(tag)
        self.dependents[depends_on].remove(tag)
        if not self.dependents[depends_on]:
            self.dependents.pop(depends_on)

    def get_dependents(self, tag: str) -> Set[str]:
        """Get all dependents (and dependents of dependents) of `tag`.

        Args:
            tag: starting tag
        """
        return _bfs(tag, self.dependents)

    def get_dependencies(self, tag: str) -> Set[str]:
        """Get all dependencies (and dependencies of dependencies) of `tag`.

        Args:
            tag: starting tag
        """
        return _bfs(tag, self.dependencies)
