"""Dummy MCP client with canned movie passage data for testing.

Implements both ``list_tools()`` and ``call_tool()`` with realistic
response shapes.  Used via the ``--dummy`` CLI flag.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["DummyMCPClient"]

log = logging.getLogger(__name__)

# Canned tool schemas (OpenAI format)

DUMMY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_passages",
            "description": "Search for passages matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_passage",
            "description": "Retrieve a single passage by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passage_id": {"type": "string", "description": "Passage ID"},
                },
                "required": ["passage_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve a full document by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID"},
                },
                "required": ["doc_id"],
            },
        },
    },
]

# Canned passage data (fake movie articles)

DUMMY_PASSAGES: list[dict[str, Any]] = [
    {
        "passage_id": "dummy_p1",
        "doc_id": "Q123456",
        "title": "Jaws (film)",
        "section": "Production",
        "text": (
            "Jaws is a 1975 American thriller film directed by Steven Spielberg. "
            "Based on Peter Benchley's 1974 novel of the same name, the film "
            "stars Roy Scheider, Robert Shaw, and Richard Dreyfuss."
        ),
        "score": 0.95,
    },
    {
        "passage_id": "dummy_p2",
        "doc_id": "Q234567",
        "title": "Christopher Nolan filmography",
        "section": "Feature films",
        "text": (
            "Christopher Nolan directed Interstellar in 2014, a science fiction "
            "film starring Matthew McConaughey. The film grossed over $677 million "
            "worldwide and received positive reviews for its visual effects."
        ),
        "score": 0.92,
    },
    {
        "passage_id": "dummy_p3",
        "doc_id": "Q234567",
        "title": "Interstellar (film)",
        "section": "Overview",
        "text": (
            "Interstellar is a 2014 epic science fiction film co-written, "
            "directed, and produced by Christopher Nolan. It features a crew of "
            "astronauts traveling through a wormhole near Saturn."
        ),
        "score": 0.88,
    },
    {
        "passage_id": "dummy_p4",
        "doc_id": "Q345678",
        "title": "Titanic (1997 film)",
        "section": "Runtime",
        "text": (
            "Titanic is a 1997 American epic romance and disaster film. "
            "It has a runtime of 195 minutes (3 hours 15 minutes), making it "
            "one of the longest mainstream films at the time of its release."
        ),
        "score": 0.90,
    },
    {
        "passage_id": "dummy_p5",
        "doc_id": "Q456789",
        "title": "Avatar (2009 film)",
        "section": "Runtime",
        "text": (
            "Avatar is a 2009 American epic science fiction film. The original "
            "theatrical release has a runtime of 162 minutes. An extended "
            "collector's edition runs 171 minutes."
        ),
        "score": 0.87,
    },
    {
        "passage_id": "dummy_p6",
        "doc_id": "Q567890",
        "title": "Meryl Streep",
        "section": "Awards",
        "text": (
            "Meryl Streep has won three Academy Awards: Best Supporting Actress "
            "for Kramer vs. Kramer (1979), Best Actress for Sophie's Choice "
            "(1982), and Best Actress for The Iron Lady (2011)."
        ),
        "score": 0.93,
    },
    {
        "passage_id": "dummy_p7",
        "doc_id": "Q567890",
        "title": "Meryl Streep",
        "section": "Career",
        "text": (
            "Streep has received a record 21 Academy Award nominations. "
            "She is widely regarded as one of the greatest actresses of her "
            "generation and has appeared in over 70 films."
        ),
        "score": 0.85,
    },
    {
        "passage_id": "dummy_p8",
        "doc_id": "Q678901",
        "title": "The Manchurian Candidate (2004 film)",
        "section": "Cast",
        "text": (
            "The 2004 remake of The Manchurian Candidate stars Denzel Washington, "
            "Meryl Streep, and Liev Schreiber. Anthony Mackie appeared in a "
            "supporting role as Corporal Robert Baker."
        ),
        "score": 0.91,
    },
    {
        "passage_id": "dummy_p9",
        "doc_id": "Q789012",
        "title": "Captain America (Marvel Cinematic Universe)",
        "section": "Characters",
        "text": (
            "Anthony Mackie portrayed Sam Wilson / Falcon in the Marvel Cinematic "
            "Universe, later taking on the mantle of Captain America. He first "
            "appeared in Captain America: The Winter Soldier (2014)."
        ),
        "score": 0.89,
    },
    {
        "passage_id": "dummy_p10",
        "doc_id": "Q890123",
        "title": "Thurgood Marshall",
        "section": "Supreme Court",
        "text": (
            "Thurgood Marshall was appointed to the Supreme Court of the United "
            "States on October 2, 1967 by President Lyndon B. Johnson. He served "
            "until his retirement on October 1, 1991, a tenure spanning nearly "
            "24 years."
        ),
        "score": 0.94,
    },
]

# Index by passage_id for quick lookup
_PASSAGE_BY_ID: dict[str, dict[str, Any]] = {p["passage_id"]: p for p in DUMMY_PASSAGES}

# Index by doc_id for document-level lookup
_DOC_PASSAGES: dict[str, list[dict[str, Any]]] = {}
for _p in DUMMY_PASSAGES:
    _DOC_PASSAGES.setdefault(_p["doc_id"], []).append(_p)


class DummyMCPClient:
    """MCP client returning canned passage data for testing.

    Implements both ``list_tools()`` and ``call_tool()`` with
    realistic return shapes matching the real MCP server.
    """

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return dummy tool schemas in OpenAI format."""
        return list(DUMMY_TOOLS)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Return canned results for the given tool.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            timeout: Ignored.

        Returns:
            Tool result matching the real MCP server's response shape.
        """
        if name == "search_passages":
            query = arguments.get("query", "").lower()
            top_k = arguments.get("top_k", 5)
            # Simple keyword matching for dummy search
            matches = [
                p
                for p in DUMMY_PASSAGES
                if any(
                    word in p["text"].lower() or word in p["title"].lower()
                    for word in query.split()
                    if len(word) > 2
                )
            ]
            if not matches:
                matches = DUMMY_PASSAGES[:3]
            return {
                "passages": matches[:top_k],
                "total": len(matches),
                "query": arguments.get("query", ""),
            }

        if name == "get_passage":
            pid = arguments.get("passage_id", "")
            passage = _PASSAGE_BY_ID.get(pid)
            if passage:
                return dict(passage)
            return {"error": f"Passage not found: {pid}"}

        if name == "get_document":
            doc_id = arguments.get("doc_id", "")
            passages = _DOC_PASSAGES.get(doc_id)
            if passages:
                return {
                    "doc_id": doc_id,
                    "title": passages[0]["title"],
                    "passages": passages,
                }
            return {"error": f"Document not found: {doc_id}"}

        return {"error": f"Unknown tool: {name}"}
