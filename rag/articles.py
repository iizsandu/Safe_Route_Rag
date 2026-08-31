"""Article bodies live in DynamoDB, not Qdrant (F-079).

Qdrant's job is finding the nearest vectors; it never needs to read
body_text to do that. Storing it as payload anyway bloated a 551 MB
vector index into a measured 2.3 GB -- 4x past Qdrant Cloud's free tier.
This module is the fix: article_id -> text, kept separately, matching
D-031's "article store, a separate service" from the start.
"""

from __future__ import annotations

import os

import boto3

TABLE_NAME = os.environ.get("ARTICLES_TABLE", "safe-route-articles")

# DynamoDB's hard limit per BatchGetItem call. Asking for more must be
# chunked or refused -- silently dropping articles past this line is
# exactly the quiet failure this project's guardrails work was about
# avoiding.
BATCH_GET_LIMIT = 100


def connect():
    """The DynamoDB resource, reused across requests -- loaded once at
    startup, same pattern as rag/app.py's Qdrant client.

    Region and credentials come from the environment (`aws configure`'s
    saved config, or AWS_DEFAULT_REGION) -- never hard-coded here.
    """
    return boto3.resource("dynamodb")


def put_article(resource, article_id: int, body_text: str) -> None:
    """Write one article. Used while loading, never at query time."""
    resource.Table(TABLE_NAME).put_item(
        Item={"article_id": article_id, "body_text": body_text})


def get_bodies(resource, article_ids: list[int]) -> dict[int, str]:
    """Fetch body_text for many articles in as few round trips as possible.

    Returns only the ids actually found. A missing id is simply absent
    from the result -- not an empty string standing in for real text.
    Callers must check for it, the same way search() already returns
    only what it actually found.
    """
    if len(article_ids) > BATCH_GET_LIMIT:
        raise ValueError(
            f"{len(article_ids)} ids requested, DynamoDB's BatchGetItem "
            f"caps out at {BATCH_GET_LIMIT}. Chunk the call rather than "
            f"silently dropping ids past the limit.")
    if not article_ids:
        return {}

    response = resource.batch_get_item(
        RequestItems={
            TABLE_NAME: {"Keys": [{"article_id": aid} for aid in article_ids]}
        }
    )
    items = response["Responses"][TABLE_NAME]
    return {int(item["article_id"]): item["body_text"] for item in items}
