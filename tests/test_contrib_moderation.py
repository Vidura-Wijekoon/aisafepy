from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aisafepy.contrib.azure_content_safety import AzureContentSafetyGuard
from aisafepy.contrib.openai_moderation import OpenAIModerationGuard
from aisafepy.stream.pipeline import Context

# ---- fakes ---------------------------------------------------------------------


class FakeModerations:
    def __init__(self, result: dict, *, is_async: bool = False, raise_exc: Exception | None = None):
        self.result = result
        self.is_async = is_async
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc
        payload = {"results": [self.result]}
        if self.is_async:

            async def _coro():
                return payload

            return _coro()
        # Mimic the SDK object shape with model_dump().
        return SimpleNamespace(model_dump=lambda: payload)


def openai_client(result: dict, **kw) -> SimpleNamespace:
    return SimpleNamespace(moderations=FakeModerations(result, **kw))


class FakeAzure:
    def __init__(self, analysis, *, is_async: bool = False, raise_exc: Exception | None = None):
        self.analysis = analysis
        self.is_async = is_async
        self.raise_exc = raise_exc
        self.calls = []

    def analyze_text(self, options):
        self.calls.append(options)
        if self.raise_exc:
            raise self.raise_exc
        resp = SimpleNamespace(categories_analysis=self.analysis)
        if self.is_async:

            async def _coro():
                return resp

            return _coro()
        return resp


# ---- OpenAI --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_moderation_blocks_on_api_flag():
    client = openai_client(
        {
            "flagged": True,
            "categories": {"violence": True, "hate": False},
            "category_scores": {"violence": 0.91, "hate": 0.02},
        }
    )
    guard = OpenAIModerationGuard(client=client)
    d = await guard(Context(buffer="some text", prompt="p"))
    assert d.is_blocked and d.tier == 2
    assert d.evidence["categories"] == ["violence"]
    assert d.confidence == pytest.approx(0.91)
    assert d.severity == "high"
    assert client.moderations.calls[0] == {"model": "omni-moderation-latest", "input": "some text"}


@pytest.mark.asyncio
async def test_openai_moderation_threshold_blocks_when_api_does_not_flag():
    result = {
        "flagged": False,
        "categories": {"self-harm": False},
        "category_scores": {"self-harm": 0.35},
    }
    guard = OpenAIModerationGuard(client=openai_client(result), thresholds={"self-harm": 0.3})
    d = await guard(Context(buffer="text"))
    assert d.is_blocked and d.evidence["threshold_hits"] == ["self-harm"]
    assert d.severity == "medium"

    guard = OpenAIModerationGuard(client=openai_client(result))
    d = await guard(Context(buffer="text"))
    assert d.action.value == "allow"
    assert d.confidence == pytest.approx(0.65)
    assert d.evidence["scores"] == {"self-harm": 0.35}


@pytest.mark.asyncio
async def test_openai_moderation_async_client_and_include_prompt():
    client = openai_client({"flagged": False, "categories": {}, "category_scores": {}}, is_async=True)
    guard = OpenAIModerationGuard(client=client, include_prompt=True, model="text-moderation-latest")
    d = await guard(Context(buffer="answer", prompt="question"))
    assert d.action.value == "allow"
    assert client.moderations.calls[0]["input"] == "question\nanswer"
    assert client.moderations.calls[0]["model"] == "text-moderation-latest"


@pytest.mark.asyncio
async def test_openai_moderation_fails_open_on_error_timeout_and_bad_client():
    guard = OpenAIModerationGuard(client=openai_client({}, raise_exc=ConnectionError("down")))
    d = await guard(Context(buffer="x"))
    assert d.action.value == "allow" and "down" in d.evidence["error"]

    class Slow:
        class moderations:  # mimics the SDK attribute name
            @staticmethod
            async def create(**kw):
                await asyncio.sleep(0.5)
                return {"results": []}

    d = await OpenAIModerationGuard(client=Slow(), timeout_s=0.01)(Context(buffer="x"))
    assert d.action.value == "allow" and "error" in d.evidence

    d = await OpenAIModerationGuard(client=object())(Context(buffer="x"))
    assert d.action.value == "allow" and "moderations.create" in d.evidence["error"]

    d = await OpenAIModerationGuard(client=object())(Context(buffer="  "))
    assert d.action.value == "allow" and d.rationale == "empty"


# ---- Azure ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_azure_blocks_at_threshold_and_reports_severities():
    analysis = [
        {"category": "Hate", "severity": 2},
        SimpleNamespace(category="Violence", severity=6),
        {"category": "Sexual", "severity": None},
    ]
    client = FakeAzure(analysis)
    guard = AzureContentSafetyGuard(client=client)
    d = await guard(Context(buffer="text"))
    assert d.is_blocked
    assert d.evidence["hits"] == {"Violence": 6}
    assert d.evidence["severities"] == {"Hate": 2, "Violence": 6}
    assert d.severity == "critical"
    assert d.confidence == pytest.approx(6 / 7)
    # Fallback options shape when the Azure SDK is absent.
    assert client.calls[0] == {"text": "text"}


@pytest.mark.asyncio
async def test_azure_custom_thresholds_allow_and_async_client():
    client = FakeAzure([{"category": "Hate", "severity": 3}], is_async=True)
    guard = AzureContentSafetyGuard(client=client, thresholds={"Hate": 4})
    d = await guard(Context(buffer="text"))
    assert d.action.value == "allow"
    assert d.confidence == pytest.approx(1 - 3 / 7)

    guard = AzureContentSafetyGuard(client=client, thresholds={"Hate": 2})
    d = await guard(Context(buffer="text"))
    assert d.is_blocked and d.severity == "high"


@pytest.mark.asyncio
async def test_azure_truncates_from_front_and_fails_open():
    client = FakeAzure([])
    guard = AzureContentSafetyGuard(client=client, max_chars=5)
    d = await guard(Context(buffer="0123456789"))
    assert d.action.value == "allow"
    assert client.calls[0]["text"] == "56789"

    d = await AzureContentSafetyGuard(client=FakeAzure([], raise_exc=RuntimeError("x")))(
        Context(buffer="t")
    )
    assert d.action.value == "allow" and "error" in d.evidence
    d = await AzureContentSafetyGuard(client=object())(Context(buffer="t"))
    assert d.action.value == "allow" and "analyze_text" in d.evidence["error"]
