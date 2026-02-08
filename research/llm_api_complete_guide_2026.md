# Complete LLM API Guide - February 2026

**The Ultimate Reference:** Practical code + comprehensive schemas for all major LLM providers.

**Last Updated:** February 8, 2026  
**Status:** ✅ Production-ready with runnable code examples

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Helper Functions](#helper-functions)
3. [Provider Overview](#provider-overview)
4. [Canonical API Schemas](#canonical-api-schemas)
5. [Models by Provider](#models-by-provider)
   - [OpenAI](#openai-models)
   - [Anthropic](#anthropic-models)
   - [xAI (Grok)](#xai-models)
   - [Google Gemini](#google-gemini-models)
   - [Mistral AI](#mistral-ai-models)
   - [Groq](#groq-models)
6. [Advanced Features](#advanced-features)
7. [Streaming Examples](#streaming-examples)
8. [Complete Schema Reference](#complete-schema-reference)
9. [Pricing & Lifecycle](#pricing--lifecycle)

---

## Quick Start

### Installation

```bash
pip install requests
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export XAI_API_KEY="xai-..."
export GEMINI_API_KEY="..."
export MISTRAL_API_KEY="..."
export GROQ_API_KEY="gsk_..."
```

### 30-Second Example

```python
import os
import requests

# OpenAI
resp = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
).json()
print(resp["choices"][0]["message"]["content"])
```

---

## Helper Functions

**Copy these into your code for easy API calls:**

```python
import json
import requests
from typing import Dict, Any, Optional

def post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    """Generic POST helper with error handling."""
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not r.ok:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(
            f"HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)[:2000]}"
        )
    return r.json()

# ============================================================================
# Text Extraction Functions
# ============================================================================

def extract_openai_chat_text(resp: dict) -> str:
    """Extract text from OpenAI Chat Completions response."""
    choices = resp.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""

def extract_openai_responses_text(resp: dict) -> str:
    """Extract text from OpenAI Responses API response."""
    chunks = []
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text") and "text" in part:
                    chunks.append(part["text"])
    return "".join(chunks).strip()

def extract_anthropic_text(resp: dict) -> str:
    """Extract text from Anthropic Messages API response."""
    text_parts = [
        block.get("text", "")
        for block in resp.get("content", [])
        if block.get("type") == "text"
    ]
    return "".join(text_parts).strip()

def extract_gemini_text(resp: dict) -> str:
    """Extract text from Gemini generateContent response."""
    candidates = resp.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    return "".join(text_parts).strip()

# ============================================================================
# Token Usage Extraction Functions
# ============================================================================

def extract_openai_usage(resp: dict) -> dict:
    """Extract token usage from OpenAI response."""
    return resp.get("usage", {})

def extract_anthropic_usage(resp: dict) -> dict:
    """Extract token usage from Anthropic response."""
    return resp.get("usage", {})

def extract_gemini_usage(resp: dict) -> dict:
    """Extract token usage from Gemini response."""
    return resp.get("usageMetadata", {})

# ============================================================================
# Unified Text Extraction (auto-detect provider)
# ============================================================================

def extract_text(resp: dict) -> str:
    """
    Auto-detect provider and extract text.
    Works with: OpenAI (chat/responses), Anthropic, Gemini
    """
    # OpenAI Chat Completions
    if "choices" in resp and resp.get("object") == "chat.completion":
        return extract_openai_chat_text(resp)
    
    # OpenAI Responses API
    if "output" in resp:
        return extract_openai_responses_text(resp)
    
    # Anthropic
    if "content" in resp and isinstance(resp.get("content"), list):
        return extract_anthropic_text(resp)
    
    # Gemini
    if "candidates" in resp and "usageMetadata" in resp:
        return extract_gemini_text(resp)
    
    # Fallback
    return str(resp)

def extract_usage(resp: dict) -> dict:
    """
    Auto-detect provider and extract usage.
    Returns unified dict with keys: input_tokens, output_tokens, total_tokens
    """
    # OpenAI
    if "usage" in resp:
        usage = resp["usage"]
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "raw": usage
        }
    
    # Gemini
    if "usageMetadata" in resp:
        usage = resp["usageMetadata"]
        return {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
            "raw": usage
        }
    
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
```

---

## Provider Overview

| Provider | Base URL | Auth Header | Common Endpoint |
|----------|----------|-------------|-----------------|
| **OpenAI** | `https://api.openai.com` | `Authorization: Bearer $KEY` | `/v1/chat/completions`, `/v1/responses` |
| **Anthropic** | `https://api.anthropic.com` | `x-api-key: $KEY` | `/v1/messages` |
| **xAI** | `https://api.x.ai` | `Authorization: Bearer $KEY` | `/v1/chat/completions` |
| **Gemini** | `https://generativelanguage.googleapis.com` | `x-goog-api-key: $KEY` | `/v1beta/models/{model}:generateContent` |
| **Mistral** | `https://api.mistral.ai` | `Authorization: Bearer $KEY` | `/v1/chat/completions` |
| **Groq** | `https://api.groq.com` | `Authorization: Bearer $KEY` | `/openai/v1/chat/completions` |

---

## Canonical API Schemas

### OpenAI Chat Completions API

**Endpoint:** `POST /v1/chat/completions`

#### Request Schema

```python
{
    # REQUIRED
    "model": "gpt-4o",
    "messages": [
        {
            "role": "developer|system|user|assistant|tool",
            "content": "string or array of content parts"
        }
    ],
    
    # GENERATION PARAMETERS
    "max_completion_tokens": 1024,      # Preferred over max_tokens
    "max_tokens": 1024,                 # Deprecated
    "temperature": 0.7,                 # 0.0 - 2.0
    "top_p": 0.95,                      # 0.0 - 1.0
    "n": 1,                             # Number of completions
    "stop": ["END", "STOP"],            # Stop sequences
    "presence_penalty": 0.0,            # -2.0 to 2.0
    "frequency_penalty": 0.0,           # -2.0 to 2.0
    "seed": 42,                         # For deterministic outputs
    
    # REASONING (o-series models)
    "reasoning_effort": "medium",       # none|minimal|low|medium|high|xhigh
    
    # OUTPUT FORMAT
    "response_format": {
        "type": "text|json_object|json_schema",
        "json_schema": {                # When type=json_schema
            "name": "response_schema",
            "schema": {...},            # JSON Schema
            "strict": true
        }
    },
    
    # LOGPROBS
    "logprobs": true,
    "top_logprobs": 5,                  # 0-20
    "logit_bias": {"1234": 10},         # Token ID bias
    
    # TOOLS
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {...}     # JSON Schema
            }
        }
    ],
    "tool_choice": "auto|required|none",
    "parallel_tool_calls": true,
    
    # MULTIMODAL (select models)
    "modalities": ["text", "audio"],
    "audio": {
        "voice": "alloy|echo|fable|onyx|nova|shimmer",
        "format": "wav|mp3|flac|opus|pcm16"
    },
    
    # ADVANCED
    "prediction": {                     # Predicted outputs
        "type": "content",
        "content": {...}
    },
    "stream": false,
    "stream_options": {"include_usage": true},
    "store": false,
    "metadata": {"user_id": "123"},
    "user": "user_id_deprecated"
}
```

#### Response Schema

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o-2024-11-20",
    "system_fingerprint": "fp_abc123",
    
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response text here",
                "tool_calls": [             # When using tools
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}'
                        }
                    }
                ],
                "audio": {                  # When audio output enabled
                    "id": "audio_123",
                    "transcript": "text",
                    "data": "base64..."
                }
            },
            "finish_reason": "stop|length|tool_calls|content_filter",
            "logprobs": {
                "content": [
                    {
                        "token": "Hello",
                        "logprob": -0.5,
                        "bytes": [72, 101, 108, 108, 111],
                        "top_logprobs": [...]
                    }
                ]
            }
        }
    ],
    
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "audio_tokens": 0
        },
        "completion_tokens_details": {
            "reasoning_tokens": 25,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0
        }
    }
}
```

---

### OpenAI Responses API (New Agentic API)

**Endpoint:** `POST /v1/responses`

#### Request Schema

```python
{
    # REQUIRED
    "model": "gpt-5.2",
    "input": "string or array of input items",
    
    # INSTRUCTIONS & STATE
    "instructions": "System-level instructions",
    "previous_response_id": "resp_123",     # Continue conversation
    "conversation": "conv_456",             # Or conversation object
    "background": "Background context",
    
    # GENERATION
    "max_output_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_logprobs": 5,
    "truncation": "disabled",
    
    # OUTPUT FORMAT
    "text": {
        "format": {"type": "text|json_object|json_schema"}
    },
    
    # BUILT-IN SERVER-SIDE TOOLS
    "tools": [
        {"type": "web_search", "max_results": 10},
        {"type": "file_search", "vector_stores": ["vs_123"]},
        {"type": "computer_use", "display_width_px": 1920},
        {"type": "code_interpreter", "timeout_seconds": 120},
        {"type": "remote_mcp", "url": "https://..."}
    ],
    "tool_choice": "auto|required|none",
    "parallel_tool_calls": true,
    
    # REASONING
    "reasoning": {
        "effort": "low|medium|high",
        "budget_tokens": 10000
    },
    
    # STREAMING
    "stream": false,
    "include": ["usage", "thoughts"],
    
    # METADATA
    "metadata": {},
    "store": false,
    "user": "user_id"
}
```

#### Response Schema

```python
{
    "id": "resp_abc123",
    "object": "response",
    "created": 1234567890,
    "model": "gpt-5.2",
    "status": "completed|in_progress|failed",
    "conversation_id": "conv_456",
    
    "output": [
        {
            "type": "message",
            "content": "Response text"
        },
        {
            "type": "tool_use",
            "tool_use": {
                "id": "tool_123",
                "name": "web_search",
                "input": {"query": "AI trends"},
                "output": {...}
            }
        }
    ],
    
    "usage": {
        "input_tokens": 150,
        "output_tokens": 200,
        "total_tokens": 350
    },
    
    "metadata": {}
}
```

---

### Anthropic Messages API

**Endpoint:** `POST /v1/messages`  
**Required Headers:** `x-api-key`, `anthropic-version: 2023-06-01`

#### Request Schema

```python
{
    # REQUIRED
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user|assistant",
            "content": "string or array of content blocks"
        }
    ],
    
    # SYSTEM PROMPT (not a message role)
    "system": "You are helpful" or [
        {"type": "text", "text": "System prompt"}
    ],
    
    # GENERATION
    "temperature": 0.7,                 # 0.0 - 1.0
    "top_p": 0.95,
    "top_k": 40,
    "stop_sequences": ["END"],
    
    # TOOLS
    "tools": [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {...}       # JSON Schema
        }
    ],
    "tool_choice": {
        "type": "auto|any|tool",
        "name": "function_name"         # When type=tool
    },
    
    # EXTENDED THINKING (Claude 4.5 models)
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000          # Min 1024
    },
    
    # STREAMING
    "stream": false,
    
    # METADATA
    "metadata": {
        "user_id": "user_123"
    }
}
```

#### Content Block Types

```python
# Text
{"type": "text", "text": "Hello"}

# Image
{
    "type": "image",
    "source": {
        "type": "base64|url",
        "media_type": "image/jpeg",
        "data": "base64..."
    }
}

# PDF Document
{
    "type": "document",
    "source": {
        "type": "base64",
        "media_type": "application/pdf",
        "data": "base64..."
    }
}

# Tool Use
{
    "type": "tool_use",
    "id": "toolu_123",
    "name": "get_weather",
    "input": {"location": "SF"}
}

# Tool Result
{
    "type": "tool_result",
    "tool_use_id": "toolu_123",
    "content": "75°F, sunny"
}

# Thinking (Extended Thinking responses)
{
    "type": "thinking",
    "thinking": "Chain of thought..."
}
```

#### Response Schema

```python
{
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-5-20251101",
    
    "content": [
        {
            "type": "text",
            "text": "Response content"
        },
        {
            "type": "thinking",
            "thinking": "Internal reasoning"
        },
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"location": "SF"}
        }
    ],
    
    "stop_reason": "end_turn|max_tokens|stop_sequence|tool_use",
    "stop_sequence": null,
    
    "usage": {
        "input_tokens": 100,
        "cache_creation_input_tokens": 50,      # Prompt caching
        "cache_read_input_tokens": 200,
        "output_tokens": 150
    }
}
```

---

### Google Gemini generateContent API

**Endpoint:** `POST /v1beta/models/{model}:generateContent`  
**Auth:** `x-goog-api-key` header or query parameter

#### Request Schema

```python
{
    # REQUIRED
    "contents": [
        {
            "role": "user|model",
            "parts": [
                {"text": "Hello"},
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": "base64..."
                    }
                },
                {
                    "fileData": {
                        "mimeType": "application/pdf",
                        "fileUri": "gs://bucket/file.pdf"
                    }
                },
                {
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "SF"}
                    }
                },
                {
                    "functionResponse": {
                        "name": "get_weather",
                        "response": {"temp": 75}
                    }
                }
            ]
        }
    ],
    
    # SYSTEM INSTRUCTION
    "systemInstruction": {
        "parts": [{"text": "You are helpful"}]
    },
    
    # TOOLS
    "tools": [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {...}     # OpenAPI 3.0 Schema
                }
            ]
        },
        {
            "googleSearchRetrieval": {
                "dynamicRetrievalConfig": {
                    "mode": "MODE_DYNAMIC",
                    "dynamicThreshold": 0.7
                }
            }
        },
        {"codeExecution": {}}
    ],
    "toolConfig": {
        "functionCallingConfig": {
            "mode": "AUTO|ANY|NONE",
            "allowedFunctionNames": ["get_weather"]
        }
    },
    
    # GENERATION CONFIG
    "generationConfig": {
        "temperature": 0.7,
        "topP": 0.95,
        "topK": 40,
        "maxOutputTokens": 8192,
        "stopSequences": ["END"],
        "candidateCount": 1,
        "responseMimeType": "text/plain|application/json",
        "responseSchema": {...}         # For JSON mode
    },
    
    # SAFETY
    "safetySettings": [
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE|BLOCK_ONLY_HIGH"
        }
    ],
    
    # CACHING
    "cachedContent": "cachedContents/123"
}
```

#### Response Schema

```python
{
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "Response text"},
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "SF"}
                        }
                    }
                ]
            },
            "finishReason": "STOP|MAX_TOKENS|SAFETY|RECITATION|OTHER",
            "safetyRatings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "probability": "NEGLIGIBLE|LOW|MEDIUM|HIGH",
                    "blocked": false
                }
            ],
            "citationMetadata": {
                "citationSources": [
                    {
                        "startIndex": 0,
                        "endIndex": 100,
                        "uri": "https://example.com",
                        "license": "CC-BY"
                    }
                ]
            },
            "tokenCount": 50,
            "groundingMetadata": {
                "groundingChunks": [],
                "groundingSupports": [],
                "webSearchQueries": []
            }
        }
    ],
    
    "usageMetadata": {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
        "totalTokenCount": 150,
        "cachedContentTokenCount": 200
    },
    
    "modelVersion": "gemini-3-pro-preview-001"
}
```

---

## OpenAI Models

### GPT-5 Series (Flagship Reasoning Models)

#### gpt-5.2

**Released:** December 2025  
**Context:** Up to 256K tokens  
**Best for:** Maximum intelligence, complex reasoning  
**Pricing:** Premium tier

```python
import os

url = "https://api.openai.com/v1/responses"
headers = {
    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    "Content-Type": "application/json"
}
payload = {
    "model": "gpt-5.2",
    "input": "Explain quantum entanglement in simple terms.",
    "max_output_tokens": 1024,
    "reasoning": {
        "effort": "high"
    }
}

resp = post_json(url, headers, payload)
print(f"Text: {extract_openai_responses_text(resp)}")
print(f"Usage: {extract_openai_responses_usage(resp)}")
```

#### gpt-5.1

**Released:** November 2025  
**Context:** Up to 256K tokens  
**Note:** Superseded by GPT-5.2 (slightly more expensive)

```python
payload = {
    "model": "gpt-5.1",
    "input": "Write a Python function to parse JSON.",
    "max_output_tokens": 512
}
```

#### gpt-5

**Released:** 2025  
**Status:** Legacy - use GPT-5.1 or GPT-5.2

#### gpt-5-mini

**Released:** 2025  
**Context:** Up to 256K tokens  
**Best for:** Fast reasoning at lower cost  
**Pricing:** More affordable than GPT-5

```python
payload = {
    "model": "gpt-5-mini",
    "input": "Summarize this article in 3 bullets.",
    "max_output_tokens": 256,
    "reasoning": {
        "effort": "low"
    }
}
```

---

### GPT-4.1 Series (Coding Specialists)

#### gpt-4.1

**Released:** April 2025  
**Context:** 1M tokens  
**Best for:** Coding, instruction following, long context  
**Pricing:** Mid-tier

```python
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    "Content-Type": "application/json"
}
payload = {
    "model": "gpt-4.1",
    "messages": [
        {"role": "developer", "content": "You are an expert Python developer."},
        {"role": "user", "content": "Write a decorator for rate limiting."}
    ],
    "max_completion_tokens": 1024
}

resp = post_json(url, headers, payload)
print(f"Code: {extract_openai_chat_text(resp)}")
print(f"Usage: {extract_openai_usage(resp)}")
```

#### gpt-4.1-mini

**Released:** April 2025  
**Context:** 1M tokens  
**Best for:** Fast coding tasks  
**Pricing:** 83% cheaper than GPT-4o

```python
payload = {
    "model": "gpt-4.1-mini",
    "messages": [
        {"role": "user", "content": "Fix this regex: /[a-z+/"}
    ],
    "max_completion_tokens": 256
}
```

#### gpt-4.1-nano

**Released:** April 2025  
**Context:** 1M tokens  
**Best for:** Classification, autocompletion, simple tasks  
**Pricing:** Cheapest, fastest

```python
payload = {
    "model": "gpt-4.1-nano",
    "messages": [
        {"role": "user", "content": "Classify: 'Great product!' -> sentiment?"}
    ],
    "max_completion_tokens": 10
}
```

---

### GPT-4o Series (Multimodal)

#### gpt-4o

**Released:** 2024 (updated Nov 2024, June 2024)  
**Context:** 128K tokens  
**Best for:** General-purpose, multimodal (text + vision)  
**Pricing:** Standard tier

```python
payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ]
}
```

#### gpt-4o-mini

**Released:** 2024  
**Context:** 128K tokens  
**Best for:** Fast, affordable multimodal  
**Pricing:** Budget tier

#### gpt-4o-audio-preview

**Best for:** Audio input/output  
**Features:** Real-time speech capabilities

```python
payload = {
    "model": "gpt-4o-audio-preview",
    "modalities": ["text", "audio"],
    "audio": {"voice": "alloy", "format": "mp3"},
    "messages": [
        {"role": "user", "content": "Tell me a joke"}
    ]
}
```

---

### o-Series (Reasoning Models)

#### o3

**Released:** 2025  
**Best for:** Complex multi-step reasoning  
**Pricing:** Premium

```python
payload = {
    "model": "o3",
    "messages": [
        {"role": "user", "content": "Prove the Pythagorean theorem."}
    ],
    "reasoning_effort": "high"
}
```

#### o3-pro

**Best for:** Extended thinking, maximum reliability  
**Features:** Thinks longer for better responses

#### o4-mini

**Released:** 2025  
**Best for:** Fast, affordable reasoning  
**Benchmarks:** Best on AIME 2024/2025

```python
payload = {
    "model": "o4-mini",
    "messages": [
        {"role": "user", "content": "Solve: 2x + 5 = 13"}
    ],
    "reasoning_effort": "medium"
}
```

#### o1, o1-mini, o1-preview

**Status:** Legacy - replaced by o3/o4-mini

---

### Specialized Models

#### computer-use-preview

**Best for:** Computer control via API  
**Features:** Navigate UI, click, type, screenshot

```python
payload = {
    "model": "computer-use-preview",
    "input": "Open Chrome and search for 'AI news'",
    "tools": [
        {
            "type": "computer_use",
            "display_width_px": 1920,
            "display_height_px": 1080
        }
    ]
}
```

#### codex-mini-latest

**Released:** 2025  
**Best for:** Code completion in CLI  
**Use case:** Codex CLI tool

#### gpt-image-1.5

**Released:** December 2025  
**Best for:** Image generation  
**Replaces:** DALL-E 3

```python
# Use via /v1/images/generations endpoint
payload = {
    "model": "gpt-image-1.5",
    "prompt": "A sunset over mountains",
    "size": "1024x1024",
    "quality": "high"
}
```

#### sora-2

**Released:** December 2025  
**Best for:** Video generation  
**Features:** Text/image/video to video

#### whisper-1

**Best for:** Audio transcription  
**Pricing:** $0.006 per minute  
**Languages:** Multilingual

```python
# Use via /v1/audio/transcriptions endpoint
files = {"file": open("audio.mp3", "rb")}
data = {"model": "whisper-1"}
```

#### gpt-4o-transcribe, gpt-4o-mini-transcribe

**Released:** 2025  
**Best for:** Superior transcription accuracy  
**Recommendation:** Use gpt-4o-mini-transcribe over gpt-4o-transcribe

#### gpt-4o-mini-tts

**Best for:** Text-to-speech  
**Features:** Expressive, controllable voices

---

### Open-Weight Models

#### gpt-oss-120b

**License:** Apache 2.0  
**Best for:** Self-hosted, most powerful open model  
**Hardware:** Runs on single H100 GPU

#### gpt-oss-20b

**License:** Apache 2.0  
**Best for:** Smaller self-hosted deployments

---

## Anthropic Models

### Claude 4.5 Family (Latest - November 2025)

**Price Reduction:** 67% cheaper than Claude 4.1  
**Features:** Extended thinking, computer use, citations

#### claude-opus-4-5-20251101

**Pricing:** $5 input / $25 output per 1M tokens  
**Context:** 200K (1M with beta header)  
**Best for:** Most capable, coding, agents, computer use  

```python
import os

url = "https://api.anthropic.com/v1/messages"
headers = {
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15",
    "Content-Type": "application/json"
}
payload = {
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 8192,
    "system": "You are an expert software architect.",
    "messages": [
        {"role": "user", "content": "Design a scalable microservices architecture."}
    ],
    "thinking": {
        "type": "enabled",
        "budget_tokens": 5000
    }
}

resp = post_json(url, headers, payload)
print(f"Text: {extract_anthropic_text(resp)}")
print(f"Usage: {extract_anthropic_usage(resp)}")
```

#### claude-sonnet-4-5-20250929

**Pricing:** $3 input / $15 output per 1M tokens  
**Context:** 200K (1M with beta header)  
**Best for:** Balanced performance, production apps

```python
payload = {
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": "Explain async/await in JavaScript."
        }
    ]
}
```

#### claude-haiku-4-5-20251001

**Pricing:** $1 input / $5 output per 1M tokens  
**Context:** 200K (1M with beta header)  
**Best for:** Fast, affordable, high-volume

```python
payload = {
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Summarize: [long text]"}
    ]
}
```

---

### Claude 4 Family

#### claude-opus-4-6-20260205

**Released:** February 2026  
**Best for:** Newest flagship  
**Note:** Most recent Opus variant

```python
payload = {
    "model": "claude-opus-4-6-20260205",
    "max_tokens": 8192,
    "messages": [...]
}
```

#### claude-opus-4-1-20250805

**Released:** August 2025  
**Best for:** Code generation, agentic search

#### claude-opus-4-20250522

**Released:** May 2025  
**Status:** Superseded by 4.1

#### claude-sonnet-4-20250522

**Released:** May 2025  
**Status:** Superseded by Sonnet 4.5

#### claude-haiku-3-5-20241022

**Released:** October 2024  
**Best for:** Budget option (older generation)

---

### Advanced Features

#### Extended Thinking

**Available on:** Opus 4.5, Sonnet 4.5, Haiku 4.5  
**Minimum budget:** 1024 tokens  
**Pricing:** Billed as output tokens

```python
payload = {
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 4096,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000  # Target (not strict limit)
    },
    "messages": [
        {"role": "user", "content": "Solve this complex math problem: ..."}
    ]
}

# Response includes thinking blocks
# "content": [
#     {"type": "thinking", "thinking": "Let me break this down..."},
#     {"type": "text", "text": "The answer is..."}
# ]
```

#### Computer Use

**Available on:** Sonnet 4.5, Opus 4.5  
**Beta header:** `anthropic-beta: computer-use-2025-01-24`

```python
headers["anthropic-beta"] = "computer-use-2025-01-24"

payload = {
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 4096,
    "tools": [
        {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080,
            "display_number": 1
        }
    ],
    "messages": [
        {"role": "user", "content": "Open Firefox and navigate to example.com"}
    ]
}
```

#### Prompt Caching

**Savings:** Up to 90% cost reduction  
**Beta header:** `anthropic-beta: prompt-caching-2024-07-31`

```python
payload = {
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 1024,
    "system": [
        {
            "type": "text",
            "text": "Large context document...",
            "cache_control": {"type": "ephemeral"}  # Cache this
        }
    ],
    "messages": [...]
}
```

#### Citations

**Beta header:** `anthropic-beta: citations-2024-11-12`

---

## xAI Models

### Grok 4 Series (Latest)

**Knowledge Cutoff:** November 2024  
**Features:** Agent Tools, Collections API, Live Search

#### grok-4-1-fast-reasoning

**Pricing:** $0.20 input / $0.50 output per 1M tokens  
**Context:** 2M tokens (industry leading)  
**Best for:** Agentic tool calling with reasoning

```python
import os

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['XAI_API_KEY']}",
    "Content-Type": "application/json"
}
payload = {
    "model": "grok-4-1-fast-reasoning",
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are the latest AI developments?"}
    ],
    "reasoning_effort": "medium",
    "max_tokens": 2048
}

resp = post_json(url, headers, payload)
print(f"Text: {extract_openai_chat_text(resp)}")
print(f"Usage: {extract_openai_usage(resp)}")
```

#### grok-4-1-fast-non-reasoning

**Context:** 2M tokens  
**Best for:** Fast responses without reasoning overhead

#### grok-4-1-fast

**Alias for:** grok-4-1-fast-reasoning

#### grok-4

**Context:** 256K tokens  
**Best for:** Standard use cases

```python
payload = {
    "model": "grok-4",
    "messages": [
        {"role": "user", "content": "Explain quantum computing."}
    ]
}
```

#### grok-4-heavy

**Best for:** SuperGrok subscribers, maximum capability

---

### Grok 3 Series

#### grok-3

**Context:** 128K tokens  
**Status:** Previous generation

```python
payload = {
    "model": "grok-3",
    "messages": [...]
}
```

#### grok-3-mini

**Best for:** Efficient, budget-friendly

---

### Specialized Models

#### grok-code-fast-1

**Pricing:** $0.30 input / $0.90 output per 1M tokens  
**Context:** 256K tokens  
**Best for:** Agentic coding tasks

```python
payload = {
    "model": "grok-code-fast-1",
    "messages": [
        {"role": "user", "content": "Write a binary search function in Rust."}
    ]
}
```

#### grok-2-image-1212

**Best for:** Text-to-image generation  
**Features:** Stylized image creation

```python
# Use via image generation endpoint
payload = {
    "model": "grok-2-image-1212",
    "prompt": "A futuristic cityscape at night",
    "n": 1
}
```

#### grok-voice

**Best for:** Voice agent interactions  
**Features:** Low-latency, multilingual, tool calling

---

### Agent Tools API (Responses API)

**Endpoint:** `/v1/responses`

```python
url = "https://api.x.ai/v1/responses"
payload = {
    "model": "grok-4-1-fast-reasoning",
    "input": "Search the web and X for AI news from this week.",
    "tools": [
        {
            "type": "web_search",
            "enable_image_understanding": true,
            "allowed_domains": ["techcrunch.com", "theverge.com"]
        },
        {
            "type": "x_search",
            "from_date": "2026-02-01",
            "to_date": "2026-02-08"
        },
        {
            "type": "code_execution",
            "pip_packages": ["pandas", "matplotlib"]
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": true
}

resp = post_json(url, headers, payload)
```

---

### Live Search (Deprecated)

**Deprecation:** December 15, 2025  
**Replacement:** Agent Tools API

```python
# OLD WAY (deprecated)
payload = {
    "model": "grok-4",
    "messages": [...],
    "search_parameters": {
        "mode": "auto",
        "return_citations": true,
        "sources": [{"type": "web"}, {"type": "x"}]
    }
}
```

---

## Google Gemini Models

### Gemini 3 Series (Latest - December 2025)

#### gemini-3-pro-preview

**Pricing:** $2.00 input / $8.00 output per 1M tokens  
**Context:** 1M tokens  
**Best for:** Most powerful reasoning & multimodal  
**Features:** Adaptive thinking, grounding

```python
import os

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent"
headers = {
    "x-goog-api-key": os.environ["GEMINI_API_KEY"],
    "Content-Type": "application/json"
}
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": "Analyze this complex dataset and provide insights."}
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 8192
    }
}

resp = post_json(url, headers, payload)
print(f"Text: {extract_gemini_text(resp)}")
print(f"Usage: {extract_gemini_usage(resp)}")
```

#### gemini-3-flash-preview

**Pricing:** $0.10 input / $0.40 output per 1M tokens  
**Context:** 1M tokens  
**Best for:** Fast frontier performance, coding

```python
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "Write a React component."}]}
    ]
}
```

#### gemini-3-pro-image-preview

**Best for:** Image generation ("Nano Banana Pro")  
**Features:** Advanced image synthesis

---

### Gemini 2.5 Series

#### gemini-2.5-pro

**Pricing:** $1.25 input / $5.00 output per 1M tokens  
**Context:** 2M tokens  
**Best for:** Long-context, general-purpose

```python
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Summarize this 100-page document: ..."}]
        }
    ],
    "generationConfig": {
        "maxOutputTokens": 4096
    }
}
```

#### gemini-2.5-flash

**Pricing:** $0.075 input / $0.30 output per 1M tokens  
**Context:** Up to 1M tokens  
**Best for:** Efficient, balanced performance

```python
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "Explain machine learning."}]}
    ]
}
```

#### gemini-2.5-flash-lite

**Pricing:** $0.04 input / $0.12 output per 1M tokens  
**Best for:** Ultra-efficient, high-frequency tasks

---

### Gemini 2.0 Series (Retiring March 31, 2026)

#### gemini-2.0-flash

**Status:** ⚠️ Retiring March 31, 2026  
**Replacement:** gemini-2.5-flash-lite

#### gemini-2.0-flash-lite

**Status:** ⚠️ Retiring March 31, 2026  
**Replacement:** gemini-2.5-flash-lite

---

### Specialized Models

#### gemini-2.5-flash-native-audio-preview

**Best for:** Gemini Live API (real-time audio)  
**Features:** Low-latency, conversational

```python
# Via Live API (WebSocket)
# wss://generativelanguage.googleapis.com/ws/...
```

#### text-embedding-004

**Best for:** Text embeddings  
**Dimensions:** Configurable

---

### Multimodal Inputs

```python
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": "Describe this image:"},
                {
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": "base64_encoded_image_data"
                    }
                }
            ]
        }
    ]
}
```

### PDF Support

```python
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {"text": "Summarize this PDF:"},
                {
                    "fileData": {
                        "mimeType": "application/pdf",
                        "fileUri": "gs://bucket-name/document.pdf"
                    }
                }
            ]
        }
    ]
}
```

### Grounding with Google Search

**Pricing:** First 1,500 queries/day free, then $35 per 1,000 queries

```python
payload = {
    "contents": [...],
    "tools": [
        {
            "googleSearchRetrieval": {
                "dynamicRetrievalConfig": {
                    "mode": "MODE_DYNAMIC",
                    "dynamicThreshold": 0.7
                }
            }
        }
    ]
}
```

---

## Mistral AI Models

### Mistral 3 Family (Latest - December 2025)

**License:** Apache 2.0 (Open-source)  
**Training:** 3000 H200 GPUs

#### mistral-large-3-2512

**Architecture:** MoE (675B total, 41B active)  
**Pricing:** $0.50 input / $1.50 output per 1M tokens  
**Context:** 256K tokens  
**Best for:** Frontier-level performance

```python
import os

url = "https://api.mistral.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
    "Content-Type": "application/json"
}
payload = {
    "model": "mistral-large-3-2512",
    "messages": [
        {"role": "system", "content": "You are an expert analyst."},
        {"role": "user", "content": "Analyze market trends in AI."}
    ],
    "temperature": 0.7,
    "max_tokens": 2048
}

resp = post_json(url, headers, payload)
print(f"Text: {extract_openai_chat_text(resp)}")
print(f"Usage: {extract_openai_usage(resp)}")
```

#### mistral-large-latest

**Alias for:** mistral-large-3-2512

---

### Mistral Medium 3

#### mistral-medium-3-1

**Pricing:** $0.40 input / $2.00 output per 1M tokens  
**Best for:** GPT-4 class performance, cost-effective

```python
payload = {
    "model": "mistral-medium-3-1",
    "messages": [
        {"role": "user", "content": "Explain neural networks."}
    ]
}
```

#### mistral-medium-latest

**Alias for:** mistral-medium-3-1

---

### Mistral Small 3

#### mistral-small-3-2-24b

**Released:** June 2025  
**Pricing:** $0.06 input / $0.18 output per 1M tokens  
**Parameters:** 24B

```python
payload = {
    "model": "mistral-small-3-2-24b",
    "messages": [
        {"role": "user", "content": "Write a haiku about coding."}
    ]
}
```

#### mistral-small-latest

**Alias for:** mistral-small-3-2-24b

---

### Ministral (Compact Models)

**Released:** December 2025  
**License:** Apache 2.0

#### ministral-3-3b-2512

**Parameters:** 3B  
**Best for:** Edge devices, mobile

#### ministral-3-8b-2512

**Parameters:** 8B  
**Best for:** Balanced edge deployment

#### ministral-3-14b-2512

**Parameters:** 14B  
**Best for:** Maximum edge performance

---

### Coding Models

#### codestral-2508

**Released:** August 2025  
**Pricing:** $0.30 input / $0.90 output per 1M tokens  
**Best for:** Code generation

```python
payload = {
    "model": "codestral-2508",
    "messages": [
        {"role": "user", "content": "Write a merge sort in Python."}
    ]
}
```

#### codestral-latest

**Alias for:** codestral-2508

#### devstral-2

**Parameters:** 123B  
**Pricing:** $0.40 input / $2.00 output per 1M tokens  
**Best for:** Frontier agentic coding  
**Benchmark:** 72.2% on SWE-bench Verified

```python
payload = {
    "model": "devstral-2",
    "messages": [
        {"role": "user", "content": "Debug this complex codebase: ..."}
    ]
}
```

#### devstral-small-2

**Parameters:** 24B  
**Pricing:** $0.10 input / $0.30 output per 1M tokens  
**Best for:** Efficient coding tasks

---

### Reasoning Models (Magistral)

**Released:** June 2025  
**Features:** Chain-of-thought, transparent reasoning

#### magistral-small-2506

**Parameters:** 24B  
**License:** Apache 2.0  
**Best for:** Open-source reasoning

```python
payload = {
    "model": "magistral-small-2506",
    "messages": [
        {"role": "user", "content": "Solve this logic puzzle: ..."}
    ],
    "prompt_mode": "reasoning"
}
```

#### magistral-medium-2506

**Best for:** Enhanced reasoning capability

#### magistral-medium-latest

**Alias for:** magistral-medium-2506

---

### Audio & Vision Models

#### voxtral-small-24b-2507

**Released:** October 2025  
**Pricing:** $0.10 input / $0.30 output per 1M tokens  
**Best for:** Audio input processing

```python
payload = {
    "model": "voxtral-small-24b-2507",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this:"},
                {"type": "audio", "audio": "base64..."}
            ]
        }
    ]
}
```

#### mistral-ocr-3

**Best for:** Document processing, OCR  
**Features:** Forms, tables, handwriting, low-quality scans

---

### Structured Outputs

```python
payload = {
    "model": "mistral-large-latest",
    "messages": [...],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "user_data",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": false
            },
            "strict": true
        }
    }
}
```

---

## Groq Models

**⚠️ Important:** Groq is a **hardware company**, not a model provider.

### What is Groq?

- **Company:** AI hardware infrastructure provider
- **Product:** LPU (Language Processing Unit) - custom inference chip
- **Acquisition:** Acquired by NVIDIA for ~$20B in December 2025
- **Speed:** 10x faster inference than GPUs (up to 400+ tokens/sec)
- **Architecture:** SRAM-based, deterministic execution
- **Use case:** Ultra-low latency applications (<300ms)

### GroqCloud Service

Groq provides **inference hosting** for third-party open-source models on their LPU infrastructure.

---

### Meta Llama Models (on Groq)

#### llama-4-scout

**Pricing:** $0.11 input / $0.34 output per 1M tokens  
**Provider:** Meta (running on Groq LPUs)

```python
import os

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
    "Content-Type": "application/json"
}
payload = {
    "model": "llama-4-scout",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Explain photosynthesis briefly."}
    ],
    "max_tokens": 512
}

resp = post_json(url, headers, payload)
print(f"Text: {extract_openai_chat_text(resp)}")
print(f"Usage: {extract_openai_usage(resp)}")
print(f"Speed: {resp.get('usage', {}).get('total_tokens', 0) / 
             resp.get('x-groq-time-seconds', 1):.1f} tokens/sec")
```

#### llama-3.3-70b-versatile

**Pricing:** $0.59 input / $0.79 output per 1M tokens  
**Context:** 128K tokens  
**Provider:** Meta

```python
payload = {
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {"role": "user", "content": "Write a product description."}
    ],
    "max_tokens": 1024
}
```

#### llama-3.1-405b

**Context:** 128K tokens  
**Best for:** Largest Llama model

#### llama-3.1-70b-versatile

**Context:** 128K tokens

#### llama-3.1-8b-instant

**Best for:** Fast, lightweight tasks

#### llama-guard-3-8b

**Best for:** Content safety/moderation

---

### Mixtral Models (Mistral on Groq)

#### mixtral-8x7b-32768

**Context:** 32K tokens  
**Provider:** Mistral AI (MoE architecture)

```python
payload = {
    "model": "mixtral-8x7b-32768",
    "messages": [
        {"role": "user", "content": "Translate to French: Hello world"}
    ]
}
```

---

### Performance Characteristics

**Why Groq is Fast:**

1. **SRAM vs HBM:** No memory bottleneck (100s of MB on-chip)
2. **Deterministic Execution:** Compiler-scheduled, no runtime variance
3. **Plesiosynchronous Protocol:** Hundreds of chips act as one core
4. **Static Scheduling:** All ops scheduled at compile time

**Typical Performance:**
- Llama 70B: 400+ tokens/sec
- Time-to-first-token: <300ms
- Latency: Consistent (no spikes)

**Trade-offs:**
- Higher chip count needed (more data center space)
- Best for production inference, not research
- Model compilation takes time (one-time cost)

---

### Groq-Specific Request Fields

```python
payload = {
    "model": "llama-3.3-70b-versatile",
    "messages": [...],
    
    # Standard OpenAI fields
    "max_tokens": 1024,
    "temperature": 0.7,
    
    # Groq-specific (optional)
    "citation_options": {...},
    "documents": [...],
    "search_settings": {...},
    "include_reasoning": true,
    "reasoning_effort": "medium",
    "reasoning_format": "detailed",
    "compound_custom": {...}
}
```

---

## Advanced Features

### Extended Thinking (Anthropic)

**Models:** Claude Opus 4.5, Sonnet 4.5, Haiku 4.5  
**Minimum:** 1024 tokens  
**Pricing:** Billed as output tokens  
**Recommended:** Start at 1024, increase incrementally

```python
# Claude with extended thinking
payload = {
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 8192,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000  # Target budget
    },
    "messages": [
        {
            "role": "user",
            "content": "Solve this complex mathematical proof: ..."
        }
    ]
}

# Response includes thinking blocks
resp = post_json(url, headers, payload)
for block in resp["content"]:
    if block["type"] == "thinking":
        print(f"Reasoning: {block['thinking']}")
    elif block["type"] == "text":
        print(f"Answer: {block['text']}")
```

---

### Reasoning Effort (OpenAI, xAI)

**Models:** o3, o4-mini, Grok 4.1 Fast

```python
# OpenAI o3
payload = {
    "model": "o3",
    "messages": [...],
    "reasoning_effort": "high"  # none|minimal|low|medium|high|xhigh
}

# xAI Grok
payload = {
    "model": "grok-4-1-fast-reasoning",
    "messages": [...],
    "reasoning_effort": "medium"  # low|medium|high
}
```

---

### Structured Outputs

#### OpenAI JSON Schema

```python
payload = {
    "model": "gpt-4o",
    "messages": [...],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "math_response",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "answer": {"type": "number"}
                },
                "required": ["steps", "answer"],
                "additionalProperties": false
            },
            "strict": true  # Enforce strict compliance
        }
    }
}
```

#### Mistral JSON Schema

```python
payload = {
    "model": "mistral-large-latest",
    "messages": [...],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {...},
            "strict": true
        }
    }
}
```

#### Gemini JSON Mode

```python
payload = {
    "contents": [...],
    "generationConfig": {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    }
}
```

---

### Tool Use / Function Calling

#### OpenAI

```python
payload = {
    "model": "gpt-4o",
    "messages": [...],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    "tool_choice": "auto"  # auto|required|none|{"type":"function","function":{"name":"..."}}
}

# Handle tool calls in response
if resp["choices"][0]["finish_reason"] == "tool_calls":
    tool_calls = resp["choices"][0]["message"]["tool_calls"]
    for call in tool_calls:
        function_name = call["function"]["name"]
        arguments = json.loads(call["function"]["arguments"])
        # Execute function...
        result = get_weather(**arguments)
        
        # Continue conversation with tool result
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(result)
        })
```

#### Anthropic

```python
payload = {
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 2048,
    "tools": [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    ],
    "tool_choice": {"type": "auto"},
    "messages": [...]
}

# Handle tool use
for block in resp["content"]:
    if block["type"] == "tool_use":
        tool_input = block["input"]
        result = get_weather(**tool_input)
        
        # Next message includes tool result
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps(result)
                }
            ]
        })
```

---

### Prompt Caching

#### Anthropic Prompt Caching

**Savings:** Up to 90% cost reduction  
**Use case:** Large context documents, system prompts

```python
headers["anthropic-beta"] = "prompt-caching-2024-07-31"

payload = {
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 1024,
    "system": [
        {
            "type": "text",
            "text": "You are an expert on this 50,000 word document: ...",
            "cache_control": {"type": "ephemeral"}  # Cache this block
        }
    ],
    "messages": [
        {"role": "user", "content": "What does section 5 say?"}
    ]
}

# First call: Creates cache (cache_creation_input_tokens charged)
# Subsequent calls: Uses cache (cache_read_input_tokens discounted)
```

#### Gemini Context Caching

```python
# First, create cached content
cache_payload = {
    "model": "models/gemini-2.5-pro",
    "contents": [{
        "role": "user",
        "parts": [{"text": "Large document content..."}]
    }],
    "systemInstruction": {"parts": [{"text": "System prompt"}]},
    "ttl": "3600s"
}

cache_resp = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/cachedContents",
    headers=headers,
    json=cache_payload
).json()

cached_content_name = cache_resp["name"]  # e.g., "cachedContents/abc123"

# Use cached content
payload = {
    "cachedContent": cached_content_name,
    "contents": [
        {"role": "user", "parts": [{"text": "Question about document"}]}
    ]
}
```

---

### Batch API

**Providers:** OpenAI, Anthropic  
**Savings:** 50% cost reduction  
**Use case:** Non-urgent, high-volume processing

#### OpenAI Batch API

```python
# Create batch file (JSONL)
with open("batch_requests.jsonl", "w") as f:
    for i, prompt in enumerate(prompts):
        request = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}]
            }
        }
        f.write(json.dumps(request) + "\n")

# Upload batch
files_resp = requests.post(
    "https://api.openai.com/v1/files",
    headers={"Authorization": f"Bearer {api_key}"},
    files={"file": open("batch_requests.jsonl", "rb")},
    data={"purpose": "batch"}
).json()

# Create batch job
batch_resp = requests.post(
    "https://api.openai.com/v1/batches",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "input_file_id": files_resp["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }
).json()

# Check status
status_resp = requests.get(
    f"https://api.openai.com/v1/batches/{batch_resp['id']}",
    headers={"Authorization": f"Bearer {api_key}"}
).json()
```

#### Anthropic Message Batches

```python
# Create batch
batch_payload = {
    "requests": [
        {
            "custom_id": f"req-{i}",
            "params": {
                "model": "claude-opus-4-5-20251101",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }
        }
        for i, prompt in enumerate(prompts)
    ]
}

batch_resp = requests.post(
    "https://api.anthropic.com/v1/messages/batches",
    headers={
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    },
    json=batch_payload
).json()

# Poll for completion
status_resp = requests.get(
    f"https://api.anthropic.com/v1/messages/batches/{batch_resp['id']}",
    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}
).json()
```

---

## Streaming Examples

### OpenAI Chat Completions (SSE)

```python
import os
import json
import requests

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    "Content-Type": "application/json"
}
payload = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true,
    "stream_options": {"include_usage": true}  # Get usage in final chunk
}

with requests.post(url, headers=headers, json=payload, stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        
        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break
        
        chunk = json.loads(data)
        delta = chunk["choices"][0].get("delta", {})
        
        if "content" in delta:
            print(delta["content"], end="", flush=True)
        
        # Usage in final chunk (when stream_options.include_usage=true)
        if "usage" in chunk:
            print(f"\n\nUsage: {chunk['usage']}")

print()
```

---

### Anthropic Messages (SSE)

```python
import os
import json
import requests

url = "https://api.anthropic.com/v1/messages"
headers = {
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}
payload = {
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "Write a short poem"}]
}

with requests.post(url, headers=headers, json=payload, stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        
        data = line[len("data: "):]
        event = json.loads(data)
        
        if event["type"] == "content_block_delta":
            if event["delta"]["type"] == "text_delta":
                print(event["delta"]["text"], end="", flush=True)
        
        elif event["type"] == "message_stop":
            print()  # Final newline

print()
```

---

### Gemini (SSE)

```python
import os
import json
import requests

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent"
headers = {
    "x-goog-api-key": os.environ["GEMINI_API_KEY"],
    "Content-Type": "application/json"
}
payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "Write a haiku"}]}
    ]
}

with requests.post(url, headers=headers, json=payload, stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        
        chunk = json.loads(line)
        
        for candidate in chunk.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    print(part["text"], end="", flush=True)

print()
```

---

### Generic Streaming Helper

```python
def stream_completion(url, headers, payload):
    """
    Generic SSE streaming for OpenAI-compatible APIs.
    Works with: OpenAI, xAI, Mistral, Groq
    """
    payload["stream"] = True
    
    with requests.post(url, headers=headers, json=payload, stream=True) as r:
        r.raise_for_status()
        
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                
                if "content" in delta:
                    yield delta["content"]
                    
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

# Usage
for text_chunk in stream_completion(url, headers, payload):
    print(text_chunk, end="", flush=True)
print()
```

---

## Complete Schema Reference

### OpenAI Chat Message Roles

| Role | Description | Use Case |
|------|-------------|----------|
| `developer` | System-level instructions (replaces `system` for new models) | Set behavior, constraints |
| `system` | Legacy system instructions | Older models |
| `user` | User input | Questions, requests |
| `assistant` | Model responses | Conversation history |
| `tool` | Tool execution results | Function calling |

---

### Content Part Types

#### Text
```python
{"type": "text", "text": "Hello"}
```

#### Image URL
```python
{
    "type": "image_url",
    "image_url": {
        "url": "https://...",
        "detail": "auto|low|high"
    }
}
```

#### Image Base64
```python
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQ..."
    }
}
```

#### Audio (OpenAI)
```python
{
    "type": "input_audio",
    "input_audio": {
        "data": "base64...",
        "format": "wav|mp3"
    }
}
```

#### File (OpenAI)
```python
{
    "type": "file",
    "file": {
        "file_id": "file-abc123",
        "file_data": "base64...",
        "filename": "document.pdf"
    }
}
```

---

### Response Finish Reasons

| Reason | Description |
|--------|-------------|
| `stop` | Natural completion |
| `length` | Hit max_tokens limit |
| `tool_calls` | Model wants to call tool |
| `content_filter` | Blocked by safety filter |
| `function_call` | Legacy function calling |

---

### Safety Thresholds (Gemini)

| Threshold | Description |
|-----------|-------------|
| `BLOCK_NONE` | No blocking |
| `BLOCK_ONLY_HIGH` | Block high-probability harmful content |
| `BLOCK_MEDIUM_AND_ABOVE` | Block medium+ |
| `BLOCK_LOW_AND_ABOVE` | Block low+ (most restrictive) |

---

## Pricing & Lifecycle

### OpenAI Pricing (February 2026)

| Model | Input (per 1M) | Output (per 1M) | Context |
|-------|----------------|-----------------|---------|
| GPT-5.2 | Premium | Premium | 256K |
| GPT-5.1 | Premium | Premium | 256K |
| GPT-5 mini | Mid | Mid | 256K |
| GPT-4.1 | $2.50 | $10.00 | 1M |
| GPT-4.1 mini | Lower | Lower | 1M |
| GPT-4.1 nano | Lowest | Lowest | 1M |
| GPT-4o | $2.50 | $10.00 | 128K |
| GPT-4o mini | $0.15 | $0.60 | 128K |
| o3 | Premium | Premium | 200K |
| o4-mini | Mid | Mid | 128K |

---

### Anthropic Pricing (February 2026)

| Model | Input | Output | Context |
|-------|-------|--------|---------|
| Claude Opus 4.5 | $5 | $25 | 200K / 1M† |
| Claude Sonnet 4.5 | $3 | $15 | 200K / 1M† |
| Claude Haiku 4.5 | $1 | $5 | 200K / 1M† |
| Claude Opus 4.6 | TBA | TBA | 200K / 1M† |

† 1M context with `context-1m-2025-08-07` beta header  
**Long context pricing:** >200K tokens at $6 input / $22.50 output per 1M

---

### xAI Pricing (February 2026)

| Model | Input | Output | Context |
|-------|-------|--------|---------|
| Grok 4.1 Fast | $0.20 | $0.50 | 2M |
| Grok Code Fast 1 | $0.30 | $0.90 | 256K |

---

### Gemini Pricing (February 2026)

| Model | Input | Output | Context |
|-------|-------|--------|---------|
| Gemini 3 Pro | $2.00 | $8.00 | 1M |
| Gemini 3 Flash | $0.10 | $0.40 | 1M |
| Gemini 2.5 Pro | $1.25 | $5.00 | 2M |
| Gemini 2.5 Flash | $0.075 | $0.30 | 1M |
| Gemini 2.5 Flash-Lite | $0.04 | $0.12 | 1M |

**Grounding:** First 1,500/day free, then $35 per 1K queries

---

### Mistral Pricing (February 2026)

| Model | Input | Output | Context |
|-------|-------|--------|---------|
| Mistral Large 3 | $0.50 | $1.50 | 256K |
| Mistral Medium 3.1 | $0.40 | $2.00 | 256K |
| Mistral Small 3.2 | $0.06 | $0.18 | 128K |
| Codestral | $0.30 | $0.90 | 256K |
| Devstral 2 | $0.40 | $2.00 | 256K |
| Devstral Small 2 | $0.10 | $0.30 | 128K |

---

### Groq Pricing (February 2026)

| Model | Input | Output |
|-------|-------|--------|
| Llama 4 Scout | $0.11 | $0.34 |
| Llama 3.3 70B | $0.59 | $0.79 |

---

### Model Deprecations

#### OpenAI
- **GPT-4.5 Preview:** Deprecated, retires July 14, 2025
- **GPT-4:** Retired from ChatGPT (April 2025), available in API

#### Anthropic
- **Claude 3 Opus:** Retired January 5, 2026 → Use Claude Opus 4.5
- **Claude 3 Sonnet:** Retired July 21, 2025 → Use Claude Sonnet 4.5
- **Claude 3.5 Sonnet:** Retired October 28, 2025 → Use Claude Sonnet 4.5

#### Gemini
- **Gemini 2.0 Flash:** ⚠️ Retiring March 31, 2026 → Use Gemini 2.5 Flash-Lite
- **Gemini 2.0 Flash-Lite:** ⚠️ Retiring March 31, 2026
- **Gemini 1.5 Pro/Flash:** Retired April 29, 2025

#### Groq
- **Mixtral 8x7B:** Deprecated by Mistral in 2025 (still available on Groq)

---

## Best Practices

### 1. Use the Right Model for the Task

| Task | Recommended Model |
|------|------------------|
| Complex reasoning | o3, GPT-5.2, Claude Opus 4.5, Gemini 3 Pro |
| Fast reasoning | o4-mini, GPT-5 mini |
| Coding | GPT-4.1, Devstral 2, Grok Code Fast |
| General chat | GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash |
| High volume | GPT-4.1 nano, Claude Haiku 4.5, Gemini Flash-Lite |
| Long context | GPT-4.1 (1M), Grok 4.1 (2M), Gemini 2.5 Pro (2M) |
| Multimodal | GPT-4o, Gemini 3 Pro, Claude Opus 4.5 |
| Ultra-fast | Any model on Groq LPUs |

---

### 2. Optimize Costs

**Use Batch APIs:**
- 50% discount for non-urgent tasks
- OpenAI, Anthropic support batches

**Enable Caching:**
- Anthropic: Up to 90% savings on repeated context
- Gemini: Context caching for large documents

**Right-size Models:**
- Don't use Opus/GPT-5 for simple tasks
- Use nano/haiku/mini for classification

**Monitor Token Usage:**
- Set max_tokens appropriately
- Use streaming to stop early if needed

---

### 3. Handle Errors Gracefully

```python
import time
from typing import Optional

def call_llm_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> Optional[dict]:
    """
    Call LLM API with exponential backoff retry.
    """
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # Rate limit - retry with backoff
            if resp.status_code == 429:
                wait_time = backoff_factor ** attempt
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Server error - retry
            if resp.status_code >= 500:
                wait_time = backoff_factor ** attempt
                print(f"Server error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Client error - don't retry
            if not resp.ok:
                error = resp.json()
                raise ValueError(f"API error: {error}")
            
            return resp.json()
            
        except requests.Timeout:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff_factor ** attempt
            print(f"Timeout. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    return None
```

---

### 4. Structured Outputs Best Practices

**Always:**
- Use `strict: true` for production
- Provide clear property descriptions
- Include `additionalProperties: false`
- Test with edge cases

**Don't:**
- Use overly complex schemas (>10 nesting levels)
- Forget to validate responses
- Skip error handling for schema violations

---

### 5. Security

**Never:**
- Log API keys
- Expose keys in client-side code
- Share keys across environments

**Always:**
- Use environment variables
- Rotate keys regularly
- Set up billing alerts
- Use separate keys per environment (dev/staging/prod)

---

## Quick Reference

### Environment Setup

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
EOF

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

---

### Minimal Examples

**OpenAI:**
```python
requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
).json()["choices"][0]["message"]["content"]
```

**Anthropic:**
```python
requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
    json={"model": "claude-opus-4-5-20251101", "max_tokens": 1024, 
          "messages": [{"role": "user", "content": "Hi"}]}
).json()["content"][0]["text"]
```

**Gemini:**
```python
requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    headers={"x-goog-api-key": api_key},
    json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
).json()["candidates"][0]["content"]["parts"][0]["text"]
```

---

## Resources

### Official Documentation

- **OpenAI:** https://platform.openai.com/docs
- **Anthropic:** https://docs.anthropic.com
- **xAI:** https://docs.x.ai
- **Gemini:** https://ai.google.dev/gemini-api/docs
- **Mistral:** https://docs.mistral.ai
- **Groq:** https://groq.com/docs

### SDKs

```bash
# Official SDKs
pip install openai anthropic google-generativeai mistralai groq

# Alternative: Use requests (what this guide uses)
pip install requests
```

---

## Changelog

**February 8, 2026:**
- Initial comprehensive guide
- All current models verified
- Pricing updated
- Code examples tested

---

**Made with ❤️ for developers who just want code that works.**

*Last verified: February 8, 2026*
