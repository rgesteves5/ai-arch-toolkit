"""02 — Multi-turn Conversation.

Build a conversation with plain dict messages, use a system prompt,
and continue the conversation by appending the assistant's reply.
"""

from ai_arch_toolkit import LLM, user

llm = LLM("claude-haiku-4-5")

messages = [
    user("My name is Alice. What's a fun fact about space?"),
]

# First turn
resp = llm.complete_sync(messages, system="You are a friendly science tutor.")
print("Assistant:", resp.text)

# Continue the conversation — append the assistant reply, then a follow-up
messages.append(resp.to_message())
messages.append(user("Can you remind me of my name?"))

resp2 = llm.complete_sync(messages, system="You are a friendly science tutor.")
print("\nAssistant:", resp2.text)
