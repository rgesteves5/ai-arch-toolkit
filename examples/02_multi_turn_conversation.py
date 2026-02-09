"""02 — Multi-turn Conversation (Anthropic).

Build a conversation with explicit Message objects, use a system prompt,
and continue the conversation by appending the assistant's reply.
"""

from ai_arch_toolkit import Client, Message

client = Client("anthropic", model="claude-haiku-4-5-20251001")

messages = [
    Message(role="user", content="My name is Alice. What's a fun fact about space?"),
]

# First turn
resp = client.chat(messages, system="You are a friendly science tutor.")
print("Assistant:", resp.text)

# Continue the conversation — append the assistant reply, then a follow-up
messages.append(resp.to_message())
messages.append(Message(role="user", content="Can you remind me of my name?"))

resp2 = client.chat(messages, system="You are a friendly science tutor.")
print("\nAssistant:", resp2.text)
