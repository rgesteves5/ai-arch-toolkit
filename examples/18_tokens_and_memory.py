"""18 — Token Estimation + Conversation Memory.

No API calls: demonstrates local token heuristics and memory trimming.
"""

from ai_arch_toolkit import (
    ConversationMemory,
    Message,
    SlidingWindowMemory,
    estimate_conversation_tokens,
    estimate_text_tokens,
)

text = "Large language models process text in tokens."
print("Text:", text)
print("Estimated tokens:", estimate_text_tokens(text))

memory = ConversationMemory()
memory.add_user("Hello, my name is Alice.")
memory.add_assistant("Hi Alice, nice to meet you.")
memory.add(Message(role="user", content="Can you summarize Newton's first law?"))

history = memory.history()
print("\nConversationMemory items:", len(history))
print("ConversationMemory token estimate:", estimate_conversation_tokens(history))

window = SlidingWindowMemory(max_tokens=30)
window.add_user("A" * 80)
window.add_assistant("B" * 80)
window.add_user("C" * 20)

print("\nSlidingWindowMemory items after trim:", len(window.history()))
for i, item in enumerate(window.history(), start=1):
    content = item.content if isinstance(item.content, str) else "<multimodal>"
    print(f"  {i}. {item.role}: {str(content)[:60]}")

