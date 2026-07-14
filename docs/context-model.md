# Context Model

Several framework concepts can contain text, but they answer different questions.

| Concept | Question it answers | Typical lifetime |
|---|---|---|
| `Content` | What input type is sent to a provider? | One model call |
| `Resource` | Where did this content come from and how was it parsed? | Loaded snapshot |
| `Knowledge` | How does the application name, classify, and retrieve reference content? | Application/session |
| `Memory` | What has an agent learned or recorded over time? | Multiple runs |
| `PromptTemplate` | Which sources and variables are needed? | Reusable definition |
| `Prompt` | What resolved literal sections will be rendered? | One compiled snapshot |
| `PromptLayout` | How are sections serialized for the model? | Render policy |
| response format | How must the model structure its answer? | Model call |

The normal flow is:

```text
files -> Resources -> optional Knowledge -> PromptTemplate -> Prompt -> RenderedPrompt -> LLM
```

A PDF passed as `DocumentPart` is provider input. A Markdown file read to build system
instructions is a Resource. A style guide stored under `company.style` is Knowledge. Facts
written by an agent during earlier runs are Memory.

Input format, prompt layout, and response format are independent. A YAML resource can be
selected and serialized into an XML prompt while the model response is constrained by a
Pydantic `OutputSchema`.
