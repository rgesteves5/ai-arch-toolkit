# Image Generation

Image models draw and edit images through `LLM.generate_image()`. A chat model can also draw in
the middle of its turn with the hosted `image_generation()` tool (OpenAI). Either way, the images
come back in `Response.images` as bytes with their MIME type, and the call is metered like any
other.

```python
from ai_arch_toolkit import LLM, image

llm = LLM("gpt-image-2.5-flare")
response = llm.generate_image_sync("A small red lighthouse at sunset", aspect_ratio="16:9")
picture = response.images[0]
open("lighthouse.png", "wb").write(picture.data)  # picture.media_type == "image/png"
print(response.cost)                              # priced from the reported image tokens

# Edit: the prompt and the images to change
edited = llm.generate_image_sync("Make the sky purple", images=[image(picture.data)])
```

`generate_image()` runs on the same path as `complete()`: middleware, retries, fallbacks (to
another image model), `Response.attempts` and the meter all apply. A `generate_image_sync()`
wrapper exists, as for every coroutine.

## Image models

| Provider | Models | Edits | Billing |
|---|---|---|---|
| OpenAI (Images API) | `gpt-image-2.5-sunburst`, `gpt-image-2.5-flare`, `gpt-image-2`, `gpt-image-1.5`, `chatgpt-image-latest`, `gpt-image-1`, `gpt-image-1-mini` | up to 16 input images | image tokens at their own rates |
| Gemini | `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3-pro-image` | input images in the prompt | image output tokens at their own rate |
| xAI | `grok-imagine-image-2.0`, `grok-imagine-image`, `grok-imagine-image-quality` | up to 5 input images | the cost xAI reports |
| Meta | `muse-image-1.0` | one input image | $0.01 per image |

Anthropic's models do not generate images. An OpenAI-compatible server raises `RequestError`.
On Gemini the image models need a paid project: the free tier has no quota for them. Imagen and
`gemini-2.5-flash-image` are shut down.

The model of the `LLM` is the image model. A chat model refuses `generate_image()` with
`RequestError`, and an image model refuses `complete()`, except on Gemini and Meta, where an
image model also answers a chat turn with its images.

## Options

The options are portable. Each model raises `RequestError`, before anything is sent, for what it
does not take:

| Option | Values | Where it applies |
|---|---|---|
| `images` | `image(...)` parts to edit or draw from | all |
| `n` | how many images | OpenAI and Meta up to 10, xAI up to 10, Gemini 1 |
| `aspect_ratio` | `"16:9"`, `"1:1"`… | each model's own list (below) |
| `resolution` | `"512"`, `"1K"`, `"2K"`, `"4K"` | OpenAI, Gemini, xAI (`1K`/`2K`); not Meta |
| `quality` | the provider's levels | OpenAI (`low` to `high`, plus `xhigh` and `max` on the 2.5 models), xAI (`low`, `medium`) |
| `output_format` | `"png"`, `"jpeg"`, `"webp"` | OpenAI and Meta |

A resolution is the area of a square with that edge, as Gemini sizes its `1K`/`2K`/`4K` images.
On OpenAI, the aspect ratio and the resolution together become the `size`:
- On `gpt-image-2` and the 2.5 models, `aspect_ratio="16:9"` gives `1360x768`, and with
  `resolution="4K"` it gives `3840x2160`. Sizes come in multiples of 16, between 655,360 and
  8,294,400 pixels, with ratios from 1:3 to 3:1.
- The older GPT Image models take only `1:1`, `2:3` and `3:2`, at `1K`.
- Meta reads only the ratio.

Gemini takes its documented ratios (`1:4` to `8:1` on 3.1 Flash, `1:1` to `21:9` on the others)
and sizes (`512` on 3.1 Flash only, `1K` only on Flash Lite). xAI takes the ratios its SDK
converts.

## Cost

`Usage` keeps image tokens apart from text tokens: `image_input_tokens`, `image_output_tokens`,
and `image_count` for a provider that bills per image. `ModelPricing` has the matching rates:
`image_input`, `image_output`, their `batch_` variants, and `per_image`. An image rate left out
falls back to the text rate. Image tokens count against a budget's token caps. A strict budget
reserves, for each image asked, the model's per-image price plus the image output tokens its
provider publishes for the model, the quality and the size asked: OpenAI's table for the GPT
Image models before gpt-image-2 and its calculator for gpt-image-2 and later, Gemini's count per
image size. A quality or a size left to the model reserves the dearest it can pick (`auto` has
no published count), and a model with no published count reserves 24,000 tokens, above the
dearest published image (23,719, a 2880x2880 one at gpt-image-2's `high`). A model billed per
image reserves no image tokens. Gemini's image models think on every call, billed at the text
rate: their output token limit is reserved for it. See [Pricing](pricing.md).

## Images inside a turn

On OpenAI's own host, a chat model can draw while it answers:

```python
from ai_arch_toolkit import LLM, image_generation

llm = LLM("gpt-5.5")
tool = image_generation(model="gpt-image-2.5-flare", quality="low", aspect_ratio="16:9")
response = await llm.complete("Draw the logo we discussed", tools=[tool])
response.images  # the drawn image(s), next to response.text
```

- `model` is required: it names the image model that draws, since OpenAI's own default is an old
  one. The other options are `generate_image()`'s.
- OpenAI reports the tool's tokens apart from the turn's. `Response.cost` and the meter add
  them, priced at the image model's rates.
- The hosted tool needs OpenAI's host. Meta, xAI and the compatible servers refuse it.

Streamed with `stream_events()`, the images arrive as `StreamEvent(kind="image")`. A preview has
`partial=True`: OpenAI's partial images, with `image_generation(partial_images=1..3)`, and Gemini's
interim thought images. A finished image has `partial=False`. `Response.images` holds the finished
images only. A non-streamed request drops `partial_images`, which OpenAI takes only in a stream.

### Editing in the next turn

`response.to_message()` keeps the turn's `_raw`, so the next turn can edit what was drawn:
- On OpenAI the image goes back as an input image, because OpenAI refuses to replay the drawn
  item when nothing is stored (`store: false`).
- On Meta the image goes back by reference.

A history rebuilt without `_raw` (from a database, say) carries no image. To edit it, attach it
to the next user message with `image(...)`.
