# Florence2 in ComfyUI

> Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. 
Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. 
It leverages our FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. 
The model's sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model.

## New Feature: Document Visual Question Answering (DocVQA)

This fork includes support for Document Visual Question Answering (DocVQA) using the Florence2 model. DocVQA allows you to ask questions about the content of document images, and the model will provide answers based on the visual and textual information in the document. This feature is particularly useful for extracting information from scanned documents, forms, receipts, and other text-heavy images.

## Installation:

Clone this repository to 'ComfyUI/custom_nodes` folder.

Install the dependencies in requirements.txt, transformers version 4.38.0 minimum is required:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Florence2-Tile-prompt\requirements.txt`

![image](https://github.com/kijai/ComfyUI-Florence2/assets/40791699/4d537ac7-5490-470f-92f5-3007da7b9cc7)
![image](https://github.com/kijai/ComfyUI-Florence2/assets/40791699/512357b7-39ee-43ee-bb63-7347b0a8d07d)

Supports most Florence2 models, which can be automatically downloaded with the `DownloadAndLoadFlorence2Model` to `ComfyUI/models/LLM`:

Official:

https://huggingface.co/microsoft/Florence-2-base

https://huggingface.co/microsoft/Florence-2-base-ft

https://huggingface.co/microsoft/Florence-2-large

https://huggingface.co/microsoft/Florence-2-large-ft

https://huggingface.co/HuggingFaceM4/Florence-2-DocVQA

Tested finetunes:

https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5

https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5

https://huggingface.co/thwri/CogFlorence-2.2-Large

https://huggingface.co/HuggingFaceM4/Florence-2-DocVQA

https://huggingface.co/gokaygokay/Florence-2-SD3-Captioner

https://huggingface.co/gokaygokay/Florence-2-Flux-Large

https://huggingface.co/NikshepShetty/Florence-2-pixelpros

## Using DocVQA

To use the DocVQA feature:
1. Load a document image into ComfyUI.
2. Connect the image to the Florence2 DocVQA node.
3. Input your question about the document.
4. The node will output the answer based on the document's content.

Example questions:
- "What is the total amount on this receipt?"
- "What is the date mentioned in this form?"
- "Who is the sender of this letter?"

Note: The accuracy of answers depends on the quality of the input image and the complexity of the question.

## Tile Prompt Workflow (Florence2 + SimpleTiles Uprez - bundled)

This fork adds **per-tile captioning and conditioning** for intelligent tiled upscaling. Each tile receives its own context-aware prompt from Florence2, enabling better detail enhancement in different regions of your image.

### Prerequisites

1. **Florence2 Model** - Download via `DownloadAndLoadFlorence2Model` node
   - Recommended: `microsoft/Florence-2-large` or `Florence-2-large-ft` for best captions
   - Use `FL2MODEL` output type (not `FLORENCE2`)

### Bundled Tiles Nodes (SimpleTiles Uprez)

This repo includes a vendored copy of `ComfyUI_SimpleTiles_Uprez`, so you do **not** need a separate install for tiling/blending.

- Node types: `SimpleTilesUprezDynamicTileSplit`, `SimpleTilesUprezDynamicTileMerge`
- Display names: `TileSplit (SimpleTiles Uprez Dynamic)`, `TileMerge (SimpleTiles Uprez Dynamic)`

If you also have `C:\ComfyUI\custom_nodes\ComfyUI_SimpleTiles_Uprez` installed, remove/disable one of the two to avoid duplicate node registrations.

### Workflow Overview

Here's the complete node chain for per-tile prompted upscaling:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Image                                                    │
│    └─> LoadImage node                                           │
├─────────────────────────────────────────────────────────────────┤
│ 2. Split into Tiles                                              │
│    └─> TileSplit (SimpleTiles Uprez Dynamic)                    │
│        - Output: tiles (IMAGE batch), tile_calc (metadata)      │
├─────────────────────────────────────────────────────────────────┤
│ 3. Add Position Metadata (if needed)                            │
│    └─> Tile Calc Add Positions                                  │
│        - Only needed if the split node omits positions           │
├─────────────────────────────────────────────────────────────────┤
│ 4. Caption Each Tile                                             │
│    └─> Florence2 Batch Caption (Tiles)                          │
│        - Input: tiles, Florence2 model (FL2MODEL)               │
│        - Output: prompts_list (one caption per tile)            │
├─────────────────────────────────────────────────────────────────┤
│ 5. Convert Prompts to Conditioning                              │
│    └─> Prompt List -> Conditioning                              │
│        - Input: prompts_list, CLIP model                        │
│        - Output: conditioning_list (CLIP embeddings per tile)   │
├─────────────────────────────────────────────────────────────────┤
│ 6. Process Tiles with Per-Tile Prompts                          │
│    └─> Tiled Sampler with Prompt List                           │
│        - Input: tiles, per_tile_positive, negative, VAE, model  │
│        - Output: processed_tiles (enhanced tiles)               │
├─────────────────────────────────────────────────────────────────┤
│ 7. Merge Tiles Back Together                                    │
│    └─> TileMerge (SimpleTiles Uprez Dynamic)                    │
│        - Input: processed_tiles, tile_calc                      │
│        - Output: final upscaled image                           │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start Tips

- **Caption Quality**: Use `detailed_caption` or `more_detailed_caption` task modes
- **Prompt Formatting**: Use prepend/append text to add quality tags
  - Example prepend: `"high quality, detailed, "`
  - Example append: `", sharp focus, 8k"`
- **Fallback Conditioning**: Always provide `fallback_positive` to TiledSampler for safety
- **Preview Captions**: Use `TilePromptPreview` node to inspect generated captions
- **Edit Captions**: Use `PromptListEditor` to manually override specific tile prompts

### New Nodes in This Package

| Node | Purpose |
|------|---------|
| **Florence2 Batch Caption (Tiles)** | Caption each tile with Florence2 |
| **Prompt List -> Conditioning** | Convert prompts to CLIP embeddings |
| **Tiled Sampler with Prompt List** | Sample tiles with per-tile conditioning |
| **Tile Calc Add Positions** | Add position metadata to tile_calc |
| **Tile Prompt Preview** | Preview tiles with their captions |
| **Prompt List Editor** | Manually edit individual tile prompts |

All nodes are automatically registered when this custom node is enabled. See `FLORENCE2_TILES_DESIGN.md` for detailed documentation.

### Sample Workflows

- `example_workflow.json`
- `workflows/florence2_tile_prompt_uprez.json`
- `workflows/simpletiles_uprez_dynamic_upscale.json`
