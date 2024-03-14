# ttnn_functional_whisper Demo

## How to Run

Use `pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_functional_whisper_for_audio_classification[models.experimental.functional_whisper.tt.ttnn_optimized_functional_whisper]` to run the ttnn optimized functional whispher demo for audio classification.

Use `pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_functional_whisper_for_audio_classification[models.experimental.functional_whisper.tt.ttnn_functional_whisper]` to run the ttnn functional whispher demo for audio classification.

Use `pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_functional_whisper_for_conditional_generation[models.experimental.functional_whisper.tt.ttnn_optimized_functional_whisper]` to run the ttnn optimized functional whispher demo for conditional generation.

Use `pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_functional_whisper_for_conditional_generation[models.experimental.functional_whisper.tt.ttnn_functional_whisper]` to run the ttnn functional whispher demo for conditional generation.

## Inputs

Inputs for Audio classification is taken from `google/fleurs` dataset and Inputs for Conditional generation is taken from `hf-internal-testing/librispeech_asr_dummy` dataset.
