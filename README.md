# Test case for the [tract](https://github.com/sonos/tract) library
This is an experiment on how to wrap a PyTorch-trained ONNX U-Net model into a standalone Rust executable.

## Notes
- Input/Output image size is fixed for now (1211 x 900), but arbitrary sizes can be extended easily.
- Execution is entirely single core! That means rather high latency, but almost linear scalability.
- The model could be embedded in the executable as well using the [include_bytes!](https://doc.rust-lang.org/std/macro.include_bytes.html) macro.
- For the curious: The bundled mini model does some (bad) pixel segmentation on binary sheet music images. The [full model](https://github.com/sonovice/smude/releases) is alot bigger and better.
