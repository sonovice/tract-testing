use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        .model_for_path("model.onnx")?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 900, 1211)))?
        .into_optimized()?
        .into_runnable()?;

    let img = image::open("input.png").unwrap().to_rgb();
    let image: Tensor =
        tract_ndarray::Array4::from_shape_fn((1, 3, 900, 1211), |(_, c, y, x)| {
            img[(x as _, y as _)][c] as f32 / 255.0
        }).into();

    model.run(tvec!(image))?;
    Ok(())
}