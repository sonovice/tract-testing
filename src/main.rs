use image::GrayImage;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        .model_for_path("model.onnx")?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 1211, 900)))?
        .into_optimized()?
        .into_runnable()?;

    let input_image = image::open("input.png").unwrap().to_rgb();
    let input_tensor: Tensor =
        tract_ndarray::Array4::from_shape_fn((1, 3, 1211, 900), |(_, c, y, x)| {
            input_image[(x as _, y as _)][c] as f32 / 255.0
        }).into();

    let mut outputs = model.run(tvec!(input_tensor)).unwrap();
    let output = outputs.pop().unwrap(); // Get first and only output
    let output_tensor = output
        .to_array_view::<i64>()
        .unwrap()
        .into_shape((1, 1211, 900)) // Fix tensor shape
        .unwrap()
        .permuted_axes([2, 1, 0]); // CHW -> WHC

    let classes = ["background", "upper_stafflines", "lower_stafflines", "barlines"];
    for (i, name) in classes.iter().enumerate() {
        let class_tensor = output_tensor
            .mapv(|a| (a == i as i64) as u8 * 255)
            .into_raw_vec();
        let output_image = GrayImage::from_raw(900, 1211, class_tensor).unwrap();
        output_image.save(format!("{}.png", name)).unwrap();
    }

    Ok(())
}