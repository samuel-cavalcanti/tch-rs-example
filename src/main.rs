use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, vision::dataset::Dataset, Device};

// This should rearch at least 94% accuracy.

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn main() -> Result<()> {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device.clone());
    let net = net(&vs.root());

    let dataset = tch::vision::mnist::load_dir("data")?;
    let dataset = dataset_to_device(dataset, &device);

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    let time = std::time::SystemTime::now();
    for epoch in 1..200 {
        let loss = net
            .forward(&dataset.train_images)
            .cross_entropy_for_logits(&dataset.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&dataset.test_images)
            .accuracy_for_logits(&dataset.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}% is cuda: {}",
            epoch,
            f64::try_from(&loss)?,
            100. * f64::try_from(&test_accuracy)?,
            device.is_cuda(),
        );
    }
    let delta_t = time.elapsed()?;
    println!("Training time {} seconds", delta_t.as_secs_f64());
    Ok(())
}

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            IMAGE_DIM,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

fn dataset_to_device(dataset: Dataset, device: &Device) -> Dataset {
    let train_labels = dataset.train_labels.to_device(device.clone());
    let train_images = dataset.train_images.to_device(device.clone());
    let test_labels = dataset.test_labels.to_device(device.clone());
    let test_images = dataset.test_images.to_device(device.clone());

    Dataset {
        test_images,
        test_labels,
        train_images,
        train_labels,
        labels: dataset.labels,
    }
}
