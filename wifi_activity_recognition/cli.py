"""Command-line interface for WiFi activity recognition.

This module provides CLI commands for common tasks like training models,
running inference, and streaming real-time data.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import click

from .utils.config import load_config
from .utils.logging import setup_logging
from .version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="wifi-activity-recognition")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to configuration file"
)
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """Command group for CSI-based activity recognition."""
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    global logger
    logger = logging.getLogger(__name__)

    # Load configuration
    if config:
        ctx.obj["config"] = load_config(config)
    else:
        ctx.obj["config"] = {}

    ctx.obj["verbose"] = verbose


@cli.command()
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
    help="Hardware platform to use",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Hardware configuration file",
)
@click.option(
    "--duration", "-d", default=10, type=int, help="Duration to stream in seconds"
)
@click.option("--output", "-o", type=click.Path(), help="Output file to save CSI data")
@click.pass_context
def stream(
    ctx, hardware: str, config_file: Optional[str], duration: int, output: Optional[str]
):
    """Stream real-time CSI data from hardware."""
    try:
        import time

        from .hardware import CSIReader
        from .utils.io import save_csi_data

        hw_config = load_config(config_file) if config_file else {}
        click.echo(f"Starting CSI stream from {hardware}...")

        reader = CSIReader(hardware, hw_config)
        csi_data_list = []

        with reader:
            start_time = time.time()
            packet_count = 0

            click.echo("Streaming CSI data (press Ctrl+C to stop)...")

            try:
                for csi_data in reader.stream():
                    packet_count += 1
                    csi_data_list.append(csi_data)

                    if ctx.obj["verbose"]:
                        click.echo(f"Packet {packet_count}: {csi_data.shape}")

                    if time.time() - start_time >= duration:
                        break

            except KeyboardInterrupt:
                click.echo("\nStopping stream...")

            if output and csi_data_list:
                save_csi_data(csi_data_list, output)
                click.echo(f"Saved {len(csi_data_list)} packets to {output}")

            click.echo(f"Stream completed. Total packets: {packet_count}")

    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Streaming failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


# ---------------------------------------------------------------------------
# Research workflow helpers
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--hardware", "-h", type=str, help="Hardware platform to use")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Hardware configuration file",
)
@click.option(
    "--duration", "-d", default=5, type=int, help="Duration to collect in seconds"
)
@click.option("--packets", "-p", type=int, help="Number of packets to collect")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Destination file (.h5 or .json) for collected CSI",
)
@click.option(
    "--interactive/--no-interactive", default=False, help="Run interactive wizard"
)
def collect(
    hardware: Optional[str],
    config_file: Optional[str],
    duration: int,
    packets: Optional[int],
    output: str,
    interactive: bool,
):
    """Collect CSI data with optional interactive wizard."""
    try:
        from .hardware import CSIReader, list_supported_hardware
        from .utils.config import load_config
        from .utils.io import save_csi_data

        hw_config = load_config(config_file) if config_file else {}

        # Interactive prompts -------------------------------------------------
        if interactive or not hardware:
            available = list_supported_hardware()
            click.echo("Available hardware:")
            for idx, hw in enumerate(available, start=1):
                click.echo(f"  {idx}. {hw}")
            hardware = click.prompt(
                "Select hardware", type=click.Choice(available), default=available[0]
            )
            duration = click.prompt("Duration (s)", type=int, default=duration)

        if packets is None:
            packets = duration

        if hardware is None:
            raise click.ClickException("Hardware type required")

        required_keys = ["sampling_rate", "channel"]
        for key in required_keys:
            if key not in hw_config:
                hw_config[key] = hw_config.get(key, 1)

        reader = CSIReader(hardware, hw_config)
        collected = []
        with reader:
            stream_iter = reader.stream()
            with click.progressbar(range(packets), label="Collecting") as bar:
                for _ in bar:
                    try:
                        pkt = next(stream_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "CSI stream ended before the requested packet count "
                            f"({packets}) was collected."
                        ) from exc
                    collected.append(pkt)
        if not collected:
            raise RuntimeError("No CSI packets collected")
        save_csi_data(collected, output)
        click.echo(
            click.style(f"Saved {len(collected)} packets to {output}", fg="green")
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Collection failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data directory",
)
@click.option(
    "--labels",
    "-l",
    required=True,
    type=click.Path(exists=True),
    help="Path to labels file",
)
@click.option(
    "--model",
    "-m",
    default="cnn2d",
    type=click.Choice(["cnn2d", "cnn3d", "resnet", "transformer"]),
    help="Model architecture to use",
)
@click.option("--epochs", "-e", default=100, type=int, help="Number of training epochs")
@click.option(
    "--batch-size", "-b", default=32, type=int, help="Batch size for training"
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for trained model",
)
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
    help="Hardware platform used for data collection",
)
@click.pass_context
def train(
    ctx,
    data: str,
    labels: str,
    model: str,
    epochs: int,
    batch_size: int,
    output: str,
    hardware: str,
):
    """Train an activity recognition model."""
    try:
        from .datasets import Dataset
        from .models import create_model
        from .training import Trainer

        click.echo(f"Loading dataset from {data}...")

        # Load dataset
        dataset = Dataset.from_files(
            data_path=data, labels_path=labels, hardware_type=hardware
        )

        click.echo(
            f"Dataset loaded: {len(dataset)} samples, {len(dataset.classes)} classes"
        )
        click.echo(f"Classes: {dataset.classes}")

        # Create model
        click.echo(f"Creating {model} model...")
        in_channels = dataset.input_shape[0] if dataset.input_shape else 1
        model_instance = create_model(
            model,
            num_classes=len(dataset.classes),
            in_channels=in_channels,
        )

        # Setup training
        trainer = Trainer(model=model_instance, dataset=dataset, batch_size=batch_size)

        # Train model
        click.echo(f"Starting training for {epochs} epochs...")

        with click.progressbar(length=epochs, label="Training") as bar:

            def progress_callback(epoch, metrics):
                """Update the CLI progress bar after each training epoch.

                Args:
                    epoch: One-based epoch index reported by the trainer.
                    metrics: Metric dictionary collected for the completed epoch.
                """
                bar.update(1)
                if ctx.obj["verbose"]:
                    click.echo(f"\nEpoch {epoch}: {metrics}")

            trainer.train(epochs=epochs, progress_callback=progress_callback)

        # Save model
        trainer.save_model(output)
        click.echo(f"Model saved to {output}")

        # Print final metrics
        metrics = trainer.get_metrics()
        click.echo("Final training accuracy: {:.3f}".format(metrics["train_accuracy"]))
        click.echo("Final validation accuracy: {:.3f}".format(metrics["val_accuracy"]))

    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Training failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command("autotrain")
@click.option("--data", "-d", required=True, type=click.Path(exists=True))
@click.option("--labels", "-l", required=True, type=click.Path(exists=True))
@click.option(
    "--model", "-m", default="cnn2d", type=click.Choice(["cnn2d", "resnet", "cnn3d"])
)
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
)
@click.option("--epochs", "-e", default=1, type=int, help="Epochs per trial")
@click.option("--learning-rates", default="1e-3", help="Comma-separated learning rates")
@click.option("--batch-sizes", default="32", help="Comma-separated batch sizes")
@click.option(
    "--output", "-o", required=True, type=click.Path(), help="Path to best model"
)
def autotrain(
    data: str,
    labels: str,
    model: str,
    hardware: str,
    epochs: int,
    learning_rates: str,
    batch_sizes: str,
    output: str,
):
    """Automated training with simple hyperparameter search."""
    try:
        import itertools

        from .datasets import Dataset
        from .models import build_model_artifact, create_model
        from .training import Trainer

        dataset = Dataset.from_files(
            data_path=data, labels_path=labels, hardware_type=hardware
        )
        lrs = [float(x) for x in learning_rates.split(",") if x]
        batches = [int(x) for x in batch_sizes.split(",") if x]
        combos = list(itertools.product(lrs, batches))
        best_acc = -1.0
        best_artifact = None

        with click.progressbar(combos, label="Hyperparameter search") as bar:
            for lr, bs in bar:
                model_instance = create_model(
                    model,
                    num_classes=len(dataset.classes),
                    in_channels=dataset.input_shape[0],
                )
                trainer = Trainer(
                    model=model_instance,
                    dataset=dataset,
                    batch_size=bs,
                    learning_rate=lr,
                )
                trainer.train(epochs=epochs)
                acc = trainer.get_metrics().get("val_accuracy", 0.0)
                if acc > best_acc:
                    best_acc = acc
                    best_artifact = build_model_artifact(
                        trainer.model,
                        model_name=model,
                        model_kwargs={
                            "num_classes": len(dataset.classes),
                            "in_channels": dataset.input_shape[0],
                        },
                        metadata={
                            "best_val_accuracy": float(best_acc),
                            "learning_rate": lr,
                            "batch_size": bs,
                        },
                    )
                if best_artifact is None:
                    continue

        if best_artifact is None:
            raise RuntimeError("No successful training runs")
        torch.save(best_artifact, output)
        click.echo(
            click.style("Best validation accuracy: {:.3f}".format(best_acc), fg="green")
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Autotrain failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model file",
)
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to CSI data file for prediction",
)
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
    help="Hardware platform used for data collection",
)
@click.option("--output", "-o", type=click.Path(), help="Output file for predictions")
@click.option(
    "--threshold",
    "-t",
    default=0.5,
    type=float,
    help="Confidence threshold for predictions",
)
@click.pass_context
def predict(
    ctx, model: str, input: str, hardware: str, output: Optional[str], threshold: float
):
    """Run activity prediction on CSI data."""
    try:
        from .inference import ActivityRecognizer
        from .hardware.base import CSIData
        from .models import load_model
        from .utils.io import load_csi_data, save_predictions

        click.echo(f"Loading model from {model}...")
        model_instance = load_model(model)

        click.echo(f"Loading CSI data from {input}...")
        csi_data = load_csi_data(input)
        if not isinstance(csi_data, list) or any(
            not isinstance(sample, CSIData) for sample in csi_data
        ):
            raise ValueError(
                "Prediction input must contain serialized CSIData packets. "
                "Use JSON or HDF5 files created from CSI packet collections."
            )

        click.echo(f"Running predictions on {len(csi_data)} samples...")

        # Create recognizer
        recognizer = ActivityRecognizer(model_instance)

        predictions = []
        confidences = []

        with click.progressbar(csi_data, label="Predicting") as bar:
            for csi_sample in bar:
                activity, confidence = recognizer.predict(csi_sample)

                if confidence >= threshold:
                    predictions.append(activity)
                    confidences.append(confidence)
                else:
                    predictions.append("uncertain")
                    confidences.append(confidence)

        # Display results
        click.echo("\nPrediction Results:")
        unique_activities = {}
        for pred, conf in zip(predictions, confidences):
            if pred not in unique_activities:
                unique_activities[pred] = []
            unique_activities[pred].append(conf)

        for activity, confs in unique_activities.items():
            avg_conf = sum(confs) / len(confs)
            count = len(confs)
            click.echo(
                "  {}: {} samples (avg confidence: {:.3f})".format(
                    activity, count, avg_conf
                )
            )

        # Save predictions if requested
        if output:
            save_predictions(predictions, confidences, output)
            click.echo(f"Predictions saved to {output}")

    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Prediction failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
    help="Hardware platform to use",
)
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model file",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Hardware configuration file",
)
@click.option(
    "--threshold",
    "-t",
    default=0.7,
    type=float,
    help="Confidence threshold for activity detection",
)
@click.option(
    "--window-size",
    "-w",
    default=100,
    type=int,
    help="Window size for activity detection (packets)",
)
@click.pass_context
def live(
    ctx,
    hardware: str,
    model: str,
    config_file: Optional[str],
    threshold: float,
    window_size: int,
):
    """Run live activity recognition from hardware stream."""
    try:
        import time

        from .hardware import CSIReader
        from .inference import StreamingPredictor
        from .models import load_model

        # Load hardware config
        if config_file:
            hw_config = load_config(config_file)
        else:
            hw_config = {}

        click.echo(f"Loading model from {model}...")
        model_instance = load_model(model)

        click.echo(f"Connecting to {hardware}...")
        reader = CSIReader(hardware, hw_config)

        # Create streaming predictor
        predictor = StreamingPredictor(
            model=model_instance,
            window_size=window_size,
            confidence_threshold=threshold,
        )

        click.echo("Starting live activity recognition (press Ctrl+C to stop)...")
        click.echo(f"Confidence threshold: {threshold}")
        click.echo(f"Window size: {window_size} packets")
        click.echo("-" * 50)

        with reader:
            try:
                for csi_data in reader.stream():
                    result = predictor.update(csi_data)

                    if result:
                        activity, confidence, timestamp = result
                        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

                        # Color coding based on confidence
                        if confidence >= 0.9:
                            color = "green"
                        elif confidence >= 0.7:
                            color = "yellow"
                        else:
                            color = "red"

                        click.echo(
                            f"[{time_str}] "
                            + click.style(f"{activity}", fg=color, bold=True)
                            + " (confidence: {:.3f})".format(confidence)
                        )

            except KeyboardInterrupt:
                click.echo("\nStopping live recognition...")

    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Live recognition failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
)
@click.option(
    "--model", "-m", type=click.Path(exists=True), help="Optional model for predictions"
)
@click.option("--num-packets", "-n", default=10, type=int, help="Packets to visualize")
@click.option("--save", "-s", type=click.Path(), help="Save final heatmap to file")
@click.option("--config-file", "-c", type=click.Path(exists=True))
def visualize(
    hardware: str,
    model: Optional[str],
    num_packets: int,
    save: Optional[str],
    config_file: Optional[str],
):
    """Visualize CSI packets and optional predictions in real time."""
    try:
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .hardware import CSIReader
        from .inference import ActivityRecognizer
        from .models import load_model
        from .utils.config import load_config
        from .utils.visualization import plot_csi_heatmap

        hw_config = load_config(config_file) if config_file else {}
        reader = CSIReader(hardware, hw_config)
        if model:
            recognizer = ActivityRecognizer(load_model(model))
        else:
            recognizer = None

        packets: list[np.ndarray] = []
        with reader:
            with click.progressbar(range(num_packets), label="Streaming") as bar:
                for _ in bar:
                    pkt = next(reader.stream())
                    if recognizer is not None:
                        act, conf = recognizer.predict(pkt)
                        color = "green" if conf > 0.5 else "yellow"
                        click.echo(
                            click.style("{} ({:.2f})".format(act, conf), fg=color)
                        )
                    packets.append(pkt.amplitude.mean(axis=(0, 1)))

        arr = np.stack(packets)
        ax = plot_csi_heatmap(arr)
        if save:
            ax.figure.savefig(save)
            click.echo(click.style(f"Saved visualization to {save}", fg="green"))
        plt.close(ax.figure)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Visualization failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option(
    "--hardware",
    "-h",
    type=click.Choice(["intel_5300", "esp32", "atheros", "all"]),
    default="all",
    help="Hardware platform to show info for",
)
def info(hardware: str):
    """Show information about supported hardware and models."""
    from .hardware import CSIReader, get_hardware_info, list_supported_hardware
    from .models import list_available_models

    click.echo("WiFi Activity Recognition - System Information")
    click.echo("=" * 50)

    # Package info
    click.echo(f"Version: {__version__}")
    click.echo()

    # Supported hardware
    supported_hw = list_supported_hardware()
    click.echo("Supported Hardware:")

    if hardware == "all":
        for hw in supported_hw:
            info = get_hardware_info(hw)
            click.echo(f"  • {info.get('name', hw)}")
            if info.get("notes"):
                click.echo(f"    {info['notes']}")
    else:
        if hardware in supported_hw:
            info = get_hardware_info(hardware)
            click.echo(f"  Hardware: {info.get('name', hardware)}")
            click.echo(f"  Subcarriers: {info.get('subcarriers', 'Unknown')}")
            click.echo(f"  Max Antennas: {info.get('max_antennas', 'Unknown')}")
            click.echo(
                f"  Sampling Rate: {info.get('typical_sampling_rate', 'Unknown')} Hz"
            )
            click.echo(
                f"  Bandwidth Options: {info.get('bandwidth_options', 'Unknown')} MHz"
            )
            if info.get("notes"):
                click.echo(f"  Notes: {info['notes']}")
        else:
            click.echo(f"  Hardware '{hardware}' not supported")

    click.echo()

    # Available models
    try:
        models = list_available_models()
        click.echo("Available Pre-trained Models:")
        for model_name, model_info in models.items():
            click.echo(
                f"  • {model_name}: {model_info.get('description', 'No description')}"
            )
    except Exception:
        click.echo("Available Models: Check documentation")

    click.echo()

    # System status
    click.echo("System Status:")

    # Check dependencies
    try:
        import torch

        click.echo(f"  ✓ PyTorch: {torch.__version__}")
    except ImportError:
        try:
            import tensorflow as tf

            click.echo(f"  ✓ TensorFlow: {tf.__version__}")
        except ImportError:
            click.echo("  ✗ No ML backend found (install PyTorch or TensorFlow)")

    try:
        import cv2

        click.echo(f"  ✓ OpenCV: {cv2.__version__}")
    except ImportError:
        click.echo("  ✗ OpenCV not found")

    # Check hardware availability
    click.echo()
    click.echo("Hardware Status:")
    for hw in supported_hw:
        try:
            reader = CSIReader(hw, {})
            status = "Available" if reader else "Not detected"
            click.echo(f"  {hw}: {status}")
        except Exception:
            click.echo(f"  {hw}: Not available")


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--data", "-d", required=True, type=click.Path(exists=True))
@click.option("--labels", "-l", required=True, type=click.Path(exists=True))
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
)
@click.option("--output", "-o", required=True, type=click.Path())
def benchmark(model: str, data: str, labels: str, hardware: str, output: str) -> None:
    """Run accuracy, latency, and memory benchmarks."""
    try:
        import torch

        from benchmarks.performance_report import generate_performance_report

        from .datasets import Dataset
        from .hardware.base import CSIData
        from .models import load_model
        from .training import Trainer

        dataset = Dataset.from_files(
            data_path=data, labels_path=labels, hardware_type=hardware
        )
        model_instance = load_model(model)
        trainer = Trainer(model_instance, dataset)
        loader = trainer.val_loader

        sample = next(iter(loader))[0][0]
        flat = sample.reshape(-1)
        packet = CSIData(
            timestamp=0.0,
            amplitude=flat.numpy()[None, None, :],
            phase=flat.numpy()[None, None, :],
            frequency=0.0,
            bandwidth=0.0,
            n_tx=1,
            n_rx=1,
            n_subcarriers=flat.numel(),
        )

        def predictor(csi):
            """Run the benchmark predictor on a single CSI packet.

            Args:
                csi: Benchmark CSI packet converted into model input.

            Returns:
                Raw model output tensor for the packet.
            """
            return model_instance(torch.tensor(csi.amplitude).float())

        packets = [packet]

        def consumer(pkt_iter):
            """Materialize a packet iterator for memory benchmarking.

            Args:
                pkt_iter: Iterable of packets produced by the benchmark harness.

            Returns:
                List containing all packets yielded by the iterator.
            """
            return list(pkt_iter)

        generate_performance_report(
            model_instance,
            {"val": loader},
            predictor,
            packets,
            consumer,
            output,
        )
        click.echo(click.style(f"Benchmark report saved to {output}", fg="green"))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Benchmark failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model",
)
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation data array (.npy)",
)
@click.option(
    "--labels",
    "-l",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation label array (.npy)",
)
@click.option(
    "--hardware",
    "-h",
    required=True,
    type=click.Choice(["intel_5300", "esp32", "atheros"]),
    help="Hardware platform used for data",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output file for evaluation results"
)
@click.pass_context
def evaluate(
    ctx,
    model: str,
    data: str,
    labels: str,
    hardware: str,
    output: Optional[str],
):
    """Evaluate model performance on test dataset."""
    try:
        from .datasets import Dataset
        from .models import load_model
        from .training import Trainer
        from .utils.io import save_evaluation_results

        click.echo(f"Loading model from {model}...")
        model_instance = load_model(model)

        click.echo(f"Loading evaluation dataset from {data}...")
        dataset = Dataset.from_files(
            data_path=data,
            labels_path=labels,
            hardware_type=hardware,
        )

        click.echo(f"Evaluating on {len(dataset.test[0])} test samples...")

        trainer = Trainer(model_instance, dataset)
        results = trainer.evaluate(split="test")

        # Display results
        click.echo("Evaluation Results:")
        click.echo("  Accuracy: {:.3f}".format(results["accuracy"]))
        click.echo("  Precision: {:.3f}".format(results["precision"]))
        click.echo("  Recall: {:.3f}".format(results["recall"]))
        click.echo("  F1-Score: {:.3f}".format(results["f1_score"]))

        # Per-class results
        if "per_class_metrics" in results:
            click.echo("\nPer-class Results:")
            for class_name, metrics in results["per_class_metrics"].items():
                click.echo("  {}:".format(class_name))
                click.echo("    Precision: {:.3f}".format(metrics["precision"]))
                click.echo("    Recall: {:.3f}".format(metrics["recall"]))
                click.echo("    F1-Score: {:.3f}".format(metrics["f1_score"]))

        # Save results if requested
        if output:
            save_evaluation_results(results, output)
            click.echo(f"\nResults saved to {output}")

    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Evaluation failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option(
    "--target", "-t", required=True, type=click.Choice(["mobile", "edge", "cloud"])
)
@click.option("--input-shape", required=True, help="Input tensor shape, e.g. 1,1,8,8")
@click.option("--output", "-o", required=True, type=click.Path())
def export(model: str, target: str, input_shape: str, output: str) -> None:
    """Export a trained model for deployment."""
    try:
        from deployment.edge.optimization import convert_to_onnx, quantize_dynamic

        from .models import load_model

        model_instance = load_model(model)
        shape = tuple(int(s) for s in input_shape.split(","))
        import torch

        sample = torch.randn(*shape)
        convert_to_onnx(model_instance, sample, Path(output))
        if target == "edge":
            quantize_dynamic(model_instance)
        click.echo(click.style(f"Exported model to {output}", fg="green"))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Export failed: %s", exc)
        raise click.ClickException(str(exc)) from exc


def main():
    """Run the CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
