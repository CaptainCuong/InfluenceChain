# === Standard Library ===
import os
import math
import time
import copy
import argparse

# === Third-party Libraries ===
import cv2
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

# === PyTorch ===
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# === Project Modules ===
from utils import distributed_log
from data import get_metadata, get_dataset, fix_legacy_dict, skewed_mnist, SynthesizedDataset, leave_out_classes_random
import unets


unsqueeze3x = lambda x: x[..., None, None, None]


class GuassianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze3x(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze3x((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze3x(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze3x(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze3x(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze3x((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it to feed class label for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502). With very small number of
            sampling steps, use ddim sampling for better image quality.

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze3x(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()
        return final
    
    def predict_x0_mean_variance(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False
    ):
        """
        Predicts x_0 mean and variance after full reverse process.

        Returns:
            x0_mean: Predicted x0 from final reverse step.
            x0_var: Posterior variance used at final step (x0 space approximation).
        """
        model.eval()
        final = xT

        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            current_t = torch.tensor([t] * len(final), device=final.device)
            current_sub_t = torch.tensor([i] * len(final), device=final.device)

            with torch.no_grad():
                pred_epsilon = model(final, current_t, **model_kwargs)
                pred_x0 = self.get_x0_from_xt_eps(final, pred_epsilon, current_sub_t, scalars)
                pred_mean = self.get_pred_mean_from_x0_xt(final, pred_x0, current_sub_t, scalars)

                if i == 1:
                    # At final step, compute predictive variance:
                    beta_tilde = scalars["beta_tilde"][current_sub_t].view(-1, 1, 1, 1)
                    noise_var = beta_tilde  # scalar variance added in last step

                    # Convert to variance in x0-space (approx):
                    x0_var = noise_var * (1.0 / scalars["alpha_bar"][current_sub_t].view(-1, 1, 1, 1))
                    return pred_mean, x0_var

                if ddim:
                    final = (
                        unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1]).sqrt() * pred_x0 +
                        (1 - unsqueeze3x(scalars["alpha_bar"][current_sub_t - 1])).sqrt() * pred_epsilon
                    )
                else:
                    final = pred_mean + unsqueeze3x(
                        scalars["beta_tilde"][current_sub_t].sqrt()
                    ) * torch.randn_like(final)

                final = final.detach()


def mc_nll(args, model, loader, diffusion, num_mc_samples=10, sampling_steps=250):
    model.to(args.device).eval()
    total_nll = 0.0
    total_samples = 0
    log_two_pi = math.log(2 * math.pi)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n_samples_per_gpu = math.ceil(num_mc_samples / world_size)
    num_mc_samples = n_samples_per_gpu * world_size

    with torch.no_grad():
        pbar = tqdm(total=(len(loader)), desc="Monte Carlo Sampling") if rank == 0 else None
        for imgs, _ in loader:
            imgs = imgs.to(args.device)  # [B, C, H, W]
            imgs = imgs * 2 - 1  # Scale to [-1, 1]
            B, C, H, W = imgs.shape
            log_probs = []
            
            for _ in range(n_samples_per_gpu):
                # Step 1: Sample x_T ~ N(0, I)
                # distributed_log(args, "Step 1: Sampling xT from N(0, I)")
                xT = torch.randn_like(imgs, device=args.device)  # [B, C, H, W]

                # Step 2: Predict mean and variance of x0 given xT
                # distributed_log(args, "Step 2: Predicting x0 mean and variance")
                x0_mean, x0_var = diffusion.predict_x0_mean_variance(model, xT, sampling_steps, model_kwargs={"y": None}, ddim=args.ddim)

                # Step 3: Compute Gaussian log-likelihood log p(x0 | xT)
                # Shape: [B, C, H, W]
                # distributed_log(args, "Step 3: Computing Gaussian log-likelihood")
                log_px = -0.5 * (
                    ((imgs - x0_mean) ** 2) / x0_var + x0_var.log() + log_two_pi
                )
    
                # Step 4: Sum over all dimensions (per image log-prob) → shape: [B]
                # distributed_log(args, "Step 4: Summing over all dimensions")
                log_px = log_px.view(B, -1).sum(dim=1)
                log_probs.append(log_px.unsqueeze(0))  # shape: [1, B]

            # Step 5: Gather log probabilities across all GPUs
            # distributed_log(args, "Step 5: Gathering log probabilities across all GPUs")
            log_probs = torch.cat(log_probs, dim=0) # shape: [n_samples_per_gpu, B]
            gathered_log_probs = [torch.zeros_like(log_probs) for _ in range(world_size)]
            dist.all_gather(gathered_log_probs, log_probs, group=dist.group.WORLD)

            # Step 6: Monte Carlo log marginal likelihood estimate (log-mean-exp)
            log_probs = torch.cat(gathered_log_probs, dim=0) # shape: [num_mc_samples, B]
            log_p = torch.logsumexp(log_probs, dim=0) - math.log(num_mc_samples)

            # Step 6: Accumulate total NLL
            nll = -log_p.sum().item()
            total_nll += nll
            total_samples += B

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    # Step 7: Return average NLL per image
    return total_nll / total_samples

class LossLogger:
    """Logger to track loss values with exponential moving average (EMA)."""

    def __init__(self, max_steps, ema_weight=0.9):
        """
        Args:
            max_steps (int): Total number of training steps.
            ema_weight (float): Weight for EMA smoothing (default: 0.9).
        """
        self.max_steps = max_steps
        self.ema_weight = ema_weight
        self.losses = []
        self.ema_loss = None
        self.start_time = time.time()

    def log(self, value, display=False):
        """
        Log a new loss value and optionally display the EMA loss and elapsed time.

        Args:
            value (float): New loss value to log.
            display (bool): If True, print logging info to stdout.
        """
        self.losses.append(value)

        if self.ema_loss is None:
            self.ema_loss = value
        else:
            self.ema_loss = self.ema_weight * self.ema_loss + (1 - self.ema_weight) * value

        if display:
            elapsed_hours = (time.time() - self.start_time) / 3600
            print(
                f"Step: {len(self.losses):>4}/{self.max_steps}  "
                f"Loss (EMA): {self.ema_loss:.4f}  "
                f"Elapsed Time: {elapsed_hours:.2f} hr"
            )


def train_one_epoch(model, dataloader, diffusion, optimizer, logger, lrs, args):
    """
    Train the model for one epoch on the given dataloader.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        diffusion (DiffusionProcess): Diffusion process with sampling methods.
        optimizer (Optimizer): Optimizer for model parameters.
        logger (LossLogger): Logger for tracking training loss.
        lrs (Scheduler or None): Learning rate scheduler.
        args (Namespace): Training configuration and hyperparameters.
    """
    model.train()

    for step, (images, labels) in enumerate(dataloader):
        # Ensure images are in [0, 1] range before scaling
        assert images.min().item() >= 0 and images.max().item() <= 1

        # Scale images to [-1, 1], move to device
        images = 2 * images.to(args.device) - 1
        labels = labels.to(args.device) if args.class_cond else None

        # Sample timesteps and generate noisy images
        t = torch.randint(low=0, high=diffusion.timesteps, size=(len(images),), dtype=torch.int64).to(args.device)
        xt, eps = diffusion.sample_from_forward_process(images, t)
        pred_eps = model(xt, t, y=labels)

        # Compute MSE loss and backpropagate
        loss = ((pred_eps - eps) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lrs is not None:
            lrs.step()

        # EMA model update (only on rank 0)
        if args.local_rank == 0 and hasattr(args, "ema_dict"):
            new_state = model.state_dict()
            for k, v in args.ema_dict.items():
                args.ema_dict[k] = args.ema_w * v + (1 - args.ema_w) * new_state[k]

            # Log loss every 100 steps
            logger.log(loss.item(), display=(step % 100 == 0))


def generate_initial_noise(batch_size, num_channels, image_size, device, xT=None):
    """Generate or use provided initial noise tensor xT."""
    if xT is not None:
        return xT
    return torch.randn(batch_size, num_channels, image_size, image_size).float().to(device)


def generate_class_labels(batch_size, num_classes, device):
    """Generate random class labels."""
    return torch.randint(num_classes, (batch_size,), dtype=torch.int64).to(device)


def gather_distributed_tensor(tensor, world_size, group):
    """Gather tensors from all processes in distributed setup."""
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered).cpu().numpy()


def postprocess_images(images, target_count):
    """Convert to uint8 format and crop to required count."""
    images = np.concatenate(images, axis=0).transpose(0, 2, 3, 1)  # NCHW → NHWC
    images = (127.5 * (images + 1)).clip(0, 255).astype(np.uint8)
    return images[:target_count]


def postprocess_labels(labels, target_count):
    """Concatenate and crop labels if available."""
    return np.concatenate(labels, axis=0)[:target_count] if labels else None


def sample_N_images(
    N,
    model,
    diffusion,
    xT=None,
    sampling_steps=250,
    batch_size=64,
    num_channels=3,
    image_size=32,
    num_classes=None,
    args=None,
):
    """
    Sample N images from a diffusion model using the reverse diffusion process.

    Returns:
        samples (np.ndarray): (N, H, W, C) uint8 images in [0, 255].
        labels (np.ndarray or None): Class labels if class_cond is True.
    """
    samples, labels = [], []
    num_samples = 0

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.group.WORLD
    total_batches = math.ceil(N / (batch_size * world_size))

    pbar = tqdm(total=total_batches, desc="Sampling") if rank == 0 else None

    while num_samples < N:
        current_xT = generate_initial_noise(batch_size, num_channels, image_size, args.device, xT)

        if args.class_cond:
            y = generate_class_labels(batch_size, num_classes, args.device)
        else:
            y = None

        gen_images = diffusion.sample_from_reverse_process(
            model=model,
            xT=current_xT,
            timesteps=sampling_steps,
            model_kwargs={"y": y},
            ddim=args.ddim,
        )

        gathered_images = gather_distributed_tensor(gen_images, world_size, group)
        samples.append(gathered_images)

        if args.class_cond:
            gathered_labels = gather_distributed_tensor(y, world_size, group)
            labels.append(gathered_labels)

        num_samples += batch_size * world_size
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    final_images = postprocess_images(samples, N)
    final_labels = postprocess_labels(labels, N) if args.class_cond else None

    return final_images, final_labels


def load_pretrained_model(model, ckpt_path, device, delete_keys=None):
    """Load a pretrained checkpoint into the model with optional key deletion."""
    print(f"Loading pretrained model from {ckpt_path}")
    checkpoint = fix_legacy_dict(torch.load(ckpt_path, map_location=device))
    model_state = model.state_dict()

    if delete_keys:
        for key in delete_keys:
            if key in checkpoint:
                print(
                    f"Deleting key '{key}' due to shape mismatch: "
                    f"ckpt ({checkpoint[key].shape}) vs model ({model_state[key].shape})"
                )
                del checkpoint[key]

    model.load_state_dict(checkpoint, strict=False)
    print("Mismatched keys:", set(checkpoint.keys()) ^ set(model_state.keys()))
    print(f"Successfully loaded pretrained model from {ckpt_path}")
    return model


def save_sample_images(images, save_dir, filename):
    """Save a batch of sampled images concatenated horizontally."""
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, filename)
    concatenated = np.concatenate(images, axis=1)[:, :, ::-1]  # Convert to BGR for cv2
    cv2.imwrite(image_path, concatenated)


def save_model_checkpoints(model, ema_dict, args, base_name=None):
    """Save model and EMA checkpoint files."""
    os.makedirs(args.save_dir, exist_ok=True)
    if base_name is None:
        base_name = f"{args.arch}_{args.dataset}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}-experiment-{args.experiment_type}"
    
    model_path = os.path.join(args.save_dir, base_name + ".pt")
    ema_path = os.path.join(args.save_dir, base_name + f"_ema_{args.ema_w}.pt")

    torch.save(model.state_dict(), model_path)
    torch.save(ema_dict, ema_path)


def train_model(args, model, diffusion, train_loader, sampler, metadata, logger):
    """Train a diffusion model with optional pretrained initialization."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Optionally load pretrained model
    if args.pretrained_ckpt:
        return load_pretrained_model(model, args.pretrained_ckpt, args.device, args.delete_keys)

    # Begin training
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, diffusion, optimizer, logger, lrs=None, args=args)

        # # Periodic sampling
        # if epoch % 1 == 0:
        #     sampled_images, _ = sample_N_images(
        #         N=16,
        #         model=model,
        #         diffusion=diffusion,
        #         xT=None,
        #         sampling_steps=args.sampling_steps,
        #         batch_size=args.batch_size,
        #         num_channels=metadata.num_channels,
        #         image_size=metadata.image_size,
        #         num_classes=metadata.num_classes,
        #         args=args,
        #     )
        #     if args.local_rank == 0:
        #         filename = (
        #             f"{args.arch}_{args.dataset}-{args.diffusion_steps}_steps-"
        #             f"{args.sampling_steps}-sampling_steps-experiment_type-{args.experiment_type}-class_condn_{args.class_cond}.png"
        #         )
        #         save_sample_images(sampled_images, args.save_dir, filename)

        # Save model and EMA checkpoints
        if args.local_rank == 0:
            save_model_checkpoints(model, args.ema_dict, args, base_name=metadata.model_name)

    return model

def create_model(args, metadata):
    """Create and return a U-Net model instance based on the given arguments and metadata."""
    num_classes = metadata.num_classes if args.class_cond else None

    model_cls = unets.__dict__[args.arch]
    model = model_cls(
        image_size=metadata.image_size,
        in_channels=metadata.num_channels,
        out_channels=metadata.num_channels,
        num_classes=num_classes,
    )

    return model.to(args.device)

def distributed_setup(args):
    """
    Sets up distributed training if multiple GPUs are available.

    Args:
        args: Argument namespace with attributes like local_rank and batch_size.
        model: The model to be wrapped with DDP if needed.
        train_set: Dataset used for training.

    Returns:
        model: Possibly wrapped in DistributedDataParallel.
        sampler: DistributedSampler if using DDP, else None.
    """
    num_gpus = torch.cuda.device_count()
    is_distributed = num_gpus > 1

    if is_distributed:
        distributed_log(args, f"Using distributed training on {num_gpus} GPUs.")

        args.batch_size //= num_gpus
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        raise Exception("Distributed training requires multiple GPUs.")

def argument_parse():
    """
    Parses command-line arguments for configuring diffusion model training and sampling.
    """
    parser = argparse.ArgumentParser(description="Minimal implementation of diffusion models")

    # === Diffusion Model Settings ===
    diffusion = parser.add_argument_group("Diffusion Model")
    diffusion.add_argument("--arch", type=str, default="UNet", help="Neural network architecture to use")
    diffusion.add_argument("--class-cond", action="store_true", help="Enable class-conditioned diffusion")
    diffusion.add_argument("--diffusion-steps", type=int, default=1000, help="Total diffusion timesteps")
    diffusion.add_argument("--sampling-steps", type=int, default=50, help="Steps for the sampling process")
    diffusion.add_argument("--ddim", action="store_true", help="Use DDIM sampling instead of default")

    # === Dataset Settings ===
    dataset = parser.add_argument_group("Dataset")
    dataset.add_argument("--dataset", type=str, default="mnist", help="Name of dataset to use")
    dataset.add_argument("--data-dir", type=str, default="~/datasets/", help="Root directory for datasets")
    dataset.add_argument("--experiment-type", type=str, default="skew_minor", choices=["skew_minor", "skew_major"], help="Type of experiment to run")

    # === Optimizer Settings ===
    optimizer = parser.add_argument_group("Optimizer")
    optimizer.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU")
    optimizer.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    optimizer.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    optimizer.add_argument("--ema-w", type=float, default=0.9995, help="EMA (exponential moving average) weight")

    # === Sampling / Fine-tuning ===
    sampling = parser.add_argument_group("Sampling / Fine-tuning")
    sampling.add_argument("--pretrained-ckpt", type=str, help="Path to pretrained checkpoint")
    sampling.add_argument("--delete-keys", nargs="+", help="Keys to remove from pretrained model")
    sampling.add_argument("--sampling-only", action="store_true", help="Skip training and only sample")
    sampling.add_argument("--num-sampled-images", type=int, default=11000, help="Number of images to sample")

    # === Miscellaneous Settings ===
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--save-dir", type=str, default="./trained_models/", help="Directory to save outputs")
    misc.add_argument("--local-rank", type=int, default=0, help="Local rank for DDP (set automatically by launcher)")
    misc.add_argument("--seed", type=int, default=112233, help="Random seed for reproducibility")

    # === Experiment Settings ===
    exp = parser.add_argument_group("Experiment")
    exp.add_argument("--influence-sampling-time", type=int, default=100, help="Number of leave-out experiments to run")
    exp.add_argument("--minority-classes", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="List of minority classes for skewed datasets")
    exp.add_argument("--minority-count", type=int, default=100, help="Number of minority class samples")
    exp.add_argument("--majority-count", type=int, default=1000, help="Number of majority class samples")
    exp.add_argument("--val-per-class", type=int, default=10, help="Number of validation samples per class")

    return parser.parse_args()

def main():
    args = argument_parse()
    if args.experiment_type == "skew_minor":
        EXP_NAME = "Minority"
    else:
        EXP_NAME = "Majority"
    args.majority_classes = [i for i in range(10) if i not in args.minority_classes]
    metadata = get_metadata(args.dataset)
    torch.backends.cudnn.benchmark = True
    args.device = f"cuda:{args.local_rank}"
    distributed_setup(args)
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    if args.local_rank == 0:
        print(args)
        print("We are assuming that model input/ouput pixel range is [-1, 1]. Please adhere to it.")

    # Load dataset
    train_set, val_set = skewed_mnist(args, minority_classes=args.minority_classes, minority_count=args.minority_count, majority_count=args.majority_count, val_per_class=args.val_per_class)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    
    leave_out_base = [] # \theta_base_z
    leave_out_syn = [] # \theta_final_z
    pbar = tqdm(total=args.influence_sampling_time, desc=f"{EXP_NAME} Influence Sampling") if args.local_rank == 0 else None
    for _ in range(args.influence_sampling_time):
        
        if args.experiment_type == "skew_minor":
            train_set_leave_out = leave_out_classes_random(train_set, leave_out_classes=args.minority_classes, max_per_class=50)
        else:
            train_set_leave_out = leave_out_classes_random(train_set, leave_out_classes=args.majority_classes, max_per_class=50)

        # Create model and diffusion process
        model_base = create_model(args, metadata)
        metadata.model_name = f"{args.arch}_{args.dataset}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}-experiment-{args.experiment_type}_base"
        diffusion = GuassianDiffusion(args.diffusion_steps, args.device)

        model_base = DDP(model_base, device_ids=[args.local_rank], output_device=args.local_rank)
        sampler = DistributedSampler(train_set_leave_out)
        train_leave_out_loader = DataLoader(
            train_set_leave_out,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

        distributed_log(args, f"Training dataset loaded: Number of batches: {len(train_leave_out_loader)}, Number of images: {len(train_set_leave_out)}")

        logger = LossLogger(len(train_leave_out_loader) * args.epochs)
        args.ema_dict = copy.deepcopy(model_base.state_dict())

        model_base = train_model(args, model_base, diffusion, train_leave_out_loader, sampler, metadata, logger)

        distributed_log(args, "Training complete. Evaluating base model on MC NLL...")
        mc_nll_base = mc_nll(args, model_base, val_loader, diffusion, sampling_steps=args.sampling_steps)
        distributed_log(args, f"MC NLL for base model: {mc_nll_base}")
        leave_out_base.append(mc_nll_base)

        ################### Train model with synthesized images ####################    
        # Synthesize images using the trained model
        syn_train_set, _ = sample_N_images(
                                            N=len(train_set_leave_out),
                                            model=model_base,
                                            diffusion=diffusion,
                                            xT=None,
                                            sampling_steps=args.sampling_steps,
                                            batch_size=args.batch_size,
                                            num_channels=metadata.num_channels,
                                            image_size=metadata.image_size,
                                            num_classes=metadata.num_classes,
                                            args=args,
                                        )
        
        model_syn = create_model(args, metadata)

        syn_train_set = SynthesizedDataset(syn_train_set)

        
        metadata.model_name = f"{args.arch}_{args.dataset}-timesteps_{args.diffusion_steps}-class_condn_{args.class_cond}-experiment-{args.experiment_type}_syn"
        model_syn = DDP(model_syn, device_ids=[args.local_rank], output_device=args.local_rank)
        sampler = DistributedSampler(syn_train_set)
        
        syn_train_loader = DataLoader(
            syn_train_set,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

        distributed_log(args, f"Synthetic training dataset loaded: Number of batches: {len(syn_train_loader)}, Number of images: {len(syn_train_set)}")
        
        logger = LossLogger(len(syn_train_loader) * args.epochs)
        args.ema_dict = copy.deepcopy(model_syn.state_dict())

        model_syn = train_model(args, model_syn, diffusion, syn_train_loader, sampler, metadata, logger)

        distributed_log(args, "Training complete. Evaluating synthetic model on MC NLL...")
        mc_nll_syn = mc_nll(args, model_syn, val_loader, diffusion, sampling_steps=args.sampling_steps)
        distributed_log(args, f"MC NLL for synthetic model: {mc_nll_syn}")
        leave_out_syn.append(mc_nll_syn)

        if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    if args.local_rank == 0:
        leave_out_base = np.array(leave_out_base)
        leave_out_syn = np.array(leave_out_syn)

        # Save results
        np.save(os.path.join(args.save_dir, f"leave_out_base_{args.experiment_type}.npy"), leave_out_base)
        np.save(os.path.join(args.save_dir, f"leave_out_syn_{args.experiment_type}.npy"), leave_out_syn)

        distributed_log(args, f"Leave-out base MC NLL {EXP_NAME} Group: {leave_out_base.mean():.4f} ± {leave_out_base.std():.4f}")
        distributed_log(args, f"Leave-out synthetic MC NLL {EXP_NAME} Group: {leave_out_syn.mean():.4f} ± {leave_out_syn.std():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()