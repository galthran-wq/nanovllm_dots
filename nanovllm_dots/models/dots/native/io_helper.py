"""Latent normalize/denormalize (ported from the reference DotsTtsCore.IOHelper).
Stats come from the checkpoint's latent_stats.pt."""
import torch


class IOHelper:
    def __init__(self, latent_stats_path=None):
        if latent_stats_path is not None:
            latent_stats = torch.load(latent_stats_path, weights_only=False)
            self.global_mean = torch.as_tensor(latent_stats["mean"])
            self.global_var = torch.as_tensor(latent_stats["var"])
        else:
            self.global_mean = None
            self.global_var = None

    def normalize(self, x):
        if self.global_mean is not None and self.global_var is not None:
            x = (x - self.global_mean.to(x.device)) / torch.sqrt(self.global_var.to(x.device))
        return x

    def denormalize(self, x):
        if self.global_mean is not None and self.global_var is not None:
            x = x * torch.sqrt(self.global_var.to(x.device)) + self.global_mean.to(x.device)
        return x

    @staticmethod
    def sample_from_latent(latent):
        mean, log_std = latent.chunk(2, 1)
        z = mean + torch.randn_like(mean) * torch.exp(log_std)
        return z.transpose(1, 2)
