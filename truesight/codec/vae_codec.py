"""TrueSight VAE codec (Step 7): GDN autoencoder + hyperprior + discretised Gaussian-mixture entropy model.

    x --g_a--> y --h_a--> z --EntropyBottleneck--> z_hat --h_s--> (w, mu, sigma) per latent element
    y_hat = round(y) is coded with p(y_hat) = sum_k w_k [Phi((y_hat+.5-mu_k)/s_k) - Phi((y_hat-.5-mu_k)/s_k)]
    x_hat = g_s(y_hat)

Rate R = -sum log2 p(y_hat) - sum log2 p(z_hat) (bits an ideal arithmetic coder needs), so minimising the rate is
maximum-likelihood estimation of the entropy model. Loss = bpp + lambda * 255^2 * MSE (a weighted ELBO).
Input height / width must be multiples of 64 (the hyper-encoder downsamples by another factor of 4).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

from .layers import conv, deconv

SCALE_MIN = 0.11          # smallest allowed sigma: keeps the likelihood of a bin finite and the gradient bounded
LOG_SCALE_MAX = 10.0      # exp(10) ~ 2e4 is far beyond any real latent range; guards against fp overflow -> inf/NaN
LAMBDAS = (0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483)   # the six rate-distortion points (Section 7, Step 7)


class GMMConditional(nn.Module):
    """Discretised Gaussian-mixture likelihood for integer-valued (or noisy) latents."""

    def __init__(self, K=3):
        super().__init__()
        self.K = K

    def likelihood(self, y, w_logits, mu, sigma):
        """y: (B,C,H,W); w_logits, mu, sigma: (B,K,C,H,W). Returns p(y) of shape (B,C,H,W), floored at 1e-9."""
        w = F.softmax(w_logits, dim=1)
        sigma = sigma.clamp(min=SCALE_MIN)
        n = torch.distributions.Normal(mu, sigma)
        yk = y.unsqueeze(1)
        p = n.cdf(yk + 0.5) - n.cdf(yk - 0.5)
        return (w * p).sum(dim=1).clamp(min=1e-9)


class TrueSightCodec(nn.Module):
    """N = hidden width, M = latent channels (192 in the guide), K = number of mixture components."""

    DOWN = 64             # required divisibility of H and W

    def __init__(self, N=128, M=192, K=3):
        super().__init__()
        self.N, self.M, self.K = N, M, K
        self.g_a = nn.Sequential(conv(3, N), GDN(N), conv(N, N), GDN(N),
                                 conv(N, N), GDN(N), conv(N, M))
        self.g_s = nn.Sequential(deconv(M, N), GDN(N, inverse=True), deconv(N, N), GDN(N, inverse=True),
                                 deconv(N, N), GDN(N, inverse=True), deconv(N, 3))
        self.h_a = nn.Sequential(conv(M, N, 3, 1), nn.LeakyReLU(inplace=True),
                                 conv(N, N), nn.LeakyReLU(inplace=True), conv(N, N))
        self.h_s = nn.Sequential(deconv(N, M), nn.LeakyReLU(inplace=True),
                                 deconv(M, M * 3 // 2), nn.LeakyReLU(inplace=True),
                                 conv(M * 3 // 2, 3 * K * M, 3, 1))
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gmm = GMMConditional(K)

    def gmm_params(self, z_hat, shape):
        """Mixture parameters for a latent of ``shape`` = (B, M, H, W): (w_logits, mu, sigma), each (B,K,M,H,W)."""
        B, _, H, W = shape
        p = self.h_s(z_hat).view(B, 3, self.K, self.M, H, W)
        return p[:, 0], p[:, 1], torch.exp(p[:, 2].clamp(max=LOG_SCALE_MAX))

    def forward(self, x):
        if x.shape[-1] % self.DOWN or x.shape[-2] % self.DOWN:
            raise ValueError(f"H and W must be multiples of {self.DOWN}, got {tuple(x.shape[-2:])}")
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_lik = self.entropy_bottleneck(z)                  # noise in train, rounding in eval
        w, mu, sigma = self.gmm_params(z_hat, y.shape)
        if self.training:
            y_rate = y + torch.empty_like(y).uniform_(-0.5, 0.5)   # uniform noise: differentiable stand-in for rounding
            y_dec = y + (torch.round(y) - y).detach()              # straight-through rounding for the decoder
        else:
            y_rate = y_dec = torch.round(y)
        y_lik = self.gmm.likelihood(y_rate, w, mu, sigma)
        x_hat = self.g_s(y_dec)
        return {"x_hat": x_hat, "y": y, "y_hat": y_dec, "likelihoods": {"y": y_lik, "z": z_lik}}

    def aux_loss(self):
        """Loss of the entropy-bottleneck quantiles; optimised separately from the main loss."""
        return self.entropy_bottleneck.loss()

    def split_parameters(self):
        """(main_params, aux_params): the aux optimiser owns only the entropy-bottleneck quantiles."""
        aux = [p for n, p in self.named_parameters() if n.endswith(".quantiles")]
        main = [p for n, p in self.named_parameters() if not n.endswith(".quantiles")]
        return main, aux


def bpp_terms(likelihoods, num_px):
    """Bits per pixel contributed by each likelihood tensor: {name: bpp}; their sum is the total rate."""
    return {k: torch.log(l).sum() / (-math.log(2) * num_px) for k, l in likelihoods.items()}


def rd_loss(out, x, lmbda):
    """Rate-distortion loss: returns (loss, bpp, mse). loss = bpp + lambda * 255^2 * MSE."""
    B, _, H, W = x.shape
    bpp = sum(bpp_terms(out["likelihoods"], B * H * W).values())
    mse = F.mse_loss(out["x_hat"], x)
    return lmbda * 255 ** 2 * mse + bpp, bpp, mse
