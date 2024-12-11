import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvVAE(nn.Module):
    """
    Conv1 VAE를 활용하여, fixed window size의 Timeseries Data를 fixed embedding size로 mapping하는 Neural Network
    (batch size, num_steps, num_features) -- [model] --> (batch size, num embeddings)
    """

    def __init__(
            self,
            num_steps=168,
            num_features=3,
            num_hiddens=64,
            num_embeddings=12
    ):
        super(ConvVAE, self).__init__()
        self.num_steps = num_steps
        self.num_features = num_features
        self.num_hiddens = num_hiddens

        self.enc_conv1 = nn.Conv1d(
            in_channels=num_features, out_channels=num_hiddens, kernel_size=3, stride=2, padding=1
        )
        self.enc_conv2 = nn.Conv1d(
            in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=2, padding=1
        )
        self.enc_conv3 = nn.Conv1d(
            in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=2, padding=1
        )

        enc_feature_dim = math.ceil(num_steps / (2 ** 3)) * num_hiddens

        self.fc_mu = nn.Linear(enc_feature_dim, num_embeddings)
        self.fc_logvar = nn.Linear(enc_feature_dim, num_embeddings)

        self.dec_fc = nn.Linear(num_embeddings, enc_feature_dim)

        self.dec_deconv1 = nn.ConvTranspose1d(
            in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_deconv2 = nn.ConvTranspose1d(
            in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_deconv3 = nn.ConvTranspose1d(
            in_channels=num_hiddens, out_channels=num_features, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def encode(self, x):
        """ x 인코딩하기
        """
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))

        # Flatten
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            # 학습 시
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 추론 시
            return mu

    def decode(self, z):
        """ latent vector(z)로부터 x 복원(x_hat)
        """
        h = self.dec_fc(z)
        h = h.view(h.size(0), self.num_hiddens, -1)
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        h = self.dec_deconv3(h)
        return h

    def forward(self, x):
        """ feed forward 처리
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def embed(self, x):
        """ Embed 시에는 Mean Variable만 추론
        """
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))

        h = h.view(h.size(0), -1)
        return self.fc_mu(h)

    def get_loss(self, x):
        """ 손실함수 구하기 """
        x_hat, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x)
        kl_loss = - torch.mean(torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), dim=1))
        return recon_loss, kl_loss
