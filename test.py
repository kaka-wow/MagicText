import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image


class SimpleDiffusionModel(nn.Module):
    def __init__(self, num_timesteps=1000):
        super(SimpleDiffusionModel, self).__init__()
        self.num_timesteps = num_timesteps
        self.register_buffer('betas', torch.linspace(1e-4, 2e-2, num_timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

    def forward_diffusion_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def reverse_diffusion_process(self, model_output, t, x):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bars_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        mean = (x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bars_t)) * model_output) / torch.sqrt(alphas_t)
        variance = betas_t
        noise = torch.randn_like(x)

        if t[-1] == 0:
            return mean
        else:
            return mean + torch.sqrt(variance) * noise


# 带类别条件的去噪器模型
class ConditionalDenoiseModel(nn.Module):
    def __init__(self,in_channels=1,out_channels=1):
        super(ConditionalDenoiseModel, self).__init__()
        # 编码器（下采样）
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 解码器（上采样）
        # 修改转置卷积层的输入和输出通道数
        self.dec1 = self.up_conv_block(512, 256)
        self.dec2 = self.up_conv_block(512, 128)
        self.dec3 = self.up_conv_block(256, 64)
        self.dec4 = self.up_conv_block(128, 64)
        # 最终卷积层以确保输出大小为32x32
        self.outc = nn.Conv2d(64, out_channels, kernel_size=2,stride=2)
        self.sigmoid = nn.Sigmoid()


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(kernel_size=2, stride=2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(kernel_size=2, stride=2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(kernel_size=2, stride=2)(enc3))

        # 解码器
        dec1 = self.dec1(enc4)
        dec1 = torch.cat((dec1, enc3), dim=1)
        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)
        dec4 = self.dec4(dec3)
        # 最终卷积层
        out = self.outc(dec4)

        return out


def train(name,
        portrait,
        diffusion_model,
        denoise_model_name,
        denoise_model_portrait,
        optimizer_name,
        optimizer_portrait,
        scheduler_name,
        scheduler_portrait,
        device,
        batch_size,
        num_epochs=1,
        ):
    
    diffusion_model.to(device)
    denoise_model_name.to(device)
    denoise_model_portrait.to(device)
    diffusion_model.eval()  # 扩散模型不需要训练
    image = [name, portrait]
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        for i in range(2):
            image[i] = image[i].to(device)

            # 正向扩散过程
            t = torch.randint(0, diffusion_model.num_timesteps, (batch_size, ), device=device).long()
            x_noisy, _ = diffusion_model.forward_diffusion_sample(image[i], t)
            # 反向扩散过程
            if i == 0:
                predicted_noise = denoise_model_name(x_noisy)
            if i == 1:
                predicted_noise = denoise_model_portrait(x_noisy)

            # 反向传播和优化
            loss = loss_fn(predicted_noise,
                            x_noisy - torch.sqrt(1 - diffusion_model.alpha_bars[t].view(-1, 1, 1, 1)) * image[i])
            if i == 0:
                optimizer_name.zero_grad()
                loss.backward()
                optimizer_name.step()
                scheduler_name.step()
                print(f"NAME Epoch:{epoch},loss:{loss}")
            if i == 1:
                optimizer_portrait.zero_grad()
                loss.backward()
                optimizer_portrait.step()
                scheduler_portrait.step()
                #if epoch % 20 == 0:
                    #print(f"PORTRAIT Epoch:{epoch},loss:{loss}")

        
        # 释放未使用的 GPU 内存
        torch.cuda.empty_cache()

        # 保存一些生成的样本
    with torch.no_grad():
        
        sample_images = generate_samples(diffusion_model, denoise_model_name, denoise_model_portrait, device)
        grid = make_grid(sample_images, nrow=2)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title(f'Generated Samples at Epoch {epoch + 1} ')
        plt.axis('off')
        plt.show()

        torch.save(denoise_model_name.state_dict(), 'name.pth')  
        torch.save(denoise_model_portrait.state_dict(), 'portrait.pth')  



def generate_samples(diffusion_model, denoise_model_name, denoise_model_portrait, device, num_samples=1):
    diffusion_model.to(device)
    denoise_model_name.to(device)
    denoise_model_portrait.to(device)
    diffusion_model.eval()
    denoise_model_name.eval()
    denoise_model_portrait.eval()

    img_channels = 1
    img_height = 128
    img_width = 128
    mix = torch.randn(num_samples, img_channels, img_height, img_width, device=device)
    name = mix
    portrait = mix

    for i in reversed(range(diffusion_model.num_timesteps)):
        t_tensor = torch.full((num_samples,), i, device=device, dtype=torch.long)

        mix_noise = 0.7*denoise_model_portrait(mix) + 0.3*denoise_model_name(mix)
        name_noise = denoise_model_name(name)
        portrait_noise = denoise_model_portrait(portrait)
        
        mix = diffusion_model.reverse_diffusion_process(mix_noise, t_tensor, mix)
        name = diffusion_model.reverse_diffusion_process(name_noise, t_tensor, name)
        portrait = diffusion_model.reverse_diffusion_process(portrait_noise, t_tensor, portrait)

    if name.dim() == 3:
        mix.unsqueeze(0)
        name.unsqueeze(0)
        portrait.unsqueeze(0)

    return nn.functional.sigmoid(torch.cat((mix,name,portrait),dim=0))

 

if __name__ == "__main__":
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize([128,128]),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    name = "./name.png"
    portrait = "./portrait.jpg"

    with Image.open(name) as img:
        name = transform(img.convert('RGB'))
        plt.imshow(np.squeeze(name.numpy()), cmap='gray')
        plt.show()

    with Image.open(portrait) as img:
        portrait = transform(img)
        plt.imshow(np.squeeze(portrait.numpy()), cmap='gray')
        plt.show()

    # 初始化模型和优化器
    diffusion_model = SimpleDiffusionModel()
    #for i in range(10):
    #    x,_ = diffusion_model.forward_diffusion_sample(portrait, i)
    #    plt.imshow(np.squeeze(x.numpy()), cmap='gray')
    #    plt.show()

    denoise_model_name = ConditionalDenoiseModel()
    denoise_model_portrait = ConditionalDenoiseModel()

    optimizer_name = optim.Adam(denoise_model_name.parameters(), lr=0.001)
    optimizer_portrait = optim.Adam(denoise_model_portrait.parameters(), lr=0.001)

    scheduler_name = optim.lr_scheduler.StepLR(optimizer_name, step_size=50, gamma=0.95)
    scheduler_portrait = optim.lr_scheduler.StepLR(optimizer_portrait, step_size=50, gamma=0.99)

    # 训练模型
    device = torch.device("cuda:0") #if torch.cuda.is_available() else "cpu"
    train(name,
          portrait,
          diffusion_model,
          denoise_model_name,
          denoise_model_portrait,
          optimizer_name,
          optimizer_portrait,
          scheduler_name,
          scheduler_portrait,
          device,
          batch_size=8,
          num_epochs=1000,
          )



