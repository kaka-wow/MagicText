import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from Unet import UNetModel
from main import DiffusionModel, Min_Max_normalization
def Normalization(data):
    return (data - torch.mean(data)) / (torch.std(data) + 1e-5)
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
    mix = torch.randn(num_samples, img_channels, img_height, img_width, device=device)  # 这里生成高斯噪声，均值为0，方差为1
    name = mix.clone()
    portrait = mix.clone()

    # 去噪过程
    for i in reversed(range(diffusion_model.num_timesteps)):
        t_tensor = torch.full((num_samples,), i, device=device, dtype=torch.long)

        with torch.no_grad():  # 禁用梯度计算以节省显存
            mix_noise = 0.7 * denoise_model_portrait(mix, t_tensor) + 0.3 * denoise_model_name(mix, t_tensor)
            name_noise = denoise_model_name(name, t_tensor)
            portrait_noise = denoise_model_portrait(portrait, t_tensor)

            mix = diffusion_model.reverse_diffusion_process(mix_noise, t_tensor, mix)
            name = diffusion_model.reverse_diffusion_process(name_noise, t_tensor, name)
            portrait = diffusion_model.reverse_diffusion_process(portrait_noise, t_tensor, portrait)

            # 下面三行用来防止均值偏移，可以确保显示图像，但是会引起很多噪点
            mix = Normalization(mix)
            name = Normalization(name)
            portrait = Normalization(portrait)

    if name.dim() == 3:
        mix = mix.unsqueeze(0)
        name = name.unsqueeze(0)
        portrait = portrait.unsqueeze(0)

    return Min_Max_normalization(torch.cat((mix, name, portrait), dim=0))

if __name__ == "__main__":
    diffusion_model = DiffusionModel(batch_size=8, num_timesteps=1000)
    name_model = UNetModel(image_size=[1, 128, 128],
                           in_channels=1,
                           model_channels=32,
                           out_channels=1,
                           num_res_blocks=3,
                           )
    portrait_model = UNetModel(image_size=[1, 128, 128],
                               in_channels=1,
                               model_channels=32,
                               out_channels=1,
                               num_res_blocks=3,
                               )
    name_model.load_state_dict(torch.load('./model_weight/name.pth', map_location='cpu', weights_only=True))
    portrait_model.load_state_dict(torch.load('./model_weight/portrait.pth', map_location='cpu', weights_only=True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name_model.to(device)
    portrait_model.to(device)

    sample_images = generate_samples(diffusion_model, name_model, portrait_model, device, num_samples=1)  # 减少样本数量
    grid = make_grid(sample_images, nrow=2)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title(f'test ')
    plt.axis('off')
    plt.show()