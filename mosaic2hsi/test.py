import torch
import matplotlib.pyplot as plt
import skimage.measure as measure
from torch.utils.data import DataLoader
from model import ResidualNet
from model import ParallelMultiscaleNet
from model import MultiscaleNet
import dataset

dtype = 'float32'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Net define
# Residual
PATH = './trainedWeights/ResNetWeights.pt'
Resnet = ResidualNet.ResNet().to(device)
Resnet.load_state_dict(torch.load(PATH, map_location=device))
Resnet.eval()
# Multiscale
MSnet = MultiscaleNet.MultiscaleNet().to(device)
PATH = './trainedWeights/MultiscaleNetWeights.pt'
MSnet.load_state_dict(torch.load(PATH, map_location=device))
MSnet.eval()
# PMS
PMSnet = ParallelMultiscaleNet.PMSNet().to(device)
PATH = './trainedWeights/PMSNetWeights.pt'
PMSnet.load_state_dict(torch.load(PATH, map_location=device))
PMSnet.eval()

# validation # '.\ExampleData\BGU' # or '.\ExampleData\camp'#
valSet = dataset.MyDataSet('.\ExampleData\BGU')
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=0)
with torch.no_grad():
    for i, data in enumerate(valLoader, 0):
        valmasaicPic, valgdPic, picName = data['masaic'].to(device),\
                                          data['gd'].to(device),\
                                          data['name'][0]
        Resvaloutput = Resnet(valmasaicPic)
        MSvaloutput = MSnet(valmasaicPic)
        PMSvaloutput = PMSnet(valmasaicPic)
        # show the reconstruction of a certain channel
        channel = 22
        fig = plt.figure()
        gd = fig.add_subplot(141)
        gd.title.set_text('Ref')
        gdImg = valgdPic[0, channel, 0:, 0:].cpu().detach().numpy()
        gd.imshow(gdImg, cmap='gray')
        plt.axis('off')
        Resre = fig.add_subplot(142)
        ResreImg = Resvaloutput[0, channel, 0:, 0:].cpu().detach().numpy()
        Respsnr = measure.compare_psnr(gdImg, ResreImg)
        Resre.title.set_text('Res\nPSNR: ' + str(round(Respsnr, 2)))
        Resre.imshow(ResreImg, cmap='gray')
        plt.axis('off')
        MSre = fig.add_subplot(143)
        MSreImg = MSvaloutput[0, channel, 0:, 0:].cpu().detach().numpy()
        MSpsnr = measure.compare_psnr(gdImg, MSreImg)
        MSre.title.set_text('MS\nPSNR: ' + str(round(MSpsnr, 2)))
        MSre.imshow(MSreImg, cmap='gray')
        plt.axis('off')
        PMSre = fig.add_subplot(144)
        PMSre.title.set_text('PMS')
        PMSreImg = PMSvaloutput[0, channel, 0:, 0:].cpu().detach().numpy()
        PMSpsnr = measure.compare_psnr(gdImg, PMSreImg)
        PMSre.title.set_text('PMS\nPSNR: ' + str(round(PMSpsnr, 2)))
        PMSre.imshow(PMSreImg, cmap='gray')
        plt.axis('off')
        plt.ioff()
        plt.show()
