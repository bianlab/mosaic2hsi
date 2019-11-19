import scipy.io as sio
import os
from torch.utils.data import Dataset

dtype = 'float32'


# data set
class MyDataSet(Dataset):
    def __init__(self, dataPath, transform=None):
        if not os.path.isdir(dataPath):
            print('Error: "', dataPath,
                  '" is not a directory or does not exist.')
            return
        self.picName = os.listdir(dataPath)
        self.picNum = len(self.picName)
        self.transform = transform
        self.dataPath = dataPath

    def __len__(self):
        return self.picNum

    def __getitem__(self, idx):
        filePath = self.dataPath + '\\' + self.picName[idx]
        tempData = sio.loadmat(filePath)['dataMat'].astype(dtype)
        masaic = tempData[0:3, 0:, 0:]
        gd = tempData[3:, 0:, 0:]
        sample = {'masaic': masaic, 'gd': gd, 'name': self.picName[idx]}
        if self.transform:
            sample.transform(sample)

        return sample

