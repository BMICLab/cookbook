from torch.utils.data import Dataset
from glob import glob
import sys


class ImgDataset(Dataset):
    def __init__(self, img_fld, gt_fld) -> None:
        super().__init__()

        self.img_fld = img_fld
        self.gt_fld = gt_fld

        self.imgs = sorted(glob(self.img_fld + "/*.nii.gz"))
        self.gts = sorted(glob(self.gt_fld + "/*.nii.gz"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img =   # TODO: Load image
        gt =    # TODO: Load Label

       
        return img, gt




if __name__ == "__main__":

    img_dataset = ImgDataset(
                img_fld="data/imgs",
                gt_fld="data/labels",
            )
    print(len(img_dataset))
    try:
        img,gt = next(iter(img_dataset))
        print("Img: {}\tGT: {}".format(img.shape,gt.shape))
        print("Success")
    except Exception as e: 
        print(e,file=sys.stderr)
