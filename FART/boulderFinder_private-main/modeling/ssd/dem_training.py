'''
For the dem data do training

- fill in later

use most of the stuff from experimenting

'''
from torch_snippets import *
import pandas as pd
import torch 
from PIL import Image
from sklearn.model_selection import train_test_split
from modeling.ssd.model_utils import SSD300, MultiBoxLoss
from modeling.ssd.utils import modify_pixels
from torchvision.transforms import v2
    
IMAGE_ROOT = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/'
    
normalize = v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

data_transforms = {
    'train': v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ])
}

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()


class OpenDataset(torch.utils.data.Dataset):
    w, h = 300, 300
    def __init__(self, df, image_dir=IMAGE_ROOT, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.image_infos = df.source.unique()
        logger.info(f'{len(self)} items loaded')

        self.transform = transform
        
    def __getitem__(self, ix):
        # load images and masks
        image_id = self.image_infos[ix]
        img_path = IMAGE_ROOT + image_id
        img = Image.open(img_path).convert("RGB") #This part seems wrong for my dataset

        if self.transform:
            img = self.transform(img)

        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.

        data = df[df['source'] == image_id]
        labels = data['label'].values.tolist()

        data = data[['XMin','YMin','XMax','YMax']].values
        
        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        
        boxes = data.astype(np.uint32).tolist() # convert to absolute coordinates
       
        return img, boxes, labels, image_id

    def collate_fn(self, batch):
        images, boxes, labels, image_names = [], [], [], []
        for item in batch:
            img, image_boxes, image_labels, image_id = item
            img = preprocess_image(img)[None]
            images.append(img)
            boxes.append(torch.tensor(image_boxes).float().to(device)/300.)
            labels.append(torch.tensor([label2target[c] for c in image_labels]).long().to(device))
            image_names.append(image_id)
        images = torch.cat(images).to(device)
        return images, boxes, labels, image_names

    def __len__(self):
        return len(self.image_infos)


# train_ds.image_infos[ix]
# image_id =train_ds.image_infos[ix]
# img_path = find(image_id, train_ds.files)
# img = Image.open(img_path).convert("RGB") #This part seems wrong for my dataset

# if self.transform:
#     img = self.transform(img)

# img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR))/255.




if __name__ == '__main__':

    DATA_ROOT = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/labeled_data/'
    MODEL_LOC = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/models/'
    IMAGE_ROOT = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/'

    df = pd.read_pickle(f'{DATA_ROOT}boulders_dem_tiled.pkl')
    df.rename(columns={'x_min':'XMin',
                       'x_max':'XMax',
                       'y_min':'YMin',
                       'y_max':'YMax'}, inplace=True)

    #Two 'concerning' things.
    #One is how did we get negative values? I think it's edge cases
    #we also get all 0's sometimes
    df[df['XMin'] < 0] = 0
    df[df['YMin'] < 0] = 0

    df = df[~(df['label'] == 0)]

    df = modify_pixels(df)

    DF_RAW = df

    df = df[df['source'].isin(df['source'].unique().tolist())]

    label2target = {l:t+1 for t,l in enumerate(DF_RAW['label'].unique())}
    label2target['background'] = 0
    target2label = {t:l for l,t in label2target.items()}
    background_class = label2target['background']
    num_classes = len(label2target)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trn_ids, val_ids = train_test_split(df.source.unique(), test_size=0.10, random_state=99)
    trn_df, val_df = df[df['source'].isin(trn_ids)], df[df['source'].isin(val_ids)]
    len(trn_df), len(val_df)

    train_ds = OpenDataset(trn_df, transform=data_transforms['train'])

    test_ds = OpenDataset(val_df)
    
    t1, t2, t3, t4 = train_ds[40]

    # something like this next
    train_loader = DataLoader(train_ds, batch_size=32,
                              collate_fn=train_ds.collate_fn, drop_last=True)

    test_loader = DataLoader(test_ds, batch_size=32, 
                             collate_fn=test_ds.collate_fn, drop_last=True)

    def train_batch(inputs, model, criterion, optimizer):
        model.train()
        N = len(train_loader)
        images, boxes, labels, _ = inputs
        _regr, _clss = model(images)
        loss = criterion(_regr, _clss, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
        
    @torch.no_grad()
    def validate_batch(inputs, model, criterion):
        model.eval()
        images, boxes, labels, _ = inputs
        _regr, _clss = model(images)
        loss = criterion(_regr, _clss, boxes, labels)
        return loss

    ##Based on trail and error 40 is where val_loss is minimized and train loss is relatively small
    n_epochs = 200
    model = SSD300(num_classes,device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.00001, max_lr = 0.1, cycle_momentum=False)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device)

    log = Report(n_epochs=n_epochs)

    for epoch in range(1 , n_epochs):
        _n = len(train_loader)
        for ix, inputs in enumerate(train_loader):
            loss = train_batch(inputs, model, criterion, optimizer)
            pos = (epoch + (ix+1)/_n)
            log.record(pos, trn_loss=loss.item(), end='\r')

        _n = len(test_loader)
        for ix,inputs in enumerate(test_loader):
            val_loss = validate_batch(inputs, model, criterion)
            pos = (epoch + (ix+1)/_n)
            log.record(pos, val_loss=val_loss.item(), end='\r')

        # ##Use with preloaded model
        # if(epoch % 10) == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss,
        #         'log':log
        #         }, DATA_ROOT + 'SSD300_DEM_' + str(epoch) +'bf_in_prog.pt')

    #----------------------------------------------------------------------

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'log':log,
                }, MODEL_LOC + 'SSD300_DEM_test_two_200_ep.pt')

    log.plot()
