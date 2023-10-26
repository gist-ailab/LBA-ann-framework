import torch
import numpy as np

from loader.crop_img import CropCOCO

from models.sam_embed import SAM_embed

import time
import os


def main():
    
    epochs = 10
    batch_size = 2
    lr = 1e-4
    
    train_dataset = CropCOCO()
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            pin_memory=True, num_workers=4, shuffle=True)
    
    
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    model = SAM_embed().to(device)
    
    save_path = "checkpoints/{}".format(int(time.time()))
    os.mkdir(save_path)
    # model_parameters = filter(lambda p: p.requires_grad, list(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    
    for i in range(epochs):
        for ep_num, (img1, img2) in enumerate(dataloader):
            img1, img2 = img1.to(device), img2.to(device)
            
            optimizer.zero_grad()
            
            total_img = torch.cat([img1, img2])
            feature = model(total_img)
            
            feature1 = feature[:batch_size]
            feature2 = feature[batch_size:]
            
            loss = cos_sim(feature1, feature2) 
            loss = torch.mean(loss)
            
            loss.backward()
            
            if ep_num % 10 == 0:
                print("{}/{}, Loss : {}".format(ep_num, len(dataloader), loss.item()))

            # print(loss)
            
            optimizer.step()
        torch.save(model.state_dict(), os.path.join(save_path, "epoch{}.pth".format(str(i))))
            
    
main()