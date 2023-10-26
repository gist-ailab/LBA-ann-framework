import os
import time
import argparse
import warnings

import torch

import numpy as np

warnings.filterwarnings(action='ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



from loader.coco import CocoDataset, collate_fn
from models.ssl_model import InstFrame

def main(exp_name, args):
    is_save = args.save
    if is_save:
        save_path = os.path.join(args.save_dir, '{}'.format(exp_name)).replace(" ", "_")
        save_path = save_path.replace(":", "-")
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        f = open(os.path.join(save_path, "info.txt"), 'w')
        f.write(str(args))

        f.close()
        
    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device("cpu")
    
    train_dataset = CocoDataset(root="/mnt/d/Datasets/coco2017/train2017", 
                                json="/mnt/d/Datasets/coco2017/annotations/instances_train2017.json")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False,
                                               num_workers=4, shuffle=True, collate_fn=collate_fn)
    
        
    model = InstFrame(num_classes=91, args=args)
    parameters = [p for p in model.parameters() if p.requires_grad]


    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # scheduler = load_scheduler(args.scheduler, optimizer, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps)
    
    print_freq = 5
    for epoch in range(args.epochs):
        
        loss_sum = 0
        cls_loss_sum = 0
        seg_loss_sum = 0
        
        model.train()
        for i, data in enumerate(train_loader):
            
            
            lr_scheduler = None
            if epoch == 0:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(train_loader) - 1)

                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                        )

            labeled_image, labeled_target = data
            
    
            labeled_image = list(image.to(args.device) for image in labeled_image)
            labeled_target = [{k: v.to(args.device) for k, v in t.items()} for t in labeled_target]


            optimizer.zero_grad()
            
            output, losses = model(labeled_image, labeled_target)
            # print(losses)
            loss = sum(loss for loss in losses.values())
            
            loss.backward()
            optimizer.step()
            
  
            loss_sum += loss.item()
            loss_avg = loss_sum / (i+1)
            
            if i % print_freq == 0:
                print(loss_avg)
            
            # cls_loss_sum += losses["loss_classifier"].detach().item()
            # cls_loss_avg = cls_loss_sum / (i+1)
            
            # seg_loss_sum += losses["loss_mask"].detach().item()
            # seg_loss_avg = seg_loss_sum / (i+1)
            
            
            # if i % print_freq == 0:
            #     print("Epoch [{}/{}] | Batch [{}/{}] | Loss : {:.6f}, Cls : {:.6f}, Seg : {:.6f}".format(
            #                                                             epoch+1, args.epochs, 
            #                                                             str(i+1).zfill(3), len(train_loader), 
            #                                                             loss_avg, cls_loss_avg, seg_loss_avg))

            
            if lr_scheduler is not None:
                lr_scheduler.step()
                
        scheduler.step()
        print("-"*80)
        
        if args.save:
            torch.save(model.state_dict(), os.path.join(save_path, "epoch{}.pth".format(str(epoch))))    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument("--scheduler", type=str, default="StepLR", help="learning rate scheduler")
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    parser.add_argument("--pretrained", type=bool, default=True, help="small model pretrained by imagenet")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-1)
    parser.add_argument(
        "--lr-steps",
        default=[20, 30],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()
    exp_name = time.strftime('%c', time.localtime(time.time()))

    main(exp_name, args)