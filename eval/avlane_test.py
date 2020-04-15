# KEVIN O'BRIEN 3/25/20
# REVISED BY EVA MO 4/10/20
#######################

import numpy as np
import torch
import os
import importlib

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import avlane
from erfnet import ERFNet
from erfnet_imagenet import ERFNet as ERFNet_imagenet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval_binary

import visdom


NUM_CHANNELS = 3
NUM_CLASSES = 1 # 20

image_transform = ToPILImage()
input_transform_lot = Compose([
    Resize((512,1024),Image.BILINEAR),
    ToTensor(),
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
label_transform_lot = Compose([
    Resize((512,1024),Image.NEAREST),
    ToLabel(),
])
pred_transform_lot = Compose([
    # Resize((512,1024),Image.NEAREST),
    Relabel(1, 255),
    ToPILImage(),
    Resize((480,640), Image.NEAREST),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    #Import ERFNet model from the folder
    #Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
    if args.pretrainedEncoder:
        pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
        #pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dcit'])
        pretrainedEnc = next(pretrainedEnc.children()).features.encoder
        if (not args.cuda):
            pretrainedEnc = pretrainedEnc.cpu()
        model = ERFNet(NUM_CLASSES, encoder=pretrainedEnc)
    else:
        model = ERFNet(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if args.cuda:
        model = model.cuda()

    #model.load_state_dict(torch.load(args.state))
    #model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    dataset_test = avlane(args.datadir, input_transform_lot, label_transform_lot, 'test')
    loader = DataLoader(dataset_test, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()

    fig = plt.figure()
    ax = fig.gca()
    h = ax.imshow(Image.new('RGB', (640*2, 480), 0))

    print(len(loader.dataset))
    
    iouEvalTest = iouEval_binary(NUM_CLASSES)

    with torch.no_grad():
        for step, (images, labels, filename) in enumerate(loader):

            #print(images.shape)
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = labels
            outputs = model(inputs)

            preds = torch.where(outputs > 0.5, torch.ones([1], dtype=torch.long).cuda(), torch.zeros([1], dtype=torch.long).cuda())
            #preds = torch.where(outputs > 0.5, torch.ones([1], dtype=torch.uint8).cuda(), torch.zeros([1], dtype=torch.uint8).cuda()) # b x 1 x h x w

            #label = outputs[0].max(0)[1].byte().cpu().data
            #label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
            #label_color = Colorize()(label.unsqueeze(0))

            # iou
            iouEvalTest.addBatch(preds[:,0], targets[:,0]) # no_grad handles it already
            iouTest = iouEvalTest.getIoU()
            iouStr = "test IOU: " + '{:0.2f}'.format(iouTest.item()*100) + "%"
            print (iouStr)
            
            # save the output
            filenameSave = os.path.join(args.loadDir,'test_results', filename[0].split("test/")[1])
            os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
            #image_transform(label.byte()).save(filenameSave)      
            #label_save = ToPILImage()(label)  
            pred = preds.to(torch.uint8).squeeze(0).cpu() # 1xhxw
            pred_save = pred_transform_lot(pred)
            #pred_save.save(filenameSave) 
            
            # concatenate data & result
            im1 = Image.open(os.path.join(args.datadir, 'data/test', filename[0].split("test/")[1])).convert('RGB')
            im2 = pred_save
            dst = Image.new('RGB', (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            filenameSaveConcat = os.path.join(args.loadDir,'test_results_concat', filename[0].split("test/")[1])
            os.makedirs(os.path.dirname(filenameSaveConcat), exist_ok=True)
            #dst.save(filenameSaveConcat)
            
            # wrtie iou on dst
            font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 36)
            d = ImageDraw.Draw(dst)
            d.text((900, 0), iouStr, font=font, fill=(255,255,0))

            # show video
            h.set_data(dst)
            plt.draw()
            plt.axis('off')
            plt.pause(1e-2)

            if (args.visualize):         
                vis.image(label_save.numpy())
                
            print (step, filenameSave)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../save/autovalet_training1/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="test")  #can be val, test, train, demoSequence
    parser.add_argument('--pretrainedEncoder') # default="../trained_models/erfnet_encoder_pretrained.pth.tar"

    parser.add_argument('--datadir', default=os.getenv("HOME") + "/Documents/datasets/AVLane/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cuda', action='store_true', default=True)

    parser.add_argument('--visualize', default=False,action='store_true')
    main(parser.parse_args())
