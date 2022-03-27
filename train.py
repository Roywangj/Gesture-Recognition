import argparse
import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from utils import GestureData
import torchvision.transforms as transform
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
from model import gesture_cls


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--model', default='ResNet101', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message
    path_data = os.path.abspath(os.path.dirname(__file__))
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    model_path = args.checkpoint = 'checkpoints/' + args.model + message + '/model/'
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    #   Building model
    printf(f"args: {args}")
    printf('==> Building model..')
    start_epoch = 0
    net = gesture_cls().to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)

    best_train_acc = 0.
    best_train_acc_avg = 0.
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    #   Preparing data
    print('==> Preparing data..')
    mytrainTransform = transform.Compose([
                                        transform.ToPILImage(),
                                        transform.RandomHorizontalFlip(p=0.5),
                                        transform.RandomGrayscale(p=0.4),
                                        transform.RandomVerticalFlip(p=0.5),
                                        transform.ColorJitter(
                                            brightness=(0, 60), contrast=(0, 20), saturation=(0, 25), hue=(-0.5, 0.5)
                                        ),
                                        transform.ToTensor(),
                                        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
    trainset = GestureData(path_data, train=True, transform=mytrainTransform )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)


    #   train
    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, trainloader, optimizer, criterion, device)  # {"loss", "acc", "acc_avg", "time"}
        scheduler.step()
        if train_out["acc"] > best_train_acc:
            best_train_acc = train_out["acc"]
            torch.save(net.state_dict(),  model_path + "best_performance.pth" )
            printf('save new model ---')
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }



if __name__ == '__main__':
    main()