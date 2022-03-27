import argparse
import os
import logging
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import GestureData
from model import gesture_cls


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--model', default='ResNet101', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=10, type=int, help='default value for classes of ScanObjectNN')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument('--pretrained_model_path', default='/ResNet101-pretain/model/', type=str, help='model_path')   #   checkpoints/xxx/model/
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
    path_data = os.path.abspath(os.path.dirname(__file__))

    test_path ='checkpoints/' + args.pretrained_model_path
    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(test_path, "testout.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)


    #   Building model
    printf('==> Building model..')
    printf(f"args: {args}")
    net = gesture_cls().to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    #   load model_path

    model_path = test_path+ "best_performance.pth"
    net.load_state_dict(torch.load(model_path))
    testset = GestureData(path_data, train=False, transform=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers ,drop_last=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    printf("Accuracy of the network on test images: %d %%" % (100 * correct / total))



if __name__ == "__main__":
    main()