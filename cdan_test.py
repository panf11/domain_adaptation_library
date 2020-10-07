import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.transforms import ResizeImage
from tools.lr_scheduler import StepwiseLR
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='logs/cdan_test1')


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        ResizeImage(512),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=(-0.1, 0.1)),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        ResizeImage(512),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, download=False, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=False, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task='val', download=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, evaluate=True, download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_dataset = dataset(root=args.root, task='test', download=False, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    print(train_source_iter.__len__())
    print(train_target_iter.__len__())

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = train_source_dataset.num_classes
    classifier = ImageClassifier(backbone, num_classes).to(device)
    classifier_feature_dim = classifier.features_dim

    domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()

    # define optimizer and lr scheduler
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=False
    ).to(device)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args, epoch)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
            torch.save({'epoch': epoch + 1,
                        'state_dict': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': acc1}, 'weights/windmodel_{}_{:.3f}.pth'.format(epoch+1, acc1))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier, args, epoch)
    print("test_acc1 = {:3.1f}".format(acc1))
    writer.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_adv: ConditionalDomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_losses = AverageMeter('Source Cls Loss', ':6.2f')
    cls_losses_target = AverageMeter('Target cls loss', '3.1f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    cls_accs_target = AverageMeter('Target Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        cls_loss_target = F.cross_entropy(y_t, labels_t)
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + cls_loss_target + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]
        cls_acc_target = accuracy(y_t, labels_t)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        cls_accs_target.update(cls_acc_target.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
        losses.update(loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        cls_losses_target.update(cls_loss_target.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    writer.add_scalar('Loss/Train', losses.avg, epoch)
    writer.add_scalar('Loss_source_class/Train', cls_losses.avg, epoch)
    writer.add_scalar('Loss_target_class/Train', cls_losses_target.avg, epoch)
    writer.add_scalar('Loss_domain_confusion/Train', trans_losses.avg, epoch)
    writer.add_scalar('Accuracy_class/Train', cls_accs.avg, epoch)
    writer.add_scalar('Accuracy_class_target/Train', cls_accs_target.avg, epoch)
    writer.add_scalar('Accuracy_domain/Train', domain_accs.avg, epoch)
    writer.add_scalar('Learning Rate', lr_scheduler.get_lr(), epoch)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace, epoch: int) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    writer.add_scalar('Loss/Validation', losses.avg, epoch)
    writer.add_scalar('Accuracy_class_target/Validation', top1.avg, epoch)
    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--entropy', default=False, action='store_true', help='use entropy conditioning')

    args = parser.parse_args()
    print(args)
    main(args)

# CUDA_VISIBLE_DEVICES=0 python examples/cdan.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20
# CUDA_VISIBLE_DEVICES=0 python cdan_wind.py data/wind -d Wind -s R -t C -a resnet50 -b 16 --epochs 20 -i 2500
