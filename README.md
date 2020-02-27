# pytorch_tutorial

train with pytorch exercise

```shell script
nohup python -u pytorch_imagenet.py --lr 0.0001 --model-type alexnet --model-structure 1 --checkpoint-path ./cifar10_m1 > log1.txt 2>&1&
nohup python -u pytorch_imagenet.py --model-type VGG8 --decreasing-lr 80,120 --epochs 150 --detail --checkpoint-path ./cifar10_ > cifar10_vgg8_log.txt 2>&1&
nohup python -u pytorch_imagenet.py --model-type VGG8 --Quantized --qbit 4,8 --decreasing-lr 40,60 --epochs 80 --detail --checkpoint-path ./cifar10_w4a8 > cifar10_w4a8_vgg8_log.txt 2>&1&
nohup python -u pytorch_imagenet.py --model-type VGG8 --Quantized --qbit 2,4 --lr 0.0001 --decreasing-lr 40,60 --epochs 80 --detail --checkpoint-path ./cifar10_w2a4 > cifar10_w2a4_vgg8_log.txt 2>&1&
nohup python -u imagenet_pytorx.py --train-batch-size 32 --test-batch-size 32 --lr 0.001 --model-type VGG8 --decreasing-lr 40,60 --epochs 80 --detail --checkpoint-path ./cifar10_crxb_ideal > cifar10_crxb_ideal_vgg8_log.txt 2>&1&
nohup python -u imagenet_pytorx.py --test-batch-size 32 --resume ../model/cifar10_crxb_vgg8_ideal_best.pth > cifar10_crxb_vgg8_ideal_test_log.txt 2>&1&
 ```
