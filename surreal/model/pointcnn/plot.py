#! -*- utf-8 -*-
import matplotlib.pyplot as plt

ftrain = open('../../../../exp/log/train_epoch_acc.txt', 'r')
ftest = open('../../../../exp/log/test_epoch_acc.txt', 'r')

train_accs = []
for line in ftrain.readlines():
    train_accs.append(float(line.strip().split(' ')[2]))

test_accs = []
for line in ftest.readlines():
    test_accs.append(float(line.strip().split(' ')[2]))

x = list(range(100))

plt.plot(x, train_accs, 'b', label='Train')
plt.plot(x, test_accs, 'g', label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="right")
plt.show()