import re
import matplotlib.pyplot as plt

# file="logs/eval/eval-shiftNAS-cifar100-212-loss1/log.txt"
file="logs/search/search-C100-7641-lrrenew/log.txt"


train_acc=[]
test_acc=[]

with open(file, "r") as fd:
    for line in fd:
        match1=re.match(r".* train_acc (.*)", line)
        match2=re.match(r".* valid_acc (.*)", line)

        if match1:
            train_acc.append(float(match1.group(1)))
            # print(train_acc[-1])
        elif match2:
            test_acc.append(float(match2.group(1)))

x=list(range(len(train_acc)))
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
# lns1 = ax.plot(x, train_acc, marker='^', color='green', label='train_acc')
# lns2 = ax.plot(x, test_acc, marker='s', color='red', label='test_acc')
lns1 = ax.plot(x, train_acc, color='green', label='train_acc')
lns2 = ax.plot(x, test_acc, color='red', label='test_acc')

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc = 0)

plt.xlabel("epoch", fontsize=12)
# plt.xlim(-8,-2,1)
plt.ylabel("Accuracy (%)", fontsize=12)
# plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('fig_c100_1.jpg')
# plt.savefig('fig_clean.jpg')
plt.show()
