import matplotlib.pyplot as plt

maxData = 15
maxEnsemble = 35

inFile = open("compare.txt", "r")
allInts = []
for val in inFile.read().split():
    allInts.append(float(val))
inFile.close()
# print allInts
k = 0
x = []
y1 = []
y2 = []
y3 = []
y4 = []
for i in range(1, maxEnsemble+1):
    a = [0.0,0.0,0.0,0.0]
    for j in range(1, maxData+1):
        # print allInts[k], allInts[k+1], allInts[k+2], allInts[k+3]
        if(allInts[k]>=allInts[k+1] and allInts[k]>=allInts[k+2] and allInts[k]>=allInts[k+3]):
            a[0] += 1.0
        elif(allInts[k+1]>=allInts[k] and allInts[k+1]>=allInts[k+2] and allInts[k+1]>=allInts[k+3]):
            a[1] += 1.0
        elif(allInts[k+2]>=allInts[k+1] and allInts[k+2]>=allInts[k] and allInts[k+2]>=allInts[k+3]):
            a[2] += 1.0
        elif(allInts[k+3]>=allInts[k+1] and allInts[k+3]>=allInts[k+2] and allInts[k+3]>=allInts[k]):
            a[3] += 1.0
        k += 4
    x.append(i)
    y1.append((100.0)*(float(a[0])/15.0))
    y2.append((100.0)*(float(a[1])/15.0))
    y3.append((100.0)*(float(a[2])/15.0))
    y4.append((100.0)*(float(a[3])/15.0))
l1, = plt.plot(x, y1, label='Rot Forest');
l2, = plt.plot(x, y2, label = 'AdaBoost');
l3, = plt.plot(x, y3, label = 'Bagging');
l4, = plt.plot(x, y4, label = 'Random Forest');
plt.legend(handles = [l1, l2, l3, l4])
plt.show()