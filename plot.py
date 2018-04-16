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
    y1.append((100.0)*(float(a[1])/15.0))
    y2.append((100.0)*(float(a[2])/15.0))
    y3.append((100.0)*(float(a[3])/15.0))
areaLabels = ['Rot Forest', 'AdaBoost', 'Bagging', 'Random Forest']
fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3)
loc = y1.index(max(y1))
ax.text(loc+2, y1[loc]*0.25, areaLabels[1])
loc = y2.index(max(y2))
ax.text(loc, y1[loc] + y2[loc]*0.33, areaLabels[2])
loc = y3.index(max(y3))
ax.text(loc, y1[loc] + y2[loc] + y3[loc]*0.5, areaLabels[3])
loc =y3.index(min(y3))
ax.text(loc+2, (y1[loc] + y2[loc] + y3[loc])*1.25, areaLabels[0]) 
ax.set_xlabel("Number of classifiers in ensemble")
axes = plt.gca()
axes.set_xlim(1,35)
axes.set_ylim(0,100)
plt.show()