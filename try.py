import KNN
import DT
import k_means
import random_forest
import perceptron
import matplotlib.pyplot as plt
 
data = {'DT':DT.acc*100, 'KNN':KNN.score*100, 'k_means':k_means.acc*100,
        'random_forest':random_forest.final_acc,'perceptron':perceptron.accuracy}
algo = list(data.keys())
acc = list(data.values())
plt.bar(algo, acc, color ='red',width = 0.3)
 
plt.xlabel("algorithm")
plt.ylabel("accuracy")
plt.title("accuracy of all algorithms")
plt.show()
