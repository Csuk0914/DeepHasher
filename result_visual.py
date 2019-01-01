import matplotlib.pyplot as plt
import pickle

with open("training_hist.txt", "rb") as fp:
    hist = pickle.load(fp)
plt.plot(hist[0], hist[1])
plt.show()