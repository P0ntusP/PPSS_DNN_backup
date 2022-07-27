import autoencoder_phythia


sigma = 100
# sigma2 test[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 100]


accu = []
scorez = []

res = autoencoder_phythia.autoencoder_pythia(sigma_1=sigma, sigma_2=sigma)
scorez.append(res[0])
accu.append(res[1])

f = open("sigmaboth_score.txt", "a")
score_string = ", ".join(str(e) for e in scorez)
f.write(score_string)
f.write(", ")
f.close()


g = open("sigmaboth_acc.txt", "a")
acc_string = ", ".join(str(e) for e in accu)
g.write(acc_string)
g.write(" ,")
g.close()



