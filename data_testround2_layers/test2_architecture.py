#Test round 2, adding intermediate layers on winners from test1. Resulting on 6 new archs, new parameters added:
#Large filter layer before first hidden layer to see how that effects, removing 2 hidden layers, number 2 and 4.
#Lastly adding one layer between every encoded one with width. dim(n) = (dim(n-1)+dim(n+1))/2.
#parameters that is the same: bottleneck dim for first test round is 4, epochs 200
#batchsize 256. Activation relu in all hidden layers except bottle which is linear. sigmoid in end

#OBS number of hidden parameters are not the same anymore

#X² with intermediate layers: Hidden param= 1.648 million, in nonsym decoded was as in test1
encoded = Dense(units=794, activation='relu')(input_img)
encoded = Dense(units=628, activation='relu')(encoded)
encoded = Dense(units=461, activation='relu')(encoded)
encoded = Dense(units=321, activation='relu')(encoded)
encoded = Dense(units=180, activation='relu')(encoded)
encoded = Dense(units=113, activation='relu')(encoded)
encoded = Dense(units=45, activation='relu')(encoded)
encoded = Dense(units=37, activation='relu')(encoded)
encoded = Dense(units=29, activation='relu')(encoded)
encoded = Dense(units=16, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=16, activation='relu')(encoded)
decoded = Dense(units=29, activation='relu')(decoded)
decoded = Dense(units=37, activation='relu')(decoded)
decoded = Dense(units=45, activation='relu')(decoded)
decoded = Dense(units=113, activation='relu')(decoded)
decoded = Dense(units=180, activation='relu')(decoded)
decoded = Dense(units=321, activation='relu')(decoded)
decoded = Dense(units=461, activation='relu')(decoded)
decoded = Dense(units=628, activation='relu')(decoded)
decoded = Dense(units=794, activation='relu')(decoded)

#X² with removed layers: Hidden param=0.772 million
encoded = Dense(units=794, activation='relu')(input_img)
encoded = Dense(units=180, activation='relu')(encoded)
encoded = Dense(units=29, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=29, activation='relu')(encoded)
decoded = Dense(units=45, activation='relu')(decoded)
decoded = Dense(units=180, activation='relu')(decoded)
decoded = Dense(units=461, activation='relu')(decoded)
decoded = Dense(units=794, activation='relu')(decoded)

#X² with enlarging layer: Hidden param= 3.831 million
encoded = Dense(units=3000, activation='relu')(input_img)
encoded = Dense(units=461, activation='relu')(encoded)
encoded = Dense(units=180, activation='relu')(encoded)
encoded = Dense(units=45, activation='relu')(encoded)
encoded = Dense(units=29, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=29, activation='relu')(encoded)
decoded = Dense(units=45, activation='relu')(decoded)
decoded = Dense(units=180, activation='relu')(decoded)
decoded = Dense(units=461, activation='relu')(decoded)
decoded = Dense(units=794, activation='relu')(decoded)

#Triangle with intermediate layers: Hidden param=1.684 million
encoded = Dense(units=700, activation='relu')(input_img)
encoded = Dense(units=600, activation='relu')(encoded)
encoded = Dense(units=500, activation='relu')(encoded)
encoded = Dense(units=400, activation='relu')(encoded)
encoded = Dense(units=300, activation='relu')(encoded)
encoded = Dense(units=200, activation='relu')(encoded)
encoded = Dense(units=100, activation='relu')(encoded)
encoded = Dense(units=75, activation='relu')(encoded)
encoded = Dense(units=50, activation='relu')(encoded)
encoded = Dense(units=27, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=50, activation='relu')(encoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=300, activation='relu')(decoded)
decoded = Dense(units=500, activation='relu')(decoded)
decoded = Dense(units=700, activation='relu')(decoded)

#Triangle with removed layers: Hidden param= 0.775 million
encoded = Dense(units=700, activation='relu')(input_img)
encoded = Dense(units=300, activation='relu')(encoded)
encoded = Dense(units=50, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=50, activation='relu')(encoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=300, activation='relu')(decoded)
decoded = Dense(units=500, activation='relu')(decoded)
decoded = Dense(units=700, activation='relu')(decoded)


#triangle with enlarging layer: Hidden param= 4.043 million
encoded = Dense(units=3000, activation='relu')(input_img)
encoded = Dense(units=500, activation='relu')(encoded)
encoded = Dense(units=300, activation='relu')(encoded)
encoded = Dense(units=100, activation='relu')(encoded)
encoded = Dense(units=50, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=50, activation='relu')(encoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=300, activation='relu')(decoded)
decoded = Dense(units=500, activation='relu')(decoded)
decoded = Dense(units=700, activation='relu')(decoded)