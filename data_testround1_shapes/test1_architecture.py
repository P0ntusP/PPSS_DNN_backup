#Total parameters  in hidden layer is about 1.1 million, layers 13, bottleneck dim for first test round is 4, epochs 200
#batchsize 256. Activation relu in all hidden layers except bottle which is linear. sigmoid in end



#Triangle hidden layers test 1.086 million hidden param in encode layer
#Winner
encoded = Dense(units=700, activation='relu')(input_img)
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


#Square hidden layers test1 1.123 million hidden param in encode layer
encoded = Dense(units=440, activation='relu')(input_img)
encoded = Dense(units=440, activation='relu')(encoded)
encoded = Dense(units=440, activation='relu')(encoded)
encoded = Dense(units=440, activation='relu')(encoded)
encoded = Dense(units=440, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=440, activation='relu')(encoded)
decoded = Dense(units=440, activation='relu')(decoded)
decoded = Dense(units=440, activation='relu')(decoded)
decoded = Dense(units=440, activation='relu')(decoded)
decoded = Dense(units=440, activation='relu')(decoded)

#x² hidden layers test 1.082 million param in encoder layer
#Winner
encoded = Dense(units=794, activation='relu')(input_img)
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

#log hidden layers test 1.114 million param in encoder
encoded = Dense(units=608, activation='relu')(input_img)
encoded = Dense(units=568, activation='relu')(encoded)
encoded = Dense(units=492, activation='relu')(encoded)
encoded = Dense(units=23, activation='relu')(encoded)
encoded = Dense(units=15, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=15, activation='relu')(encoded)
decoded = Dense(units=23, activation='relu')(decoded)
decoded = Dense(units=492, activation='relu')(decoded)
decoded = Dense(units=568, activation='relu')(decoded)
decoded = Dense(units=608, activation='relu')(decoded)
