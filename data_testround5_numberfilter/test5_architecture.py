#epochs 200
#batchsize 256. Activation relu in all hidden layers except bottle which is linear. sigmoid in end

#Test arch for bottleneck dim 2,3,4,5,6,8,10. x² with filter layer
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