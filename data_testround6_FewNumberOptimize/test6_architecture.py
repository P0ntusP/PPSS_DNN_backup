#No filter few layers: 97%!!!!!!!!!!!!!!!!!!!!!!!!!!!! with MSE loss
encoded = Dense(units=400, activation='relu')(input_img)
encoded = Dense(units=100, activation='relu')(encoded)
encoded = Dense(units=29, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=29, activation='relu')(encoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=400, activation='relu')(decoded)


0.9343984793285401
input_img = Input(shape=(784,))
# architecture
encoded = Dense(units=300, activation='relu')(input_img)
encoded = Dense(units=100, activation='relu')(encoded)
encoded = Dense(units=29, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=29, activation='relu')(encoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=300, activation='relu')(decoded)
# output layer
decoded = Dense(units=784, activation='sigmoid')(decoded)

#Marcin. 0.6782952395422491
encoded = Dense(units=500, activation='relu')(input_img)
encoded = Dense(units=500, activation='relu')(encoded)
encoded = Dense(units=2000, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=2000, activation='relu')(encoded)
decoded = Dense(units=500, activation='relu')(decoded)
decoded = Dense(units=500, activation='relu')(decoded)

