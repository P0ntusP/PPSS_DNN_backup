#1
encoded = Dense(units=400, activation='relu')(input_img)
encoded = Dense(units=100, activation='relu')(encoded)
encoded = Dense(units=29, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=29, activation='relu')(encoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=400, activation='relu')(decoded)

#2
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


#3
encoded = Dense(units=500, activation='relu')(input_img)
encoded = Dense(units=500, activation='relu')(encoded)
encoded = Dense(units=2000, activation='relu')(encoded)
encoded = Dense(units=bottle_dim, activation='linear')(encoded)

decoded = Dense(units=2000, activation='relu')(encoded)
decoded = Dense(units=500, activation='relu')(decoded)
decoded = Dense(units=500, activation='relu')(decoded)


#4
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
#5