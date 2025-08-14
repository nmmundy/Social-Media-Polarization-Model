import matplotlib.pyplot as plt

# Data
zPrime = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
gap = [0.0038561934810719964, -4.7136207264517e-16, 0.02300381854383549,
       0.1996977985221863, 0.4745612153363549, 0.6676643222275842,
       0.3163375373429102, 0.25859720098944255, 0.005210837138646454,
       0.04959192850319837, 0.0]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(zPrime, gap, marker='o', linestyle='-', color='b')
plt.xlabel('zPrime')
plt.ylabel('Spectral Gap')
plt.title('Spectral Gap vs zPrime')
plt.grid(True)
plt.show()
