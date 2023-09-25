import random

def perceptron(x1, x2, w1, w2, b):
    return 1 if x1 * w1 + x2 * w2 + b > 0 else -1

def main():
    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    b = random.uniform(-1, 1)
    images = [[0.31565, 0.83101], [0.36484, 0.8518], [0.16975, 0.84049], [0.08838, 0.62068], [0.098166, 0.79092]]
    T = [ 1, 1, 1 , -1, -1 ]
    E = 0
    n = 0.2
    Total_error = 1

    nb_boucle = 0

    while Total_error > 0:
        Total_error = 0
        nb_boucle += 1
        for i in range(0, 5):
            Y = perceptron(images[i][0], images[i][1], w1, w2, b)
            ex = T[i] - Y
            w1 += n * ex * images[i][0]
            w2 += n * ex * images[i][1]
            b += n * ex * 1

        for i in range(0, 5):
            Y = perceptron(images[i][0], images[i][1], w1, w2, b)
            ex = T[i] - Y
            Total_error += abs(ex)
    print("E = ", E, "for nb_boucle = ", nb_boucle)


main()