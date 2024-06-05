import numpy as np

def barycentric_weights(x1, y1, x2, y2, x3, y3, x4, y4, xn, yn):
    def area(x1, y1, x2, y2, x3, y3):
        return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    A = area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x3, y3, x4, y4)
    
    A1 = area(xn, yn, x2, y2, x3, y3) + area(xn, yn, x3, y3, x4, y4)
    A2 = area(x1, y1, xn, yn, x3, y3) + area(x1, y1, x3, y3, x4, y4)
    A3 = area(x1, y1, x2, y2, xn, yn) + area(x1, y1, xn, yn, x4, y4)
    A4 = area(x1, y1, x2, y2, x3, y3) + area(x1, y1, x3, y3, xn, yn)
    
    lambda1 = A1 / A
    lambda2 = A2 / A
    lambda3 = A3 / A
    lambda4 = A4 / A
    
    return lambda1, lambda2, lambda3, lambda4

# Example usage
x1, y1 = 0, 0
x2, y2 = 1, 0
x3, y3 = 1, 1
x4, y4 = 0, 1
xn, yn = 0.25, 0.25

lambdas = barycentric_weights(x1, y1, x2, y2, x3, y3, x4, y4, xn, yn)
print(lambdas)