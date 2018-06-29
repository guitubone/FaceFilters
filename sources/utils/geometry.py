import numpy as np
from math import asin, pi

# Classe para representar um ponto/vetor, facilitar codigo para as funcoes de operacoes geometricas
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return str(self.x) + " " + str(self.y)

    # Soma de dois pointos
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    # Subtracao de dois pointos
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    # Multiplicacao de ponto por escalar
    def __mul__(self, t):
        return Point(self.x * t, self.y * t)

    # Tamanho do vetor (0,0)->(self.x,self.y)
    def len(self):
        return np.sqrt(self.x*self.x + self.y*self.y)

    def tuple(self):
        return int(self.x), int(self.y)

    # Projecao do vetor self em other
    def relative_proj(self, other):
        return dot(self,other)/(other.len() * other.len())

    # Retorna o ponto da linha que liga self a reta representada que passa por a e b
    def intersect_line(self, a, b):
        p = self
        if(a == b):
            return a
        ap = p - a
        ab = b - a
        u = ap.relative_proj(ab)
        return a + ab*u

# Produto escalar entre pontos a e b
def dot(a, b):
    return a.x*b.x + a.y*b.y
