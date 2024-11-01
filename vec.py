import math

class float3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def add(v0, v1):
        return float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

    def __add__(self, v):
        return float3(self.x + v.x, self.y + v.y, self.z + v.z)


    @staticmethod
    def subtract(v0, v1):
        return float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

    def __sub__(self, v):
        return float3(self.x - v.x, self.y - v.y, self.z - v.z)

    @staticmethod
    def multiply(v0, v1):
        return float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z)

    def __mul__(self, v):
        return float3(self.x * v, self.y * v, self.z * v)

    def __idiv__(self, v):
        return float3(self.x / v, self.y / v, self.z / v)

    @staticmethod
    def dot(v0, v1):
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

    @staticmethod
    def length(v):
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    @staticmethod
    def normalize(v):
        magnitude = float3.length(v)
        if magnitude == 0.0:
            return float3(0.0, 0.0, 0.0)

        ret = float3(0.0, 0.0, 0.0)
        ret.x = v.x / magnitude
        ret.y = v.y / magnitude
        ret.z = v.z / magnitude

        return ret

    @staticmethod
    def cross(v0, v1):
        return float3(
            v0.y * v1.z - v0.z * v1.y,
            v0.z * v1.x - v0.x * v1.z,
            v0.x * v1.y - v0.y * v1.x)

    @staticmethod
    def scalar_multiply(v, scalar):
        return float3(v.x * scalar, v.y * scalar, v.z * scalar)
     
    ##
    def project(self, v):
        dp = float3.dot(v, self)
        mag_squared = float3.dot(v, v)
        projection = float3(
            v.x * (dp / mag_squared),
            v.y * (dp / mag_squared),
            v.z * (dp / mag_squared))

        return projection