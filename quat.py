import math
from vec import *
from mat4 import *

##
class quaternion:

    ##
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    ##
    @staticmethod
    def from_angle_axis(axis, angle):
        return quaternion(
            axis.x * math.sin(angle * 0.5),
            axis.y * math.sin(angle * 0.5),
            axis.z * math.sin(angle * 0.5),
            math.cos(angle * 0.5)
        )
    
    ##
    @staticmethod
    def from_axis_angle(axis, angle):
        return quaternion(
            axis.x * math.sin(angle * 0.5),
            axis.y * math.sin(angle * 0.5),
            axis.z * math.sin(angle * 0.5),
            math.cos(angle * 0.5)
        )

    @staticmethod
    def to_angle_axis(q):
        diff = 1.0 - q.w * q.w
        if diff < 0.0:
            diff = 0.0
        fDenom = math.sqrt(diff)
        ret = float3(q.x, q.y, q.z)
        if fDenom > 0.00001:
            ret.x /= fDenom
            ret.y /= fDenom
            ret.z /= fDenom
           
        if q.w > 1.0:
            q.w = 1.0
        w = 2.0 * math.acos(q.w)
        
        return ret, w

    @staticmethod
    def to_axis_angle(q):
        diff = 1.0 - q.w * q.w
        if diff < 0.0:
            diff = 0.0
        fDenom = math.sqrt(diff)
        ret = float3(q.x, q.y, q.z)
        if fDenom > 0.00001:
            ret.x /= fDenom
            ret.y /= fDenom
            ret.z /= fDenom
           
        if q.w > 1.0:
            q.w = 1.0
        w = 2.0 * math.acos(q.w)
        
        return ret, w

    @staticmethod
    def multiply(q0, q1):
        return quaternion(
            q0.w * q1.x + q0.x * q1.w + q0.y * q1.z - q0.z * q1.y,
            q0.w * q1.y + q0.y * q1.w + q0.z * q1.x - q0.x * q1.z,
            q0.w * q1.z + q0.z * q1.w + q0.x * q1.y - q0.y * q1.x,
            q0.w * q1.w - q0.x * q1.x - q0.y * q1.y - q0.z * q1.z
        )
    
    ##
    def __mul__(self, r):
        ret = quaternion(0.0, 0.0, 0.0, 0.0)
        ret.w = r.w * self.w - r.x * self.x - r.y * self.y - r.z * self.z
        ret.x = r.w * self.x + (r.x * self.w) - r.y * self.z + r.z * self.y
        ret.y = r.w * self.y + r.x * self.z + (r.y * self.w) - (r.z * self.x)
        ret.z = r.w * self.z - (r.x * self.y) + (r.y * self.x) + (r.z * self.w)

        return ret

    ##
    def __add__(self, r):
        return quaternion(self.x + r.x, self.y + r.y, self.z + r.z, self.w + r.w)

    ##
    def __sub__(self, r):
        return quaternion(self.x - r.x, self.y - r.y, self.z - r.z, self.w - r.w)

    ##
    def to_matrix(self):
        fXSquared = self.x * self.x
        fYSquared = self.y * self.y
        fZSquared = self.z * self.z
    
        fXMulY = self.x * self.y
        fXMulZ = self.x * self.z
        fXMulW = self.x * self.w
    
        fYMulZ = self.y * self.z
        fYMulW = self.y * self.w
    
        fZMulW = self.z * self.w
        
        afVal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        afVal[0] = 1.0 - 2.0 * fYSquared - 2.0 * fZSquared
        afVal[1] = 2.0 * fXMulY - 2.0 * fZMulW
        afVal[2] = 2.0 * fXMulZ + 2.0 * fYMulW
        afVal[3] = 0.0
        
        afVal[4] = 2.0 * fXMulY + 2.0 * fZMulW
        afVal[5] = 1.0 - 2.0 * fXSquared - 2.0 * fZSquared
        afVal[6] = 2.0 * fYMulZ - 2.0 * fXMulW
        afVal[7] = 0.0
        
        afVal[8] = 2.0 * fXMulZ - 2.0 * fYMulW
        afVal[9] = 2.0 * fYMulZ + 2.0 * fXMulW
        afVal[10] = 1.0 - 2.0 * fXSquared - 2.0 * fYSquared
        afVal[11] = 0.0
        
        afVal[12] = afVal[13] = afVal[14] = 0.0
        afVal[15] = 1.0
        
        return float4x4(afVal)

    ##
    def normalize(self):
        magnitude = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
        return quaternion(self.x / magnitude, self.y / magnitude, self.z / magnitude, self.w / magnitude)

    ##
    @staticmethod
    def from_euler(euler):
        roll = euler.z
        pitch = euler.x
        yaw = euler.y

        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)

        return quaternion(qx, qy, qz, qw)

    ##
    def to_euler(self):
        sqw = self.w * self.w
        sqx = self.x * self.x
        sqy = self.y * self.y
        sqz = self.z * self.z

        ret = float3(0.0, 0.0, 0.0)

        unit = sqx + sqy + sqz + sqw
        test = self.x * self.y + self.z * self.w
        if(test > 0.499 * unit):
            ret.y = 2.0 * math.atan2(self.x, self.w)
            ret.x = math.pi * 0.5
            ret.z = 0

            return ret
        
        if(test < -0.499 * unit):
            ret.y = -2.0 * math.atan2(self.x, self.w)
            ret.x = -math.pi * 0.5
            ret.z = 0
            return ret
        
        ret.y = math.atan2(2.0 * self.y * self.w - 2 * self.x * self.z, sqx - sqy - sqz + sqw)
        ret.x = math.asin(2.0 * test / unit)
        ret.z = math.atan2(2.0 * self.x * self.w - 2.0 * self.y * self.z, -sqx + sqy - sqz + sqw)

        return ret

    ##
    def conjugate(self):
        return quaternion(-self.x, -self.y, -self.z, self.w)

    ##
    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    ##
    def invert(self):
        return quaternion(-self.x, -self.y, -self.z, self.w)

    ##
    def decompose(self, direction):
        vector = float3(self.x, self.y, self.z)
        projection = vector.project(direction)
        twist = quaternion(projection.x, projection.y, projection.z, self.w)
        twist = twist.normalize()
        twist_inverted = twist.invert()
        swing = self * twist_inverted

        return twist, swing

    ##
    ## see: https://stackoverflow.com/questions/32813626/constrain-pitch-yaw-roll
    def constrain(self, radian):
        max_magnitude = math.sin(radian * 0.5)
        max_magnitude_squared = max_magnitude * max_magnitude
        v = float3.normalize(float3(self.x, self.y, self.z))
        v_magnitude_squared = v.x * v.x + v.y * v.y + v.z * v.z
        
        ret = self
        if v_magnitude_squared > max_magnitude_squared:
            v = v * max_magnitude

            sign = 1.0
            if self.w < 0.0:
                sign = -1.0

            ret.x = v.x
            ret.y = v.y
            ret.z = v.z
            ret.w = math.sqrt(1.0 - max_magnitude_squared) * sign

        return ret

    @staticmethod
    def slerp(a, b, t):
        q = quaternion(0.0, 0.0, 0.0, 1.0)

        cosHalfTheta = (a.w * b.w) + (a.x * b.x) + (a.y * b.y) + (a.z * b.z)
        if math.fabs(cosHalfTheta) >= 1.0:
            q.w = a.w
            q.x = a.x
            q.y = a.y
            q.z = a.z

            return q
        
        halfTheta = math.acos(cosHalfTheta)
        sinHalfTheta = math.sqrt(1.0 - cosHalfTheta * cosHalfTheta)

        if math.fabs(sinHalfTheta) < 0.001:
            q.w = (a.w * 0.5 + b.w * 0.5)
            q.x = (a.x * 0.5 + b.x * 0.5)
            q.y = (a.y * 0.5 + b.y * 0.5)
            q.z = (a.z * 0.5 + b.z * 0.5)

            return q

        ratioA = math.sin((1 - t) * halfTheta) / sinHalfTheta
        ratioB = math.sin(t * halfTheta) / sinHalfTheta

        q.w = (a.w * ratioA + b.w * ratioB)
        q.x = (a.x * ratioA + b.x * ratioB)
        q.y = (a.y * ratioA + b.y * ratioB)
        q.z = (a.z * ratioA + b.z * ratioB)

        return q