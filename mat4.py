import math
from vec import *

##
class float4x4:

    ##
    def __init__(self, entries = [
        1.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 1.0]):
        self.entries = entries

    ##
    def identity(self):
        self.entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0
        ]

    ##
    @staticmethod
    def translate(translation):
        return float4x4([
            1.0, 0.0, 0.0, translation.x, 
            0.0, 1.0, 0.0, translation.y, 
            0.0, 0.0, 1.0, translation.z, 
            0.0, 0.0, 0.0, 1.0])

    ##
    @staticmethod
    def scale(scale):
        return float4x4([
            scale.x, 0.0, 0.0, 0.0, 
            0.0, scale.y, 0.0, 0.0, 
            0.0, 0.0, scale.z, 0.0, 
            0.0, 0.0, 0.0, 1.0])

    ##
    @staticmethod
    def rotate(q):
        return q.to_matrix()

    ##
    @staticmethod
    def multiply(m0, m1):
        entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        for i in range(0, 4):

            for j in range(0, 4):
                fResult = 0.0

                for k in range(0, 4):
                
                    iIndex0 = (i << 2) + k
                    iIndex1 = (k << 2) + j
                    fResult += (m0.entries[iIndex0] * m1.entries[iIndex1])
                
                entries[(i << 2) + j] = fResult
            
        return float4x4(entries)
        
     ##
    @staticmethod
    def transpose(m):
        return float4x4([
            m.entries[0], m.entries[4], m.entries[8], m.entries[12],
            m.entries[1], m.entries[5], m.entries[9], m.entries[13],
            m.entries[2], m.entries[6], m.entries[10], m.entries[14],
            m.entries[3], m.entries[7], m.entries[11], m.entries[15]
        ])

    ##
    def __mul__(self, m):
        entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        for i in range(0, 4):

            for j in range(0, 4):
                fResult = 0.0

                for k in range(0, 4):
                
                    iIndex0 = (i << 2) + k
                    iIndex1 = (k << 2) + j
                    fResult += (self.entries[iIndex0] * m.entries[iIndex1])
                
                entries[(i << 2) + j] = fResult
            
        return float4x4(entries)

    ##
    @staticmethod
    def concat_matrices(matrices):
        curr_matrix = matrices[0]
        for i in range(1, len(matrices)):
            concat_matrices = float4x4.multiply(curr_matrix, matrices[i])
            curr_matrix = concat_matrices

        return float4x4(curr_matrix.entries)

    ##
    def apply(self, v):
        return float3(
            float3.dot(float3(self.entries[0], self.entries[1], self.entries[2]), v) + self.entries[3],
            float3.dot(float3(self.entries[4], self.entries[5], self.entries[6]), v) + self.entries[7],
            float3.dot(float3(self.entries[8], self.entries[9], self.entries[10]), v) + self.entries[11]
        )

    ##
    def apply2(self, v):
        return float3(
            float3.dot(float3(self.entries[0], self.entries[1], self.entries[2]), v) + self.entries[3],
            float3.dot(float3(self.entries[4], self.entries[5], self.entries[6]), v) + self.entries[7],
            float3.dot(float3(self.entries[8], self.entries[9], self.entries[10]), v) + self.entries[11]
        ), self.entries[12] * v.x + self.entries[13] * v.y + self.entries[14] * v.z + self.entries[15]

    ##
    @staticmethod
    def from_angle_axis(axis, angle):
        fCosAngle = math.cos(angle)
        fSinAngle = math.sin(angle)
        fT = 1.0 - angle

        entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        fCosAngle = math.cos(angle)
        fSinAngle = math.sin(angle)
        fT = 1.0 - fCosAngle

        entries[0] = fT * axis.x * axis.x + fCosAngle
        entries[5] = fT * axis.y * axis.y + fCosAngle
        entries[10] = fT * axis.z * axis.z + fCosAngle

        fTemp0 = axis.x * axis.y * fT
        fTemp1 = axis.z * fSinAngle

        entries[4] = fTemp0 + fTemp1
        entries[1] = fTemp0 - fTemp1

        fTemp0 = axis.x * axis.z * fT
        fTemp1 = axis.y * fSinAngle

        entries[8] = fTemp0 - fTemp1
        entries[2] = fTemp0 + fTemp1

        fTemp0 = axis.y * axis.z * fT
        fTemp1 = axis.x * fSinAngle

        entries[9] = fTemp0 + fTemp1
        entries[6] = fTemp0 - fTemp1

        return float4x4(entries)

    ##
    @staticmethod
    def from_axis_angle(axis, angle):
        fCosAngle = math.cos(angle)
        fSinAngle = math.sin(angle)
        fT = 1.0 - angle

        entries = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        fCosAngle = math.cos(angle)
        fSinAngle = math.sin(angle)
        fT = 1.0 - fCosAngle

        entries[0] = fT * axis.x * axis.x + fCosAngle
        entries[5] = fT * axis.y * axis.y + fCosAngle
        entries[10] = fT * axis.z * axis.z + fCosAngle

        fTemp0 = axis.x * axis.y * fT
        fTemp1 = axis.z * fSinAngle

        entries[4] = fTemp0 + fTemp1
        entries[1] = fTemp0 - fTemp1

        fTemp0 = axis.x * axis.z * fT
        fTemp1 = axis.y * fSinAngle

        entries[8] = fTemp0 - fTemp1
        entries[2] = fTemp0 + fTemp1

        fTemp0 = axis.y * axis.z * fT
        fTemp1 = axis.x * fSinAngle

        entries[9] = fTemp0 + fTemp1
        entries[6] = fTemp0 - fTemp1

        return float4x4(entries)

    ##
    @staticmethod
    def to_angle_axis(m):
        epsilon = 0.01
        epsilon2 = 0.1
        
        if ((math.fabs(m.entries[0]-m.entries[4])< epsilon)
        and (math.fabs(m.entries[2]-m.entries[8])< epsilon)
        and (math.fabs(m.entries[6]-m.entries[9])< epsilon)): 
            
            if ((math.fabs(m.entries[1]+m.entries[4]) < epsilon2)
            and (math.fabs(m.entries[2]+m.entries[8]) < epsilon2)
            and (math.fabs(m.entries[6]+m.entries[9]) < epsilon2)
            and (math.fabs(m.entries[0]+m.entries[5]+m.entries[10]-3) < epsilon2)):
                return [0,1,0,0]
            
            angle = math.pi
            xx = (m.entries[0]+1)/2
            yy = (m.entries[5]+1)/2
            zz = (m.entries[10]+1)/2
            xy = (m.entries[1]+m.entries[4])/4
            xz = (m.entries[2]+m.entries[8])/4
            yz = (m.entries[6]+m.entries[9])/4
            if ((xx > yy) and (xx > zz)):
                if (xx < epsilon):
                    x = 0
                    y = 0.7071
                    z = 0.7071
                else:
                    x = math.sqrt(xx)
                    y = xy/x
                    z = xz/x
                
            elif (yy > zz):
                if (yy< epsilon):
                    x = 0.7071
                    y = 0
                    z = 0.7071
                else:
                    y = math.sqrt(yy)
                    x = xy/y
                    z = yz/y
            else:
                if (zz< epsilon):
                    x = 0.7071
                    y = 0.7071
                    z = 0
                else:
                    z = math.sqrt(zz)
                    x = xz/z
                    y = yz/z

            return [angle, x, y, z]
        
        s = math.sqrt((m.entries[9] - m.entries[6])*(m.entries[9] - m.entries[6])
            +(m.entries[2] - m.entries[8])*(m.entries[2] - m.entries[8])
            +(m.entries[4] - m.entries[1])*(m.entries[4] - m.entries[1]))
        if (math.fabs(s) < 0.001):
             s=1
            
        num = (m.entries[0] + m.entries[5] + m.entries[10] - 1)/2
        if num < -1.0:
            num = -1.0
        elif num > 1.0:
            num = 1.0
        angle = math.acos(num)
        x = (m.entries[9] - m.entries[6])/s
        y = (m.entries[2] - m.entries[8])/s
        z = (m.entries[4] - m.entries[1])/s

        return [angle, x, y, z]

    ##
    @staticmethod
    def invert(m):
        inv = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0] 
        invOut = [
            1.0, 0.0, 0.0, 0.0, 
            0.0, 1.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, 0.0, 
            0.0, 0.0, 0.0, 1.0]
        
        inv[0] = (m.entries[5] * m.entries[10] * m.entries[15] -
            m.entries[5]  * m.entries[11] * m.entries[14] -
            m.entries[9]  * m.entries[6]  * m.entries[15] +
            m.entries[9]  * m.entries[7]  * m.entries[14] +
            m.entries[13] * m.entries[6]  * m.entries[11] -
            m.entries[13] * m.entries[7]  * m.entries[10])
        
        inv[4] = (-m.entries[4]  * m.entries[10] * m.entries[15] +
        m.entries[4]  * m.entries[11] * m.entries[14] +
        m.entries[8]  * m.entries[6]  * m.entries[15] -
        m.entries[8]  * m.entries[7]  * m.entries[14] -
        m.entries[12] * m.entries[6]  * m.entries[11] +
        m.entries[12] * m.entries[7]  * m.entries[10])
        
        inv[8] = (m.entries[4]  * m.entries[9] * m.entries[15] -
        m.entries[4]  * m.entries[11] * m.entries[13] -
        m.entries[8]  * m.entries[5] * m.entries[15] +
        m.entries[8]  * m.entries[7] * m.entries[13] +
        m.entries[12] * m.entries[5] * m.entries[11] -
        m.entries[12] * m.entries[7] * m.entries[9])
        
        inv[12] = (-m.entries[4]  * m.entries[9] * m.entries[14] +
        m.entries[4]  * m.entries[10] * m.entries[13] +
        m.entries[8]  * m.entries[5] * m.entries[14] -
        m.entries[8]  * m.entries[6] * m.entries[13] -
        m.entries[12] * m.entries[5] * m.entries[10] +
        m.entries[12] * m.entries[6] * m.entries[9])
        
        inv[1] = (-m.entries[1]  * m.entries[10] * m.entries[15] +
        m.entries[1]  * m.entries[11] * m.entries[14] +
        m.entries[9]  * m.entries[2] * m.entries[15] -
        m.entries[9]  * m.entries[3] * m.entries[14] -
        m.entries[13] * m.entries[2] * m.entries[11] +
        m.entries[13] * m.entries[3] * m.entries[10])
        
        inv[5] = (m.entries[0]  * m.entries[10] * m.entries[15] -
        m.entries[0]  * m.entries[11] * m.entries[14] -
        m.entries[8]  * m.entries[2] * m.entries[15] +
        m.entries[8]  * m.entries[3] * m.entries[14] +
        m.entries[12] * m.entries[2] * m.entries[11] -
        m.entries[12] * m.entries[3] * m.entries[10])
        
        inv[9] = (-m.entries[0]  * m.entries[9] * m.entries[15] +
        m.entries[0]  * m.entries[11] * m.entries[13] +
        m.entries[8]  * m.entries[1] * m.entries[15] -
        m.entries[8]  * m.entries[3] * m.entries[13] -
        m.entries[12] * m.entries[1] * m.entries[11] +
        m.entries[12] * m.entries[3] * m.entries[9])
        
        inv[13] = (m.entries[0]  * m.entries[9] * m.entries[14] -
        m.entries[0]  * m.entries[10] * m.entries[13] -
        m.entries[8]  * m.entries[1] * m.entries[14] +
        m.entries[8]  * m.entries[2] * m.entries[13] +
        m.entries[12] * m.entries[1] * m.entries[10] -
        m.entries[12] * m.entries[2] * m.entries[9])
        
        inv[2] = (m.entries[1]  * m.entries[6] * m.entries[15] -
        m.entries[1]  * m.entries[7] * m.entries[14] -
        m.entries[5]  * m.entries[2] * m.entries[15] +
        m.entries[5]  * m.entries[3] * m.entries[14] +
        m.entries[13] * m.entries[2] * m.entries[7] -
        m.entries[13] * m.entries[3] * m.entries[6])
        
        inv[6] = (-m.entries[0]  * m.entries[6] * m.entries[15] +
        m.entries[0]  * m.entries[7] * m.entries[14] +
        m.entries[4]  * m.entries[2] * m.entries[15] -
        m.entries[4]  * m.entries[3] * m.entries[14] -
        m.entries[12] * m.entries[2] * m.entries[7] +
        m.entries[12] * m.entries[3] * m.entries[6])
        
        inv[10] = (m.entries[0]  * m.entries[5] * m.entries[15] -
        m.entries[0]  * m.entries[7] * m.entries[13] -
        m.entries[4]  * m.entries[1] * m.entries[15] +
        m.entries[4]  * m.entries[3] * m.entries[13] +
        m.entries[12] * m.entries[1] * m.entries[7] -
        m.entries[12] * m.entries[3] * m.entries[5])
        
        inv[14] = (-m.entries[0]  * m.entries[5] * m.entries[14] +
        m.entries[0]  * m.entries[6] * m.entries[13] +
        m.entries[4]  * m.entries[1] * m.entries[14] -
        m.entries[4]  * m.entries[2] * m.entries[13] -
        m.entries[12] * m.entries[1] * m.entries[6] +
        m.entries[12] * m.entries[2] * m.entries[5])
        
        inv[3] = (-m.entries[1] * m.entries[6] * m.entries[11] +
        m.entries[1] * m.entries[7] * m.entries[10] +
        m.entries[5] * m.entries[2] * m.entries[11] -
        m.entries[5] * m.entries[3] * m.entries[10] -
        m.entries[9] * m.entries[2] * m.entries[7] +
        m.entries[9] * m.entries[3] * m.entries[6])
        
        inv[7] = (m.entries[0] * m.entries[6] * m.entries[11] -
        m.entries[0] * m.entries[7] * m.entries[10] -
        m.entries[4] * m.entries[2] * m.entries[11] +
        m.entries[4] * m.entries[3] * m.entries[10] +
        m.entries[8] * m.entries[2] * m.entries[7] -
        m.entries[8] * m.entries[3] * m.entries[6])
        
        inv[11] = (-m.entries[0] * m.entries[5] * m.entries[11] +
        m.entries[0] * m.entries[7] * m.entries[9] +
        m.entries[4] * m.entries[1] * m.entries[11] -
        m.entries[4] * m.entries[3] * m.entries[9] -
        m.entries[8] * m.entries[1] * m.entries[7] +
        m.entries[8] * m.entries[3] * m.entries[5])
        
        inv[15] = (m.entries[0] * m.entries[5] * m.entries[10] -
        m.entries[0] * m.entries[6] * m.entries[9] -
        m.entries[4] * m.entries[1] * m.entries[10] +
        m.entries[4] * m.entries[2] * m.entries[9] +
        m.entries[8] * m.entries[1] * m.entries[6] -
        m.entries[8] * m.entries[2] * m.entries[5])
        
        det = m.entries[0] * inv[0] + m.entries[1] * inv[4] + m.entries[2] * inv[8] + m.entries[3] * inv[12]
        if abs(det) <= 1.0e-5:
            for i in range(0, 16):
                invOut[i] = 1.0e9
        
        else:
            det = 1.0 / det
            for i in range(0, 16):
                invOut[i] = inv[i] * det
        
        return float4x4(invOut)

    @staticmethod
    def angle_axis_to_euler(axis, angle):
        s = math.sin(angle)
        c = math.cos(angle)
        t = 1-c
        
        if ((axis.x * axis.y * t + axis.z * s) > 0.998):
            heading = 2 * math.atan2(axis.x * math.sin(angle/2), math.cos(angle/2))
            attitude = math.pi/2
            bank = 0
            return
        
        if ((axis.x * axis.y * t + axis.z * s) < -0.998):
            heading = -2 * math.atan2(axis.x * math.sin(angle / 2), math.cos(angle / 2))
            attitude = -math.pi / 2
            bank = 0
            return
        
        heading = math.atan2(axis.y * s - axis.x * axis.z * t , 1 - (axis.y * axis.y+ axis.z * axis.z ) * t)
        attitude = math.asin(axis.x * axis.y * t + axis.z * s)
        bank = math.atan2(axis.x * s - axis.y * axis.z * t , 1 - (axis.x * axis.x + axis.z * axis.z) * t)

        return [attitude, heading, bank]

    @staticmethod
    def euler_to_axis_angle(euler):
        # Assuming the angles are in radians.
        c1 = math.cos(euler.z/2)
        s1 = math.sin(euler.z/2)
        c2 = math.cos(euler.x/2)
        s2 = math.sin(euler.x/2)
        c3 = math.cos(euler.y/2)
        s3 = math.sin(euler.y/2)
        c1c2 = c1*c2
        s1s2 = s1*s2
        w = c1c2*c3 - s1s2*s3
        x = c1c2*s3 + s1s2*c3
        y = s1*c2*c3 + c1*s2*s3
        z = c1*s2*c3 - s1*c2*s3
        angle = 2 * math.acos(w)
        
        # normalize 
        norm = x*x+y*y+z*z
        if norm < 0.001:
            # when all euler angles are zero angle =0 so
            # we can set axis to anything to avoid divide by zero
            x = 1
            y = 0
            z = 0
        else:
            norm = math.sqrt(norm)
            x /= norm
            y /= norm
            z /= norm

        return [z, x, y, angle]

    @staticmethod
    def view_matrix(
        eye_position,
        look_at,
        up):

        dir = look_at - eye_position
        dir = float3.normalize(dir)
        
        tangent = float3.normalize(float3.cross(up, dir))
        binormal = float3.normalize(float3.cross(dir, tangent))
        
        xform_matrix = float4x4(
            entries = [
                tangent.x, tangent.y, tangent.z, 0.0,
                binormal.x, binormal.y, binormal.z, 0.0,
                -dir.x, -dir.y, -dir.z, 0.0,
                0.0, 0.0, 0.0, 1.0
            ])
        
        
        translation_matrix = float4x4(
            entries = [
                1.0, 0.0, 0.0, -eye_position.x,
                0.0, 1.0, 0.0, -eye_position.y,
                0.0, 0.0, 1.0, -eye_position.z,
                0.0, 0.0, 0.0, 1.0
            ])

        return xform_matrix * translation_matrix


    @staticmethod
    def perspective_projection_matrix(
        field_of_view,
        view_width,
        view_height,
        far,
        near):

        d = 1.0 / math.tan(field_of_view * 0.5)
        aspect = view_width / view_height
        one_over_aspect = 1.0 / aspect
        one_over_far_minus_near = 1.0 / (far - near)
        
        m00 = -d * one_over_aspect
        m11 = d
        m22 = -far * one_over_far_minus_near
        m23 = -1.0
        m32 = -far * near * one_over_far_minus_near

        return float4x4(
            entries = [
                m00,    0.0,    0.0,    0.0,
                0.0,    m11,    0.0,    0.0,
                0.0,    0.0,    m22,    m23,
                0.0,    0.0,    m32,    0.0                   
            ]   
        )

    @staticmethod
    def orthographic_projection_matrix(
        left,
        right,
        top,
        bottom,
        far,
        near,
        inverted):

        width = right - left
        height = top - bottom
        far_minus_near = far - near

        m00 = 2.0 / width
        m03 = -(right + left) / (right - left)
        m11 = 2.0 / height
        m13 = -(top + bottom) / (top - bottom)
        m22 = -2.0 / far_minus_near
        m23 = -(far + near) / far_minus_near
        
        if inverted:
            m11 = -m11

        return float4x4(
            entries = [
                m00, 0.0, 0.0, m03,
                0.0, m11, 0.0, m13,
                0.0, 0.0, m22, m23,
                0.0, 0.0, 0.0, 1.0
            ]
        )

    ##
    @staticmethod
    def get_tangent_matrix(normal):
        up = float3(0.0, 1.0, 0.0)
        dp = float3.dot(normal, up)
        if math.fabs(dp) >= 0.985:
            up = float3(0.0, 0.0, 1.0)

        tangent = float3.normalize(float3.cross(up, normal))
        binormal = float3.normalize(float3.cross(normal, tangent))
        xform_matrix = float4x4(
            [
                tangent.x, tangent.y, tangent.z, 0.0,
                binormal.x, binormal.y, binormal.z, 0.0,
                normal.x, normal.y, normal.z, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
        )
        
        return xform_matrix