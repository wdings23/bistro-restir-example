from mat4 import *
from quat import *

class Joint:
    
    ##
    def __init__(self, name, rotation, scale, translation):
        self.name = name
        self.rotation = rotation
        self.scale = scale
        self.translation = translation
        self.children = []
        self.parent = None
        self.is_root_joint = False
        
        rotation_matrix = float4x4.rotate(rotation)
        translation_matrix = float4x4.translate(translation)
        scale_matrix = float4x4.scale(scale)
        
        #translation_rotation_matrix = float4x4.multiply(translation_matrix, rotation_matrix)
        #self.local_matrix = float4x4.multiply(translation_rotation_matrix, scale_matrix)

        self.local_matrix = float4x4.concat_matrices([translation_matrix, rotation_matrix, scale_matrix])

        self.total_matrix = float4x4()
        self.anim_matrix = float4x4()
        self.total_anim_matrix = float4x4()

        self.total_matrix.identity()
        self.anim_matrix.identity()
        self.total_anim_matrix.identity()

        self.bind_matrix = float4x4()
        self.inverse_bind_matrix = float4x4()

        self.move_orientation = quaternion(x = 0.0, y = 0.0, z = 0.0, w = 1.0)

        self.local_rotation = quaternion(0.0, 0.0, 0.0, 1.0)
        self.local_translation = float3(0.0, 0.0, 0.0)

        self.matching_anim_rotation = quaternion(0.0, 0.0, 0.0, 1.0)
        self.matching_anim_translation = float3(0.0, 0.0, 0.0)
        self.matching_anim_matrix = float4x4()

    ##
    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    ##
    def get_world_position(self):
        return float3(self.total_matrix.entries[3], self.total_matrix.entries[7], self.total_matrix.entries[11])

##
class Joint_Hierarchy:

    ##
    def __init__(self, joints = []):
        self.joints = joints
        self.root_joints = []

        self.joint_dict = {}

        for joint in joints:
            if joint.parent == None:
                self.root_joints.append(joint)
                joint.is_root_joint = True

            self.joint_dict[joint.name] = joint

        #print('\n')
        for root_joint in self.root_joints:
            Joint_Hierarchy.traverse_joint(root_joint)
        #print('\n')

        for joint in joints:
            joint.bind_matrix = joint.total_matrix
            joint.inverse_bind_matrix = float4x4.invert(joint.bind_matrix)

    ##
    @staticmethod
    def traverse_joint(curr_joint):
        parent_joint = curr_joint.parent
        if parent_joint != None:
            curr_joint.total_matrix = float4x4.concat_matrices(
                [parent_joint.total_matrix,
                 curr_joint.local_matrix,
                 curr_joint.anim_matrix])
        else:
            curr_joint.total_matrix = float4x4.concat_matrices(
                [curr_joint.local_matrix,
                 curr_joint.anim_matrix])

        #print('\"{}\" position ({}, {}, {})'.format(
        #    curr_joint.name, 
        #    curr_joint.total_matrix.entries[3],
        #    curr_joint.total_matrix.entries[7],
        #    curr_joint.total_matrix.entries[11]))

        if len(curr_joint.children) > 0:
            for child in curr_joint.children:
                Joint_Hierarchy.traverse_joint(child)

    ##
    def apply_scale(self, scale_value):
        for root_joint in self.root_joints:
            updated_anim_matrix = float4x4.concat_matrices([
                root_joint.anim_matrix,
                float4x4.scale(float3(scale_value, scale_value, scale_value))])
            root_joint.anim_matrix = updated_anim_matrix

        #print('\n')
        for root_joint in self.root_joints:
            self.traverse_joint(root_joint)
        #print('\n')

    ##
    def apply_rotation(self, axis, angle):
        for root_joint in self.root_joints:
            total_matrix = root_joint.total_matrix
            q = quaternion.from_angle_axis(axis, angle)
            angle_axis_rotation_matrix = q.to_matrix()
            updated_anim_matrix = float4x4.concat_matrices([
                root_joint.anim_matrix,
                float4x4.rotate(q)])
            root_joint.anim_matrix = updated_anim_matrix

        #print('\n')
        for root_joint in self.root_joints:
            self.traverse_joint(root_joint)
        #print('\n')

    ##
    def apply_rotation_to_joint(self, joint_name, axis, angle):
        for joint in self.joints:
            if joint.name == joint_name:
                q = quaternion.from_angle_axis(axis, angle)
                angle_axis_rotation_matrix = q.to_matrix()
                updated_local_matrix = float4x4.multiply(angle_axis_rotation_matrix, joint.local_matrix)
                joint.local_matrix = updated_local_matrix 

        #print('\n')
        for root_joint in self.root_joints:
            self.traverse_joint(root_joint)
        #print('\n')