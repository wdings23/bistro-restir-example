import gltflib

from quat import *

from rig_anim_utils import *
from joint_rig import *

import struct
import os
import sys

##
class KeyFrameChannel (object):

    ##
    def __init__(self, times, data, joint_index, type):
        self.times = times
        self.data = data
        self.joint_index = joint_index
        self.channel_type = type


##
def load_vector_data(
    model,
    buffer_contents,
    accessor_index,
    directory):
    
    # accessor => buffer_view_index => buffer_index
    # accessor has byte length and offset
    buffer_accessor = model.accessors[accessor_index]
    buffer_view_index = buffer_accessor.bufferView
    buffer_view = model.bufferViews[buffer_view_index]
    buffer_index = buffer_view.buffer
    buffer_byte_length = buffer_view.byteLength
    buffer_byte_offset = buffer_view.byteOffset
    buffer = model.buffers[buffer_index]

    # content of buffer from file
    buffer_content = None
    if buffer.uri in buffer_contents:
        buffer_content = buffer_contents[buffer.uri]
    else:
        file = open(os.path.join(directory, buffer.uri), "rb")
        buffer_content = file.read()
        file.close()
        buffer_contents[buffer.uri] = buffer_content

    # type size, byte, short, and float/int
    component_type_size = 4
    if buffer_accessor.componentType == 5120 or buffer_accessor.componentType == 5121:
        component_type_size = 1
    elif buffer_accessor.componentType == 5122 or buffer_accessor.componentType == 5123:
        component_type_size = 2

    # type of vector
    component_size = 3
    if buffer_accessor.type == 'SCALAR':
        component_size = 1
    elif buffer_accessor.type == 'VEC2':
        component_size = 2
    elif buffer_accessor.type == 'VEC4':
        component_size = 4
    elif buffer_accessor.type == 'MAT4':
        component_size = 16

    vector_size = component_size * component_type_size
    
    # data with check of number of data entries
    position_data = buffer_content[buffer_byte_offset:buffer_byte_offset + buffer_byte_length]
    num_positions = int(buffer_byte_length / vector_size) 
    assert(num_positions == buffer_accessor.count)

    data = []

    # format to parse, from char to short, defaulting to float
    format = 'f'
    if buffer_accessor.componentType == 5120:
        format = 'b'
    elif buffer_accessor.componentType == 5121:
        format = 'B'
    elif buffer_accessor.componentType == 5122:
        format = 'h'
    elif buffer_accessor.componentType == 5123:
        format = 'H'

    # parse out data from byte array
    curr_byte_index = 0
    for i in range(num_positions):
        if component_size == 1:
            if format == 'f':
                x = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x])
            else:
                x = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x])

        elif component_size == 2:
            if format == 'f':
                x = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                y = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x, y])
            else:
                x = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                y = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x, y])

        elif component_size == 3:
            if format == 'f':
                x = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                y = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                z = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x, y, z])
            else:
                x = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                y = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                z = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x, y, z])

        elif component_size == 4:
            if format == 'f':
                x = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                y = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                z = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                w = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x, y, z, w])
            else:
                x = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                y = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                z = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                w = int(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
                data.append([x, y, z, w])

        elif component_size == 16:
            entries = [0.0] * 16
            for matrix_index in range(16):
                entries[matrix_index] = float(struct.unpack(format, position_data[curr_byte_index: curr_byte_index + component_type_size])[0])
                curr_byte_index += component_type_size
            
            data.append(entries)

    return data

##
def load_mesh(
    gltf,
    directory):
    model = gltf.model

    buffer_contents = {}
    
    ret_position_data = []
    ret_normal_data = []
    ret_texcoord_data = []
    ret_joint_index_data = []
    ret_joint_weight_data = []
    ret_triangle_index_data = []

    joint_to_node_mappings = []

    if model.meshes != None:
        for mesh_index in range(len(model.meshes)):
            mesh = model.meshes[mesh_index]

            for primitive_index in range(len(mesh.primitives)):
                primitive = mesh.primitives[primitive_index]
                
                # position data
                position_accessor_index = primitive.attributes.POSITION
                position_data = load_vector_data(
                    model = model, 
                    buffer_contents = buffer_contents, 
                    accessor_index = position_accessor_index,
                    directory = directory)

                # normal data
                normal_acccessor_index = primitive.attributes.NORMAL
                normal_data = load_vector_data(
                    model = model, 
                    buffer_contents = buffer_contents, 
                    accessor_index = normal_acccessor_index,
                    directory = directory)

                # uv data
                texcoord_acccessor_index = primitive.attributes.TEXCOORD_0
                texcoord_data = load_vector_data(
                    model = model, 
                    buffer_contents = buffer_contents, 
                    accessor_index = texcoord_acccessor_index,
                    directory = directory)

                # skin to joint percentage
                joint_acccessor_index = primitive.attributes.JOINTS_0
                joint_data = load_vector_data(
                    model = model, 
                    buffer_contents = buffer_contents, 
                    accessor_index = joint_acccessor_index,
                    directory = directory)

                # skin to joint percentage
                joint_weight_acccessor_index = primitive.attributes.WEIGHTS_0
                joint_weight_data = load_vector_data(
                    model = model, 
                    buffer_contents = buffer_contents, 
                    accessor_index = joint_weight_acccessor_index,
                    directory = directory)

                # triangle indices
                triangle_indices_accesssor_index = primitive.indices
                triangle_index_data = load_vector_data(
                    model = model,
                    buffer_contents = buffer_contents,
                    accessor_index = triangle_indices_accesssor_index,
                    directory = directory)

                # output
                ret_position_data.append(position_data)
                ret_normal_data.append(normal_data)
                ret_texcoord_data.append(texcoord_data)
                ret_joint_index_data.append(joint_data)
                ret_joint_weight_data.append(joint_weight_data)
                ret_triangle_index_data.append(triangle_index_data)

    return ret_position_data, ret_normal_data, ret_texcoord_data, ret_joint_index_data, ret_joint_weight_data, ret_triangle_index_data

##
def load_anim_hierarchy(
    model,
    directory):

    joint_indices = model.skins[0].joints
    num_joints = len(joint_indices)
    nodes = model.nodes
    joint_names = [''] * num_joints
    joint_rotations = [quaternion(0.0, 0.0, 0.0, 1.0)] * num_joints
    joint_scales = [float3(1.0, 1.0, 1.0)] * num_joints
    joint_translations = [float3(0.0, 0.0, 0.0)] * num_joints
    joints = [None] * num_joints

    for joint_index in joint_indices:
        node = nodes[joint_index]
        joint_names[joint_index] = nodes[joint_index].name
        joint_rotations[joint_index] = quaternion(
            node.rotation[0], 
            node.rotation[1], 
            node.rotation[2], 
            node.rotation[3])
        if node.scale != None:
            joint_scales[joint_index] = float3(
                node.scale[0], 
                node.scale[1],
                node.scale[2])
        if node.translation != None:
            joint_translations[joint_index] = float3(
                node.translation[0], 
                node.translation[1],
                node.translation[2])

        # create joint
        joint = Joint(
            joint_names[joint_index],
            joint_rotations[joint_index],
            joint_scales[joint_index],
            joint_translations[joint_index])
        joints[joint_index] = joint

    
    # set joint children and parent
    for joint_index in range(0, num_joints):
        node = nodes[joint_index]
        joint = joints[joint_index]
        assert(joint.name == node.name)
        if node.children != None:
            for child_index in node.children:
                joint.children.append(joints[child_index])
                joints[child_index].parent = joint
        
    # rig from joints
    rig = Joint_Hierarchy(joints)

    byte_buffers = [None] * len(model.buffers)

    for i in range(len(model.buffers)):
        asset_file = open(os.path.join(directory, model.buffers[i].uri), 'rb')
        byte_buffers[i] = asset_file.read()
        asset_file.close()

    num_animations = len(model.animations)
    
    keyframe_channels = {}

    anim_translations = [None] * num_joints
    anim_scales = [None] * num_joints
    anim_rotations = [None] * num_joints

    animation_index = 0
    for animation_index in range(0, num_animations):
        animation = model.animations[animation_index]
        num_animation_channels = len(animation.channels)
        assert(len(animation.samplers) == num_animation_channels)
        for channel_index in range(0, num_animation_channels):
            animation_sampler = animation.samplers[channel_index]
            
            # time accessor
            time_accessor_index = animation_sampler.input
            
            # time buffer view
            time_buffer_view = model.bufferViews[time_accessor_index]
            time_total_buffer_index = time_buffer_view.buffer
            time_buffer_view_length = time_buffer_view.byteLength
            time_buffer_view_offset = time_buffer_view.byteOffset
            time_buffer = byte_buffers[time_total_buffer_index][time_buffer_view_offset: time_buffer_view_offset + time_buffer_view_length]
            
            # load frame times
            frame_times = []
            curr_byte_index = 0
            while True:
                if curr_byte_index >= time_buffer_view_length:
                    break

                time = float(struct.unpack('f', time_buffer[curr_byte_index: curr_byte_index + 4])[0])
                frame_times.append(time)
                curr_byte_index += 4
            
            # channel
            channel_accessor_index = animation_sampler.output
            animation_channel = animation.channels[channel_index]
            channel_node_index = animation_channel.target.node
            channel_index_type = animation_channel.target.path
            channel_buffer_view = model.bufferViews[channel_accessor_index]
            channel_total_buffer_index = channel_buffer_view.buffer
            channel_buffer_view_length = channel_buffer_view.byteLength
            channel_buffer_view_offset = channel_buffer_view.byteOffset
            channel_buffer = byte_buffers[time_total_buffer_index][channel_buffer_view_offset: channel_buffer_view_offset + channel_buffer_view_length]

            joint_index = channel_node_index
            joint_name = joints[joint_index].name

            # accessor info
            accessor = model.accessors[channel_accessor_index]
           
            curr_byte_index = 0
            component_index = 0
            v3 = float3(0.0, 0.0, 0.0)
            v4 = quaternion(0.0, 0.0, 0.0, 0.0)

            # load data from buffer as given in buffer view pointed to by accessor
            while True:
                if curr_byte_index >= channel_buffer_view_length:
                    break
                
                # 4-byte to float
                data = float(struct.unpack('f', channel_buffer[curr_byte_index: curr_byte_index + 4])[0])

                # set the float3 or quaternion data
                ready_for_append = False
                if accessor.type == 'VEC3':
                    if component_index % 3 == 0:
                        v3.x = data
                    elif component_index % 3 == 1:
                        v3.y = data
                    elif component_index % 3 == 2:
                        v3.z = data
                        ready_for_append = True
                elif accessor.type == 'VEC4':
                    if component_index % 4 == 0:
                        v4.x = data
                    elif component_index % 4 == 1:
                        v4.y = data
                    elif component_index % 4 == 2:
                        v4.z = data
                    elif component_index % 4 == 3:
                        v4.w = data     
                        ready_for_append = True    
                    else:
                        assert(False)   # not handled

                # add key frames
                if ready_for_append == True:
                    
                    if channel_index_type == 'translation':
                        if anim_translations[joint_index] == None:
                            anim_translations[joint_index] = []

                        anim_translations[joint_index].append(float3(v3.x, v3.y, v3.z))
                    
                    elif channel_index_type == 'scale':
                        if anim_scales[joint_index] == None:
                            anim_scales[joint_index] = []

                        anim_scales[joint_index].append(float3(v3.x, v3.y, v3.z))
                    
                    elif channel_index_type == 'rotation':
                        if anim_rotations[joint_index] == None:
                            anim_rotations[joint_index] = []           

                        anim_rotations[joint_index].append(quaternion(v4.x, v4.y, v4.z, v4.w))
                        v4.x = 0.0
                        v4.y = 0.0
                        v4.z = 0.0
                        v4.w = 0.0

                curr_byte_index += 4    
                component_index += 1

            if not joint_name in keyframe_channels:
                keyframe_channels[joint_name] = []
            
            if channel_index_type == 'translation':
                keyframe_channels[joint_name].append(KeyFrameChannel(frame_times, anim_translations[joint_index], joint_index, channel_index_type))
            elif channel_index_type == 'scale':
                keyframe_channels[joint_name].append(KeyFrameChannel(frame_times, anim_scales[joint_index], joint_index, channel_index_type))
            elif channel_index_type == 'rotation':
                keyframe_channels[joint_name].append(KeyFrameChannel(frame_times, anim_rotations[joint_index], joint_index, channel_index_type))

    return rig, keyframe_channels

##
def traverse_joint_blender_debug(joint):
    axis, angle = quaternion.to_angle_axis(joint.local_rotation)
    local_rotation_matrix = float4x4.from_angle_axis(axis, angle)
    local_translation_matrix = float4x4.translate(joint.local_translation)
    local_matrix = float4x4.concat_matrices(
        [
            local_translation_matrix,
            local_rotation_matrix
        ]
    )

    parent_total_matrix = float4x4()
    if joint.parent == None:
        parent_total_matrix.identity()
    else:
        parent_total_matrix = joint.parent.total_matrix

    joint.total_matrix = float4x4.concat_matrices(
        [
            parent_total_matrix,
            local_matrix
        ]
    )
    
    color = float3(255.0, 0.0, 0.0)
    print('draw_sphere([{}, {}, {}], 0.1, {}, {}, {}, 255, "{}") '.format(
            joint.total_matrix.entries[3],
            joint.total_matrix.entries[7],
            joint.total_matrix.entries[11],
            color.x,
            color.y,
            color.z,
            joint.name))

    for child in joint.children:
        traverse_joint_blender_debug(child)

##
def get_local_transformations(
    joint
):
    
    parent_joint = joint.parent
    if parent_joint == None:
        # no parent, just use transform from bind matrix

        angle_axis = float4x4.to_angle_axis(joint.bind_matrix)
        joint.local_rotation = quaternion.from_angle_axis(float3(angle_axis[1], angle_axis[2], angle_axis[3]), angle_axis[0])
        joint.local_translation = float3(
            joint.bind_matrix.entries[3],
            joint.bind_matrix.entries[7],
            joint.bind_matrix.entries[11])
    else:
        if len(joint.children) > 0: 
            # get transfrom from bind matrix
            
            # apply parent's inverse bind to joint's bind to get the local transform
            local_bind_matrix = parent_joint.inverse_bind_matrix * joint.bind_matrix
            joint.local_translation = float3(
                local_bind_matrix.entries[3],
                local_bind_matrix.entries[7],
                local_bind_matrix.entries[11]
            )

            # local rotation matrix
            local_rotation_matrix = float4x4(
                [
                    local_bind_matrix.entries[0], local_bind_matrix.entries[1], local_bind_matrix.entries[2], 0.0,
                    local_bind_matrix.entries[4], local_bind_matrix.entries[5], local_bind_matrix.entries[6], 0.0,
                    local_bind_matrix.entries[8], local_bind_matrix.entries[9], local_bind_matrix.entries[10], 0.0,
                    0.0,                          0.0,                          0.0,                           1.0,
                ]
            )

            # convert to quaternion
            angle_axis = float4x4.to_angle_axis(local_rotation_matrix)
            joint.local_rotation = quaternion.from_angle_axis(
                float3(angle_axis[1], angle_axis[2], angle_axis[3]),
                angle_axis[0]
            )


    # traverse into children
    for child_joint in joint.children:
        get_local_transformations(child_joint)

##
def load_gltf(file_path):
    directory_end = file_path.rfind('\\')
    if directory_end < 0:
        directory_end = file_path.rfind('/')
    directory = file_path[:directory_end]
    gltf = gltflib.GLTF.load(file_path)
    model = gltf.model

    mesh_positions, mesh_normals, mesh_texcoords, mesh_joint_indices, mesh_joint_weights, mesh_triangle_indices = load_mesh(
        gltf = gltf,
        directory = directory)
    rig, keyframe_channels = load_anim_hierarchy(
        model = model, 
        directory = directory)

    # load joint mappings and inverse bind skin matrices
    joint_to_node_mappings = []
    for skin_index in range(len(model.skins)):
        skin = model.skins[skin_index]
        
        # joint index to node
        for joint_index in range(len(skin.joints)):
            node_index = skin.joints[joint_index]
            assert(node_index <= len(rig.joints))
            joint_to_node_mappings.append(node_index)

        # inverse bind matrix
        buffer_contents = {}
        inverse_bind_accessor_index = skin.inverseBindMatrices
        inverse_bind_matrix_data = load_vector_data(
            model = model,
            buffer_contents = buffer_contents,
            accessor_index = inverse_bind_accessor_index,
            directory = directory)

    # set the initial bind rotation and translation by inverting the inverse bind to get the bind matrix and extract rotation and translation
    for inverse_bind_matrix_index in range(len(inverse_bind_matrix_data)):
        inverse_bind_matrix_data_entry = inverse_bind_matrix_data[inverse_bind_matrix_index]

        node_index = joint_to_node_mappings[inverse_bind_matrix_index]
        joint = rig.joints[node_index]

        inverse_bind_matrix = float4x4(entries = [
            inverse_bind_matrix_data_entry[0], inverse_bind_matrix_data_entry[4], inverse_bind_matrix_data_entry[8], inverse_bind_matrix_data_entry[12], 
            inverse_bind_matrix_data_entry[1], inverse_bind_matrix_data_entry[5], inverse_bind_matrix_data_entry[9], inverse_bind_matrix_data_entry[13], 
            inverse_bind_matrix_data_entry[2], inverse_bind_matrix_data_entry[6], inverse_bind_matrix_data_entry[10], inverse_bind_matrix_data_entry[14], 
            inverse_bind_matrix_data_entry[3], inverse_bind_matrix_data_entry[7], inverse_bind_matrix_data_entry[11], inverse_bind_matrix_data_entry[15], 
        ])
        joint.inverse_bind_matrix = inverse_bind_matrix
        joint.bind_matrix = float4x4.invert(joint.inverse_bind_matrix)
        
        rotation_matrix = float4x4(entries = 
            [
                joint.bind_matrix.entries[0],  joint.bind_matrix.entries[1],   joint.bind_matrix.entries[2], 0.0,
                joint.bind_matrix.entries[4],  joint.bind_matrix.entries[5],   joint.bind_matrix.entries[6], 0.0,
                joint.bind_matrix.entries[8],  joint.bind_matrix.entries[9],   joint.bind_matrix.entries[10], 0.0,
                joint.bind_matrix.entries[12], joint.bind_matrix.entries[13],  joint.bind_matrix.entries[14], 1.0,
            ]
        )
        angle_axis = float4x4.to_angle_axis(rotation_matrix)
        joint.bind_rotation = quaternion.from_axis_angle(
            axis = float3(angle_axis[1], angle_axis[2], angle_axis[3]),
            angle = angle_axis[0])
        joint.bind_translation = float3(joint.bind_matrix.entries[3], joint.bind_matrix.entries[7], joint.bind_matrix.entries[11])
        joint.bind_scale = float3(1.0, 1.0, 1.0)

    for root_joint in rig.root_joints:
        get_local_transformations(root_joint)
    
    #traverse_joint_blender_debug(rig.root_joints[0])


    return mesh_positions, mesh_normals, mesh_texcoords, mesh_joint_indices, mesh_joint_weights, mesh_triangle_indices, rig, keyframe_channels, joint_to_node_mappings, inverse_bind_matrix_data
    