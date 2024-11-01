import os 
import sys
import json

from gltf_loader import *
from rig_anim_utils import *

blender_debug_output = '''
import bpy
import json
import math
import bmesh
import os
import random

##
def draw_sphere(pos, box_size, r, g, b, a, name):
    
    ret_x = pos[0]
    ret_y = pos[1]
    ret_z = pos[2]

    ret = bpy.ops.mesh.primitive_ico_sphere_add(radius = box_size, location = (ret_x, -ret_z, ret_y))
    
    mat = bpy.data.materials.new(name="MaterialName")
    mat.diffuse_color = (r/255,g/255,b/255, a/255)
    obj = bpy.context.object
    obj.data.materials.append(mat)
    obj.name = name

    return ret

'''

##
def traverse_matching_anim(joint, output_file_path = ''):
    
    matching_axis, matching_angle = quaternion.to_angle_axis(joint.matching_anim_rotation)
    matching_rotation_matrix = float4x4.from_angle_axis(matching_axis, matching_angle)
    matching_translation_matrix = float4x4.translate(joint.matching_anim_translation)
    matching_anim_matrix = float4x4.concat_matrices([
        matching_translation_matrix,
        matching_rotation_matrix,
    ])

    local_axis, local_angle = quaternion.to_angle_axis(joint.local_rotation)
    local_rotation_matrix = float4x4.from_angle_axis(local_axis, local_angle)
    local_translation_matrix = float4x4.translate(joint.local_translation)
    local_matrix = float4x4.concat_matrices([
        local_translation_matrix,
        local_rotation_matrix,
    ])

    parent_total_match_anim_matrix = float4x4()
    if joint.parent == None:
        parent_total_match_anim_matrix.identity()
    else:
        parent_total_match_anim_matrix = joint.parent.matching_anim_matrix

    joint.matching_anim_matrix = float4x4.concat_matrices([
        parent_total_match_anim_matrix,
        local_matrix,
        matching_anim_matrix,
    ])

    color = float3(255.0, 0.0, 0.0)

    if output_file_path != '':
        output_line = 'draw_sphere([{}, {}, {}], 0.05, {}, {}, {}, 255, "{}")\n'.format(
                joint.matching_anim_matrix.entries[3],
                joint.matching_anim_matrix.entries[7],
                joint.matching_anim_matrix.entries[11],
                color.x,
                color.y,
                color.z,
                joint.name)
        output_file = open(output_file_path, 'a')
        output_file.write(output_line)
        output_file.close()

    # print('draw_sphere([{}, {}, {}], 0.05, {}, {}, {}, 255, "{}") '.format(
    #         joint.matching_anim_matrix.entries[3],
    #         joint.matching_anim_matrix.entries[7],
    #         joint.matching_anim_matrix.entries[11],
    #         color.x,
    #         color.y,
    #         color.z,
    #         joint.name))

    for child_joint in joint.children:
        traverse_matching_anim(child_joint, output_file_path)

##
def match_joint_transforms(
    dest_joint,
    rig_src,
    rig_dest,
    joint_mappings,
    last_mapped_src_joint):

    ## steps
    # bring src and parent joint's world position into the dest parent joint space, by 
    # setting the current matching-animation matrix of the dest joint's translation part to the src parent joint's world position.
    # We can use the inverse of the matrix to bring the src's joint world position locally to the src parent joint with the dest parent joint's orientation local space.
    # This can be used to calculate the vector from the parent joint local position (0, 0, 0) to the joint local position
    # Taking the above vector along with the local translation vector of the dest joint, we can calculate the axis and angle in 
    # this local space of the current animation.
    # Axis is the cross product between the src local translation vector with dest local translation vector.
    # Angle is the dot product of the above two vectors.
    # Verify at the end.  

    if dest_joint.name in joint_mappings and dest_joint.parent != None:
        src_joint_name = joint_mappings[dest_joint.name]
        
        if src_joint_name != None:
            # mapped joint

            # get the src joint and its parent current position
            src_joint = None
            for joint_index in range(len(rig_src.joints)):
                if rig_src.joints[joint_index].name == src_joint_name:
                    src_joint = rig_src.joints[joint_index]
                    break
            assert(src_joint != None)
            src_bone_length = float3.length(src_joint.translation)
            
            if src_bone_length > 0.0:
                
                # src and src parent joint world positions
                src_joint_world_position = float3(
                    src_joint.total_matrix.entries[3],
                    src_joint.total_matrix.entries[7],
                    src_joint.total_matrix.entries[11]
                )

                src_joint_parent_world_position = float3(
                    src_joint.parent.total_matrix.entries[3],
                    src_joint.parent.total_matrix.entries[7],
                    src_joint.parent.total_matrix.entries[11]
                )
                
                # length of source bone and dest bone
                src_parent_to_child_length = float3.length(src_joint.local_translation)
                dest_parent_to_child_length = float3.length(dest_joint.local_translation)

                # ratio between the bone
                src_to_dest_bone_length_ratio = 0.0
                if dest_parent_to_child_length > 0.0:
                    src_to_dest_bone_length_ratio = src_parent_to_child_length / dest_parent_to_child_length
                
                # current dest parent joint local space orientation with src parent joint position 
                dest_parent_matrix = dest_joint.parent.matching_anim_matrix
                dest_parent_joint_matching_matrix = float4x4(
                    [
                        dest_parent_matrix.entries[0], dest_parent_matrix.entries[1], dest_parent_matrix.entries[2], src_joint_parent_world_position.x,  
                        dest_parent_matrix.entries[4], dest_parent_matrix.entries[5], dest_parent_matrix.entries[6], src_joint_parent_world_position.y,  
                        dest_parent_matrix.entries[8], dest_parent_matrix.entries[9], dest_parent_matrix.entries[10], src_joint_parent_world_position.z,  
                        dest_parent_matrix.entries[12], dest_parent_matrix.entries[13], dest_parent_matrix.entries[14], dest_parent_matrix.entries[15]
                    ]
                )
                inverse_dest_parent_joint_local_matching_matrix = float4x4.invert(dest_parent_joint_matching_matrix)

                # transform src joint into src parent's local space
                src_joint_local_matching_anim_position = inverse_dest_parent_joint_local_matching_matrix.apply(src_joint_world_position) 
                
                # local axis angle
                v0 = float3.normalize(dest_joint.local_translation)
                v1 = float3.normalize(src_joint_local_matching_anim_position)
                anim_matching_local_axis = float3.normalize(float3.cross(v0, v1))
                dp = float3.dot(v0, v1)
                anim_matching_local_angle = math.acos(dp)
                anim_matching_local_matrix = float4x4.from_angle_axis(anim_matching_local_axis, anim_matching_local_angle)

                dest_joint.parent.matching_anim_rotation = quaternion.from_angle_axis(anim_matching_local_axis, anim_matching_local_angle)
                
                # verify angle
                verify_anim_dest_joint_translation = anim_matching_local_matrix.apply(v0)
                verify_dp = float3.dot(verify_anim_dest_joint_translation, v1)
                assert(((verify_dp - 1.0) * (verify_dp - 1.0)) <= 1.0e-6)

                #print('\n#### matching anim ####\n')


                #output_file_path = 'd:\\test\\python-webgpu\\debug-blender.py'
                #output_file = open(output_file_path, 'w')
                #output_file.write(output)
                #output_file.close()
                
                # need to refresh the matching animation matrices with the updated values
                for dest_root_joint in rig_dest.root_joints:
                    traverse_matching_anim(
                        joint = dest_root_joint
                    )
                
        


    for child_joint in dest_joint.children:
        match_joint_transforms(
            dest_joint = child_joint,
            rig_src = rig_src,
            rig_dest = rig_dest,
            joint_mappings = joint_mappings,
            last_mapped_src_joint = last_mapped_src_joint)

##
def print_joint_position(joint):

    joint_position = float3(
        joint.total_matrix.entries[3],
        joint.total_matrix.entries[7],
        joint.total_matrix.entries[11]
    )

    color = float3(255.0, 0.0, 0.0)
    print('draw_sphere([{}, {}, {}], 0.2, {}, {}, {}, 255, "{}")'.format(
        joint_position.x,
        joint_position.y,
        joint_position.z,
        color.x,
        color.y,
        color.z,
        joint.name))

    for child_joint in joint.children:
        print_joint_position(child_joint)

##
def get_all_anim_transformations(
    joint,
    matching_anim_rotations,
    matching_anim_translations):

    matching_anim_rotations[joint.name] = joint.local_rotation * joint.matching_anim_rotation
    matching_anim_translations[joint.name] = joint.local_translation
    for child_joint in joint.children:
        get_all_anim_transformations(
            joint = child_joint, 
            matching_anim_rotations = matching_anim_rotations,
            matching_anim_translations = matching_anim_translations)

##
def get_rig_extent(rig):
    largest_position = float3(-9999.0, -9999.0, -9999.0)
    smallest_position = float3(9999.0, 9999.0, 9999.0)

    for root_joint in rig.root_joints:
        traverse_rig_bind(root_joint)

    for joint_index in range(len(rig.joints)):
        position = float3(
            rig.joints[joint_index].total_matrix.entries[3],
            rig.joints[joint_index].total_matrix.entries[7],
            rig.joints[joint_index].total_matrix.entries[11])
        
        if position.x > largest_position.x:
            largest_position.x = position.x
        if position.y > largest_position.y:
            largest_position.y = position.y
        if position.z > largest_position.z:
            largest_position.z = position.z

        if position.x < smallest_position.x:
            smallest_position.x = position.x
        if position.y < smallest_position.y:
            smallest_position.y = position.y
        if position.z < smallest_position.z:
            smallest_position.z = position.z

    root_position = float3(
        rig.root_joints[0].total_matrix.entries[3],
        rig.root_joints[0].total_matrix.entries[7],
        rig.root_joints[0].total_matrix.entries[11]
    )

    return smallest_position, largest_position, root_position

##
def save_matching_keyframes(
    keyframe_channels,
    src_keyframe_file_path,
    dest_keyframe_file_path,
    keyframe_output_directory):

    keys = list(keyframe_channels.keys())
    num_keys = len(keys)

    json_dict = {}
    for key_index in range(num_keys):
        key = keys[key_index]

        keyframe_info_array = []
        for type_index in range(3):

            type_str = 'translation'
            if type_index == 1:
                type_str = 'rotation'
            elif type_index == 2:
                type_str = 'scale'

            data = keyframe_channels[key][type_index].data
            times = keyframe_channels[key][type_index].times

            keyframe_info_array.append([])

            num_frames = len(times)
            for frame_index in range(num_frames):
                keyframe_data_dict = {
                    'type': type_str,
                    'x': data[frame_index].x,
                    'y': data[frame_index].y,
                    'z': data[frame_index].z,
                    'time': times[frame_index]
                }

                if type_index == 1:
                    keyframe_data_dict['w'] = data[frame_index].w
                
                keyframe_info_array[type_index].append(keyframe_data_dict)
        
        key = keys[key_index]
        json_dict[key] = keyframe_info_array

    src_base_name = os.path.basename(src_keyframe_file_path)
    src_base_name = src_base_name[:src_base_name.rfind('.')]
    dest_base_name = os.path.basename(dest_keyframe_file_path)
    dest_base_name = dest_base_name[:dest_base_name.rfind('.')]
    file_name = src_base_name + '-to-' + dest_base_name

    json_file_content = json.dumps(json_dict, indent = 4)
    output_file_path = os.path.join(keyframe_output_directory, file_name + '-matching-animation.json')
    json_file = open(output_file_path, 'wb')
    json_file.write(json_file_content.encode('utf-8'))
    json_file.close()

##
def load_matching_keyframes(file_path):
    file = open(file_path, 'r')
    file_content = file.read()
    file.close()
    keyframe_dict = json.loads(file_content)

    keys = list(keyframe_dict.keys())

    key_frame_channels = {}
    for joint_index in range(len(keys)):
        key = keys[joint_index]
        
        key_frame_channels[key] = []
        for type_index in range(3):
            type_str = 'translation'
            if type_index == 1:
                type_str = 'rotation'
            elif type_index == 2:
                type_str = 'scale'
            
            info = keyframe_dict[key][type_index]

            data_array = []
            time_array = []
            for entry in info:
                
                if type_index == 1:
                    channel_data = quaternion(
                        entry['x'],
                        entry['y'],
                        entry['z'],
                        entry['w']
                    )
                    data_array.append(channel_data)
                else:
                    channel_data = float3(
                        entry['x'],
                        entry['y'],
                        entry['z']
                    )
                    data_array.append(channel_data)

                time = entry['time']
                time_array.append(time)

            key_frame_channels[key].append(KeyFrameChannel(
                times = time_array,
                data = data_array,
                joint_index = joint_index,
                type = type_str))

    return key_frame_channels

##
def translate_keyframe_channels2(
    src_file_path,
    dest_file_path,
    save_directory = ''):

    joint_mappings = {
        'mixamorig:Spine' : 'Spine',
        'mixamorig:Head' : 'Head',
        'mixamorig:Spine2' : 'Spine1',
        'mixamorig:LeftForeArm' : 'LeftShoulder',
        'mixamorig:LeftForeArm' : 'LeftForeArm',
        'mixamorig:LeftHand': 'LeftHand',
        'mixamorig:RightForeArm' : 'RightShoulder',
        'mixamorig:RightForeArm' : 'RightForeArm',
        'mixamorig:RightHand' : 'RightHand',
        'mixamorig:LeftLeg' : 'LeftLeg',
        'mixamorig:LeftFoot' : 'LeftFoot',
        'mixamorig:LeftToeBase' : 'LeftToeBase',
        'mixamorig:RightLeg' : 'RightLeg',
        'mixamorig:RightFoot' : 'RightFoot',
        'mixamorig:RightToeBase' : 'RightToeBase',
    }

    mesh_positions_src, mesh_normals_src, mesh_texcoords_src, mesh_joint_indices_src, mesh_joint_weights_src, mesh_triangle_indices_src, rig_src, keyframe_channels_src, joint_to_node_mappings_src, inverse_bind_matrix_data_src = load_gltf(src_file_path)
    mesh_positions_dest, mesh_normals_dest, mesh_texcoords_dest, mesh_joint_indices_dest, mesh_joint_weights_dest, mesh_triangle_indices_dest, rig_dest, keyframe_channels_dest, joint_to_node_mappings_dest, inverse_bind_matrix_data_dest = load_gltf(dest_file_path)

    joint_positions = {}
    joint_rotations = {}

    keyframe_times = keyframe_channels_src[rig_src.root_joints[0].name][0].times

    total_matching_anim_rotations = {}
    total_matching_anim_translations = {}
    total_times = []

    # get rig scaling factor
    smallest_position_src, largest_position_src, root_bind_position_src = get_rig_extent(rig_src)
    smallest_position_dest, largest_position_dest, root_bind_position_dest = get_rig_extent(rig_dest) 

    rig_src_to_dest_scale = float3(
        (largest_position_src.x - smallest_position_src.x) / (largest_position_src.x - smallest_position_dest.x),
        (largest_position_src.y - smallest_position_src.y) / (largest_position_src.y - smallest_position_dest.y),
        (largest_position_src.z - smallest_position_src.z) / (largest_position_src.z - smallest_position_dest.z)
    )
    rig_translation_scale = rig_src_to_dest_scale.x
    if rig_src_to_dest_scale.y > rig_translation_scale:
        rig_translation_scale = rig_src_to_dest_scale.y
    if rig_src_to_dest_scale.z > rig_translation_scale:
        rig_translation_scale = rig_src_to_dest_scale.z

    # rig translations is the world position of joint in bind pose
    for time in keyframe_times:
        
        # play animation at current time to get the source joint locations
        for src_root_joint in rig_src.root_joints:
            traverse_rig(
                curr_joint = src_root_joint,
                rig = rig_src,
                keyframe_channels = keyframe_channels_src,
                time = time,
                joint_positions = joint_positions,
                joint_rotations = joint_rotations,
                root_joint_name = src_root_joint.name)

        #print('\n####\n')
        #for src_root_joint in rig_src.root_joints:
        #    print_joint_position(src_root_joint)

        # initialize matching animation matrix with bind pose
        #print('\n#### matching anim ####\n')

        # initial matching animation transformation to default
        for joint in rig_dest.joints:
            joint.matching_anim_rotation = quaternion(0.0, 0.0, 0.0, 1.0)
            joint.matching_anim_translation = float3(0.0, 0.0, 0.0)
            joint.matching_anim_matrix.identity()
        
        # matching animation matrices to bind pose
        for dest_root_joint in rig_dest.root_joints:
            traverse_matching_anim(dest_root_joint)

        # translate matching animation from src rig to dest rig
        for dest_root_joint in rig_dest.root_joints:
            match_joint_transforms(
                dest_joint = dest_root_joint,
                rig_src = rig_src,
                rig_dest = rig_dest,
                joint_mappings = joint_mappings,
                last_mapped_src_joint = None)

        # set the scaled root joint position
        frame_index = len(total_times)
        translation_src = keyframe_channels_src[rig_src.root_joints[0].name][0].data[frame_index] - root_bind_position_src
        translation_dest = float3(
            translation_src.x / rig_translation_scale,
            translation_src.y / rig_translation_scale,
            translation_src.z / rig_translation_scale
        )
        
        matching_anim_rotations_at_time = {}
        matching_anim_translations_at_time = {}
        for dest_root_joint in rig_dest.root_joints:
            get_all_anim_transformations(
                joint = dest_root_joint, 
                matching_anim_rotations = matching_anim_rotations_at_time,
                matching_anim_translations = matching_anim_translations_at_time)

        # set the dest root joint's location keyframe, scaled by the rig's extent
        matching_anim_translations_at_time[rig_dest.root_joints[0].name] = float3(
            translation_dest.x,
            translation_dest.y,
            translation_dest.z)

        for key in matching_anim_rotations_at_time:
            if not key in total_matching_anim_rotations:
                total_matching_anim_rotations[key] = []
            
            total_matching_anim_rotations[key].append(matching_anim_rotations_at_time[key])

        for key in matching_anim_translations_at_time:
            if not key in total_matching_anim_translations:
                total_matching_anim_translations[key] = []

            total_matching_anim_translations[key].append(matching_anim_translations_at_time[key])

        

        total_times.append(time)

        print('frames at time: {} done'.format(time))

    total_matching_anim_scales = [float3(1.0, 1.0, 1.0)] * len(total_times)

    keyframe_channels_dest = {}
    keys = list(total_matching_anim_rotations.keys())
    for i in range(len(keys)):
        key = keys[i]
        keyframe_channels_dest[key] = []
        keyframe_channels_dest[key].append(KeyFrameChannel(
            times = total_times,
            data = total_matching_anim_translations[key],
            joint_index = i,
            type = 'translation'))

        keyframe_channels_dest[key].append(KeyFrameChannel(
            times = total_times,
            data = total_matching_anim_rotations[key],
            joint_index = i,
            type = 'rotation'))

        keyframe_channels_dest[key].append(KeyFrameChannel(
            times = total_times,
            data = total_matching_anim_scales,
            joint_index = i,
            type = 'scale'))

    # save matching animation key frames
    if save_directory != '':
        save_matching_keyframes(
            keyframe_channels = keyframe_channels_dest,
            src_keyframe_file_path = src_file_path,
            dest_keyframe_file_path = dest_file_path,
            keyframe_output_directory = save_directory)

    return keyframe_channels_dest

        