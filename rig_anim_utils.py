import math

from vec import *
from mat4 import *
from quat import *


##
def traverse_rig(
    curr_joint,
    rig,
    keyframe_channels,
    time,
    joint_positions,
    joint_rotations,
    root_joint_name = None):

    # check if joint is in the keyframes
    if not curr_joint.name in keyframe_channels:
        anim_translation = curr_joint.translation
        anim_rotation = curr_joint.rotation
        anim_scale = curr_joint.scale
    else:
        anim_translation, anim_rotation, anim_scale = get_key_frame_data(
            curr_joint, 
            keyframe_channels, 
            time)

    # default if given root joint name
    if curr_joint.name == root_joint_name:
        anim_translation = float3(0.0, 0.0, 0.0)
        anim_rotation = quaternion(0.0, 0.0, 0.0, 1.0)
        anim_scale = float3(1.0, 1.0, 1.0)

    translation_matrix = float4x4.translate(anim_translation)
    axis_angle = quaternion.to_angle_axis(anim_rotation)
    rotation_matrix = float4x4.from_angle_axis(axis_angle[0], axis_angle[1])
    #rotation_matrix = anim_rotation.to_matrix()
    scale_matrix = float4x4.scale(anim_scale)

    local_translation_matrix = float4x4.translate(anim_translation)
    local_scale_matrix = float4x4.scale(curr_joint.scale)

    local_anim_matrix = float4x4.concat_matrices([
        local_translation_matrix,
        rotation_matrix,
        local_scale_matrix
    ])

    parent_joint = curr_joint.parent
    if parent_joint != None:
        curr_joint.total_matrix = float4x4.concat_matrices(
            [parent_joint.total_matrix,
             local_anim_matrix])
    else:
        curr_joint.total_matrix = local_anim_matrix

    # for debugging
    curr_joint.debug_anim_rotation = anim_rotation
    curr_joint.debug_axis_angle = axis_angle
    curr_joint.debug_rotation_matrix = rotation_matrix
    curr_joint.debug_time = time

    world_position = float3(
        curr_joint.total_matrix.entries[3],
        curr_joint.total_matrix.entries[7],
        curr_joint.total_matrix.entries[11])

    joint_positions[curr_joint.name] = world_position

    angle_axis = float4x4.to_angle_axis(curr_joint.total_matrix)
    joint_rotations[curr_joint.name] = angle_axis

    for child_joint in curr_joint.children:
        traverse_rig(
            child_joint,
            rig,
            keyframe_channels,
            time,
            joint_positions,
            joint_rotations,
            root_joint_name)

##
def traverse_rig_with_world_orientation_and_translation(
    curr_joint,
    rig,
    keyframe_channels,
    time,
    world_orientation,
    world_translation):

    anim_translation = float3(0.0, 0.0, 0.0)
    anim_rotation = quaternion(0.0, 0.0, 0.0, 1.0)
    anim_scale = float3(1.0, 1.0, 1.0)

    start_anim_translation = keyframe_channels[curr_joint.name][0].data[0]
    start_anim_rotation = keyframe_channels[curr_joint.name][1].data[0]

    # get the keyframe index
    keyframe_channel = keyframe_channels[curr_joint.name]
    for i in range(0, 3):
        num_keyframes = len(keyframe_channel[i].times)
        time_pct = 0.0
        prev_keyframe = 0
        next_keyframe = 0

        for j in range(0, num_keyframes):
            if keyframe_channel[i].times[j] > time:
                prev_keyframe = j - 1
                if prev_keyframe < 0:
                    prev_keyframe = 0

                next_keyframe = j
                denom = (keyframe_channel[i].times[next_keyframe] - keyframe_channel[i].times[prev_keyframe]) 
                if math.fabs(denom) > 0.0:
                    time_pct =  (time - keyframe_channel[i].times[prev_keyframe]) / denom
                else:
                    time_pct = 0.0

                break
        
        # key frame data
        if i != 1:
            key_frame_data = keyframe_channel[i].data[prev_keyframe] + (keyframe_channel[i].data[next_keyframe] - keyframe_channel[i].data[prev_keyframe]) * time_pct
        else:
            d = (keyframe_channel[i].data[next_keyframe] - keyframe_channel[i].data[prev_keyframe])
            d.x *= time_pct
            d.y *= time_pct
            d.z *= time_pct

            key_frame_data = keyframe_channel[i].data[prev_keyframe] + d

        if i == 0:
            anim_translation = key_frame_data
        elif i == 1:
            anim_rotation = key_frame_data
        elif i == 2:
            anim_scale = key_frame_data

    # relative to the start of animation
    relative_translation = start_anim_translation - anim_translation
    relative_rotation = start_anim_rotation - anim_rotation    

    # bring to world space
    world_joint_translation = anim_translation
    world_joint_orientation = anim_rotation

    if curr_joint.name == rig.root_joints[0].name:
        world_joint_translation = float3(0.0, 0.0, 0.0)
        world_joint_orientation = world_orientation # quaternion(0.0, 0.0, 0.0, 1.0)

    # rotation, scale, and translation matrices
    axis_angle = quaternion.to_angle_axis(world_joint_orientation)
    rotation_matrix = float4x4.from_angle_axis(axis_angle[0], axis_angle[1])
    local_translation_matrix = float4x4.translate(world_joint_translation)
    local_scale_matrix = float4x4.scale(curr_joint.scale)

    local_anim_matrix = float4x4.concat_matrices([
        local_translation_matrix,
        rotation_matrix,
        local_scale_matrix
    ])

    parent_joint = curr_joint.parent
    if parent_joint != None:
        curr_joint.total_matrix = float4x4.concat_matrices(
            [parent_joint.total_matrix,
             local_anim_matrix])
    else:
        curr_joint.total_matrix = local_anim_matrix

    for child_joint in curr_joint.children:
        traverse_rig_with_world_orientation_and_translation(
            child_joint,
            rig,
            keyframe_channels,
            time,
            world_orientation,
            world_translation)

##
def get_joint_locations(
    rig, 
    time,
    keyframe_channels,
    root_joint_name = None):
    
    joint_positions = {}
    joint_rotations = {}

    for root_joint in rig.root_joints:
        traverse_rig(
            root_joint,
            rig,
            keyframe_channels,
            time,
            joint_positions,
            joint_rotations,
            root_joint_name)

    return joint_positions, joint_rotations

##
def get_root_joint_location_at_time(
    rig,
    time,
    keyframe_channels):

    root_joint_name = rig.root_joints[0].name
    joint = rig.joint_dict[root_joint_name]

    anim_translation, anim_rotation, anim_scale = get_key_frame_data(
        rig.root_joints[0],
        keyframe_channels,
        time)

    axis_angle = quaternion.to_angle_axis(anim_rotation)
    rotation_matrix = float4x4.from_angle_axis(axis_angle[0], axis_angle[1])

    local_translation_matrix = float4x4.translate(anim_translation)
    local_scale_matrix = float4x4.scale(joint.scale)

    local_anim_matrix = float4x4.concat_matrices([
        local_translation_matrix,
        rotation_matrix,
        local_scale_matrix
    ])

    total_joint_matrix = local_anim_matrix
    world_position = float3(
        total_joint_matrix.entries[3],
        total_joint_matrix.entries[7],
        total_joint_matrix.entries[11])

    return world_position

##
def get_key_frame_data(
    curr_joint,
    keyframe_channels,
    time):

    # get the keyframe index
    keyframe_channel = keyframe_channels[curr_joint.name]
    for i in range(0, 3):
        num_keyframes = len(keyframe_channel[i].times)
        time_pct = 0.0
        prev_keyframe = -1
        next_keyframe = -1

        for j in range(0, num_keyframes):
            if keyframe_channel[i].times[j] > time:
                prev_keyframe = j - 1
                if prev_keyframe < 0:
                    prev_keyframe = 0

                next_keyframe = j
                denom = (keyframe_channel[i].times[next_keyframe] - keyframe_channel[i].times[prev_keyframe]) 
                if math.fabs(denom) > 0.0:
                    time_pct =  (time - keyframe_channel[i].times[prev_keyframe]) / denom
                else:
                    time_pct = 0.0

                break
        
        if prev_keyframe < 0:
            prev_keyframe = num_keyframes - 1

        if next_keyframe < 0:
            next_keyframe = num_keyframes - 1

        # key frame data
        if i != 1:
            key_frame_data_delta = (keyframe_channel[i].data[next_keyframe] - keyframe_channel[i].data[prev_keyframe]) * time_pct
            key_frame_data = keyframe_channel[i].data[prev_keyframe] + key_frame_data_delta
        else:
            mult = 1.0
            diff = keyframe_channel[i].data[next_keyframe] - keyframe_channel[i].data[prev_keyframe]
            
            # wrap-around rotation
            length = float3.length(diff)
            if length >= 1.0:
                mult = -1.0

            # interpolate
            key_frame_data_delta = quaternion(
                keyframe_channel[i].data[next_keyframe].x * mult - keyframe_channel[i].data[prev_keyframe].x, 
                keyframe_channel[i].data[next_keyframe].y * mult - keyframe_channel[i].data[prev_keyframe].y, 
                keyframe_channel[i].data[next_keyframe].z * mult - keyframe_channel[i].data[prev_keyframe].z, 
                keyframe_channel[i].data[next_keyframe].w - keyframe_channel[i].data[prev_keyframe].w)
            key_frame_data_delta.x *= time_pct
            key_frame_data_delta.y *= time_pct
            key_frame_data_delta.z *= time_pct
            key_frame_data_delta.w *= time_pct

            key_frame_data = keyframe_channel[i].data[prev_keyframe] + key_frame_data_delta

        if i == 0:
            anim_translation = key_frame_data
        elif i == 1:
            anim_rotation = key_frame_data
        elif i == 2:
            anim_scale = key_frame_data

    return anim_translation, anim_rotation, anim_scale

##
def traverse_rig_from_current_position(
    curr_joint,
    rig,
    keyframe_channels,
    time,
    joint_positions,
    joint_rotations,
    time_delta):

    # current frame and next frame after time delta
    anim_translation0, anim_rotation0, anim_scale0 = get_key_frame_data(
        curr_joint, 
        keyframe_channels, 
        time)

    anim_translation1, anim_rotation1, anim_scale1 = get_key_frame_data(
        curr_joint, 
        keyframe_channels, 
        time + time_delta)

    rotation_matrix = float4x4()
    rotation_matrix.identity()

    anim_translation = float3(0.0, 0.0, 0.0)

    # apply the relative animation translation between the 2 frames, applying with orientation and add to current root position
    if curr_joint.name == rig.root_joints[0].name:
        
        # re-orient the relative translation to model space for the later application of the given orientation matrix 
        anim_rotation_matrix = anim_rotation0.to_matrix()
        inverse_anim_rotation_matrix = float4x4.invert(anim_rotation_matrix)
        world_position_relative_translation = anim_translation1 - anim_translation0
        model_relative_translation = inverse_anim_rotation_matrix.apply(world_position_relative_translation)
        
        curr_root_joint_world_position = rig.root_joints[0].get_world_position()
        rotation_matrix = rotation_matrix * curr_joint.move_orientation.to_matrix()
        anim_translation = curr_joint.move_orientation.to_matrix().apply(model_relative_translation)
        anim_translation.y = 0.0        # test test test
        anim_translation = anim_translation + curr_root_joint_world_position

    local_translation_matrix = float4x4.translate(anim_translation)
    local_scale_matrix = float4x4.scale(curr_joint.scale)

    local_anim_matrix = float4x4.concat_matrices([
        local_translation_matrix,
        rotation_matrix,
        local_scale_matrix
    ])

    parent_joint = curr_joint.parent
    if parent_joint != None:
        curr_joint.total_matrix = float4x4.concat_matrices(
            [parent_joint.total_matrix,
             local_anim_matrix])
    else:
        curr_joint.total_matrix = local_anim_matrix

    world_position = float3(
        curr_joint.total_matrix.entries[3],
        curr_joint.total_matrix.entries[7],
        curr_joint.total_matrix.entries[11])

    joint_positions[curr_joint.name] = world_position

    angle_axis = float4x4.to_angle_axis(curr_joint.total_matrix)
    joint_rotations[curr_joint.name] = angle_axis

    for child_joint in curr_joint.children:
        traverse_rig(
            child_joint,
            rig,
            keyframe_channels,
            time,
            joint_positions,
            joint_rotations,
            rig.root_joints[0].name)


##
def get_future_root_joint_data_from_animation(
    rig,
    keyframe_channels,
    start_time,
    num_sample_frames,
    frame_time_between):

    root_joint_name = rig.root_joints[0].name

    duration_indices = [(-1, -1)] * 3

    translation_frames = []
    rotation_frames = []
    scale_frames = []

    # get the keyframe index
    keyframe_channel = keyframe_channels[root_joint_name]
    for i in range(0, 3):
        num_keyframes = len(keyframe_channel[i].times)
        
        start_index = -1
        end_index = -1
        last_record_time = -1.0
        frame_times = []
        frame_indices = []

        for j in range(0, num_keyframes):
            if keyframe_channel[i].times[j] > start_time:
                
                # start recording
                if start_index == -1:
                    start_index = j
                    last_record_time = keyframe_channel[i].times[j]
                    frame_indices.append(j)
                else:
                    if start_index >= 0 and keyframe_channel[i].times[j] - last_record_time >= frame_time_between:
                        last_record_time = keyframe_channel[i].times[j]
                        frame_times.append(last_record_time)
                        frame_indices.append(j)

                        if len(frame_indices) >= num_sample_frames:
                            break

        for j in range(num_sample_frames - len(frame_indices)):
            frame_indices.append(num_keyframes - 1)

        for j in range(len(frame_indices)):
            frame_index = frame_indices[j]

            if i == 0:
                translation_frames.append(keyframe_channel[i].data[frame_index])
            elif i == 1:
                rotation_frames.append(keyframe_channel[i].data[frame_index])
            elif i == 2:
                scale_frames.append(keyframe_channel[i].data[frame_index])

    return translation_frames, rotation_frames, scale_frames


##
def traverse_rig_for_blend(
    curr_joint,
    rig,
    start_blend_root_joint_world_position,
    next_keyframe_channels,
    prev_keyframe_channels,
    next_frame_time,
    prev_frame_time,
    after_next_frame_time,
    next_orientation_angle,
    prev_orientation_angle,
    blend_pct):

    anim_translation0, anim_rotation0, anim_scale0 = get_key_frame_data(
        curr_joint, 
        prev_keyframe_channels, 
        prev_frame_time)

    anim_translation1, anim_rotation1, anim_scale1 = get_key_frame_data(
        curr_joint, 
        next_keyframe_channels, 
        next_frame_time)

    anim_translation2, anim_rotation2, anim_scale2 = get_key_frame_data(
        curr_joint, 
        next_keyframe_channels, 
        after_next_frame_time)

    rotation_matrix = float4x4()
    rotation_matrix.identity()

    slerp_quat = quaternion.slerp(
        anim_rotation0, 
        anim_rotation1, 
        blend_pct)

    anim_translation = anim_translation1 * blend_pct + anim_translation0 * (1.0 - blend_pct)
    
    angle_axis1 = quaternion.to_angle_axis(slerp_quat)

    if curr_joint.name == rig.root_joints[0].name:
        # re-orient the relative translation to model space for the later application of the given orientation matrix 
        anim_rotation_matrix = anim_rotation1.to_matrix()
        inverse_anim_rotation_matrix = float4x4.invert(anim_rotation_matrix)
        world_position_relative_translation = anim_translation2 - anim_translation1
        model_relative_translation = inverse_anim_rotation_matrix.apply(world_position_relative_translation)
        end_anim_translation = curr_joint.move_orientation.to_matrix().apply(model_relative_translation)
        end_anim_translation.y = 0.0        # test test test

        curr_orientation_angle = 0.0 # (next_orientation_angle - prev_orientation_angle) * blend_pct
        rotation_matrix = float4x4.from_angle_axis(float3(0.0, 1.0, 0.0), curr_orientation_angle)

        anim_translation = start_blend_root_joint_world_position + end_anim_translation * blend_pct
        rotation_matrix = rotation_matrix * curr_joint.move_orientation.to_matrix()
    else:
        axis_angle = quaternion.to_angle_axis(slerp_quat)
        rotation_matrix = float4x4.from_angle_axis(axis_angle[0], axis_angle[1])
        
    local_translation_matrix = float4x4.translate(anim_translation)
    local_scale_matrix = float4x4.scale(curr_joint.scale)

    local_anim_matrix = float4x4.concat_matrices([
        local_translation_matrix,
        rotation_matrix,
        local_scale_matrix
    ])

    parent_joint = curr_joint.parent
    if parent_joint != None:
        curr_joint.total_matrix = float4x4.concat_matrices(
            [parent_joint.total_matrix,
             local_anim_matrix])
    else:
        curr_joint.total_matrix = local_anim_matrix

    for child_joint in curr_joint.children:
        traverse_rig_for_blend(
            child_joint,
            rig,
            start_blend_root_joint_world_position,
            next_keyframe_channels,
            prev_keyframe_channels,
            next_frame_time,
            prev_frame_time,
            after_next_frame_time,
            next_orientation_angle,
            prev_orientation_angle,
            blend_pct)

joint_count = 0

##
def traverse_rig_bind(joint):
    global joint_count

    axis, angle = quaternion.to_angle_axis(joint.local_rotation)
    local_rotation_matrix = float4x4.from_angle_axis(axis, angle)
    local_translation_matrix = float4x4.translate(joint.local_translation)
    local_scale_matrix = float4x4.scale(float3(1.0, 1.0, 1.0))
    local_matrix = local_anim_matrix = float4x4.concat_matrices([
        local_translation_matrix,
        local_rotation_matrix,
        local_scale_matrix
    ])

    joint.total_matrix.identity()

    parent_joint = joint.parent
    if parent_joint != None:
        joint.total_matrix = float4x4.concat_matrices(
            [
                parent_joint.total_matrix,
                local_anim_matrix
            ]
        )
    else:
        joint.total_matrix = local_anim_matrix

    color = float3(255.0, 0.0, 0.0)
    print('draw_sphere([{}, {}, {}], 0.02, {}, {}, {}, 255) # {} {}'.format(
            joint.total_matrix.entries[3],
            joint.total_matrix.entries[7],
            joint.total_matrix.entries[11],
            color.x,
            color.y,
            color.z,
            joint_count,
            joint.name))

    for child_joint in joint.children:
        traverse_rig_bind(child_joint)