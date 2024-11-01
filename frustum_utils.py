import math
from vec import *

##
def partition_frustum(
    xform_eye_position,
    look_at_position,
    light_direction,
    camera_near,
    camera_fov,
    larger_frustum_radius, 
    camera_start_distance):

    # camera direction and frustum radius at partition 0
    camera_direction = float3.normalize(look_at_position - xform_eye_position)
    
    # starting position of the frustum at near plane
    frustum_partition_start_postion = xform_eye_position + camera_direction * camera_near + camera_direction * camera_start_distance
    frustum_partition_mid_point = frustum_partition_start_postion + camera_direction * larger_frustum_radius * 0.5

    camera_distance = camera_start_distance + camera_near

    # width of the frustum at partition
    width_0 = 0.75 * camera_distance * math.tan(camera_fov * 0.5)
    width_1 = 1.0 * camera_distance * math.tan(camera_fov * 0.5)

    # tangent of camera
    camera_up = float3(0.0, 1.0, 0.0)
    if abs(camera_direction.y) > 0.98:
        camera_up = float3(1.0, 0.0, 0.0)            
    camera_tangent = float3.normalize(float3.cross(camera_up, camera_direction))

    top_left_light_point = frustum_partition_start_postion + (
        camera_tangent * width_0 * -0.5 + 
        camera_direction * 0.25 * larger_frustum_radius)
    top_right_light_point = frustum_partition_start_postion + (
        camera_tangent * width_0 * 0.5 + 
        camera_direction * 0.25 * larger_frustum_radius)
    bottom_left_light_point = frustum_partition_start_postion + (
        camera_tangent * width_1 * -0.5 + 
        camera_direction * 0.75 * larger_frustum_radius)
    bottom_right_light_point = frustum_partition_start_postion + (
        camera_tangent * width_1 * 0.5 + 
        camera_direction * 0.75 * larger_frustum_radius)

    top_left_light_look_at = top_left_light_point
    top_right_light_look_at = top_right_light_point
    bottom_left_light_look_at = bottom_left_light_point
    bottom_right_light_look_at = bottom_right_light_point

    top_left_light_position = top_left_light_point + light_direction * -1.0 * larger_frustum_radius * 1.1
    top_right_light_position = top_right_light_point + light_direction * -1.0 * larger_frustum_radius * 1.1
    bottom_left_light_position = bottom_left_light_point + light_direction * -1.0 * larger_frustum_radius * 1.1
    bottom_right_light_position = bottom_right_light_point + light_direction * -1.0 * larger_frustum_radius * 1.1

    return\
        frustum_partition_start_postion,\
        frustum_partition_mid_point,\
        top_left_light_look_at,\
        top_right_light_look_at,\
        bottom_left_light_look_at,\
        bottom_right_light_look_at,\
        top_left_light_position,\
        top_right_light_position,\
        bottom_left_light_position,\
        bottom_right_light_position