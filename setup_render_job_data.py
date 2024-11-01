from mat4 import *
import struct
import random

##
def setup_debug_shadow_data(
    view_projection_matrix,
    inverse_view_projection_matrix,
    light_view_projection_matrices,
    screen_width,
    screen_height,
    device,
    render_job):

    data_bytes = b''

    for i in range(16):
        data_bytes += struct.pack('f', view_projection_matrix.entries[i])
    for i in range(16):
        data_bytes += struct.pack('f', inverse_view_projection_matrix.entries[i])

    for light_index in range(len(light_view_projection_matrices)):
        for i in range(16):
            data_bytes += struct.pack('f', light_view_projection_matrices[light_index].entries[i])

    data_bytes += struct.pack('I', screen_width)
    data_bytes += struct.pack('I', screen_height)

    device.queue.write_buffer(
        buffer = render_job.uniform_buffers[0],
        buffer_offset = 0,
        data = data_bytes)
    
    data_bytes = b''
    for i in range(64):
        data_bytes += struct.pack('i', 0)

    device.queue.write_buffer(
        buffer = render_job.attachments['Counters'],
        buffer_offset = 0,
        data = data_bytes)
    
##
def setup_flood_fill_uniform_data(
    device,
    device_buffer,
    device_counter_buffer,
    position_scale,
    brick_dimension,
    brixel_dimension,
    frame_index,
    num_brixels_per_row
):
    
    uniform_bytes = b''
    uniform_bytes += struct.pack('f', position_scale)
    uniform_bytes += struct.pack('f', brick_dimension)
    uniform_bytes += struct.pack('f', brixel_dimension)
    uniform_bytes += struct.pack('I', frame_index)

    uniform_bytes += struct.pack('I', num_brixels_per_row)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)

    if frame_index == 0:
        counter_buffer = b''
        counter_buffer += struct.pack('i', 1)
        device.queue.write_buffer(
            buffer = device_counter_buffer,
            buffer_offset = 0,
            data = counter_buffer)


##
def setup_sdf_ambient_occlusion_uniform_data(
    device,
    device_buffer,
    brick_dimension,
    brixel_dimension,
    screen_width,
    screen_height,
    position_scale,
    frame_index
):

    rand0 = random.randint(1, 100) / 100.0
    rand1 = random.randint(1, 100) / 100.0

    uniform_bytes = b''
    uniform_bytes += struct.pack('f', brick_dimension)
    uniform_bytes += struct.pack('f', brixel_dimension)
    uniform_bytes += struct.pack('f', rand0)
    uniform_bytes += struct.pack('f', rand1)

    uniform_bytes += struct.pack('I', screen_width)
    uniform_bytes += struct.pack('I', screen_height)
    uniform_bytes += struct.pack('f', position_scale)
    uniform_bytes += struct.pack('I', frame_index)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)


##
def setup_sdf_brixel_temporal_restir_uniform_data(
    device,
    device_buffer,
    brick_dimension,
    brixel_dimension,
    screen_width,
    screen_height,
    position_scale,
    frame_index
):

    rand0 = random.randint(1, 100) / 100.0
    rand1 = random.randint(1, 100) / 100.0

    uniform_bytes = b''
    uniform_bytes += struct.pack('f', brick_dimension)
    uniform_bytes += struct.pack('f', brixel_dimension)
    uniform_bytes += struct.pack('f', rand0)
    uniform_bytes += struct.pack('f', rand1)

    uniform_bytes += struct.pack('I', screen_width)
    uniform_bytes += struct.pack('I', screen_height)
    uniform_bytes += struct.pack('f', position_scale)
    uniform_bytes += struct.pack('I', frame_index)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)

##
def setup_brixel_indirect_radiance_uniform_data(
    device,
    device_buffer,
    brick_dimension,
    brixel_dimension,
    screen_width,
    screen_height,
    position_scale,
    frame_index
):

    rand0 = random.randint(1, 100) / 100.0
    rand1 = random.randint(1, 100) / 100.0

    uniform_bytes = b''
    uniform_bytes += struct.pack('f', brick_dimension)
    uniform_bytes += struct.pack('f', brixel_dimension)
    uniform_bytes += struct.pack('f', rand0)
    uniform_bytes += struct.pack('f', rand1)

    uniform_bytes += struct.pack('I', screen_width)
    uniform_bytes += struct.pack('I', screen_height)
    uniform_bytes += struct.pack('f', position_scale)
    uniform_bytes += struct.pack('I', frame_index)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)

##
def setup_screen_space_brixel_radiance_uniform_data(
    device,
    device_buffer,
    brick_dimension,
    brixel_dimension,
    screen_width,
    screen_height,
    position_scale,
    frame_index
):

    rand0 = random.randint(1, 100) / 100.0
    rand1 = random.randint(1, 100) / 100.0

    uniform_bytes = b''
    uniform_bytes += struct.pack('f', brick_dimension)
    uniform_bytes += struct.pack('f', brixel_dimension)
    uniform_bytes += struct.pack('f', rand0)
    uniform_bytes += struct.pack('f', rand1)

    uniform_bytes += struct.pack('I', screen_width)
    uniform_bytes += struct.pack('I', screen_height)
    uniform_bytes += struct.pack('f', position_scale)
    uniform_bytes += struct.pack('I', frame_index)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)

##
def setup_draw_sdf_uniform_data(
    device,
    device_buffer,
    brick_dimension,
    brixel_dimension,
    screen_width,
    screen_height,
    position_scale,
    frame_index,
    camera_position,
    camera_look_direction
):

    rand0 = random.randint(1, 100) / 100.0
    rand1 = random.randint(1, 100) / 100.0

    uniform_bytes = b''
    uniform_bytes += struct.pack('f', brick_dimension)
    uniform_bytes += struct.pack('f', brixel_dimension)
    uniform_bytes += struct.pack('I', screen_width)
    uniform_bytes += struct.pack('I', screen_height)

    uniform_bytes += struct.pack('f', position_scale)
    uniform_bytes += struct.pack('I', frame_index)
    uniform_bytes += struct.pack('f', 0)
    uniform_bytes += struct.pack('f', 0)

    uniform_bytes += struct.pack('f', camera_position.x)
    uniform_bytes += struct.pack('f', camera_position.y)
    uniform_bytes += struct.pack('f', camera_position.z)
    uniform_bytes += struct.pack('f', 1.0)

    uniform_bytes += struct.pack('f', camera_look_direction.x)
    uniform_bytes += struct.pack('f', camera_look_direction.y)
    uniform_bytes += struct.pack('f', camera_look_direction.z)
    uniform_bytes += struct.pack('f', 1.0)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)

##
def setup_simple_svgf_filter_data(
    device,
    device_buffer,
    screen_width,
    screen_height,
    frame_index,
    step,
    luminance_phi,
    luminance_epsilon
):

    uniform_bytes = b''
    uniform_bytes += struct.pack('I', screen_width)
    uniform_bytes += struct.pack('I', screen_height)
    uniform_bytes += struct.pack('I', frame_index)
    uniform_bytes += struct.pack('I', step)
    uniform_bytes += struct.pack('f', luminance_phi)
    uniform_bytes += struct.pack('f', luminance_epsilon)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)
    
##
def setup_taa_uniform_data(
    device,
    device_buffer,
    view_projection_matrix,
    prev_view_projection_matrix,
    screen_width,
    screen_height,
    frame_index
):
    
    uniform_bytes = b''
    
    for i in range(16):
        uniform_bytes += struct.pack('f', view_projection_matrix.entries[i])

    for i in range(16):
        uniform_bytes += struct.pack('f', prev_view_projection_matrix.entries[i])

    uniform_bytes += struct.pack('I', screen_width)
    uniform_bytes += struct.pack('I', screen_height)
    uniform_bytes += struct.pack('I', frame_index)
    
    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)
    
##
def update_character_camera(
    device,
    device_buffer, 
    center,
    light_direction, 
    radius
):
    
    light_up = float3(0.0, 1.0, 0.0)
    if abs(light_direction.y) > 0.98:
        light_up = float3(1.0, 0.0, 0.0)

    light_position = center + light_direction * radius * 1.5
    light_view_matrix = float4x4.view_matrix(
        eye_position = float3(light_position.x, light_position.y * -1.0, light_position.z), 
        look_at = center, 
        up = light_up
    )
    
    view_width = radius * 1.5 * 2.0
    view_height = radius * 1.5 * 2.0
    view_depth = radius * 1.5 * 2.0

    light_projection_matrix = float4x4.orthographic_projection_matrix(
        left = -view_width * 0.5,
        right = view_width * 0.5,
        top = view_height * 0.5,
        bottom = -view_height * 0.5,
        far = view_depth * 0.5,
        near = -view_depth * 0.5,
        inverted = False
    )

    light_view_projection_matrix = light_projection_matrix * light_view_matrix

    uniform_bytes = b''
    for i in range(16):
        uniform_bytes += struct.pack('f', light_projection_matrix.entries[i])

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)
    
##
def update_character_shadow_data(
    device,
    device_buffer,
    view_projection_matrix
):
    uniform_bytes = b''
    for i in range(16):
        uniform_bytes += struct.pack('f', view_projection_matrix.entries[i])

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)
    

##
def update_default_data(
    device,
    device_buffer,
    screen_width,
    screen_height,
    frame_index,
    num_meshes,
    rand0,
    rand1,
    rand2,
    rand3,
    view_projection_matrix,
    prev_view_projection_matrix,
    view_matrix,
    projection_matrix,
    jittered_view_projection_matrix,
    prev_jittered_view_projection_matrix,
    camera_position,
    camera_look_at,
    light_radiance,
    light_direction,
    ambient_occlusion_distance_threshold
):
    
    uniform_bytes = b''
    
    uniform_bytes += struct.pack('i', screen_width)
    uniform_bytes += struct.pack('i', screen_height)
    uniform_bytes += struct.pack('i', frame_index)
    uniform_bytes += struct.pack('i', num_meshes)

    uniform_bytes += struct.pack('f', rand0)
    uniform_bytes += struct.pack('f', rand1)
    uniform_bytes += struct.pack('f', rand2)
    uniform_bytes += struct.pack('f', rand3)

    for i in range(16):
        uniform_bytes += struct.pack('f', view_projection_matrix.entries[i])
    for i in range(16):
        uniform_bytes += struct.pack('f', prev_view_projection_matrix.entries[i])
    for i in range(16):
        uniform_bytes += struct.pack('f', view_matrix.entries[i])
    for i in range(16):
        uniform_bytes += struct.pack('f', projection_matrix.entries[i])
    for i in range(16):
        uniform_bytes += struct.pack('f', jittered_view_projection_matrix.entries[i])
    for i in range(16):
        uniform_bytes += struct.pack('f', prev_jittered_view_projection_matrix.entries[i])

    uniform_bytes += struct.pack('f', camera_position.x)
    uniform_bytes += struct.pack('f', camera_position.y)
    uniform_bytes += struct.pack('f', camera_position.z)
    uniform_bytes += struct.pack('f', 1.0)

    camera_look_dir = float3.normalize(camera_look_at - camera_position)

    uniform_bytes += struct.pack('f', camera_look_dir.x)
    uniform_bytes += struct.pack('f', camera_look_dir.y)
    uniform_bytes += struct.pack('f', camera_look_dir.z)
    uniform_bytes += struct.pack('f', 1.0)

    uniform_bytes += struct.pack('f', light_radiance.x)
    uniform_bytes += struct.pack('f', light_radiance.y)
    uniform_bytes += struct.pack('f', light_radiance.z)
    uniform_bytes += struct.pack('f', 0.8)

    uniform_bytes += struct.pack('f', light_direction.x)
    uniform_bytes += struct.pack('f', light_direction.y)
    uniform_bytes += struct.pack('f', light_direction.z)
    uniform_bytes += struct.pack('f', 1.0)

    uniform_bytes += struct.pack('f', ambient_occlusion_distance_threshold)

    device.queue.write_buffer(
        buffer = device_buffer,
        buffer_offset = 0,
        data = uniform_bytes)