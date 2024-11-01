import struct

from vec import *

##
def save_obj(
    file_path,
    voxel_positions,
    voxel_size):

    face_indices = [
        1, 5, 7, 3,
        4, 3, 7, 8,
        8, 7, 5, 6,
        6, 2, 4, 8,
        2, 1, 3, 4,
        6, 5, 1, 2,
    ]
    face_normals = [
        float3(0.0000, 1.0000, 0.0000),
        float3(0.0000, 0.0000, 1.0000),
        float3(-1.0000, 0.0000, 0.0000),
        float3(0.0000, -1.0000, 0.0000),
        float3(1.0000, 0.0000, 0.0000),
        float3(0.0000, 0.0000, -1.0000),
    ]

    curr_output_str = ''

    total_output_str = ''

    num_voxels = len(voxel_positions)
    #if num_voxels >= 50000:
    #    num_voxels = 50000

    half_voxel_size = voxel_size * 0.5
    for voxel_index in range(num_voxels):
        voxel_position = voxel_positions[voxel_index]

        bbox_center = voxel_position
        
        pos0 = bbox_center + float3(half_voxel_size, half_voxel_size, -half_voxel_size)
        pos1 = bbox_center + float3(half_voxel_size, -half_voxel_size, -half_voxel_size)
        pos2 = bbox_center + float3(half_voxel_size, half_voxel_size, half_voxel_size)
        pos3 = bbox_center + float3(half_voxel_size, -half_voxel_size, half_voxel_size)

        pos4 = bbox_center + float3(-half_voxel_size, half_voxel_size, -half_voxel_size)
        pos5 = bbox_center + float3(-half_voxel_size, -half_voxel_size, -half_voxel_size)
        pos6 = bbox_center + float3(-half_voxel_size, half_voxel_size, half_voxel_size)
        pos7 = bbox_center + float3(-half_voxel_size, -half_voxel_size, half_voxel_size)

        curr_output_str += 'v {} {} {}\n'.format(pos0.x, pos0.y, pos0.z)
        curr_output_str += 'v {} {} {}\n'.format(pos1.x, pos1.y, pos1.z)
        curr_output_str += 'v {} {} {}\n'.format(pos2.x, pos2.y, pos2.z)
        curr_output_str += 'v {} {} {}\n'.format(pos3.x, pos3.y, pos3.z)

        curr_output_str += 'v {} {} {}\n'.format(pos4.x, pos4.y, pos4.z)
        curr_output_str += 'v {} {} {}\n'.format(pos5.x, pos5.y, pos5.z)
        curr_output_str += 'v {} {} {}\n'.format(pos6.x, pos6.y, pos6.z)
        curr_output_str += 'v {} {} {}\n'.format(pos7.x, pos7.y, pos7.z)

        if len(curr_output_str) >= 10000:
            total_output_str += curr_output_str
            curr_output_str = ''

        if voxel_index % 1000 == 0:
            print('write voxel vertex {} / {}'.format(voxel_index, num_voxels))

    for normal in face_normals:
        curr_output_str += 'vn {} {} {}\n'.format(normal.x, normal.y, normal.z)

    total_output_str += curr_output_str
    curr_output_str = ''

    for voxel_index in range(num_voxels):
        face_index = 0
        while face_index < len(face_indices):
            curr_output_str += 'f {}/1/{} {}/1/{} {}/1/{} {}/1/{}\n'.format(
                face_indices[face_index] + voxel_index * 8,
                int(face_index / 4 + 1),
                face_indices[face_index + 1] + voxel_index * 8,
                int(face_index / 4 + 1),
                face_indices[face_index + 2] + voxel_index * 8,
                int(face_index / 4 + 1),
                face_indices[face_index + 3] + voxel_index * 8,
                int(face_index / 4 + 1))
            
            face_index += 4

            if len(curr_output_str) >= 10000:
                total_output_str += curr_output_str
                curr_output_str = ''

        if voxel_index % 1000 == 0:
            print('write voxel face {} / {}'.format(voxel_index, num_voxels))

    total_output_str += curr_output_str

    file = open(file_path, 'w')
    file.write(total_output_str)
    file.close()

    print('')


##
def debug_voxels(
    device,
    device_brick_buffer,
    device_brixel_buffer,
    min_brick_position,
    brick_dimension,
    brixel_dimension):

    debug_brick_buffer = device.queue.read_buffer(device_brick_buffer)
    debug_brixel_buffer = device.queue.read_buffer(device_brixel_buffer)

    voxels = []
    for i in range(5000):
        #if i != 682:
        #    continue

        array_index = i * 16
        brick_position_x = struct.unpack('f', debug_brick_buffer[array_index:array_index + 4])[0]
        brick_position_y = struct.unpack('f', debug_brick_buffer[array_index + 4:array_index + 8])[0]
        brick_position_z = struct.unpack('f', debug_brick_buffer[array_index + 8:array_index + 12])[0]
        brixel_index = struct.unpack('I', debug_brick_buffer[array_index + 12:array_index + 16])[0]

        brick_position = float3(brick_position_x, brick_position_y, brick_position_z)

        print('debug_voxels brick position ({}, {}, {})'.format(brick_position.x, brick_position.y, brick_position.z))

        if (brick_position_x != 0.0 or brick_position_y != 0.0 or brick_position_z != 0.0):
            #voxels.append(brick_position)
            
            brixel_array_byte_index = brixel_index * 512 * 4
            for z in range(8):
                for y in range(8):
                    for x in range(8):
                        index = (z * 64 + y * 8 + x) * 4
                        index += brixel_array_byte_index
                        try:
                            brixel = struct.unpack('I', debug_brixel_buffer[index: index + 4])[0]
                            if brixel > 0:
                                brixel_position = brick_position + float3(x * brixel_dimension, y * brixel_dimension, z * brixel_dimension)
                                #brixel_position = brick_position + float3(
                                #    x * brixel_dimension + brixel_dimension * 0.5, 
                                #    -(y * brixel_dimension + brixel_dimension * 0.5), 
                                #    z * brixel_dimension + brixel_dimension * 0.5)
                                voxels.append(brixel_position)
                        except Exception as e:
                            print('exception {}'.format(e))
                            
    save_obj(
        file_path = "c:\\Users\\Dingwings\\demo-models\\voxels\\voxels.obj",
        voxel_positions = voxels,
        voxel_size = brixel_dimension
    )

    #save_obj(
    #    voxel_positions = voxels,
    #    voxel_size = brick_dimension
    #)

    print('')


##
def debug_brixels(
    device,
    device_brick_buffer,
    device_brixel_distance_buffer,
    device_brick_to_brixel_mapping_buffer,
    min_brick_position,
    max_brick_per_row,
    brick_dimension,
    brixel_dimension,
    center_position, 
    minimum_distance,
    maximum_distance):
    
    debug_brick_buffer = device.queue.read_buffer(device_brick_buffer)
    brixel_distance_buffer = device.queue.read_buffer(device_brixel_distance_buffer)
    brick_to_brixel_mapping_buffer = device.queue.read_buffer(device_brick_to_brixel_mapping_buffer)

    voxel_positions = []
    for brick_index in range(5000):
        #if brick_index != 623:
        #    continue

        #if brick_index != 622:
        #    continue

        #if brick_index != 682:
        #    continue

        brick_buffer_index = brick_index * 4

        brixel_index = struct.unpack('I', brick_to_brixel_mapping_buffer[brick_buffer_index:brick_buffer_index + 4])[0]
        if brixel_index > 0:
            brick_position_buffer_index = brick_index * 16

            brick_x = struct.unpack('f', debug_brick_buffer[brick_position_buffer_index: brick_position_buffer_index + 4])[0]
            brick_y = struct.unpack('f', debug_brick_buffer[brick_position_buffer_index + 4: brick_position_buffer_index + 8])[0]
            brick_z = struct.unpack('f', debug_brick_buffer[brick_position_buffer_index + 8: brick_position_buffer_index + 12])[0]

            brick_position = float3(
                brick_x,
                brick_y,
                brick_z)

            print('debug_brixels brick position ({}, {}, {})'.format(brick_position.x, brick_position.y, brick_position.z))

            #voxel_positions.append(brick_position + float3(brick_dimension * 0.5, -brick_dimension * 0.5, brick_dimension * 0.5))

            starting_brixel_index = brixel_index
            starting_brixel_byte_index = starting_brixel_index * 512 * 4
            for z in range(8):
                for y in range(8):
                    for x in range(8):
                        brixel_index = z * 64 + y * 8 + x
                        brixel_byte_index = starting_brixel_byte_index + brixel_index * 4
                        brixel_distance = struct.unpack('f', brixel_distance_buffer[brixel_byte_index:brixel_byte_index + 4])[0]
                        
                        if (brixel_distance >= minimum_distance and brixel_distance < maximum_distance):
                            brixel_position = brick_position + float3(
                                x * brixel_dimension, 
                                y * brixel_dimension, 
                                z * brixel_dimension)
                            voxel_positions.append(brixel_position)

    save_obj(
        file_path = 'c:\\Users\\Dingwings\\demo-models\\voxels\\brixels.obj',
        voxel_positions = voxel_positions,
        voxel_size = brixel_dimension
    )

    print('')