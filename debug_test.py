import struct
from vec import *
from mat4 import *
import numpy as np
import wgpu

##
def debug_load_float4(app, input_byte_array, struct_start):
    ret = float3(0.0, 0.0, 0.0)
    ret.x = struct.unpack('f', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4
    ret.y = struct.unpack('f', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4
    ret.z = struct.unpack('f', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4
    w = struct.unpack('f', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4

    return ret, w, struct_start

##
def debug_load_uint4(app, input_byte_array, struct_start):
    x = struct.unpack('I', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4
    y = struct.unpack('I', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4
    z = struct.unpack('I', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4
    w = struct.unpack('I', input_byte_array[struct_start:struct_start + 4])[0]
    struct_start += 4

    return x, y, z, w, struct_start

##
def debug_load_uint_array(app, input_byte_array, num_entries, struct_start):

    ret = []
    for i in range(num_entries):
        ret.append(struct.unpack('I', input_byte_array[struct_start:struct_start + 4])[0])
        struct_start += 4

    return ret, struct_start

##
def debug_load_float_array(app, input_byte_array, num_entries, struct_start):

    ret = []
    for i in range(num_entries):
        ret.append(struct.unpack('f', input_byte_array[struct_start:struct_start + 4])[0])
        struct_start += 4

    return ret, struct_start

##
def debug_load_float4_array(app, input_byte_array, num_entries, struct_start):

    ret = []
    for i in range(num_entries):
        v, _, struct_start = app.debug_load_float4(input_byte_array, struct_start)
        ret.append(v)

    return ret, struct_start

##
def debug_load_uint4_array(app, input_byte_array, num_entries, struct_start):

    ret = []
    for i in range(num_entries):
        val, struct_start = app.debug_load_uint4(input_byte_array, struct_start)
        ret.append(val)

    return ret, struct_start

##
def test_read_back_buffers(app):
    bvh_node_bytearray = app.device.queue.read_buffer(app.render_job_dict['Build BVH Compute'].attachments['BVH Nodes'])
    bvh_output_data_bytearray = app.device.queue.read_buffer(app.render_job_dict['Build BVH Compute'].uniform_buffers[4])

    bvh_triangle_index_bytearray = app.device.queue.read_buffer(app.render_job_dict['Build BVH Compute'].uniform_buffers[2])

    debug_data_array = app.device.queue.read_buffer(app.render_job_dict['Build BVH Compute'].uniform_buffers[5])
    struct_start = 0
    bounding_boxes, struct_start = app.debug_load_float4_array(debug_data_array, 48, struct_start)
    num_bin_triangles, struct_start = app.debug_load_uint_array(debug_data_array, 24, struct_start)
    
    num_bin_triangles_left, struct_start = app.debug_load_uint_array(debug_data_array, 24, struct_start)
    areas_left, struct_start = app.debug_load_float_array(debug_data_array, 24, struct_start)
    num_bin_triangles_right, struct_start = app.debug_load_uint_array(debug_data_array, 24, struct_start)
    areas_right, struct_start = app.debug_load_float_array(debug_data_array, 24, struct_start)

    bbox_left, struct_start = app.debug_load_float4_array(debug_data_array, 48, struct_start)
    bbox_right, struct_start = app.debug_load_float4_array(debug_data_array, 48, struct_start)

    left_costs, struct_start = app.debug_load_float_array(debug_data_array, 24, struct_start)
    right_costs, struct_start = app.debug_load_float_array(debug_data_array, 24, struct_start)

    node_bbox, struct_start = app.debug_load_float4_array(debug_data_array, 16, struct_start)

    x, y, z, w, struct_start = app.debug_load_uint4(debug_data_array, struct_start)

    struct_start = 0
    verify_triangle_indices = []
    for i in range(1000):
        triangle_index = struct.unpack('I', bvh_triangle_index_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 4
        verify_triangle_indices.append(triangle_index)
        #print('{} {}'.format(i, triangle_index))

    num_nodes = struct.unpack('i', bvh_output_data_bytearray[0:4])[0]

    struct_start = 0
    for i in range(num_nodes):
        bounding_box_min_x = struct.unpack('f', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 4
        bounding_box_min_y = struct.unpack('f', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 4
        bounding_box_min_z = struct.unpack('f', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 8   # add extra 4 for alignment

        bounding_box_max_x = struct.unpack('f', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 4
        bounding_box_max_y = struct.unpack('f', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 4
        bounding_box_max_z = struct.unpack('f', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 8   # add extra 4 for alignment

        left_first = struct.unpack('I', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 4

        num_triangles = struct.unpack('I', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 4

        level = struct.unpack('I', bvh_node_bytearray[struct_start:struct_start + 4])[0]
        struct_start += 8

        center = float3(
            (bounding_box_max_x + bounding_box_min_x) * 0.5,
            (bounding_box_max_y + bounding_box_min_y) * 0.5,
            (bounding_box_max_z + bounding_box_min_z) * 0.5
        )
        print('{} center: ({}, {}, {}) left: {} right: {} num triangles: {} level {}'.format(
            i,
            center.x, 
            center.y,
            center.z,
            left_first, 
            left_first + 1,
            num_triangles,
            level
        ))
        
    debug_data_bytearray = app.device.queue.read_buffer(app.render_job_dict['Build BVH Compute'].uniform_buffers[5])
    struct_start = 0
    
    debug_x0 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_y0 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_z0 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 8

    debug_x1 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_y1 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_z1 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 8

    debug_x2 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_y2 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_z2 = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 8

    debug_index0 = struct.unpack('I', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_index1 = struct.unpack('I', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    debug_index2 = struct.unpack('I', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 8

    centroid_x = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    centroid_y = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 4
    centroid_z = struct.unpack('f', debug_data_bytearray[struct_start:struct_start + 4])[0]
    struct_start += 8


##
def test_skinning_transformations(
    app,
    joint_to_node_mappings, 
    inverse_bind_matrix_data,
    device):

    vertex_positions = app.mesh_data['positions']
    vertex_normals = app.mesh_data['normals']
    vertex_texcoords = app.mesh_data['texcoords']
    vertex_joint_indices = app.mesh_data['joint_indices'] 
    vertex_joint_weights = app.mesh_data['joint_weights']

    app.update_skinning_matrices(
        joint_to_node_mappings = joint_to_node_mappings, 
        inverse_bind_matrix_data = inverse_bind_matrix_data,
        animation_time = 0.2
    )

    skin_matrices = app.mesh_data['skin_matrices']

    # buffer for the skin matrices
    matrix_index = 0
    num_matrix_batches = int(math.ceil(len(skin_matrices) / 16))
    app.uniform_skin_matrix_buffers = [] * num_matrix_batches
    app.uniform_skin_matrix_data = [] * num_matrix_batches
    for i in range(num_matrix_batches):
        if matrix_index >= len(skin_matrices):
            break

        skin_matrices_buffer_data = np.empty(
            [0, 16],
            dtype = np.float32
        )
        for j in range(16):
            if matrix_index >= len(skin_matrices):
                break

            array = np.array([np.array(skin_matrices[matrix_index].entries)], dtype = np.float32)
            skin_matrices_buffer_data = np.concatenate((skin_matrices_buffer_data, array))

            matrix_index += 1

        app.uniform_skin_matrix_buffers.append(
            device.create_buffer_with_data(
                data = skin_matrices_buffer_data, 
                usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_SRC)
        )

        app.uniform_skin_matrix_data.append(skin_matrices_buffer_data)
    
    num_meshes = len(vertex_positions)
    mesh_xform_positions = []
    mesh_xform_normals = []

    # transform vertices with skinning matrix
    for mesh_index in range(num_meshes):
        num_vertex_positions = len(vertex_positions[mesh_index])

        xform_positions = [float3(0.0, 0.0, 0.0)] * num_vertex_positions
        xform_normals = [float3(0.0, 0.0, 0.0)] * num_vertex_positions

        positions = vertex_positions[mesh_index]
        normals = vertex_normals[mesh_index]
        texcoords = vertex_texcoords[mesh_index]
        joint_indices = vertex_joint_indices[mesh_index]
        joint_weights = vertex_joint_weights[mesh_index]

        for vertex_index in range(num_vertex_positions):
            vertex_position = positions[vertex_index]
            vertex_normal = normals[vertex_index]

            # blend transformed positions with joint weights    
            total_xform = float3(0.0, 0.0, 0.0)
            total_xform_normal = float3(0.0, 0.0, 0.0)
            total_weights = 0.0
            for i in range(4):
                vertex_joint_index = joint_indices[vertex_index][i]
                vertex_joint_weight = joint_weights[vertex_index][i]

                skinning_matrix = skin_matrices[vertex_joint_index]
                normal_matrix = float4x4([
                    skinning_matrix.entries[0], skinning_matrix.entries[1], skinning_matrix.entries[2], 0.0,
                    skinning_matrix.entries[4], skinning_matrix.entries[5], skinning_matrix.entries[6], 0.0,
                    skinning_matrix.entries[8], skinning_matrix.entries[9], skinning_matrix.entries[10], 0.0,
                    skinning_matrix.entries[12], skinning_matrix.entries[13], skinning_matrix.entries[14], 1.0,
                ])

                xform = skinning_matrix.apply(float3(vertex_position[0], vertex_position[1], vertex_position[2]))
                xform_normal = normal_matrix.apply(float3(vertex_normal[0], vertex_normal[1], vertex_normal[2]))

                total_xform = total_xform + float3(
                    xform.x * vertex_joint_weight,
                    xform.y * vertex_joint_weight,
                    xform.z * vertex_joint_weight)

                total_xform_normal = total_xform_normal + float3(
                    xform_normal.x * vertex_joint_weight,
                    xform_normal.y * vertex_joint_weight,
                    xform_normal.z * vertex_joint_weight)

                total_weights += vertex_joint_weight

            xform_positions[vertex_index] = total_xform
            xform_normals[vertex_index] = total_xform_normal
            
        mesh_xform_positions.append(xform_positions)
        mesh_xform_normals.append(xform_normals)
    
    return mesh_xform_positions, mesh_xform_normals

##
def test_readback_bvh_as_obj(app):
    for debug_level in range(20):

        bvh_process_info_bytearray = app.device.queue.read_buffer(app.render_job_dict['Initialize Intermediate Nodes Compute'].attachments['BVH Process Info'])
        bvh_node_level_range_bytearray = app.device.queue.read_buffer(app.render_job_dict['Finish BVH Step Compute'].attachments['Node Level Range'])
        bvh_node_bytearray = app.device.queue.read_buffer(app.render_job_dict['Initialize Intermediate Nodes Compute'].attachments['Intermediate Nodes'])
        
        struct_start = 0
        node_level, struct_start = app.debug_load_uint_array(bvh_node_level_range_bytearray, 64, struct_start)
        struct_start = 0
        process_info, struct_start = app.debug_load_uint_array(bvh_process_info_bytearray, 20, struct_start)
        struct_start = 0
        node_bbox, struct_start = app.debug_load_float4_array(bvh_node_bytearray, node_level[debug_level * 3 + 1] * 4, struct_start)

        output_str = ''

        node_index = node_level[debug_level * 3] * 4
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

        while True:
            
            end_index = node_level[debug_level * 3 + 1] * 4
            if node_index >= end_index:
                break
            
            bbox_center = node_bbox[node_index]
            bbox_min = node_bbox[node_index + 1]
            bbox_max = node_bbox[node_index + 2]
            node_index += 4
            bbox_size = bbox_max - bbox_min
            half_bbox_size = bbox_size * 0.5

            pos0 = bbox_center + float3(half_bbox_size.x, half_bbox_size.y, -half_bbox_size.z)
            pos1 = bbox_center + float3(half_bbox_size.x, -half_bbox_size.y, -half_bbox_size.z)
            pos2 = bbox_center + float3(half_bbox_size.x, half_bbox_size.y, half_bbox_size.z)
            pos3 = bbox_center + float3(half_bbox_size.x, -half_bbox_size.y, half_bbox_size.z)

            pos4 = bbox_center + float3(-half_bbox_size.x, half_bbox_size.y, -half_bbox_size.z)
            pos5 = bbox_center + float3(-half_bbox_size.x, -half_bbox_size.y, -half_bbox_size.z)
            pos6 = bbox_center + float3(-half_bbox_size.x, half_bbox_size.y, half_bbox_size.z)
            pos7 = bbox_center + float3(-half_bbox_size.x, -half_bbox_size.y, half_bbox_size.z)

            output_str += 'v {} {} {}\n'.format(pos0.x, pos0.y, pos0.z)
            output_str += 'v {} {} {}\n'.format(pos1.x, pos1.y, pos1.z)
            output_str += 'v {} {} {}\n'.format(pos2.x, pos2.y, pos2.z)
            output_str += 'v {} {} {}\n'.format(pos3.x, pos3.y, pos3.z)

            output_str += 'v {} {} {}\n'.format(pos4.x, pos4.y, pos4.z)
            output_str += 'v {} {} {}\n'.format(pos5.x, pos5.y, pos5.z)
            output_str += 'v {} {} {}\n'.format(pos6.x, pos6.y, pos6.z)
            output_str += 'v {} {} {}\n'.format(pos7.x, pos7.y, pos7.z)

        for normal in face_normals:
            output_str += 'vn {} {} {}\n'.format(normal.x, normal.y, normal.z)

        node_index = node_level[debug_level * 3] * 4
        vertex_index = 0
        while True:
            
            end_index = node_level[debug_level * 3 + 1] * 4
            if node_index >= end_index:
                break
            
            j = 0
            while True:
                if j >= len(face_indices):
                    break 

                output_str += 'f {}/1/{} {}/1/{} {}/1/{} {}/1/{}\n'.format(
                    face_indices[j] + vertex_index * 8,
                    int(j / 4 + 1),
                    face_indices[j + 1] + vertex_index * 8,
                    int(j / 4 + 1),
                    face_indices[j + 2] + vertex_index * 8,
                    int(j / 4 + 1),
                    face_indices[j + 3] + vertex_index * 8,
                    int(j / 4 + 1))

                j += 4

            vertex_index += 1
            node_index += 4

        full_path = 'c:\\Users\\Dingwings\\demo-models\\bvh-2-level-{}.obj'.format(debug_level)
        file = open(full_path, 'w')
        file.write(output_str)
        file.close()