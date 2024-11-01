from vec import *

import os
import struct

import pywavefront

class MeshResult:
    def __init__(
        self,
        total_positions, 
        total_normals, 
        total_uvs, 
        total_face_position_indices, 
        total_face_normal_indices, 
        total_face_uv_indices, 
        material_info, 
        total_mesh_names, 
        total_materials, 
        triangle_ranges,
        mesh_vertices, 
        mesh_face_vertex_indices,
        total_position_bytes, 
        total_normal_bytes, 
        total_uv_bytes, 
        total_face_position_indices_bytes, 
        total_face_normal_indices_bytes, 
        total_face_uv_indices_bytes, 
        total_triangle_range_bytes,
        mesh_vertex_bytes,
        mesh_face_index_bytes,
        total_triangle_positions_bytes,
        mesh_triangle_vertex_count_range,
        mesh_position_ranges,
        mesh_min_positions,
        mesh_max_positions
    ):
        
        self.total_positions                     = total_positions  
        self.total_normals                       = total_normals          
        self.total_uvs                           = total_uvs           
        self.total_face_position_indices         = total_face_position_indices                   
        self.total_face_normal_indices           = total_face_normal_indices             
        self.total_face_uv_indices               = total_face_uv_indices               
        self.material_info                       = material_info              
        self.total_mesh_names                    = total_mesh_names                   
        self.total_materials                     = total_materials               
        self.triangle_ranges                     = triangle_ranges

        self.mesh_vertices                             = mesh_vertices                                                                                         
        self.mesh_face_vertex_indices                  = mesh_face_vertex_indices                                                            
        self.total_position_bytes                      = total_position_bytes                                                                          
        self.total_normal_bytes                        = total_normal_bytes                                                                                            
        self.total_uv_bytes                            = total_uv_bytes                                                                                                                
        self.total_face_position_indices_bytes         = total_face_position_indices_bytes                                                                                           
        self.total_face_normal_indices_bytes           = total_face_normal_indices_bytes                                                                                             
        self.total_face_uv_indices_bytes               = total_face_uv_indices_bytes                                                                                   
        self.total_triangle_range_bytes                = total_triangle_range_bytes  

        self.mesh_vertex_bytes                          = mesh_vertex_bytes
        self.mesh_face_index_bytes                      = mesh_face_index_bytes                                           

        self.total_triangle_positions_bytes             = total_triangle_positions_bytes
        self.mesh_triangle_vertex_count_range           = mesh_triangle_vertex_count_range
        
        self.mesh_position_ranges                       = mesh_position_ranges

        self.mesh_min_positions                         = mesh_min_positions
        self.mesh_max_positions                         = mesh_max_positions


##
def load_materials(file_path):
    file = open(file_path, 'rb')
    file_content = file.read().decode('utf-8')
    file.close()
    file_size = len(file_content)

    materials = []

    curr_position = 0
    while True:
        if curr_position >= file_size:
            break

        end_line_position = file_content.find('\n', curr_position)
        line = file_content[curr_position:end_line_position]
        curr_position = end_line_position + 1

        tokens = line.split()
        if len(tokens) <= 0:
            continue

        if tokens[0] == 'newmtl':
            materials.append({})
            curr_material = materials[len(materials) - 1]
            curr_material['name'] = tokens[1]

        elif tokens[0] == 'Ks':
            curr_material['specular'] = float3(
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]))

        elif tokens[0] == 'Kd':
            curr_material['diffuse'] = float3(
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]))

        elif tokens[0] == 'Ke':
            curr_material['emissive'] = float3(
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]))
            
        elif tokens[0] == 'd':
            curr_material['transparency'] = float(tokens[1])

        elif tokens[1] == 'map_Kd':
            curr_material['texture'] = tokens[1]

    return materials

##
def get_position_range(positions):
    
    position_length = len(positions)
    if len(positions[len(positions) - 1]) <= 0:
        position_length -= 1

    ret = [0, 0]
    num_positions = 0
    for index in range(position_length):
         num_positions += len(positions[index])
         if index == position_length - 2:
             ret[0] = num_positions

    ret[1] = num_positions

    return ret

##
def load_obj2(file_path):
    file_base_name = os.path.basename(file_path)
    file_directory = file_path[:file_path.find(file_base_name)]
    file_extension_start = file_base_name.rfind('.')
    material_base_name = file_base_name[:file_extension_start] + '.mtl'
    material_file_path = os.path.join(file_directory, material_base_name)


    file = open(file_path, 'rb')
    file_content = file.read().decode('utf-8')
    file.close()

    file_size = len(file_content)

    mesh_names = []
    material_info = []

    positions = []
    normals = []
    uvs = []

    face_position_indices = []
    face_norm_indices = []
    face_uv_indices = []

    #positions.append([])
    normals.append([])
    uvs.append([])

    total_mesh_names = []
    total_positions = []
    total_normals = []
    total_uvs = []
    total_face_position_indices = []
    total_face_normal_indices = []
    total_face_uv_indices = []
    total_face_position_indices.append([])
    total_face_normal_indices.append([])
    total_face_uv_indices.append([])
    curr_total_material_mesh_index = 0

    face_position_indices.append([])
    face_norm_indices.append([])
    face_uv_indices.append([])

    material_file = ''

    material_database = load_materials(material_file_path)

    total_materials = []

    separate_mesh_position_ranges = []
    curr_num_total_positions = 0

    curr_mesh_index = 0
    curr_position = 0
    while True:
        if curr_position >= file_size:
            break
        
        end_line_position = file_content.find('\n', curr_position)
        line = file_content[curr_position:end_line_position]
        curr_position = end_line_position + 1

        tokens = line.split()
        if tokens[0] == 'v':
            if len(positions) <= curr_mesh_index:
                positions.append([])

            positions[curr_mesh_index].append(float3(
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]))
            )

            total_positions.append(float3(
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]))
            )

            curr_num_total_positions += 1

        elif tokens[0] == 'vn':
            if len(normals) <= curr_mesh_index:
                normals.append([])

            normals[curr_mesh_index].append(float3(
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]))
            )

            total_normals.append(float3(
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]))
            )

        elif tokens[0] == 'vt':
            if len(uvs) <= curr_mesh_index:
                uvs.append([])

            uvs[curr_mesh_index].append(float3(
                float(tokens[1]),
                float(tokens[2]),
                0.0)
            )
        
            total_uvs.append(float3(
                float(tokens[1]),
                float(tokens[2]),
                0.0)
            )

        elif tokens[0] == 'f':
            if len(face_position_indices) <= curr_mesh_index:
                face_position_indices.append([])

            if len(face_uv_indices) <= curr_mesh_index:
                face_uv_indices.append([])

            if len(face_norm_indices) <= curr_mesh_index:
                face_norm_indices.append([])

            triangle_position_indices = []
            triangle_uv_indices = []
            triangle_normal_indices = []
            for i in range(1, 4):
                pos_uv_norm = tokens[i].split('/')
                face_pos = int(pos_uv_norm[0]) - 1
                triangle_position_indices.append(face_pos)

                if len(pos_uv_norm) > 1:
                    face_uv = int(pos_uv_norm[1]) - 1
                    triangle_uv_indices.append(face_uv)

                if len(pos_uv_norm) > 2:
                    face_norm = int(pos_uv_norm[2]) - 1
                    triangle_normal_indices.append(face_norm)

            face_position_indices[curr_mesh_index].append(triangle_position_indices)
            if len(triangle_uv_indices) > 0:
                face_uv_indices[curr_mesh_index].append(triangle_uv_indices)
            
            if len(triangle_normal_indices) > 0:
                face_norm_indices[curr_mesh_index].append(triangle_normal_indices)

            if len(total_face_position_indices) <= curr_total_material_mesh_index:
                total_face_position_indices.append([])
                position_count_range = get_position_range(positions)
                separate_mesh_position_ranges.append(position_count_range)

            if len(total_face_normal_indices) <= curr_total_material_mesh_index:
                total_face_normal_indices.append([])

            if len(total_face_uv_indices) <= curr_total_material_mesh_index:
                total_face_uv_indices.append([])

            total_face_position_indices[curr_total_material_mesh_index].append(triangle_position_indices)
            if len(triangle_uv_indices) > 0:
                total_face_uv_indices[curr_total_material_mesh_index].append(triangle_uv_indices)
            
            if len(total_face_normal_indices) > 0:
                total_face_normal_indices[curr_total_material_mesh_index].append(triangle_normal_indices)


        elif tokens[0] == 'o':
            positions.append([])
            mesh_names.append(tokens[1])
            curr_mesh_index = len(mesh_names) - 1

            if len(total_face_position_indices[len(total_face_position_indices) - 1]) > 0:
                total_face_position_indices.append([])                
                curr_total_material_mesh_index += 1
                position_count_range = get_position_range(positions)
                separate_mesh_position_ranges.append(position_count_range)

        elif tokens[0] == 'mtllib':
            material_file = tokens[1]

        elif tokens[0] == 'usemtl':
            if len(face_position_indices) <= curr_mesh_index:
                face_position_indices.append([])

            if(len(material_info) > 0):
                material_info[len(material_info) - 1]['end_face_index'] = len(face_position_indices[curr_mesh_index])

            material_info.append(
                {
                    'name': tokens[1], 
                    'start_face_index': len(face_position_indices[curr_mesh_index]),
                    'mesh_name': mesh_names[len(mesh_names) - 1]
                }
            )

            found_material = False
            for material in material_database:
                if tokens[1] == material['name']:
                    total_materials.append(material)
                    found_material = True 
                    break
            assert(found_material == True)

            # different material on the same mesh, split the mesh
            if len(total_face_position_indices[curr_total_material_mesh_index]) > 0:
                total_face_position_indices.append([])
                curr_total_material_mesh_index += 1
                assert(len(material_info) == len(total_face_position_indices))

                position_count_range = get_position_range(positions)
                separate_mesh_position_ranges.append(position_count_range)

                
            if len(total_mesh_names) <= curr_total_material_mesh_index:
                total_mesh_names.append(mesh_names[len(mesh_names) - 1] + '-' + tokens[1])
                assert(len(total_mesh_names) == len(material_info))

    # last mesh
    position_count_range = get_position_range(positions)
    separate_mesh_position_ranges.append(position_count_range)

    triangle_ranges = []
    start_num_triangles = 0
    num_total_triangles = 0
    for face_position_indices in total_face_position_indices:
        start_num_triangles = num_total_triangles
        num_total_triangles += len(face_position_indices)
        end_num_triangles = num_total_triangles
        triangle_ranges.append([start_num_triangles, end_num_triangles])

    mesh_position_ranges = []
    curr_num_positions = 0
    for mesh_positions in positions:
        curr_num_positions += len(mesh_positions)
        mesh_position_ranges.append(curr_num_positions)
    
    num_mesh_positions = len(total_face_position_indices)
    while True:
        erased = False
        for index in range(num_mesh_positions):
            if len(total_face_position_indices[index]) <= 0:
                total_face_position_indices.pop(index)
                erased = True
                break
        if erased == False:
            break
        num_mesh_positions = len(total_face_position_indices)

    return total_positions, total_normals, total_uvs, total_face_position_indices, total_face_normal_indices, total_face_uv_indices, material_info, total_mesh_names, total_materials, triangle_ranges, mesh_position_ranges, separate_mesh_position_ranges

class Vertex:
    def __init__(self, position, normal, uv, mesh_id):
        self.position = position
        self.normal = normal
        self.uv = uv
        self.mesh_id = mesh_id

##
def make_vertices(
    total_positions,
    total_normals,
    total_uvs,
    total_face_position_indices,
    total_face_normal_indices,
    total_face_uv_indices
):

    mesh_vertices = []
    mesh_face_vertex_indices = []

    vertex_dict = {}

    for mesh_index in range(len(total_face_position_indices)):
        total_face_position_index = total_face_position_indices[mesh_index]
        total_face_normal_index = total_face_normal_indices[mesh_index]
        total_face_uv_index = total_face_uv_indices[mesh_index]

        for face_index in range(len(total_face_position_index)):
            position_index = total_face_position_index[face_index]
            normal_index = total_face_normal_index[face_index]
            uv_index = total_face_uv_index[face_index]
            
            for k in range(3):
                pos_face_index = position_index[k]
                norm_face_index = normal_index[k]
                uv_face_index = uv_index[k]

                position = total_positions[pos_face_index]
                normal = total_normals[norm_face_index]
                uv = total_uvs[uv_face_index]

                # build dictionary
                str_x = f'{position.x:.{4}f}'
                str_y = f'{position.y:.{4}f}'
                str_z = f'{position.z:.{4}f}'
                str_normal_x = f'{normal.x:.{4}f}'
                str_normal_y = f'{normal.y:.{4}f}'
                str_normal_z = f'{normal.z:.{4}f}'
                str_uv_x = f'{uv.x:.{4}f}'
                str_uv_y = f'{uv.y:.{4}f}'
        
                vertex_key = str_x + ',' + str_y + ',' + str_z + ',' + str_normal_x + ',' + str_normal_y + ',' + str_normal_z + ',' + str_uv_x + ',' + str_uv_y

                if not vertex_key in vertex_dict:
                    mesh_vertices.append(
                        Vertex(
                            position = position,
                            normal = normal,
                            uv = uv, 
                            mesh_id = mesh_index + 1))
                    vertex_index = len(mesh_vertices) - 1

                    vertex_dict[vertex_key] = vertex_index
                    mesh_face_vertex_indices.append(vertex_index)

                else:
                    vertex_index = vertex_dict[vertex_key]
                    mesh_face_vertex_indices.append(vertex_index)

    return mesh_vertices, mesh_face_vertex_indices

##
def convert_to_bytes(
    total_positions, 
    total_normals, 
    total_uvs, 
    total_face_position_indices,
    total_face_normal_indices, 
    total_face_uv_indices,
    triangle_ranges
):

    total_position_bytes = b''
    for position in total_positions:
        total_position_bytes += struct.pack('f', position.x)
        total_position_bytes += struct.pack('f', position.y)
        total_position_bytes += struct.pack('f', position.z)
        total_position_bytes += struct.pack('f', 1.0)

    total_normal_bytes = b''
    for normal in total_normals:
        total_normal_bytes += struct.pack('f', normal.x)
        total_normal_bytes += struct.pack('f', normal.y)
        total_normal_bytes += struct.pack('f', normal.z)
        total_normal_bytes += struct.pack('f', 1.0)

    total_uv_bytes = b''
    for uv in total_uvs:
        total_uv_bytes += struct.pack('f', uv.x)
        total_uv_bytes += struct.pack('f', uv.y)
        total_uv_bytes += struct.pack('f', 0.0)
        total_uv_bytes += struct.pack('f', 0.0)

    total_face_position_indices_bytes = b''
    for face_position_indices in total_face_position_indices:
        for face_position_index in face_position_indices:
            for i in range(len(face_position_index)):
                total_face_position_indices_bytes += struct.pack('I', face_position_index[i])

    total_face_normal_indices_bytes = b''
    for face_normal_indices in total_face_normal_indices:
        for face_normal_index in face_normal_indices:
            for i in range(len(face_normal_index)):
                total_face_normal_indices_bytes += struct.pack('I', face_normal_index[i])

    total_face_uv_indices_bytes = b''
    for face_uv_indices in total_face_uv_indices:
        for face_uv_index in face_uv_indices:
            for i in range(len(face_uv_index)):
                total_face_uv_indices_bytes += struct.pack('I', face_uv_index[i])
    
    total_triangle_range_bytes = b''
    for triangle_range in triangle_ranges:
        total_triangle_range_bytes += struct.pack('I', triangle_range[0])
        total_triangle_range_bytes += struct.pack('I', triangle_range[1])

    return total_position_bytes, total_normal_bytes, total_uv_bytes, total_face_position_indices_bytes, total_face_normal_indices_bytes, total_face_uv_indices_bytes, total_triangle_range_bytes

##
def convert_vertex_buffers_to_bytes(
    mesh_vertices,
    mesh_face_vertex_indices):

    mesh_vertex_bytes = b''
    for vertex in mesh_vertices:
        mesh_vertex_bytes += struct.pack('f', vertex.position.x)
        mesh_vertex_bytes += struct.pack('f', vertex.position.y)
        mesh_vertex_bytes += struct.pack('f', vertex.position.z)
        mesh_vertex_bytes += struct.pack('f', float(vertex.mesh_id))

        mesh_vertex_bytes += struct.pack('f', vertex.uv.x)
        mesh_vertex_bytes += struct.pack('f', vertex.uv.y)
        mesh_vertex_bytes += struct.pack('f', 0.0)
        mesh_vertex_bytes += struct.pack('f', 0.0)

        mesh_vertex_bytes += struct.pack('f', vertex.normal.x)
        mesh_vertex_bytes += struct.pack('f', vertex.normal.y)
        mesh_vertex_bytes += struct.pack('f', vertex.normal.z)
        mesh_vertex_bytes += struct.pack('f', 1.0)

    face_index_bytes = b''
    for face_index in mesh_face_vertex_indices:
        face_index_bytes += struct.pack('I', face_index)

    return mesh_vertex_bytes, face_index_bytes

##
def load_obj_file(file_path):
    total_positions, total_normals, total_uvs, total_face_position_indices, total_face_normal_indices, total_face_uv_indices, material_info, total_mesh_names, total_materials, triangle_ranges, mesh_position_ranges, separate_mesh_ranges = load_obj2(file_path)
    assert(len(separate_mesh_ranges) == len(total_face_position_indices))

    num_meshes = len(total_face_position_indices)
    total_mesh_vertices = []
    for mesh_index in range(num_meshes):
        mesh_positions = []
        mesh_vertex_range = separate_mesh_ranges[mesh_index]
        num_mesh_positions = mesh_vertex_range[1] - mesh_vertex_range[0]
        for i in range(num_mesh_positions):
            position = total_positions[i + mesh_vertex_range[0]]
            mesh_positions.append(position)

        total_mesh_vertices.append(mesh_positions)

    total_num_triangle_vertices = 0
    mesh_triangle_vertex_count_range = []
    total_triangle_positions_bytes = b''
    for mesh_index in range(len(total_face_position_indices)):
        mesh_face_position_indices = total_face_position_indices[mesh_index]
        num_triangles = int(len(mesh_face_position_indices))
        num_triangle_vertices = 0
        for triangle_index in range(num_triangles):
            triangle_indices = mesh_face_position_indices[triangle_index]
            pos0 = total_positions[triangle_indices[0]]
            pos1 = total_positions[triangle_indices[1]]
            pos2 = total_positions[triangle_indices[2]]

            total_triangle_positions_bytes += struct.pack('f', pos0.x)
            total_triangle_positions_bytes += struct.pack('f', pos0.y)
            total_triangle_positions_bytes += struct.pack('f', pos0.z)
            total_triangle_positions_bytes += struct.pack('f', 1.0)

            total_triangle_positions_bytes += struct.pack('f', pos1.x)
            total_triangle_positions_bytes += struct.pack('f', pos1.y)
            total_triangle_positions_bytes += struct.pack('f', pos1.z)
            total_triangle_positions_bytes += struct.pack('f', 1.0)

            total_triangle_positions_bytes += struct.pack('f', pos2.x)
            total_triangle_positions_bytes += struct.pack('f', pos2.y)
            total_triangle_positions_bytes += struct.pack('f', pos2.z)
            total_triangle_positions_bytes += struct.pack('f', 1.0)

            num_triangle_vertices += 3

        total_num_triangle_vertices += num_triangle_vertices
        mesh_triangle_vertex_count_range.append(total_num_triangle_vertices)
        
    mesh_vertices, mesh_face_vertex_indices = make_vertices(
        total_positions = total_positions,
        total_normals = total_normals,
        total_uvs = total_uvs,
        total_face_position_indices = total_face_position_indices,
        total_face_normal_indices = total_face_normal_indices,
        total_face_uv_indices = total_face_uv_indices)

    # get separate mesh vertices and mesh triangle indices
    separate_vertex_ranges = []
    separate_vertices = []
    separate_triangle_indices = []
    last_vertex_range = 0
    last_max_vertex_index = 0
    for mesh_index in range(num_meshes):
        num_vertices = mesh_triangle_vertex_count_range[mesh_index] - last_vertex_range
        min_vertex_index = 100000
        max_vertex_index = -100000
        for vertex_index in range(num_vertices):
            min_vertex_index = min(min_vertex_index, mesh_face_vertex_indices[vertex_index + last_vertex_range])
            max_vertex_index = max(max_vertex_index, mesh_face_vertex_indices[vertex_index + last_vertex_range])

        vertices = []
        for i in range(min_vertex_index, max_vertex_index + 1):
            vertices.append(mesh_vertices[i])
        separate_vertices.append(vertices)

        indices = []
        for i in range(last_vertex_range, mesh_triangle_vertex_count_range[mesh_index]):
            indices.append(mesh_face_vertex_indices[i] - min_vertex_index)
        separate_triangle_indices.append(indices)

        separate_vertex_ranges.append((min_vertex_index, max_vertex_index))
        last_vertex_range = mesh_triangle_vertex_count_range[mesh_index]

    total_position_bytes, total_normal_bytes, total_uv_bytes, total_face_position_indices_bytes, total_face_normal_indices_bytes, total_face_uv_indices_bytes, total_triangle_range_bytes = convert_to_bytes(
        total_positions = total_positions,
        total_normals = total_normals,
        total_uvs = total_uvs,
        total_face_position_indices = total_face_position_indices,
        total_face_normal_indices = total_face_normal_indices,
        total_face_uv_indices = total_face_uv_indices,
        triangle_ranges = triangle_ranges)
    
    mesh_vertex_bytes, mesh_face_index_bytes = convert_vertex_buffers_to_bytes(
        mesh_vertices = mesh_vertices,
        mesh_face_vertex_indices = mesh_face_vertex_indices)

    min_positions = []
    max_positions = []
    for mesh_position_range in separate_mesh_ranges:
        min_position = float3(100000.0, 100000.0, 1000000.0)
        max_position = float3(-100000.0, -100000.0, -1000000.0)

        for i in range(mesh_position_range[0], mesh_position_range[1]):
            min_position.x = min(min_position.x, total_positions[i].x)
            min_position.y = min(min_position.y, total_positions[i].y)
            min_position.z = min(min_position.z, total_positions[i].z)

            max_position.x = max(max_position.x, total_positions[i].x)
            max_position.y = max(max_position.y, total_positions[i].y)
            max_position.z = max(max_position.z, total_positions[i].z)

        min_positions.append(min_position)
        max_positions.append(max_position)

    result = MeshResult(
        total_positions  = total_positions, 
        total_normals  = total_normals, 
        total_uvs  = total_uvs, 
        total_face_position_indices  = total_face_position_indices, 
        total_face_normal_indices  = total_face_normal_indices, 
        total_face_uv_indices  = total_face_uv_indices, 
        material_info  = material_info, 
        total_mesh_names  = total_mesh_names, 
        total_materials  = total_materials, 
        triangle_ranges = triangle_ranges,
        mesh_vertices  = mesh_vertices, 
        mesh_face_vertex_indices = mesh_face_vertex_indices,
        total_position_bytes  = total_position_bytes, 
        total_normal_bytes  = total_normal_bytes, 
        total_uv_bytes  = total_uv_bytes, 
        total_face_position_indices_bytes  = total_face_position_indices_bytes, 
        total_face_normal_indices_bytes  = total_face_normal_indices_bytes, 
        total_face_uv_indices_bytes  = total_face_uv_indices_bytes, 
        total_triangle_range_bytes = total_triangle_range_bytes,
        mesh_vertex_bytes = mesh_vertex_bytes,
        mesh_face_index_bytes = mesh_face_index_bytes,
        total_triangle_positions_bytes = total_triangle_positions_bytes,
        mesh_triangle_vertex_count_range = mesh_triangle_vertex_count_range,
        mesh_position_ranges = separate_mesh_ranges,
        mesh_min_positions = min_positions,
        mesh_max_positions = max_positions
    )

    return result

##
#if __name__ == '__main__':
#    #total_positions, total_normals, total_uvs, total_face_position_indices, total_face_normal_indices, total_face_uv_indices, material_info, total_mesh_names, total_materials, triangle_ranges = load_obj2('c:\\Users\\Dingwings\\demo-models\\ramen-shop.obj')
#    #assert(len(total_face_position_indices) == len(material_info))
#
#    #mesh_vertices, mesh_face_vertex_indices = make_vertices(
#    #    total_positions = total_positions,
#    #    total_normals = total_normals,
#    #    total_uvs = total_uvs,
#    #    total_face_position_indices = total_face_position_indices,
#    #    total_face_normal_indices = total_face_normal_indices,
#    #    total_face_uv_indices = total_face_uv_indices)
#
#    #total_position_bytes, total_normal_bytes, total_uv_bytes, total_face_position_indices_bytes, total_face_normal_indices_bytes, total_face_uv_indices_bytes, total_triangle_range_bytes = convert_to_bytes(
#    #    total_positions = total_positions,
#    #    total_normals = total_normals,
#    #    total_uvs = total_uvs,
#    #    total_face_position_indices = total_face_position_indices,
#    #    total_face_normal_indices = total_face_normal_indices,
#    #    total_face_uv_indices = total_face_uv_indices,
#    #    triangle_ranges = triangle_ranges)
#
#    result = load_obj_file('c:\\Users\\Dingwings\\demo-models\\ramen-shop.obj')
#
#    print('')
