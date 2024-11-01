
# test_example = true

import time

from wgpu.gui.auto import WgpuCanvas, run
import wgpu
import numpy as np
import pywavefront
import datetime

from mat4 import *
from quat import *

from render_job import *
from gltf_loader import *
from translate_keyframes import *

os.add_dll_directory('D:\\test\\python-webgpu')
import embree_build_bvh_lib

import random
import gen_poisson_disc
import threading

from load_obj import *
from frustum_utils import *

from setup_render_job_data import *

from voxel_debug import *

from brick_setup import *
from debug_test import *

##
def update_skinning_matrices_thread(
    app,
    rig,
    keyframe_channels,
    joint_to_node_mappings,
    inverse_bind_matrix_data,
    device):

    animation_time = 0.0
    keys = list(app.keyframe_channels.keys())    
    times = app.mesh_data['translated_keyframe_channels'][keys[0]][1].times
    last_time = times[len(times) - 1]

    while True:

        while app.skinning_finished == True:
            if app.canvas.is_closed():
                break

            time.sleep(0.000001)

        if app.canvas.is_closed():
            break

        joint_positions = {}
        joint_rotations = {}
        for root_joint in rig.root_joints:
            traverse_rig(
                curr_joint = root_joint,
                rig = rig,
                keyframe_channels = keyframe_channels,
                time = animation_time,
                joint_positions = joint_positions,
                joint_rotations = joint_rotations,
                root_joint_name = root_joint.name)
            
        joints = rig.joints

        # skin matrix => total matrix from animation key frames * inverse bind matrix
        skin_matrices = []
        normal_matrices = []
        for joint_index in range(len(joints)):
            node_index = joint_to_node_mappings[joint_index]

            joint = joints[node_index]

            inverse_bind_matrix_from_gltf = inverse_bind_matrix_data[joint_index]
            inverse_bind_matrix = float4x4(
                [
                    inverse_bind_matrix_from_gltf[0], inverse_bind_matrix_from_gltf[4], inverse_bind_matrix_from_gltf[8], inverse_bind_matrix_from_gltf[12],
                    inverse_bind_matrix_from_gltf[1], inverse_bind_matrix_from_gltf[5], inverse_bind_matrix_from_gltf[9], inverse_bind_matrix_from_gltf[13],
                    inverse_bind_matrix_from_gltf[2], inverse_bind_matrix_from_gltf[6], inverse_bind_matrix_from_gltf[10], inverse_bind_matrix_from_gltf[14],
                    inverse_bind_matrix_from_gltf[3], inverse_bind_matrix_from_gltf[7], inverse_bind_matrix_from_gltf[11], inverse_bind_matrix_from_gltf[15],
                ]
            )

            anim_matrix = joint.total_matrix * inverse_bind_matrix
            normal_matrix = float4x4.invert(float4x4.transpose(anim_matrix))
            skin_matrices.append(anim_matrix)
            normal_matrices.append(normal_matrix)

        animation_time += 0.03
        if animation_time >= last_time:
            animation_time = 0.0

        app.skin_matrices = skin_matrices
        app.thread_skinning_matrix_bytes = app.convert_skinning_matrices_to_bytes(skin_matrices)
        app.thread_skinning_normal_matrix_bytes = app.convert_skinning_matrices_to_bytes(normal_matrices)

        app.skinning_finished = True

class MyCanvas(WgpuCanvas):
    def __init__(self, *, parent=None, size=None, title=None, max_fps=30, **kwargs):
        super().__init__(**kwargs)
        self.left_mouse_down = False
        self.diff_x = 0.0
        self.diff_y = 0.0
        self.last_x = 0.0
        self.last_y = 0.0

        self.right_mouse_down = False
        self.pan_diff_x = 0.0
        self.pan_diff_y = 0.0
        self.pan_last_x = 0.0
        self.pan_last_y = 0.0

        self.wheel_dy = 0

        self.key_down = None

    def handle_event(self, event):
        #print('{}'.format(event['event_type']))

        if event['event_type'] == 'pointer_down':
            if event['button'] == 1:
                self.left_mouse_down = True
                self.last_x = event['x']
                self.last_y = event['y']
            
            if event['button'] == 2:
                self.right_mouse_down = True
                self.pan_last_x = event['x']
                self.pan_last_y = event['y']

        elif event['event_type'] == 'pointer_up':
            if event['button'] == 1:
                self.left_mouse_down = False
            
            if event['button'] == 2:
                self.right_mouse_down = False

                self.pan_diff_x = 0.0
                self.pan_diff_y = 0.0

        if event['event_type'] == 'pointer_move':
            if self.left_mouse_down == True:
                self.diff_x = event['x'] - self.last_x
                self.diff_y = event['y'] - self.last_y

                self.last_x = event['x']
                self.last_y = event['y']
            
            if self.right_mouse_down == True:
                self.pan_diff_x = event['x'] - self.pan_last_x
                self.pan_diff_y = event['y'] - self.pan_last_y

                self.pan_last_x = event['x']
                self.pan_last_y = event['y']

        if event['event_type'] == 'key_down':
            self.key_down = event['key']
            print('key down')
        elif event['event_type'] == 'key_up':
            self.key_down = None
            print('key up')

        if event['event_type'] == 'wheel':
            self.wheel_dy = event['dy']

class MyApp(object):
    
    ##
    def __init__(self):
    
        print("Available adapters on this system:")
        for a in wgpu.gpu.enumerate_adapters():
            print(a.summary)

        self.screen_width = 640
        self.screen_height = 480

        # Create a canvas to render to
        self.canvas = MyCanvas(size = (self.screen_width, self.screen_height), title="wgpu cube")

        # Create a wgpu device
        self.adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self.device = self.adapter.request_device()

        # Prepare present context
        self.present_context = self.canvas.get_context()
        self.render_texture_format = self.present_context.get_preferred_format(self.device.adapter)
        self.present_context.configure(device=self.device, format=self.render_texture_format)

        self.eye_position = float3(4.5, 0.5, 1.0)
        self.look_at_position = float3(0.0, 0.0, 0.0)

        self.angle_x = 0.0
        self.angle_y = 0.0

        self.jitter = (0.0, 0.0)

        self.prev_jittered_view_projection_matrix = float4x4()
        self.jittered_view_projection_matrix = float4x4()

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        self.position_scale = 4.0
        self.brick_dimension = 1.0
        self.brixel_dimension = 1.0 / 8.0

        self.default_uniform_buffer = self.device.create_buffer(
            size = 1024, 
            usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            label = 'Default Uniform Data')

        self.init_materials('d:\\Downloads\\Bistro_v4\\bistro2.mat')
        self.albedo_scene_textures, self.albedo_scene_texture_views = self.load_scene_textures(
            self.albedo_texture_paths
        )

        self.normal_scene_textures, self.normal_scene_texture_views = self.load_scene_textures(
            self.normal_texture_paths
        )

        # create render jobs with total textures in material database
        self.load_render_jobs(
            path = os.path.join(self.dir_path, 'render-jobs', 'render_jobs.json'),
            albedo_texture_array_views = self.albedo_scene_texture_views,
            normal_texture_array_views = self.normal_scene_texture_views,
            albedo_texture_array_paths = self.albedo_texture_paths,
            normal_texture_array_paths = self.normal_texture_paths)

        fp = open('d:\\Downloads\\Bistro_v4\\bistro2-triangles.bin', 'rb')
        self.binary_mesh_file_content = fp.read()
        fp.close()
        
        random.seed()

        self.animation_time = 0.0
        self.frame_index = 0

        self.num_poisson_points = 30

        self.poisson_disc_points = gen_poisson_disc.create(
            radius = 1.0,
            grid_size = 1.0,
            width = 10,
            height = 10,
            num_points = self.num_poisson_points)

        halton_sequence = [
            [0.500000, 0.333333],
            [0.250000, 0.666667],
            [0.750000, 0.111111],
            [0.125000, 0.444444],
            [0.625000, 0.777778],
            [0.375000, 0.222222],
            [0.875000, 0.555556],
            [0.062500, 0.888889],
            [0.562500, 0.037037],
            [0.312500, 0.370370],
            [0.812500, 0.703704],
            [0.187500, 0.148148],
            [0.687500, 0.481481],
            [0.437500, 0.814815],
            [0.937500, 0.259259],
            [0.031250, 0.592593]
        ]

        self.halton_sequence = []
        for val in halton_sequence:
            val[0] = (val[0] * 2.0 - 1.0)
            val[1] = (val[1] * 2.0 - 1.0)

            self.halton_sequence.append(val)

        self.view_projection_matrix = float4x4()
        self.prev_view_projection_matrix = float4x4()

        self.mesh_file_path = 'c:\\Users\\Dingwings\\demo-models\\ramen-shop-4.obj'
        #self.mesh_file_path = 'd:\\Downloads\\Bistro_v4\\bistro.obj'

        self.options = {}
        self.options['swap_chain_texture_id'] = 0
        self.options['ambient_occlusion_distance'] = 1.0
        self.options['num_temporal_restir_sample_permutations'] = 1

        self.skinning_finished = False
        self.thread_skinning_matrix_bytes = None
        self.skinning_normal_matrix_bytes = None

        self.num_indirect_draw_calls = 0
        self.num_light_indirect_draw_calls = [0, 0, 0]

        self.num_large_meshes = struct.unpack('I', self.binary_mesh_file_content[0:4])[0]
        num_total_vertices = struct.unpack('I', self.binary_mesh_file_content[4:8])[0]
        num_total_triangles = struct.unpack('I', self.binary_mesh_file_content[8:12])[0]
        vertex_size = struct.unpack('I', self.binary_mesh_file_content[12:16])[0]
        triangle_start_offset = struct.unpack('I', self.binary_mesh_file_content[16:20])[0]
        mesh_ranges = []
        start = 20
        for i in range(self.num_large_meshes):
            offset_start = struct.unpack('I', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            offset_end = struct.unpack('I', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            mesh_ranges.append([offset_start, offset_end])

        self.binary_mesh_range = self.binary_mesh_file_content[20:start]

        start_mesh_extent = start
        mesh_extents = []
        for i in range(self.num_large_meshes):
            x = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            y = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            z = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            w = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            min_position = float3(x, y, z)
            
            x = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            y = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            z = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            w = struct.unpack('f', self.binary_mesh_file_content[start:start+4])[0]
            start += 4
            max_position = float3(x, y, z)

            mesh_extents.append([min_position, max_position])

        self.binary_mesh_extent = self.binary_mesh_file_content[start_mesh_extent:start]
        file_size = len(self.binary_mesh_file_content)

        end = start + vertex_size * num_total_vertices
        self.large_vertex_buffer_bytes = self.binary_mesh_file_content[start:end]
        self.large_index_buffer_bytes = self.binary_mesh_file_content[end:]

        index0 = struct.unpack('I', self.large_index_buffer_bytes[0:4])[0]
        index1 = struct.unpack('I', self.large_index_buffer_bytes[4:8])[0]
        index2 = struct.unpack('I', self.large_index_buffer_bytes[8:12])[0]

        self.device_large_vertex_buffer = self.device.create_buffer_with_data(
            data = self.large_vertex_buffer_bytes,
            usage = wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE
        )

        self.device_large_index_buffer = self.device.create_buffer_with_data(
            data = self.large_index_buffer_bytes, 
            usage = wgpu.BufferUsage.INDEX | wgpu.BufferUsage.STORAGE
        )

    ##
    def load_render_jobs(
        self,
        path,
        albedo_texture_array_views,
        normal_texture_array_views,
        albedo_texture_array_paths,
        normal_texture_array_paths):

        file = open(path, 'rb')
        file_content = file.read()
        file.close()

        directory_end = path.rfind('\\')
        if directory_end == -1:
            directory_end = path.rfind('/')
        directory = path[:directory_end]

        self.render_job_info = json.loads(file_content.decode('utf-8'))

        self.render_jobs = []
        self.render_job_dict = {}
        for info in self.render_job_info['Jobs']:
            
            render_job_name = info['Name']

            if "JobGroup" in info:
                run_count = info['RunCount']

                prev_render_job = None
                for render_job_info in info['JobGroup']:
                    file_name = render_job_info['Pipeline']
                    full_render_job_path = os.path.join(directory, file_name)
                    render_job = RenderJob(
                        device = self.device,
                        present_context = self.canvas.get_context(),
                        render_job_file_path = full_render_job_path,
                        canvas_width = int(self.canvas._logical_size[0]),
                        canvas_height = int(self.canvas._logical_size[1]),
                        curr_render_jobs = self.render_jobs)

                    render_job.group_prev = prev_render_job
                    if prev_render_job != None:
                        prev_render_job.group_next = render_job
                    render_job.group_run_count = run_count
                    prev_render_job = render_job

                    self.render_jobs.append(render_job)
                    self.render_job_dict[render_job.name] = render_job
                   
            else:
                file_name = info['Pipeline']
                full_render_job_path = os.path.join(directory, file_name)
                
                # any overriding attachments
                attachment_info = []
                for i in range(16):
                    attachment_key = 'Attachment' + str(i)
                    if attachment_key in info:
                        attachment_info.append({
                            'index': i,
                            'name': info[attachment_key]['Name'],
                            'parent_job_name': info[attachment_key]['ParentJobName']
                        }) 

                render_job = RenderJob(
                    name = render_job_name,
                    device = self.device,
                    present_context = self.canvas.get_context(),
                    render_job_file_path = full_render_job_path,
                    canvas_width = int(self.canvas._logical_size[0]),
                    canvas_height = int(self.canvas._logical_size[1]),
                    curr_render_jobs = self.render_jobs,
                    extra_attachment_info = attachment_info)

                render_job.scissor_rect = None
                if 'Scissor' in info:
                    render_job.scissor_rect = info['Scissor']

                if 'Dispatch' in info:
                    dispatches = info['Dispatch']
                    render_job.dispatch_size = dispatches

                if 'Enabled' in info:
                    if info['Enabled'] == 'False':
                        render_job.draw_enabled = False

                self.render_jobs.append(render_job)
                self.render_job_dict[render_job.name] = render_job

        for render_job in self.render_jobs:
            render_job.finish_attachments_and_pipeline(
                self.device,
                self.render_jobs,
                self.default_uniform_buffer,
                albedo_texture_array_views,
                normal_texture_array_views,
                albedo_texture_array_paths,
                normal_texture_array_paths)

    ##
    def convert_mesh_vertices_to_bytes(
        self,
        total_mesh_positions,
        total_mesh_texcoords,
        total_mesh_normals):
        
        mesh_vertex_ranges = []

        start_vertex = 0
        total_vertex_data_bytes = b''
        for i in range(len(total_mesh_positions)):
            mesh_vertex_positions = total_mesh_positions[i]
            mesh_vertex_normals = total_mesh_normals[i]
            mesh_vertex_texcoords = total_mesh_texcoords[i]

            for j in range(len(mesh_vertex_positions)):
                mesh_position = mesh_vertex_positions[j]
                mesh_normal = mesh_vertex_normals[j]
                mesh_texcoord = mesh_vertex_texcoords[j]

                total_vertex_data_bytes += struct.pack('f', mesh_position[0])
                total_vertex_data_bytes += struct.pack('f', mesh_position[1])
                total_vertex_data_bytes += struct.pack('f', mesh_position[2])
                total_vertex_data_bytes += struct.pack('f', 1.0)

                total_vertex_data_bytes += struct.pack('f', mesh_texcoord[0])
                total_vertex_data_bytes += struct.pack('f', mesh_texcoord[1])
                total_vertex_data_bytes += struct.pack('f', 0.0)
                total_vertex_data_bytes += struct.pack('f', 0.0)

                total_vertex_data_bytes += struct.pack('f', mesh_normal[0])
                total_vertex_data_bytes += struct.pack('f', mesh_normal[1])
                total_vertex_data_bytes += struct.pack('f', mesh_normal[2])
                total_vertex_data_bytes += struct.pack('f', 1.0)

            end_vertex = (start_vertex + len(mesh_vertex_positions)) - 1
            mesh_vertex_ranges.append([start_vertex, end_vertex])
            start_vertex += len(mesh_vertex_positions)

        return total_vertex_data_bytes, mesh_vertex_ranges

    ##
    def convert_vertex_weights_to_bytes(
        self,
        total_mesh_joint_weights,
        total_mesh_joint_indices):
        
        weight_data_bytes = b''
        for i in range(len(total_mesh_joint_weights)):
            mesh_joint_weights = total_mesh_joint_weights[i]
            mesh_joint_indices = total_mesh_joint_indices[i]

            for j in range(len(mesh_joint_weights)):
                mesh_joint_weight = mesh_joint_weights[j]
                mesh_joint_index = mesh_joint_indices[j]

                weight_data_bytes += struct.pack('f', mesh_joint_weight[0])
                weight_data_bytes += struct.pack('f', mesh_joint_weight[1])
                weight_data_bytes += struct.pack('f', mesh_joint_weight[2])
                weight_data_bytes += struct.pack('f', mesh_joint_weight[3])

                weight_data_bytes += struct.pack('i', mesh_joint_index[0])
                weight_data_bytes += struct.pack('i', mesh_joint_index[1])
                weight_data_bytes += struct.pack('i', mesh_joint_index[2])
                weight_data_bytes += struct.pack('i', mesh_joint_index[3])

        return weight_data_bytes

    ## 
    def convert_skinning_matrices_to_bytes(
        self,
        matrices):

        ret_bytes = b''
        for matrix in matrices:
            for i in range(16):
                ret_bytes += struct.pack('f', matrix.entries[i])
            
        return ret_bytes

    ##
    def init_materials(
        self, 
        full_path):
        
        #self.textures = []
        #self.texture_views = []
        self.albedo_texture_paths = []
        self.normal_texture_paths = []

        base_name = os.path.basename(full_path)
        directory = full_path[:full_path.find(base_name)]

        fp = open(full_path, 'rb')
        file_content = fp.read()
        fp.close()

        curr_pos = 0
        num_materials = 0
        while True:
            diffuse_r = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            diffuse_g = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            diffuse_b = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            diffuse_a = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4

            specular_r = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            specular_g = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            specular_b = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            specular_a = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4

            emissive_r = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            emissive_g = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            emissive_b = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            emissive_a = struct.unpack('f', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4

            material_id = struct.unpack('I', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            albedo_texture_id = struct.unpack('I', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            normal_texture_id = struct.unpack('I', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
            emissive_texture_id = struct.unpack('I', file_content[curr_pos:curr_pos+4])[0]
            curr_pos += 4
        
            if material_id >= 99999:
                break

            num_materials += 1

        self.materials_bytes = file_content[:curr_pos]
        print('number of materials {}'.format(num_materials))

        num_albedo_textures = struct.unpack('I', file_content[curr_pos:curr_pos+4])[0]
        curr_pos += 4
        for i in range(num_albedo_textures):
            
            # texture file path
            texture_path_bytes = b''
            while True:
                if curr_pos >= len(file_content):
                    break
                char = struct.unpack('c', file_content[curr_pos:curr_pos+1])[0]
                curr_pos += 1
                if char == b'\n':
                    break
                texture_path_bytes += char
            texture_path = texture_path_bytes.decode('utf-8')
            file_path = os.path.join(directory, texture_path)
            self.albedo_texture_paths.append(file_path)

        num_normal_textures = struct.unpack('I', file_content[curr_pos:curr_pos+4])[0]
        curr_pos += 4
        for i in range(num_albedo_textures):
            
            # texture file path
            texture_path_bytes = b''
            while True:
                if curr_pos >= len(file_content):
                    break
                char = struct.unpack('c', file_content[curr_pos:curr_pos+1])[0]
                curr_pos += 1
                if char == b'\n':
                    break
                texture_path_bytes += char
            texture_path = texture_path_bytes.decode('utf-8')
            file_path = os.path.join(directory, texture_path)
            self.normal_texture_paths.append(file_path)

        
        '''
        group_index = 0
        
        output_str = ''
        index = 0
        texture_index = 0
        max_textures_per_group = 100
        group_offset = 3
        while True:
            if index >= num_materials:
                break

            group_index = int(index / max_textures_per_group) + group_offset
            texture_index = index % max_textures_per_group
            output_str += '@group({}) @binding({})\nvar normalTexture{}: texture_2d<f32>;\n'.format(group_index, texture_index, index)
            index += 1

        index = 0
        while True:
            if index >= num_materials:
                break

            if index == 0:
                output_str += '    if(iTextureID == {}u)\n'.format(index)
            else:
                output_str += '    else if(iTextureID == {}u)\n'.format(index)
            output_str += '    {\n'
            output_str += '        ret = textureSample(normalTexture{}, textureSampler, uv);\n'.format(index)
            output_str += '    }\n'
            
            index += 1

        file = open(os.path.join(directory, base_name + ".tmp"), 'w')
        file.write(output_str)
        file.close()
       '''

    ##
    def init_data(self):

        # vertex and index buffers for full triangle pass triangle
        full_triangle_vertex_data = np.array(
            [
                [-1.0,   3.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0],
                [-1.0,  -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [3.0,   -1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype = np.float32)

        full_triangle_index_data = np.array([
                [0, 1, 2]
            ],
            dtype = np.uint32).flatten()

        # Create vertex buffer, and upload data
        self.full_screen_triangle_vertex_buffer = self.device.create_buffer_with_data(
            data = full_triangle_vertex_data, 
            usage = wgpu.BufferUsage.VERTEX
        )

        # Create index buffer, and upload data
        self.full_screen_triangle_index_buffer = self.device.create_buffer_with_data(
            data = full_triangle_index_data, 
            usage=wgpu.BufferUsage.INDEX
        )

        gltf_file_path = os.path.join(self.dir_path, 'assets', 'chun-li-punch.gltf')
        mesh_positions, mesh_normals, mesh_texcoords, mesh_joint_indices, mesh_joint_weights, mesh_triangle_indices, rig, keyframe_channels, joint_to_node_mappings, inverse_bind_matrix_data = load_gltf(gltf_file_path)
        
        assert(len(mesh_positions[0]) == len(mesh_texcoords[0]))
        assert(len(mesh_positions[0]) == len(mesh_normals[0]))
        assert(len(mesh_positions[0]) == len(mesh_joint_indices[0]))
        assert(len(mesh_positions[0]) == len(mesh_joint_weights[0]))

        min_position = float3(99999.0, 99999.0, 999999.0)
        max_position = float3(-99999.0, -99999.0, -999999.0)
        
        for mesh in mesh_positions:
            for mesh_position in mesh:
                x = mesh_position[0]
                y = mesh_position[1]
                z = mesh_position[2]

                min_position.x = min(x, min_position.x)
                min_position.y = min(y, min_position.y)
                min_position.z = min(z, min_position.z)

                max_position.x = max(x, max_position.x)
                max_position.y = max(y, max_position.y)
                max_position.z = max(z, max_position.z)

        mesh_index = 0

        self.rig = None
        self.keyframe_channels = None
        
        src_file_path = 'd:\\test\\mediapipe\\animations2\\38_03.gltf'
        dest_file_path = 'd:\\test\\python-webgpu\\assets\\chun-li-punch.gltf'
        save_directory = 'd:\\test\\python-webgpu\\assets'

        src_name = os.path.basename(src_file_path)[:os.path.basename(src_file_path).find('.')]
        dest_name = os.path.basename(dest_file_path)[:os.path.basename(dest_file_path).find('.')]
        output_name = src_name + '-to-' + dest_name + '-matching-animation.json'

        #translated_keyframe_channels = translate_keyframe_channels2(
        #    src_file_path = src_file_path,
        #    dest_file_path = dest_file_path,
        #    save_directory = save_directory)

        matching_key_frame_animation_file_path = os.path.join(self.dir_path, 'assets', output_name)
        translated_keyframe_channels = load_matching_keyframes(
            file_path = matching_key_frame_animation_file_path
        )

        self.mesh_data = {
            'positions': mesh_positions, 
            'normals': mesh_normals, 
            'texcoords': mesh_texcoords, 
            'joint_indices': mesh_joint_indices, 
            'joint_weights': mesh_joint_weights, 
            'triangle_indices': mesh_triangle_indices, 
            'rig': rig,
            'keyframe_channels': keyframe_channels,
            'joint_to_node_mappings': joint_to_node_mappings,
            'inverse_bind_matrix_data': inverse_bind_matrix_data,
            'translated_keyframe_channels': translated_keyframe_channels,
        }

        # encode vertex data to bytes:
        total_mesh_vertices_bytes, total_mesh_vertex_ranges = self.convert_mesh_vertices_to_bytes(
            total_mesh_positions = mesh_positions,
            total_mesh_texcoords = mesh_texcoords,
            total_mesh_normals = mesh_normals)

        self.mesh_data['total_mesh_vertex_ranges'] = total_mesh_vertex_ranges

        mesh_joint_weights_bytes = self.convert_vertex_weights_to_bytes(
            total_mesh_joint_weights = mesh_joint_weights,
            total_mesh_joint_indices = mesh_joint_indices)

        mesh_xform_positions, mesh_xform_normals = test_skinning_transformations(
            self,
            joint_to_node_mappings = joint_to_node_mappings, 
            inverse_bind_matrix_data = inverse_bind_matrix_data,
            device = self.device)

        self.skinning_matrix_bytes = self.convert_skinning_matrices_to_bytes(
            self.mesh_data['skin_matrices'])
        self.skinning_normal_matrix_bytes = self.convert_skinning_matrices_to_bytes(
            self.mesh_data['skin_normal_matrices'])

        mesh_vertex_range_bytes = b''
        mesh_vertex_range_bytes += struct.pack('i', 0)
        mesh_vertex_range_bytes += struct.pack('i', len(total_mesh_vertex_ranges))
        for vertex_range in total_mesh_vertex_ranges:
            mesh_vertex_range_bytes += struct.pack('i', vertex_range[0])
            mesh_vertex_range_bytes += struct.pack('i', vertex_range[1])
        
        # get the initial bind scale
        root_joint_scaling = abs(rig.root_joints[0].inverse_bind_matrix.entries[0])
        mesh_scaling = 0.8
        scaled_min_position = min_position * root_joint_scaling
        
        uniform_bytes = b''
        uniform_bytes += struct.pack('f', 0.4)      # mesh translation
        uniform_bytes += struct.pack('f', -scaled_min_position.z * 0.5)      # mesh translation
        uniform_bytes += struct.pack('f', 0.6)      # mesh translation
        uniform_bytes += struct.pack('f', mesh_scaling)      # mesh scale

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].attachments['Skinned Vertices'],
            buffer_offset = 0,
            data = total_mesh_vertices_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = total_mesh_vertices_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].uniform_buffers[2],
            buffer_offset = 0,
            data = mesh_joint_weights_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].uniform_buffers[3],
            buffer_offset = 0,
            data = self.skinning_matrix_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].uniform_buffers[4],
            buffer_offset = 0,
            data = mesh_vertex_range_bytes
        )

        # vertex data
        vertex_data_3 = np.empty([0, 18], dtype = np.float32)
        for i in range(0, len(mesh_positions[mesh_index])):
            mesh_position = mesh_positions[mesh_index][i]
            texcoord = mesh_texcoords[mesh_index][i]
            normal = mesh_normals[mesh_index][i]

            mesh_xform_position = mesh_xform_positions[mesh_index][i]
            mesh_xform_normal = mesh_xform_normals[mesh_index][i]

            if len(mesh_joint_indices) > 0.0:
                joint_indices = mesh_joint_indices[mesh_index][i]
                joint_weights = mesh_joint_weights[mesh_index][i]

                vertex_list = [
                    #mesh_xform_position.x, mesh_xform_position.y, mesh_xform_position.z, 1.0,
                    mesh_position[0], mesh_position[1], mesh_position[2], 1.0, 
                    texcoord[0], texcoord[1], 
                    #mesh_xform_normal.x, mesh_xform_normal.y, mesh_xform_normal.z, 1.0,
                    normal[0], normal[1], normal[2], 1.0,
                    joint_weights[0], joint_weights[1], joint_weights[2], joint_weights[3],
                    float(joint_indices[0]), float(joint_indices[1]), float(joint_indices[2]), float(joint_indices[3]),
                ]
                self.rig = rig
                self.keyframe_channels = keyframe_channels
            else:
                vertex_list = [
                    mesh_position[0], mesh_position[1], mesh_position[2], 1.0, 
                    texcoord[0], texcoord[1], 
                    normal[0], normal[1], normal[2], 1.0
                ]
                
            array = np.array([np.array(vertex_list)], dtype = np.float32)
            vertex_data_3 = np.concatenate((vertex_data_3, array))
        
        # triangle index data
        index_data_3 = np.array([], dtype = np.uint32)
        for i in range(len(mesh_triangle_indices[mesh_index])):
            index = np.uint32(mesh_triangle_indices[mesh_index][i][0])
            index_data_3 = np.append(index_data_3, index)
            assert(index < len(mesh_positions[mesh_index]))

        self.index_size = index_data_3.size

        # Create vertex buffer, and upload data
        self.vertex_buffer = self.device.create_buffer_with_data(
            data = vertex_data_3, 
            usage = wgpu.BufferUsage.VERTEX
        )

        # Create index buffer, and upload data
        self.index_buffer = self.device.create_buffer_with_data(
            data = index_data_3, 
            usage=wgpu.BufferUsage.INDEX
        )
        
        self.wgpu_buffers = {}
        self.wgpu_buffers['skinned-vertex-buffer'] = self.vertex_buffer
        self.wgpu_buffers['skinned-index-buffer'] = self.index_buffer
        self.wgpu_buffers['skinned-num-vertex-indices'] = self.index_size

        self.wgpu_buffers['index-data'] = index_data_3

        self.ray_trace_mesh_data = {}
        self.ray_trace_mesh_data['vertex-buffer'] = self.device.create_buffer_with_data(
            data = self.mesh_obj_result.mesh_vertex_bytes,
            usage = wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE
        )

        self.ray_trace_mesh_data['index-buffer'] = self.device.create_buffer_with_data(
            data = self.mesh_obj_result.mesh_face_index_bytes, 
            usage = wgpu.BufferUsage.INDEX | wgpu.BufferUsage.STORAGE
        )
        
        self.ray_trace_mesh_data['num-indices'] = int(len(self.mesh_obj_result.mesh_face_index_bytes) / 4)

        # sky atmosphere uniform data
        self.light_direction = float3.normalize(float3(0.2, 0.8, 0.0))
        self.light_radiance = float3(5.0, 5.0, 5.0)

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxelize Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = mesh_vertex_range_bytes
        )

        triangle_indices = []
        index_buffer_bytes = b''
        index_buffer = self.mesh_data['triangle_indices']
        num_index_buffer_entries = len(index_buffer)
        for i in range(1):
            for j in range(len(index_buffer[i])):
                triangle_index = index_buffer[i][j][0]
                triangle_indices.append(triangle_index)
                index_buffer_bytes += struct.pack('I', triangle_index)

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxelize Compute'].uniform_buffers[2],
            buffer_offset = 0,
            data = index_buffer_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxel Fill Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = index_buffer_bytes
        )

        num_triangle_indices = len(triangle_indices)
        triangle_index_range_bytes = b''
        triangle_index_range_bytes += struct.pack('I', num_triangle_indices)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxelize Compute'].uniform_buffers[3],
            buffer_offset = 0,
            data = triangle_index_range_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['SDF Ambient Occlusion Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = index_buffer_bytes
        )

        clear_bytes = b''
        for i in range(16):
            clear_bytes += struct.pack('I', 0)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['SDF Ambient Occlusion Graphics'].attachments['Hit Brick Counters'],
            buffer_offset = 0,
            data = clear_bytes
        )

        curr_total_num_bricks = 0
        brick_dimensions = []
        brick_start_indices = []
        brick_start_positions = []
        brick_start_indices.append(curr_total_num_bricks)
        num_meshes = len(self.mesh_obj_result.mesh_max_positions)
        for mesh_index in range(num_meshes):
            max_position = self.mesh_obj_result.mesh_max_positions[mesh_index]
            min_position = self.mesh_obj_result.mesh_min_positions[mesh_index]
            brick_dimension = float3(
                math.ceil(max_position.x * 10.0) - math.floor(min_position.x * 10.0),
                math.ceil(max_position.y * 10.0) - math.floor(min_position.y * 10.0),
                math.ceil(max_position.z * 10.0) - math.floor(min_position.z * 10.0)
            )
            start_brick_position = min_position * 10.0
            curr_total_num_bricks += brick_dimension.x * brick_dimension.y * brick_dimension.z

            brick_start_positions.append(float3(
                math.floor(start_brick_position.x),
                math.floor(start_brick_position.y),
                math.floor(start_brick_position.z))
            )
            brick_dimensions.append(brick_dimension)
            brick_start_indices.append(curr_total_num_bricks)


        brick_start_position_byte_buffer = b''
        for position in brick_start_positions:
            brick_start_position_byte_buffer += struct.pack('f', position.x)
            brick_start_position_byte_buffer += struct.pack('f', position.y)
            brick_start_position_byte_buffer += struct.pack('f', position.z)

        brick_dimension_byte_buffer = b''
        for dim in brick_dimensions:
            brick_dimension_byte_buffer += struct.pack('f', dim.x)
            brick_dimension_byte_buffer += struct.pack('f', dim.y)
            brick_dimension_byte_buffer += struct.pack('f', dim.z)

        self.curr_voxelize_triangle = self.mesh_data['total_mesh_vertex_ranges'][0][0]

        material_data_bytes = b''
        for material in self.mesh_obj_result.total_materials:
            material_data_bytes += struct.pack('f', material['diffuse'].x)
            material_data_bytes += struct.pack('f', material['diffuse'].y)
            material_data_bytes += struct.pack('f', material['diffuse'].z)
            material_data_bytes += struct.pack('f', material['specular'].x)
            material_data_bytes += struct.pack('f', material['emissive'].x)
            material_data_bytes += struct.pack('f', material['emissive'].y)
            material_data_bytes += struct.pack('f', material['emissive'].z)
            material_data_bytes += struct.pack('f', material['transparency'])

        mesh_triangle_range_data_bytes = b''
        for triangle_range in self.mesh_obj_result.triangle_ranges:
            mesh_triangle_range_data_bytes += struct.pack('i', triangle_range[0])
            mesh_triangle_range_data_bytes += struct.pack('i', triangle_range[1])

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Direct Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = material_data_bytes)

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Direct Graphics'].uniform_buffers[2],
            buffer_offset = 0,
            data = mesh_triangle_range_data_bytes)
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Emissive Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = material_data_bytes)

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Emissive Graphics'].uniform_buffers[2],
            buffer_offset = 0,
            data = mesh_triangle_range_data_bytes)
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Accumulation Denoiser Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = material_data_bytes)

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Specular Filter Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = material_data_bytes)

        uniform_bytes = b''
        uniform_bytes += struct.pack('f', self.position_scale)
        uniform_bytes += struct.pack('f', self.brick_dimension)
        uniform_bytes += struct.pack('f', self.brixel_dimension)
        uniform_bytes += struct.pack('I', self.frame_index)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxelize Compute'].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_bytes)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Brick Setup Compute'].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_bytes)    
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxel Fill Compute'].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_bytes)
        
        self.counter_bytes = b''
        for i in range(256):
            self.counter_bytes += struct.pack('i', 0)

        # sdf uniform buffer
        if self.frame_index <= 0:
            self.sdf_pass_uniform_bytes = b''
            self.sdf_pass_uniform_bytes += struct.pack('f', self.brick_dimension)
            self.sdf_pass_uniform_bytes += struct.pack('f', self.brixel_dimension)
            self.sdf_pass_uniform_bytes += struct.pack('f', self.position_scale)
            self.sdf_pass_uniform_bytes += struct.pack('I', 8)

            self.device.queue.write_buffer(
                buffer = self.render_job_dict['SDF Ambient Occlusion Graphics'].uniform_buffers[0],
                buffer_offset = 0,
                data = self.sdf_pass_uniform_bytes)

            self.device.queue.write_buffer(
                buffer = self.render_job_dict['SDF Filter Graphics'].uniform_buffers[0],
                buffer_offset = 0,
                data = self.sdf_pass_uniform_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Brixel Indirect Radiance Graphics'].uniform_buffers[0],
                buffer_offset = 0,
                data = self.sdf_pass_uniform_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Screen Space Brixel Radiance Compute'].uniform_buffers[0],
                buffer_offset = 0,
                data = self.sdf_pass_uniform_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['SDF Draw Graphics'].uniform_buffers[0],
                buffer_offset = 0,
                data = self.sdf_pass_uniform_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Flood Fill Compute'].uniform_buffers[0],
                buffer_offset = 0,
                data = self.sdf_pass_uniform_bytes)
            
            counter_buffer = b''
            counter_buffer += struct.pack('i', 1)
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Flood Fill Compute'].attachments['Counters'],
                buffer_offset = 0,
                data = counter_buffer)

        # uniform data for spatial restir passes
        if self.frame_index <= 0:
            uniform_bytes = b''
            sample_radius = 30.0
            neighbor_block_check = 0
            num_samples = 20
            uniform_bytes += struct.pack('f', sample_radius)
            uniform_bytes += struct.pack('i', neighbor_block_check)
            uniform_bytes += struct.pack('i', num_samples)
            uniform_bytes += struct.pack('i', 0)

            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Spatial Restir Graphics 0'].uniform_buffers[0],
                buffer_offset = 0,
                data = uniform_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Spatial Restir Emissive Graphics'].uniform_buffers[0],
                buffer_offset = 0,
                data = uniform_bytes)
            
            uniform_bytes = b''
            sample_radius = 8.0
            neighbor_block_check = 1
            num_samples = 6
            uniform_bytes += struct.pack('f', sample_radius)
            uniform_bytes += struct.pack('i', neighbor_block_check)
            uniform_bytes += struct.pack('i', num_samples)
            uniform_bytes += struct.pack('i', 0)
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Spatial Restir Graphics 1'].uniform_buffers[0],
                buffer_offset = 0,
                data = uniform_bytes)

        # initialize skin mesh bounding box
        if self.frame_index <= 0:
            bbox_bytes = b''
            bbox_bytes += struct.pack('i', -10000)
            bbox_bytes += struct.pack('i', -10000)
            bbox_bytes += struct.pack('i', -10000)
            bbox_bytes += struct.pack('i', 10000)
            bbox_bytes += struct.pack('i', 10000)
            bbox_bytes += struct.pack('i', 10000)
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Skinning Compute'].attachments['Bounding Boxes'],
                buffer_offset = 0,
                data = bbox_bytes)
            
        self.rig_update_thread = threading.Thread(
            target = update_skinning_matrices_thread, 
            args=(
                self, 
                self.rig,
                self.mesh_data['translated_keyframe_channels'],
                self.mesh_data['joint_to_node_mappings'],
                self.mesh_data['inverse_bind_matrix_data'],
                self.device
            )
        )
        self.rig_update_thread.start()

        # material id
        fp = open('d:\\Downloads\\Bistro_v4\\bistro2.mid', 'rb')
        file_content = fp.read()
        fp.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Deferred Indirect Offscreen Graphics'].uniform_buffers[2],
            buffer_offset = 0,
            data = file_content)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Secondary Deferred Indirect Offscreen Graphics'].uniform_buffers[2],
            buffer_offset = 0,
            data = file_content)
        
        # materials
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Deferred Indirect Offscreen Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = self.materials_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Secondary Deferred Indirect Offscreen Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = self.materials_bytes
        )

        # uniform data
        indirect_uniform_bytes = b''
        indirect_uniform_bytes += struct.pack('I', self.num_large_meshes)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Deferred Indirect Offscreen Graphics'].uniform_buffers[0],
            buffer_offset = 0,
            data = indirect_uniform_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Secondary Deferred Indirect Offscreen Graphics'].uniform_buffers[0],
            buffer_offset = 0,
            data = indirect_uniform_bytes
        )

        self.setup_TLASTest_data()

    ##
    def init_draw(self):
        self.canvas.request_draw(self.draw_frame2)

    ##
    def update_camera(
        self,
        eye_position,
        look_at,
        up,
        view_width,
        view_height):

        self.camera_fov = math.pi * 0.5
        view_matrix = float4x4.view_matrix(
            eye_position = eye_position, 
            look_at = look_at, 
            up = up)
        perspective_projection_matrix = float4x4.perspective_projection_matrix(
            field_of_view = self.camera_fov * 0.5,
            view_width = view_width,
            view_height = view_height,
            far = self.camera_far,
            near = self.camera_near)

        self.view_matrix = view_matrix

        self.prev_jitter = self.jitter

        #jitter_x = self.poisson_disc_points[self.frame_index % self.num_poisson_points][0]
        #jitter_y = self.poisson_disc_points[self.frame_index % self.num_poisson_points][1]
        jitter_x = self.halton_sequence[self.frame_index % len(self.halton_sequence)][0]
        jitter_y = self.halton_sequence[self.frame_index % len(self.halton_sequence)][1]
        self.jitter_scale = 0.7
        #jitter_matrix = float4x4.translate(
        #    float3((jitter_x * self.jitter_scale) / self.screen_width, 
        #           (jitter_y * self.jitter_scale) / self.screen_height,
        #           0.0)
        #)
        self.jitter = (jitter_x * self.jitter_scale, jitter_y * self.jitter_scale)

        jittered_perspective_projection_matrix = perspective_projection_matrix
        jittered_perspective_projection_matrix.entries[3] = (jitter_x * self.jitter_scale) / float(self.screen_width)
        jittered_perspective_projection_matrix.entries[7] = (jitter_y * self.jitter_scale) / float(self.screen_height)

        return perspective_projection_matrix * view_matrix, jittered_perspective_projection_matrix * view_matrix, perspective_projection_matrix, jittered_perspective_projection_matrix

    ##
    def update_skinning_matrices(
        self,
        joint_to_node_mappings, 
        inverse_bind_matrix_data,
        animation_time):

        rig = self.mesh_data['rig']
        #keyframe_channels = self.mesh_data['keyframe_channels']
        keyframe_channels = self.mesh_data['translated_keyframe_channels']

        #start_time = datetime.datetime.now()

        joint_positions = {}
        joint_rotations = {}
        for root_joint in rig.root_joints:
            traverse_rig(
                curr_joint = root_joint,
                rig = rig,
                keyframe_channels = keyframe_channels,
                time = animation_time,
                joint_positions = joint_positions,
                joint_rotations = joint_rotations,
                root_joint_name = root_joint.name)

        #time_delta = datetime.datetime.now() - start_time
        #print('{}'.format(time_delta.microseconds / 1000))

        joints = rig.joints

        # skin matrix => total matrix from animation key frames * inverse bind matrix
        skin_matrices = []
        skin_normal_matrices = []
        for joint_index in range(len(joints)):
            node_index = joint_to_node_mappings[joint_index]

            joint = joints[node_index]

            inverse_bind_matrix_from_gltf = inverse_bind_matrix_data[joint_index]
            inverse_bind_matrix = float4x4(
                [
                    inverse_bind_matrix_from_gltf[0], inverse_bind_matrix_from_gltf[4], inverse_bind_matrix_from_gltf[8], inverse_bind_matrix_from_gltf[12],
                    inverse_bind_matrix_from_gltf[1], inverse_bind_matrix_from_gltf[5], inverse_bind_matrix_from_gltf[9], inverse_bind_matrix_from_gltf[13],
                    inverse_bind_matrix_from_gltf[2], inverse_bind_matrix_from_gltf[6], inverse_bind_matrix_from_gltf[10], inverse_bind_matrix_from_gltf[14],
                    inverse_bind_matrix_from_gltf[3], inverse_bind_matrix_from_gltf[7], inverse_bind_matrix_from_gltf[11], inverse_bind_matrix_from_gltf[15],
                ]
            )

            anim_matrix = joint.total_matrix * inverse_bind_matrix
            normal_matrix = float4x4.invert(float4x4.transpose(anim_matrix))
            skin_matrices.append(anim_matrix)
            skin_normal_matrices.append(normal_matrix)

        self.mesh_data['skin_matrices'] = skin_matrices
        self.mesh_data['skin_normal_matrices'] = skin_normal_matrices

    ##
    def update_skinning_uniform_data(
        self,
        joint_to_node_mappings,
        inverse_bind_matrix_data,
        device,
        animation_time):

        self.update_skinning_matrices(
            joint_to_node_mappings = joint_to_node_mappings, 
            inverse_bind_matrix_data = inverse_bind_matrix_data,
            animation_time = animation_time
        )

        skin_matrices = self.mesh_data['skin_matrices']

        # buffer for the skin matrices
        matrix_index = 0
        num_matrix_batches = int(math.ceil(len(skin_matrices) / 16))
        self.uniform_skin_matrix_buffers = [] * num_matrix_batches
        self.uniform_skin_matrix_data = [] * num_matrix_batches
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

            self.uniform_skin_matrix_buffers.append(
                device.create_buffer_with_data(
                    data = skin_matrices_buffer_data, 
                    usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_SRC)
            )

            self.uniform_skin_matrix_data.append(skin_matrices_buffer_data)

    ##
    def draw_frame2(self):
        start_time = datetime.datetime.now()
        
        speed = 0.002

        view_dir = float3.normalize(self.look_at_position - self.eye_position)
        up = float3(0.0, 1.0, 0.0)
        tangent = float3.cross(up, view_dir)
        binormal = float3.cross(tangent, view_dir)

        pan_speed = 0.002

        # panning, setting camera position and look at
        if self.canvas.right_mouse_down == True:
            self.eye_position = self.eye_position + binormal * -self.canvas.pan_diff_y * pan_speed + tangent * self.canvas.pan_diff_x * pan_speed
            self.look_at_position = self.look_at_position + binormal * -self.canvas.pan_diff_y * pan_speed + tangent * self.canvas.pan_diff_x * pan_speed
            
        #self.eye_position.x += speed
        #self.look_at_position.x += speed

        #self.eye_position.y += speed
        #self.look_at_position.y += speed

        rotation_speed = 0.01

        delta_x = (2.0 * math.pi) / 640.0
        delta_y = (2.0 * math.pi) / 480.0

        # update angle with mouse position delta
        if self.canvas.left_mouse_down == True:
            self.angle_x += self.canvas.diff_x * rotation_speed * delta_x
            self.angle_y += self.canvas.diff_y * rotation_speed * delta_y * -1.0
        else:
            self.angle_x = 0.0
            self.angle_y = 0.0

        self.canvas.diff_x = 0.0
        self.canvas.diff_y = 0.0

        if self.angle_x > 2.0 * math.pi:
            self.angle_x = -2.0 * math.pi 
        elif self.angle_x < -2.0 * math.pi:
            self.angle_x = 2.0 * math.pi 

        if self.angle_y >= math.pi * 0.5:
            self.angle_y = math.pi * 0.5
        elif self.angle_y <= -math.pi * 0.5:
            self.angle_y = -math.pi * 0.5

        # change debug swap chain output 
        num_debug_textures = len(self.render_job_dict['Swap Chain Graphics'].attachment_info) - 1
        if abs(self.canvas.wheel_dy) > 0:
            if self.canvas.wheel_dy > 0:
                self.options['swap_chain_texture_id'] = (self.options['swap_chain_texture_id'] + 1) % num_debug_textures
            elif self.canvas.wheel_dy < 0:
                self.options['swap_chain_texture_id'] = ((num_debug_textures + self.options['swap_chain_texture_id']) - 1) % num_debug_textures

            uniform_data_bytes = b''
            uniform_data_bytes += struct.pack('I', self.options['swap_chain_texture_id'])
            uniform_data_bytes += struct.pack('I', self.screen_width)
            uniform_data_bytes += struct.pack('I', self.screen_height)

            print('texture: {}'.format(self.options['swap_chain_texture_id']))

            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Swap Chain Graphics'].uniform_buffers[0],
                buffer_offset = 0,
                data = uniform_data_bytes
            )

            self.canvas.wheel_dy = 0

        # keys to strafe and change ambient occlusion distance
        move_speed = 0.1
        if self.canvas.key_down == 'a':
            self.eye_position = self.eye_position + tangent * move_speed
            self.look_at_position = self.look_at_position + tangent * move_speed
        elif self.canvas.key_down == 'd':
            self.eye_position = self.eye_position + tangent * -move_speed
            self.look_at_position = self.look_at_position + tangent * -move_speed
        elif self.canvas.key_down == 'w':
            self.eye_position += view_dir * move_speed
            self.look_at_position += view_dir *move_speed
        elif self.canvas.key_down == 's':
            self.eye_position += view_dir * -move_speed
            self.look_at_position += view_dir * -move_speed
        elif self.canvas.key_down == 'z':
            self.options['ambient_occlusion_distance'] -= 0.1
        elif self.canvas.key_down == 'x':
            self.options['ambient_occlusion_distance'] += 0.1

        if self.options['ambient_occlusion_distance'] < 0.0:
            self.options['ambient_occlusion_distance'] = 0.0

        # rotate eye position
        quat_x = quaternion.from_angle_axis(float3(0.0, 1.0, 0.0), self.angle_x)
        quat_y = quaternion.from_angle_axis(float3(1.0, 0.0, 0.0), self.angle_y)
        total_quat = quat_x * quat_y
        total_matrix = total_quat.to_matrix()
        xform_eye_position = total_matrix.apply(self.eye_position - self.look_at_position)
        xform_eye_position += self.look_at_position

        self.eye_position = xform_eye_position

        # update camera with new eye position
        up_direction = float3(0.0, 1.0, 0.0)
        self.camera_near = 1.0
        self.camera_far = 50.0
        view_projection_matrix, \
        jittered_view_projection_matrix, \
        perspective_projection_matrix, \
        jittered_perspective_projection_matrix = self.update_camera(
            eye_position = xform_eye_position, 
            look_at = self.look_at_position,
            up = up_direction,
            view_width = self.canvas._logical_size[0],
            view_height = self.canvas._logical_size[1])

        self.light_matrices = []
        self.light_view_projection_matrices = []

        #inverse_view_projection_matrix = float4x4.invert(view_projection_matrix)
        #setup_debug_shadow_data(
        #    view_projection_matrix = view_projection_matrix,
        #    inverse_view_projection_matrix = inverse_view_projection_matrix,
        #    light_view_projection_matrices = self.light_view_projection_matrices,
        #    screen_width = self.screen_width,
        #    screen_height = self.screen_height, 
        #    device = self.device,
        #    render_job = self.render_job_dict['Debug Shadow Compute'])

        # save previous view projection matrix
        self.prev_view_projection_matrix = self.view_projection_matrix
        self.view_projection_matrix = view_projection_matrix

        self.prev_jittered_view_projection_matrix = self.jittered_view_projection_matrix
        self.jittered_view_projection_matrix = jittered_view_projection_matrix

        joint_to_node_mappings = self.mesh_data['joint_to_node_mappings']
        inverse_bind_matrix_data = self.mesh_data['inverse_bind_matrix_data']

        # default uniform data
        update_default_data(
            device = self.device,
            device_buffer = self.default_uniform_buffer,
            screen_width = self.screen_width,
            screen_height = self.screen_height,
            frame_index = self.frame_index,
            num_meshes = len(self.mesh_obj_result.total_materials),
            rand0 = float(random.randint(0, 100)) * 0.01,
            rand1 = float(random.randint(0, 100)) * 0.01,
            rand2 = float(random.randint(0, 100)) * 0.01,
            rand3 = float(random.randint(0, 100)) * 0.01,
            view_projection_matrix = view_projection_matrix,
            prev_view_projection_matrix = self.prev_view_projection_matrix,
            view_matrix = self.view_matrix,
            projection_matrix = jittered_perspective_projection_matrix,
            jittered_view_projection_matrix = self.jittered_view_projection_matrix,
            prev_jittered_view_projection_matrix = self.prev_jittered_view_projection_matrix,
            camera_position = xform_eye_position,
            camera_look_at = self.look_at_position,
            light_radiance = self.light_radiance,
            light_direction = self.light_direction,
            ambient_occlusion_distance_threshold = self.options['ambient_occlusion_distance']
        )

        # update skinning matrices
        #self.update_skinning_uniform_data(
        #    joint_to_node_mappings = joint_to_node_mappings,
        #    inverse_bind_matrix_data = inverse_bind_matrix_data,
        #    device = self.device, 
        #    animation_time = self.animation_time
        #)

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 0: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        if self.skinning_finished == True:
            self.skinning_matrix_bytes = self.thread_skinning_matrix_bytes
            self.skinning_normal_matrix_bytes = self.thread_skinning_normal_matrix_bytes 
            self.skinning_finished = False

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].uniform_buffers[3],
            buffer_offset = 0,
            data = self.skinning_matrix_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Skinning Compute'].uniform_buffers[5],
            buffer_offset = 0,
            data = self.skinning_normal_matrix_bytes
        )

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 1: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        self.options['ambient_occlusion_radius'] = 3.0
        self.options['ambient_occlusion_quality'] = 8.0
        self.options['ambient_occlusion_num_directions'] = 16.0

        # clear counter
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Light View Compute'].attachments['Counters'],
            buffer_offset = 0,
            data = self.counter_bytes)

        self.update_render_job_user_data()

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxelize Compute'].attachments['Counters'],
            buffer_offset = 0,
            data = self.counter_bytes)

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 2: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        # update the start and end triangle indices to voxelize
        num_total_triangles = int(len(self.mesh_data['triangle_indices'][0]) / 3)
        self.voxelize_num_triangles_per_frame = num_total_triangles
        end_triangle = self.curr_voxelize_triangle + self.voxelize_num_triangles_per_frame
        if end_triangle >= num_total_triangles:
            end_triangle = int(num_total_triangles)

        curr_triangle_range_bytes = b''
        curr_triangle_range_bytes += struct.pack('I', self.curr_voxelize_triangle)
        curr_triangle_range_bytes += struct.pack('I', end_triangle)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Voxelize Compute'].uniform_buffers[4],
            buffer_offset = 0,
            data = curr_triangle_range_bytes)
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Brick Setup Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = curr_triangle_range_bytes)

        # cycle back to start triangle
        self.curr_voxelize_triangle = end_triangle
        if self.curr_voxelize_triangle >= num_total_triangles:
            self.curr_voxelize_triangle = 0

        # current presentable swapchain texture
        current_present_texture = self.present_context.get_current_texture()

        # command encoder
        #command_encoder = self.device.create_command_encoder()

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 3: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        curr_job_group_run_count = 0
        start_job_group = None
        end_job_group = None 
        start_job_group_index = -1
        render_job_index = 0

        if self.frame_index >= 2:
            self.render_job_dict['Diffuse Atmosphere Graphics'].draw_enabled = False

        self.set_up_scene_light_view_matrices(
            light_direction = self.light_direction
        )

        #print('\n***********\n')
        for render_job_index in range(len(self.render_jobs)):
            command_encoder = self.device.create_command_encoder() 

            start_render_job_time = datetime.datetime.now()

            render_job = self.render_jobs[render_job_index]
            render_job_name = render_job.name
            if render_job.draw_enabled == False:
                continue

            # job group
            if render_job.group_prev == None and render_job.group_next != None:
                start_job_group = render_job
                start_job_group_index = render_job_index
            elif render_job.group_next == None and render_job.group_prev != None:
                end_job_group = render_job
                
            # check to see loop back is needed with group run count
            if start_job_group != None and end_job_group != None:
                if curr_job_group_run_count < start_job_group.group_run_count - 1:
                    render_job_index = start_job_group_index - 1
                
                start_job_group = None
                end_job_group = None

                curr_job_group_run_count += 1


            if render_job.type == 'Graphics':
                # view for the depth texture
                if render_job.depth_texture is not None:
                    current_depth_texture_view = render_job.depth_texture.create_view()

                # output color attachments for render pass, create views for the attachments
                color_attachments = []
                if render_job.pass_type == 'Swap Chain' or render_job.pass_type == 'Swap Chain Full Triangle':
                    # swap chain job, use presentable texture and view

                    attachment_view = current_present_texture.create_view()
                    color_attachments.append({
                        'view': attachment_view,
                        'resolve_target': None,
                        'clear_value': (0, 0, 0, 0),
                        'load_op': wgpu.LoadOp.clear,
                        'store_op': wgpu.StoreOp.store
                    })

                else:
                    # regular job, use its output attachments

                    for attachment_name in render_job.attachments:
                        attachment = render_job.attachments[attachment_name]

                        skip = False
                        if (not attachment_name in render_job.attachment_views or render_job.attachment_views[attachment_name] == None):
                            skip = True

                        if skip == False and attachment != None:
                            # valid output attachment

                            load_op = wgpu.LoadOp.clear

                            # don't clear on input-output attachment
                            if render_job.attachment_types[attachment_name] == 'TextureInputOutput':
                                load_op = wgpu.LoadOp.load

                            attachment_view = render_job.attachment_views[attachment_name]
                            color_attachments.append({
                                'view': attachment_view,
                                'resolve_target': None,
                                'clear_value': (0, 0, 0, 0),
                                'load_op': load_op,
                                'store_op': wgpu.StoreOp.store
                            })

                # setup and show render pass
                if render_job.depth_texture is not None:
                    depth_load_op = wgpu.LoadOp.clear
                    
                    # don't clear if depth texture is an attachment from parent job
                    if render_job.depth_attachment_parent_info != None:
                        depth_load_op = wgpu.LoadOp.load
                    
                    render_pass = command_encoder.begin_render_pass(
                        color_attachments = color_attachments,
                        depth_stencil_attachment = 
                        {
                            "view": current_depth_texture_view,
                            "depth_clear_value": 1.0,
                            "depth_load_op": depth_load_op,
                            "depth_store_op": wgpu.StoreOp.store,
                            "depth_read_only": False,
                            "stencil_clear_value": 0,
                            "stencil_load_op": wgpu.LoadOp.clear,
                            "stencil_store_op": wgpu.StoreOp.discard,
                            "stencil_read_only": False,
                        },
                        label = render_job.name
                    )
                else:
                    render_pass = command_encoder.begin_render_pass(
                        color_attachments = color_attachments,
                        label = render_job.name
                    )

                if render_job.scissor_rect != None:
                    render_pass.set_scissor_rect(
                        x = render_job.scissor_rect[0],
                        y = render_job.scissor_rect[1],
                        width = render_job.scissor_rect[2],
                        height = render_job.scissor_rect[3])

                if render_job.name == 'Deferred Offscreen Graphics':
                    render_pass.set_pipeline(render_job.render_pipeline)
                    for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

                    render_pass.set_index_buffer(self.ray_trace_mesh_data['index-buffer'], wgpu.IndexFormat.uint32)
                    render_pass.set_vertex_buffer(0, self.ray_trace_mesh_data['vertex-buffer'])
                    render_pass.draw_indexed(self.ray_trace_mesh_data['num-indices'], 1, 0, 0, 0)

                elif render_job.name == 'Deferred Indirect Offscreen Graphics':
                    render_pass.set_pipeline(render_job.render_pipeline)
                    for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

                    render_pass.set_index_buffer(self.device_large_index_buffer, wgpu.IndexFormat.uint32)
                    render_pass.set_vertex_buffer(0, self.device_large_vertex_buffer)


                    indirect_offset = 0
                    for draw_call_index in range(self.num_indirect_draw_calls):
                        render_pass.draw_indexed_indirect(
                            indirect_buffer =  self.render_job_dict['Mesh Culling Compute'].attachments['Draw Indexed Calls'],
                            indirect_offset = indirect_offset
                        )

                        indirect_offset += (4 * 5)

                elif render_job.name == 'Secondary Deferred Indirect Offscreen Graphics':
                    render_pass.set_pipeline(render_job.render_pipeline)
                    for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

                    render_pass.set_index_buffer(self.device_large_index_buffer, wgpu.IndexFormat.uint32)
                    render_pass.set_vertex_buffer(0, self.device_large_vertex_buffer)

                    # get the number of draw calls from the second mesh culling pass
                    draw_call_count_buffer = self.device.queue.read_buffer(
                        self.render_job_dict['Mesh Culling Compute'].attachments['Draw Indexed Call Count']
                    ).tobytes()
                    self.num_indirect_draw_calls = struct.unpack('I', draw_call_count_buffer[8:12])[0]

                    indirect_offset = 0
                    for draw_call_index in range(self.num_indirect_draw_calls):
                        render_pass.draw_indexed_indirect(
                            indirect_buffer =  self.render_job_dict['Secondary Mesh Culling Compute'].attachments['Draw Indexed Calls'],
                            indirect_offset = indirect_offset
                        )

                        indirect_offset += (4 * 5)

                elif render_job.name == 'Light Deferred Indirect Offscreen Graphics 0':
                    render_pass.set_pipeline(render_job.render_pipeline)
                    for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

                    render_pass.set_index_buffer(self.device_large_index_buffer, wgpu.IndexFormat.uint32)
                    render_pass.set_vertex_buffer(0, self.device_large_vertex_buffer)

                    indirect_offset = 0
                    for draw_call_index in range(self.num_light_indirect_draw_calls[0]):
                        render_pass.draw_indexed_indirect(
                            indirect_buffer =  self.render_job_dict['Light Mesh Culling Compute 0'].attachments['Draw Indexed Calls'],
                            indirect_offset = indirect_offset
                        )

                        indirect_offset += (4 * 5)

                elif render_job.name == 'Light Deferred Indirect Offscreen Graphics 1':
                    render_pass.set_pipeline(render_job.render_pipeline)
                    for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

                    render_pass.set_index_buffer(self.device_large_index_buffer, wgpu.IndexFormat.uint32)
                    render_pass.set_vertex_buffer(0, self.device_large_vertex_buffer)

                    indirect_offset = 0
                    for draw_call_index in range(self.num_light_indirect_draw_calls[1]):
                        render_pass.draw_indexed_indirect(
                            indirect_buffer =  self.render_job_dict['Light Mesh Culling Compute 1'].attachments['Draw Indexed Calls'],
                            indirect_offset = indirect_offset
                        )

                        indirect_offset += (4 * 5)

                elif render_job.name == 'Light Deferred Indirect Offscreen Graphics 2':
                    render_pass.set_pipeline(render_job.render_pipeline)
                    for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

                    render_pass.set_index_buffer(self.device_large_index_buffer, wgpu.IndexFormat.uint32)
                    render_pass.set_vertex_buffer(0, self.device_large_vertex_buffer)

                    indirect_offset = 0
                    for draw_call_index in range(self.num_light_indirect_draw_calls[2]):
                        render_pass.draw_indexed_indirect(
                            indirect_buffer =  self.render_job_dict['Light Mesh Culling Compute 2'].attachments['Draw Indexed Calls'],
                            indirect_offset = indirect_offset
                        )

                        indirect_offset += (4 * 5)

                else:

                    # vertex and index buffer based on the pass type
                    vertex_buffer = self.vertex_buffer
                    index_buffer = self.index_buffer
                    index_size = self.index_size 
                    if render_job.pass_type == 'Swap Chain Full Triangle' or render_job.pass_type == 'Full Triangle':
                        index_buffer = self.full_screen_triangle_index_buffer
                        vertex_buffer = self.full_screen_triangle_vertex_buffer
                        index_size = 3

                    # render pass pipeline, vertex and index buffers, bind groups, and draw
                    render_pass.set_pipeline(render_job.render_pipeline)
                    render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)

                    if (#render_job.name == 'Deferred Offscreen Graphics' or 
                        render_job.name == 'Simple Offscreen Graphics' or 
                        render_job.name == 'Close Up Light View Graphics'):
                        render_pass.set_vertex_buffer(0, self.render_job_dict['Skinning Compute'].attachments['Skinned Vertices'])
                    else:
                        render_pass.set_vertex_buffer(0, vertex_buffer)

                    for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
                    render_pass.draw_indexed(index_size, 1, 0, 0, 0)

                render_pass.end()

            elif render_job.type == 'Compute':
                if render_job.group_prev != None or render_job.group_next != None:

                    if render_job.group_next == None and render_job.group_prev != None:
                        render_job_name += ' ' + str(curr_job_group_run_count - 1)
                    else:
                        render_job_name += ' ' + str(curr_job_group_run_count)

                compute_pass = command_encoder.begin_compute_pass(
                    label = render_job_name
                )
                compute_pass.set_pipeline(render_job.compute_pipeline)
                for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                    compute_pass.set_bind_group(
                        index = bind_group_id,
                        bind_group = bind_group,
                        dynamic_offsets_data = [],
                        dynamic_offsets_data_start = 0,
                        dynamic_offsets_data_length = 999999)
                compute_pass.dispatch_workgroups(
                    render_job.dispatch_size[0],
                    render_job.dispatch_size[1],
                    render_job.dispatch_size[2])
                compute_pass.end()
            elif render_job.type == 'Copy':

                # copy attachments
                for attachment_index in range(len(render_job.attachment_info)):
                    dest_attachment_name = render_job.attachment_info[attachment_index]['Name']
                    
                    if render_job.attachment_info[attachment_index]['Type'] == 'TextureOutput':
                        command_encoder.copy_texture_to_texture(
                            source = {
                                'texture': render_job.copy_attachments[dest_attachment_name],
                                'mip_level': 0,
                                'origin': (0, 0, 0) 
                            },
                            destination = {
                            'texture': render_job.attachments[dest_attachment_name],
                            'mip_level': 0,
                            'origin': (0, 0, 0)
                            },
                            copy_size = (self.canvas._physical_size[0], self.canvas._physical_size[1], 1))
                    elif render_job.attachment_info[attachment_index]['Type'] == 'BufferOutput':
                        command_encoder.copy_buffer_to_buffer(
                            source = render_job.copy_attachments[dest_attachment_name],
                            source_offset = 0,
                            destination = render_job.attachments[dest_attachment_name],
                            destination_offset = 0,
                            size = render_job.attachments[dest_attachment_name].size
                        )

            render_time_delta = datetime.datetime.now() - start_render_job_time
            #print('render job end encoding {} ({}) time elapsed {} milliseconds'.format(
            #   render_job_name, render_job_index, 
            #   render_time_delta.microseconds / 1000))
            start_render_job_time = datetime.datetime.now()

            self.device.queue.submit([command_encoder.finish()])

            if render_job_name == 'Skinning Compute':
                self.set_voxelize_dispatch_size()

            render_time_delta = datetime.datetime.now() - start_render_job_time
            #print('render job run {} ({}) time elapsed {} milliseconds'.format(
            #   render_job_name, render_job_index, 
            #   render_time_delta.microseconds / 1000))

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 4: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        #self.device.queue.submit([command_encoder.finish()])
        self.canvas.request_draw()

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 4: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        keys = list(self.keyframe_channels.keys())
        times = self.mesh_data['translated_keyframe_channels'][keys[0]][1].times
        last_time = times[len(times) - 1]

        self.animation_time += 0.03
        if self.animation_time >= last_time:
            self.animation_time = 0.0

        self.frame_index += 1
        
        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 5: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        self.update_dynamic_bvh()

        self.initialize_render_job_data_per_frame()

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 7: {} milliseconds'.format(time_delta.microseconds / 1000))

    ##
    def build_bvh(self):
        #position_bytes, face_bytes, mesh_position_index_ranges = self.load_obj_file_2(
        #    file_path = 'c:\\Users\\Dingwings\\demo-models\\train6.obj')
        
        self.mesh_obj_result = load_obj_file(self.mesh_file_path)

        end_index_bytes = b''
        for range in self.mesh_obj_result.triangle_ranges:
            end_index_bytes += struct.pack('I', range[1])

        # bvh = embree_build_bvh_lib.build_bvh(
        #     self.mesh_obj_result.total_position_bytes, 
        #     self.mesh_obj_result.total_face_position_indices_bytes,
        #     len(self.mesh_obj_result.total_position_bytes),
        #     len(self.mesh_obj_result.total_face_position_indices_bytes),
        #     end_index_bytes,
        #     len(self.mesh_obj_result.triangle_ranges))

        bvh = embree_build_bvh_lib.build_bvh2(
            self.mesh_obj_result.total_triangle_positions_bytes, 
            len(self.mesh_obj_result.total_triangle_positions_bytes),
            end_index_bytes,
            len(self.mesh_obj_result.triangle_ranges)
        )

        self.scene_info = {}
        self.scene_info['bvh'] = bvh.tobytes()
        self.scene_info['vertex_positions'] = self.mesh_obj_result.total_position_bytes
        self.scene_info['faces'] = self.mesh_obj_result.total_face_position_indices_bytes
        self.scene_info['mesh_position_index_ranges'] = end_index_bytes

    ##
    def upload_bvh_data(self):

        # temporal restir
        uniform_data_bytes = b''
        uniform_data_bytes += struct.pack('i', self.options['num_temporal_restir_sample_permutations'])
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_data_bytes)
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = self.scene_info['bvh'])
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[2],
            buffer_offset = 0,
            data = self.scene_info['vertex_positions'])
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[3],
            buffer_offset = 0,
            data = self.scene_info['faces'])
        
        # spatial restir
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Spatial Restir Graphics 0'].uniform_buffers[1],
            buffer_offset = 0,
            data = self.scene_info['bvh'])
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Spatial Restir Graphics 0'].uniform_buffers[2],
            buffer_offset = 0,
            data = self.scene_info['vertex_positions'])
        
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Spatial Restir Graphics 0'].uniform_buffers[3],
            buffer_offset = 0,
            data = self.scene_info['faces'])

        # ray validation
        #self.device.queue.write_buffer(
        #    buffer = self.render_job_dict['Ray Validation Graphics'].uniform_buffers[1],
        #    buffer_offset = 0,
        #    data = self.scene_info['bvh'])
        #
        #self.device.queue.write_buffer(
        #    buffer = self.render_job_dict['Ray Validation Graphics'].uniform_buffers[2],
        #    buffer_offset = 0,
        #    data = self.scene_info['vertex_positions'])
        #
        #self.device.queue.write_buffer(
        #    buffer = self.render_job_dict['Ray Validation Graphics'].uniform_buffers[3],
        #    buffer_offset = 0,
        #    data = self.scene_info['faces'])
        

        # build irradiance cache job
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Build Irradiance Cache Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = self.scene_info['bvh']
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Build Irradiance Cache Compute'].uniform_buffers[2],
            buffer_offset = 0,
            data = self.scene_info['vertex_positions']
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Build Irradiance Cache Compute'].uniform_buffers[3],
            buffer_offset = 0,
            data = self.scene_info['faces']
        )

        mesh_triangle_range_data_bytes = b''
        for triangle_range in self.mesh_obj_result.triangle_ranges:
            mesh_triangle_range_data_bytes += struct.pack('i', triangle_range[0])
            mesh_triangle_range_data_bytes += struct.pack('i', triangle_range[1])

        material_data_bytes = b''
        for material in self.mesh_obj_result.total_materials:
            material_data_bytes += struct.pack('f', material['diffuse'].x)
            material_data_bytes += struct.pack('f', material['diffuse'].y)
            material_data_bytes += struct.pack('f', material['diffuse'].z)
            material_data_bytes += struct.pack('f', material['specular'].x)
            material_data_bytes += struct.pack('f', material['emissive'].x)
            material_data_bytes += struct.pack('f', material['emissive'].y)
            material_data_bytes += struct.pack('f', material['emissive'].z)
            material_data_bytes += struct.pack('f', material['transparency'])

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Build Irradiance Cache Compute'].uniform_buffers[4],
            buffer_offset = 0,
            data = mesh_triangle_range_data_bytes
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Build Irradiance Cache Compute'].uniform_buffers[5],
            buffer_offset = 0,
            data = material_data_bytes
        )

        prev_vertex_count = 0
        mesh_vertex_range_data_bytes = b''
        for mesh_vertex_range in self.mesh_obj_result.mesh_triangle_vertex_count_range:
            mesh_vertex_range_data_bytes += struct.pack('I', prev_vertex_count)
            mesh_vertex_range_data_bytes += struct.pack('I', mesh_vertex_range)
            prev_vertex_count = mesh_vertex_range
        
        mesh_bbox_bytes = b''
        num_meshes = len(self.mesh_obj_result.mesh_min_positions)
        for mesh_index in range(num_meshes):
            mesh_bbox_bytes += struct.pack('f', self.mesh_obj_result.mesh_min_positions[mesh_index].x)
            mesh_bbox_bytes += struct.pack('f', self.mesh_obj_result.mesh_min_positions[mesh_index].y)
            mesh_bbox_bytes += struct.pack('f', self.mesh_obj_result.mesh_min_positions[mesh_index].z)
            mesh_bbox_bytes += struct.pack('f', 1.0)


            mesh_bbox_bytes += struct.pack('f', self.mesh_obj_result.mesh_max_positions[mesh_index].x)
            mesh_bbox_bytes += struct.pack('f', self.mesh_obj_result.mesh_max_positions[mesh_index].y)
            mesh_bbox_bytes += struct.pack('f', self.mesh_obj_result.mesh_max_positions[mesh_index].z)
            mesh_bbox_bytes += struct.pack('f', 1.0)

        uniform_data_bytes = b''
        #uniform_data_bytes += struct.pack('I', num_meshes)
        uniform_data_bytes += struct.pack('I', self.num_large_meshes)

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Mesh Culling Compute'].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_data_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Mesh Culling Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = self.binary_mesh_range 
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Mesh Culling Compute'].uniform_buffers[2],
            buffer_offset = 0,
            data = self.binary_mesh_extent
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Secondary Mesh Culling Compute'].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_data_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Secondary Mesh Culling Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = self.binary_mesh_range 
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Secondary Mesh Culling Compute'].uniform_buffers[2],
            buffer_offset = 0,
            data = self.binary_mesh_extent
        )

        for i in range(3):
            render_job_name = 'Light Mesh Culling Compute {}'.format(i)
            self.device.queue.write_buffer(
                buffer = self.render_job_dict[render_job_name].uniform_buffers[0],
                buffer_offset = 0,
                data = uniform_data_bytes
            )
            self.device.queue.write_buffer(
                buffer = self.render_job_dict[render_job_name].uniform_buffers[1],
                buffer_offset = 0,
                data = self.binary_mesh_range 
            )
            self.device.queue.write_buffer(
                buffer = self.render_job_dict[render_job_name].uniform_buffers[2],
                buffer_offset = 0,
                data = self.binary_mesh_extent
            )

        num_work_groups = math.ceil(self.num_large_meshes / 256)
        self.render_job_dict['Mesh Culling Compute'].dispatch_size[0] = int(num_work_groups)
        self.render_job_dict['Secondary Mesh Culling Compute'].dispatch_size[0] = int(num_work_groups)
        for i in range(3):
            render_job_name = 'Light Mesh Culling Compute {}'.format(i)
            self.render_job_dict[render_job_name].dispatch_size[0] = int(num_work_groups)
        
    ##
    def get_world_position_from_screen_space(
        self,
        screen_space_position,
        view_projection_matrix):
        
        inverse_view_projection_matrix = float4x4.invert(view_projection_matrix)
        clip_space_position = float3(
            screen_space_position.x * 2.0  - 1.0,
            (screen_space_position.y * 2.0 - 1.0) * -1.0,
            screen_space_position.z)
        test_world_position = float3(
            inverse_view_projection_matrix.entries[0] * clip_space_position.x +  
            inverse_view_projection_matrix.entries[1] * clip_space_position.y + 
            inverse_view_projection_matrix.entries[2] * clip_space_position.z + 
            inverse_view_projection_matrix.entries[3],
            inverse_view_projection_matrix.entries[4] * clip_space_position.x +  
            inverse_view_projection_matrix.entries[5] * clip_space_position.y + 
            inverse_view_projection_matrix.entries[6] * clip_space_position.z + 
            inverse_view_projection_matrix.entries[7],
            inverse_view_projection_matrix.entries[8] * clip_space_position.x +  
            inverse_view_projection_matrix.entries[9] * clip_space_position.y + 
            inverse_view_projection_matrix.entries[10] * clip_space_position.z + 
            inverse_view_projection_matrix.entries[11])  
        test_world_position_w = (inverse_view_projection_matrix.entries[12] * clip_space_position.x +  
            inverse_view_projection_matrix.entries[13] * clip_space_position.y + 
            inverse_view_projection_matrix.entries[14] * clip_space_position.z + 
            inverse_view_projection_matrix.entries[15])
        test_world_position.x /= test_world_position_w
        test_world_position.y /= test_world_position_w
        test_world_position.z /= test_world_position_w

        return test_world_position

    ##
    def update_light_camera(
        self,
        view_projection_matrix,
        light_position,
        light_look_at,
        view_width,
        view_height,
        view_depth,
        render_job_names):

        inverse_view_projection_matrix = float4x4.invert(view_projection_matrix)
        
        light_direction = float3.normalize(light_look_at - light_position)
        light_up = float3(0.0, 1.0, 0.0)
        if abs(light_direction.y) > 0.98:
            light_up = float3(1.0, 0.0, 0.0)

        light_view_matrix = float4x4.view_matrix(
            eye_position = float3(light_position.x, light_position.y * -1.0, light_position.z), 
            look_at = light_look_at, 
            up = light_up
        )
        
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
            uniform_bytes += struct.pack('f', view_projection_matrix.entries[i])
        for i in range(16):
            uniform_bytes += struct.pack('f', light_view_projection_matrix.entries[i])
        for i in range(16):
            uniform_bytes += struct.pack('f', inverse_view_projection_matrix.entries[i])

        uniform_bytes += struct.pack('I', self.screen_width)
        uniform_bytes += struct.pack('I', self.screen_height)

        for render_job_name in render_job_names:
            self.device.queue.write_buffer(
            buffer = self.render_job_dict[render_job_name].uniform_buffers[0],
            buffer_offset = 0,
            data = uniform_bytes)

        return uniform_bytes, light_view_matrix, light_projection_matrix

    ##
    def update_render_job_user_data(self):
        for render_job_key in self.render_job_dict:
            render_job = self.render_job_dict[render_job_key]

            for user_data in render_job.shader_resource_user_data:
                data_type = user_data[3]

                uniform_bytes = b''
                if data_type == 'int':
                    uniform_bytes += struct.pack('I', user_data[1])
                elif data_type == 'float':
                    uniform_bytes += struct.pack('f', user_data[1])

                uniform_buffer_index = user_data[0]
                self.device.queue.write_buffer(
                    buffer = render_job.uniform_buffers[uniform_buffer_index],
                    buffer_offset = user_data[2],
                    data = uniform_bytes)

    ##
    def set_voxelize_dispatch_size(self):
        bbox_buffer = self.device.queue.read_buffer(
            self.render_job_dict['Skinning Compute'].attachments['Bounding Boxes']
        )
        max_bbox_x = struct.unpack('i', bbox_buffer[0:4])[0]
        max_bbox_y = struct.unpack('i', bbox_buffer[4:8])[0]
        max_bbox_z = struct.unpack('i', bbox_buffer[8:12])[0]
        min_bbox_x = struct.unpack('i', bbox_buffer[12:16])[0]
        min_bbox_y = struct.unpack('i', bbox_buffer[16:20])[0]
        min_bbox_z = struct.unpack('i', bbox_buffer[20:24])[0]

        min_brick_position = float3(
            min_bbox_x * 0.001 * self.position_scale,
            min_bbox_y * 0.001 * self.position_scale,
            min_bbox_z * 0.001 * self.position_scale
        )
        max_brick_position = float3(
            max_bbox_x * 0.001 * self.position_scale,
            max_bbox_y * 0.001 * self.position_scale,
            max_bbox_z * 0.001 * self.position_scale
        )
        min_brick_position_int = float3(
            math.floor(min_brick_position.x),
            math.floor(min_brick_position.y),
            math.floor(min_brick_position.z)
        )
        max_brick_position_int = float3(
            math.ceil(max_brick_position.x),
            math.ceil(max_brick_position.y),
            math.ceil(max_brick_position.z)
        )

        bbox = max_brick_position_int - min_brick_position_int

        self.render_job_dict['Flood Fill Compute'].dispatch_size[0] = max(int(bbox.x), 0)
        self.render_job_dict['Flood Fill Compute'].dispatch_size[1] = max(int(bbox.y), 0)
        self.render_job_dict['Flood Fill Compute'].dispatch_size[2] = max(int(bbox.z), 0)

    ##
    def update_dynamic_bvh(self):

        # just update once for static meshes, dynamic meshes are disabled
        if self.frame_index <= 1:
            # read back skinned vertices
            skinned_vertex_buffer = self.device.queue.read_buffer(
                self.render_job_dict['Skinning Compute'].attachments['Skinned Vertex Positions']
            )
            num_skinned_meshes = len(self.mesh_data['total_mesh_vertex_ranges'])
            num_vertices = self.mesh_data['total_mesh_vertex_ranges'][num_skinned_meshes - 1][1]
            num_skin_mesh_indices = len(self.wgpu_buffers['index-data'])
            num_skin_vertex_position_bytes = num_vertices * 16
            num_skin_index_bytes = num_skin_mesh_indices * 4

            skin_mesh_vertex_buffer_byte_buffer = skinned_vertex_buffer.tobytes()
            skin_mesh_index_buffer_byte_buffer = self.wgpu_buffers['index-data'].tobytes()

            num_static_meshes = int(len(self.mesh_obj_result.triangle_ranges))
            
            skin_vertex_end_index_bytes = b''
            for vertex_range in self.mesh_data['total_mesh_vertex_ranges']:
                skin_vertex_end_index_bytes += struct.pack('I', vertex_range[1])

            curr_num_skin_mesh_triangles = 0
            skin_triangle_end_index_bytes = b''
            for index in self.mesh_data['triangle_indices']:
                num_triangles = int(len(index) / 3)
                assert(int(len(index)) % 3 == 0)
                curr_num_skin_mesh_triangles += num_triangles
                skin_triangle_end_index_bytes += struct.pack('I', curr_num_skin_mesh_triangles)

            static_vertex_end_index_bytes = b''
            for mesh_index in range(num_static_meshes):
                position_range = self.mesh_obj_result.mesh_position_ranges[mesh_index]
                static_vertex_end_index_bytes += struct.pack('I', position_range[1])
            
            curr_num_static_mesh_triangles = 0
            static_triangle_end_index_bytes = b''
            for mesh_index in range(num_static_meshes):
                triangle_range = self.mesh_obj_result.triangle_ranges[mesh_index][1]
                static_triangle_end_index_bytes += struct.pack('I', triangle_range)

            embree_build_bvh_lib.add_meshes(
                self.mesh_obj_result.total_position_bytes, 
                self.mesh_obj_result.total_face_position_indices_bytes,
                len(self.mesh_obj_result.total_position_bytes),
                len(self.mesh_obj_result.total_face_position_indices_bytes),
                static_vertex_end_index_bytes,
                static_triangle_end_index_bytes,
                num_static_meshes
            )
        
            total_mesh_position = embree_build_bvh_lib.get_total_mesh_vertex_positions()
            total_mesh_triangle_indices = embree_build_bvh_lib.get_total_mesh_triangle_indices()

            total_mesh_position_bytes = total_mesh_position.tobytes()
            total_mesh_triangle_indices_bytes = total_mesh_triangle_indices.tobytes()

        
            self.scene_info['dynamic-bvh'] = embree_build_bvh_lib.build_bvh_and_clear_added_meshes()

            # upload dynamic bvh buffer
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[4],
                buffer_offset = 0,
                data = self.scene_info['dynamic-bvh'])

            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Temporal Restir Emissive Graphics'].uniform_buffers[3],
                buffer_offset = 0,
                data = self.scene_info['dynamic-bvh'])

            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[5],
                buffer_offset = 0,
                data = total_mesh_position_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[6],
                buffer_offset = 0,
                data = total_mesh_triangle_indices_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Temporal Restir Emissive Graphics'].uniform_buffers[4],
                buffer_offset = 0,
                data = total_mesh_position_bytes)
            
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Temporal Restir Emissive Graphics'].uniform_buffers[5],
                buffer_offset = 0,
                data = total_mesh_triangle_indices_bytes)

    ##
    def initialize_render_job_data_per_frame(self):
        
        draw_call_count_buffer = self.device.queue.read_buffer(
            self.render_job_dict['Mesh Culling Compute'].attachments['Draw Indexed Call Count']
        ).tobytes()
        self.num_indirect_draw_calls = struct.unpack('I', draw_call_count_buffer[0:4])[0]

        for i in range(3):
            light_cull_render_job_name = 'Light Mesh Culling Compute {}'.format(i)
            draw_call_count_buffer = self.device.queue.read_buffer(
                self.render_job_dict[light_cull_render_job_name].attachments['Draw Indexed Call Count']
            ).tobytes()
            self.num_light_indirect_draw_calls[i] = struct.unpack('I', draw_call_count_buffer[0:4])[0]

        zero_bytes = b''
        for i in range(128):
            zero_bytes += struct.pack('i', 0)

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Mesh Culling Compute'].attachments['Draw Indexed Call Count'],
            buffer_offset = 0,
            data = zero_bytes)
        
        for i in range(3):
            render_job_name = 'Light Mesh Culling Compute {}'.format(i)
            self.device.queue.write_buffer(
                buffer = self.render_job_dict[render_job_name].attachments['Draw Indexed Call Count'],
                buffer_offset = 0,
                data = zero_bytes)
        
        

    ##
    def convert_specular_images(self):
        for root, directories, files in os.walk('D:\\Downloads\\Bistro_v4\\Textures'):
            for file in files:
                if file.find('_Specular.dds') < 0:
                    continue

                full_path = os.path.join(root, file)
                image = Image.open(full_path, 'r')

                base_name_end = file.rfind('.')
                base_name = file[:base_name_end]
                output_full_path = 'd:\\Downloads\\Bistro_v4\\converted-textures\\' + base_name + '.png'
                output_image = image.save(output_full_path)

    ##
    def convert_emissive_images(self):
        for root, directories, files in os.walk('D:\\Downloads\\Bistro_v4\\Textures'):
            for file in files:
                if file.find('_Emissive.dds') < 0:
                    continue

                full_path = os.path.join(root, file)
                image = Image.open(full_path, 'r')

                base_name_end = file.rfind('.')
                base_name = file[:base_name_end]
                output_full_path = 'd:\\Downloads\\Bistro_v4\\converted-textures\\' + base_name + '.png'
                output_image = image.save(output_full_path)

    ##
    def set_up_scene_light_view_matrices(
        self, 
        light_direction):
        
        light_up = float3(0.0, 1.0, 0.0)
        if abs(light_direction.y) > 0.98:
            light_up = float3(1.0, 0.0, 0.0)

        view_direction = float3.normalize(self.look_at_position - self.eye_position)

        light_view_matrices = []
        light_projection_matrices = []
        light_view_projection_matrices = []

        # divide up frustum and set the light's look at position at the middle of the frustum division
        # then move from the center in the light direction with the radius of the frustum division
        
        # frustum corners' world position
        inverse_view_projection = float4x4.invert(self.view_projection_matrix)
        frustum_div_pct = [0.0, 0.7, 0.92, 1.0]
        frustum_corners = []
        for frustum_div in range(len(frustum_div_pct)):
            curr_z = frustum_div_pct[frustum_div]

            # front-top
            corner_pos = float3(        # left 
                -1.0,
                1.0,
                curr_z
            )
            frustum_corners.append(corner_pos)
            corner_pos = float3(        # right
                1.0,
                1.0,
                curr_z
            )
            frustum_corners.append(corner_pos)
            # front-bottom
            corner_pos = float3(        # left
                -1.0,
                -1.0,
                curr_z
            )
            frustum_corners.append(corner_pos)
            corner_pos = float3(        # right
                1.0,
                -1.0,
                curr_z
            )
            frustum_corners.append(corner_pos)

        # frustum corners to world position
        frustum_corner_world_positions = []
        for frustum_corner_pos in frustum_corners:
            world_position = inverse_view_projection.apply2(frustum_corner_pos)
            w = world_position[1]
            if w == 0.0:
                w = 1.0 / self.camera_far
            frustum_corner_world_positions.append(world_position[0] * (1.0 / w))

        # center of the frustum divisions
        frustum_center_world_position = []
        for i in range(len(frustum_div_pct) - 1):
            start_index = i * 4
            frustum_center = float3(0.0, 0.0, 0.0)
            for i in range(8):
                world_position = frustum_corner_world_positions[start_index + i]
                frustum_center += world_position
            frustum_center = frustum_center * (1.0 / 8.0)
            frustum_center_world_position.append(frustum_center)

        # light view matrices
        for i in range(3):
            light_view_matrix = float4x4.view_matrix(
                eye_position = frustum_center_world_position[i] + self.light_direction,
                look_at = frustum_center_world_position[i],
                up = light_up
            )
            light_view_matrices.append(light_view_matrix)

        # frustum corners to light view
        light_view_corners = []
        for i in range(3):

            start_index = i * 4
            corner_positions = []
            for j in range(8):
                corner_pos = frustum_corner_world_positions[start_index + j]
                corner_light_pos = light_view_matrices[i].apply2(corner_pos)
                corner_positions.append(corner_light_pos[0])
            light_view_corners.append(corner_positions)

        # min max of the frustum division corners from light view
        light_view_frustum_division_min_max_pos = []
        for i in range(3):
            min_pos = float3(9999.0, 9999.0, 9999.0)
            max_pos = float3(-9999.0, -9999.0, -9999.0)

            for j in range(8):
                index = start_index + j

                min_pos.x = min(light_view_corners[i][j].x, min_pos.x)
                min_pos.y = min(light_view_corners[i][j].y, min_pos.y)
                min_pos.z = min(light_view_corners[i][j].z, min_pos.z)

                max_pos.x = max(light_view_corners[i][j].x, max_pos.x)
                max_pos.y = max(light_view_corners[i][j].y, max_pos.y)
                max_pos.z = max(light_view_corners[i][j].z, max_pos.z)

            
            z_near_mult = 1.0
            z_far_mult = 1.0
            if i == 0:
                z_near_mult = 10.0
                z_far_mult = 10.0

                if min_pos.z >= 0.0:
                    z_near_mult *= -1.0
                if max_pos.z < 0.0:
                    z_far_mult *= -1.0

            elif i == 1:
                z_far_mult = 6.0
                z_near_mult = 6.0
            elif i == 2:
                z_near_mult = 4.0
                z_near_mult = 8.0

            min_pos.z *= z_near_mult
            max_pos.z *= z_far_mult
            light_view_frustum_division_min_max_pos.append([min_pos, max_pos])

        view_port_mult = 1.25
        for i in range(3):
            min_max_pos = light_view_frustum_division_min_max_pos[i]
            
            if i == 0:
                min_max_pos[1].z = 10.0
                min_max_pos[0].z = -10.0
            light_projection_matrix = float4x4.orthographic_projection_matrix(
                left = min_max_pos[0].x * view_port_mult,
                right = min_max_pos[1].x * view_port_mult,
                top = min_max_pos[1].y * view_port_mult,
                bottom = min_max_pos[0].y * view_port_mult,
                far = min_max_pos[1].z,
                near = min_max_pos[0].z,
                inverted = False
            )
            light_projection_matrices.append(light_projection_matrix)
            
        for i in range(3):
            light_view_projection_matrix = light_projection_matrices[i] * light_view_matrices[i]
            light_view_projection_matrices.append(light_view_projection_matrix)
            
        '''
        frustum_div_center = [2.0, 6.0, 12.0]
        radii = [6.0, 10.0, 18.0]
        curr_look_at_frustum_position = self.eye_position
        for i in range(3):
            curr_radius = radii[i]
            curr_look_at_frustum_position = curr_look_at_frustum_position + view_direction * frustum_div_center[i]
            light_position = curr_look_at_frustum_position + light_direction * curr_radius
            light_view_matrix = float4x4.view_matrix(
                eye_position = light_position, 
                look_at = curr_look_at_frustum_position, 
                up = light_up)
            light_projection_matrix = float4x4.orthographic_projection_matrix(
                left = -radius * 1.0,
                right = radius * 1.0,
                top = radius * 1.0,
                bottom = -radius * 1.0,
                far = radius * -4.0,
                near = radius * 0.0,
                inverted = False
            )
            curr_look_at_frustum_position = curr_look_at_frustum_position + view_direction * radii[i]

            light_view_projection_matrix = light_projection_matrix * light_view_matrix
            light_view_matrices.append(light_view_matrix)
            light_projection_matrices.append(light_projection_matrix)
            light_view_projection_matrices.append(light_view_projection_matrix)
        '''

        # copy the light view matrices to the light mesh culling and light mesh drawing pass
        total_light_view_projection_matrices_uniform_bytes = b''
        for pass_index in range(3):
            culling_render_job_name = 'Light Mesh Culling Compute {}'.format(pass_index)
            draw_render_job_name = 'Light Deferred Indirect Offscreen Graphics {}'.format(pass_index)

            uniform_bytes = b''
            for i in range(16):
                uniform_bytes += struct.pack('f', light_view_matrices[pass_index].entries[i])
            for i in range(16):
                uniform_bytes += struct.pack('f', light_projection_matrices[pass_index].entries[i])
            for i in range(16):
                uniform_bytes += struct.pack('f', light_view_projection_matrices[pass_index].entries[i])
            uniform_bytes += struct.pack('I', self.num_large_meshes)

            self.device.queue.write_buffer(
                buffer = self.render_job_dict[culling_render_job_name].uniform_buffers[0],
                buffer_offset = 0,
                data = uniform_bytes)

            uniform_bytes = b''
            for i in range(16):
                uniform_bytes += struct.pack('f', light_view_projection_matrices[pass_index].entries[i])
                total_light_view_projection_matrices_uniform_bytes += struct.pack('f', light_view_projection_matrices[pass_index].entries[i])
            self.device.queue.write_buffer(
                buffer = self.render_job_dict[draw_render_job_name].uniform_buffers[0],
                buffer_offset = 0,
                data = uniform_bytes)
        
        # all the light matrices to the composite pass
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Deferred Indirect Offscreen Lighting Graphics'].uniform_buffers[0],
            buffer_offset = 0,
            data = total_light_view_projection_matrices_uniform_bytes
        )

    ##
    def load_scene_textures(
        self,
        texture_paths):

        texture_array = []
        texture_view_array = []
        for texture_path in texture_paths:
            image = Image.open(texture_path, mode = 'r')
            
            # scale down the image
            flipped_image = None
            if image.width >= 256 or image.width >= 256:
                scale_width = 256.0 / float(image.width)
                scale_height = 256.0 / float(image.height)
                scaled_image = image.resize((int(image.width * scale_width), int(image.height * scale_height)))
                flipped_image = scaled_image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
            else:     
                flipped_image = image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)

            if image.mode != 'RGBA':    
                image = flipped_image.convert(mode = 'RGBA')
            else:
                image = flipped_image
                    
            # create texture
            image_byte_array = image.tobytes()
            texture_width = image.width
            texture_height = image.height
            texture_size = texture_width, texture_height, 1
            texture_format = wgpu.TextureFormat.rgba8unorm
            texture = self.device.create_texture(
                size=texture_size,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
                dimension=wgpu.TextureDimension.d2,
                format=texture_format,
                mip_level_count=1,
                sample_count=1,
                label = texture_path
            )
            texture_array.append(texture)

            # upload texture data
            self.device.queue.write_texture(
                {
                    'texture': texture_array[len(texture_array) - 1],
                    'mip_level': 0,
                    'origin': (0, 0, 0),
                },
                image_byte_array,
                {
                    'offset': 0,
                    'bytes_per_row': texture_width * 4
                },
                texture_size
            )

            texture_view_array.append(texture_array[len(texture_array) - 1].create_view())

        return texture_array, texture_view_array
    
    ## TLAS TEST ##
    def setup_TLASTest_data(self):
        
        bvh_file = open('d:\\Downloads\\Bistro_v4\\bvh-obj\\bistro2-triangle-tlas-bvh.bin', 'rb')
        bistro_tlas_bvh_bytes = bvh_file.read()
        bvh_file.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[1], 
            buffer_offset = 0,
            data = bistro_tlas_bvh_bytes
        )

        bvh_file = open('d:\\Downloads\\Bistro_v4\\bvh-obj\\bistro2-triangle-blas-node-partition-indices.bin', 'rb')
        bistro_blas_node_index_bytes = bvh_file.read()
        bvh_file.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[4], 
            buffer_offset = 0,
            data = bistro_blas_node_index_bytes
        )

        '''        
        # split up BLAS node indices to allowed memory size
        bvh_node_size = 64
        blas_node_index_array = []
        bin_array_index = 0
        total_bytes = 0
        prev_num_total_blas_node = 0
        start_node_offset = 0
        part_index = 0
        split_bytes = []

        blas_node_index_array.append([0, part_index])
        for mesh_index in range(self.num_large_meshes):
            num_total_blas_node = struct.unpack('I', bistro_blas_num_node_bytes[bin_array_index:bin_array_index+4])[0] - start_node_offset
            num_blas_node = num_total_blas_node - prev_num_total_blas_node

            # overran fixed size
            if total_bytes + num_blas_node * bvh_node_size > 200000000:
                part_index += 1
                start_node_offset = prev_num_total_blas_node
                split_bytes.append([prev_num_total_blas_node, mesh_index - 1, total_bytes])
                num_total_blas_node = struct.unpack('I', bistro_blas_num_node_bytes[bin_array_index:bin_array_index+4])[0] - start_node_offset
                prev_num_total_blas_node = 0
                total_bytes = 0
            
            blas_node_index_array.append([num_total_blas_node, part_index])
            total_bytes += num_blas_node * bvh_node_size
            prev_num_total_blas_node = num_total_blas_node
            
            bin_array_index += 4
        

        node_indices_bytes = b''
        for i in range(len(blas_node_index_array)):
            blas_node_index = blas_node_index_array[i]
            node_indices_bytes += struct.pack('I', blas_node_index[0])
            node_indices_bytes += struct.pack('I', blas_node_index[1])
        

        bvh_file = open('d:\\Downloads\\Bistro_v4\\bvh-obj\\bistro2-triangle-blas-bvh.bin', 'rb')
        bistro_blas_bvh_bytes = bvh_file.read()
        split_offset = split_bytes[0][2]
        bistro_blas_bvh_bytes_0 = bistro_blas_bvh_bytes[:split_offset]
        bistro_blas_bvh_bytes_1 = bistro_blas_bvh_bytes[split_offset:]
        bvh_file.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[2], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_0
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[3], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_1
        )
        '''

        bvh_file_0 = open('d:\\Downloads\\Bistro_v4\\bvh-obj\\bistro2-triangle-blas-bvh-0.bin', 'rb')
        bistro_blas_bvh_bytes_0 = bvh_file_0.read()
        bvh_file_0.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[2], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_0
        )

        bvh_file_1 = open('d:\\Downloads\\Bistro_v4\\bvh-obj\\bistro2-triangle-blas-bvh-1.bin', 'rb')
        bistro_blas_bvh_bytes_1 = bvh_file_1.read()
        bvh_file_1.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[3], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_1
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[5], 
            buffer_offset = 0,
            data = self.large_vertex_buffer_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[6], 
            buffer_offset = 0,
            data = self.large_index_buffer_bytes
        )

        vertex_buffer_byte_size = len(self.large_vertex_buffer_bytes)
        index_buffer_byte_size = len(self.large_index_buffer_bytes)


##
class BitField(object):

    ##
    def __init__(
        self, 
        dimension):

        self.dimension = dimension
        self.row_dimension = int(dimension / 32)
        self.ints = [0] * self.row_dimension * dimension
        
    ##
    def toggle_set_bit(
        self,
        x, 
        y, 
        value):

        list_index_x = int(x / 32)
        list_index_y = y

        total_list_index = list_index_y * self.row_dimension + list_index_x
        bit_index = x % 32
        or_value = 1 << bit_index
        self.ints[total_list_index] |= or_value 

    ##
    def get_value(
        self,
        x, 
        y):

        list_index_x = int(x / 32)
        list_index_y = y

        total_list_index = list_index_y * self.row_dimension + list_index_x
        bit_index = x % 32
        bit_mask = 1 << bit_index
        ret = self.ints[total_list_index] & bit_mask 
        ret = ret >> bit_index

        return ret


##
if __name__ == "__main__":
    # dimension = int(8192 / 128)
    # test_bit_field = BitField(dimension)
    # test_bit_field.toggle_set_bit(int(2000 / 128), int(3000 / 128), 1)
    # ret = test_bit_field.get_value(int(2000 / 128), int(3000 / 128))

    app = MyApp()
    app.build_bvh()
    app.upload_bvh_data()
    app.init_data()
    app.init_draw()

    run()