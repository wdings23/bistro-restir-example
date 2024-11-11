from wgpu.gui.auto import WgpuCanvas, run
import wgpu
import numpy as np
import datetime

from mat4 import *
from quat import *

from render_job import *

#os.add_dll_directory('D:\\test\\python-webgpu')

import random
from load_obj import *
from frustum_utils import *
from setup_render_job_data import *

from texture_atlas_thread import *

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
        self.device = self.adapter.request_device(
            required_features = ['shader-f16', 'float32-filterable']
        )

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

        # options
        self.options = {}
        self.options['mesh-directory'] = 'd:\\Downloads\\Bistro_v4'
        self.options['mesh-name'] = 'bistro2'

        # default uniform buffer for all the render passes
        self.default_uniform_buffer = self.device.create_buffer(
            size = 1024, 
            usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            label = 'Default Uniform Data')

        material_file_path = os.path.join(self.options['mesh-directory'], self.options['mesh-name'] + '.mat')
        self.init_materials(material_file_path)
        self.albedo_scene_textures, self.albedo_scene_texture_views, self.albedo_texture_dimensions = self.load_scene_textures(
            self.albedo_texture_paths
        )

        self.normal_scene_textures, self.normal_scene_texture_views, self.normal_texture_dimensions = self.load_scene_textures(
            self.normal_texture_paths
        )
        
        # get number of pages for textures
        self.num_albedo_pages = []
        self.total_num_albedo_per_texture = []
        texture_page_size = 64
        total_num_pages = 0
        prev_total_num_pages = 0
        for texture_dimension in self.albedo_texture_dimensions:
            num_page_x = int(max(texture_dimension[0] / texture_page_size, 1))
            num_page_y = int(max(texture_dimension[1] / texture_page_size, 1))
            num_total_pages = num_page_x * num_page_y
            self.num_albedo_pages.append([num_page_x, num_page_y, num_total_pages])
            
            prev_total_num_pages = total_num_pages
            total_num_pages += num_total_pages
            self.total_num_albedo_per_texture.append([prev_total_num_pages, total_num_pages])
        self.texture_page_lookup = [-1] * total_num_pages

        self.total_num_albedo_per_mip_texture = []
        self.num_mip_albedo_pages = []
        for mip in range(3):
            mip_denom = int(pow(2, mip))

            self.total_num_albedo_per_mip_texture.append([])
            self.num_mip_albedo_pages.append([])

            total_num_pages = 0
            prev_total_num_pages = 0

            for texture_dimension in self.albedo_texture_dimensions:
                mip_texture_dimension = [
                    int(texture_dimension[0] / mip_denom),
                    int(texture_dimension[1] / mip_denom)
                ]

                num_page_x = int(max(mip_texture_dimension[0] / texture_page_size, 1))
                num_page_y = int(max(mip_texture_dimension[1] / texture_page_size, 1))
                num_total_pages = num_page_x * num_page_y
                self.num_mip_albedo_pages[mip].append([num_page_x, num_page_y, num_total_pages])
                
                prev_total_num_pages = total_num_pages
                total_num_pages += num_total_pages
                self.total_num_albedo_per_mip_texture[mip].append([prev_total_num_pages, total_num_pages])

        # create render jobs with total textures in material database
        self.load_render_jobs(
            path = os.path.join(self.dir_path, 'render-jobs', 'bistro-example-render-jobs.json'),
            albedo_texture_array_views = self.albedo_scene_texture_views,
            normal_texture_array_views = self.normal_scene_texture_views,
            albedo_texture_array_paths = self.albedo_texture_paths,
            normal_texture_array_paths = self.normal_texture_paths)

        binary_mesh_file_path = os.path.join(self.options['mesh-directory'], self.options['mesh-name'] + '-triangles.bin')
        fp = open(binary_mesh_file_path, 'rb')
        self.binary_mesh_file_content = fp.read()
        fp.close()
        
        random.seed()

        self.options['swap_chain_texture_id'] = len(self.render_job_dict['Swap Chain Graphics'].attachment_info) - 1

        self.frame_index = 0
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

        self.num_indirect_draw_calls = 0
        self.num_light_indirect_draw_calls = [0, 0, 0]

        self.num_large_meshes = struct.unpack('I', self.binary_mesh_file_content[0:4])[0]
        num_total_vertices = struct.unpack('I', self.binary_mesh_file_content[4:8])[0]
        vertex_size = struct.unpack('I', self.binary_mesh_file_content[12:16])[0]
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

        self.load_initial_scaled_textures()

        self.zero_bytes = b''
        for i in range(128):
            self.zero_bytes += struct.pack('i', 0)

        self.curr_load_texture_page = [0, 0, 0, 0, 0]
        self.texture_page_load_info = []
        self.curr_mip_load_texture_page = [0] * 16

        self.prev_num_mip_texture_page_requests = 0
        self.prev_num_page_info_requests = 0

        self.request_count_bytes = None
        self.page_request_bytes = None
        self.texture_atlas_thread = threading.Thread(
            target = thread_load_texture_pages,
            args = (self,)
        )
        self.texture_page_loaded = False
        self.copy_texture_pages = []
        self.mutex_lock = threading.Lock()

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
    def init_materials(
        self, 
        full_path):
        
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

        # full screen triangle vertex buffer
        self.full_screen_triangle_vertex_buffer = self.device.create_buffer_with_data(
            data = full_triangle_vertex_data, 
            usage = wgpu.BufferUsage.VERTEX
        )

        # full screen triangle index buffer
        self.full_screen_triangle_index_buffer = self.device.create_buffer_with_data(
            data = full_triangle_index_data, 
            usage=wgpu.BufferUsage.INDEX
        )

        # sky atmosphere uniform data
        self.light_direction = float3.normalize(float3(-0.2, 0.8, 0.0))
        self.light_radiance = float3(10.0, 10.0, 10.0)

        # material id
        material_id_file_path = os.path.join(self.options['mesh-directory'], self.options['mesh-name'] + '.mid')
        fp = open(material_id_file_path, 'rb')
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
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Texture Page Queue Compute'].uniform_buffers[1],
            buffer_offset = 0,
            data = file_content
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Build Irradiance Cache Compute'].uniform_buffers[6],
            buffer_offset = 0,
            data = file_content
        )

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
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Texture Page Queue Compute'].uniform_buffers[2],
            buffer_offset = 0,
            data = self.materials_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Build Irradiance Cache Compute'].uniform_buffers[7],
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

        uniform_data_bytes = b''
        uniform_data_bytes += struct.pack('I', self.num_large_meshes)

        # mesh culling pass
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

        # secondary mesh culling pass
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

        # dispatch size based on the number of large meshes
        num_work_groups = math.ceil(self.num_large_meshes / 256)
        self.render_job_dict['Mesh Culling Compute'].dispatch_size[0] = int(num_work_groups)
        self.render_job_dict['Secondary Mesh Culling Compute'].dispatch_size[0] = int(num_work_groups)

        # albedo texture dimensions
        texture_dimension_bytes = b''
        for dimension in self.albedo_texture_dimensions:
            texture_dimension_bytes += struct.pack('i', dimension[0])
            texture_dimension_bytes += struct.pack('i', dimension[1])
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Texture Page Queue Compute'].uniform_buffers[0],
            buffer_offset = 0,
            data = texture_dimension_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Texture Atlas Graphics'].uniform_buffers[0],
            buffer_offset = 0,
            data = texture_dimension_bytes
        )

        # normal texture dimensions
        texture_dimension_bytes = b''
        for dimension in self.normal_texture_dimensions:
            texture_dimension_bytes += struct.pack('i', dimension[0])
            texture_dimension_bytes += struct.pack('i', dimension[1])
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Texture Page Queue Compute'].uniform_buffers[3],
            buffer_offset = 0,
            data = texture_dimension_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Texture Atlas Graphics'].uniform_buffers[1],
            buffer_offset = 0,
            data = texture_dimension_bytes
        )

        # setup tlas for scene
        self.setup_TLASTest_data()

    ##
    def init_draw(self):
        self.canvas.request_draw(self.draw_frame2)
        self.texture_atlas_thread.start()

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

        jitter_x = self.halton_sequence[self.frame_index % len(self.halton_sequence)][0]
        jitter_y = self.halton_sequence[self.frame_index % len(self.halton_sequence)][1]
        self.jitter_scale = 0.7
        self.jitter = (jitter_x * self.jitter_scale, jitter_y * self.jitter_scale)

        jittered_perspective_projection_matrix = perspective_projection_matrix
        jittered_perspective_projection_matrix.entries[3] = (jitter_x * self.jitter_scale) / float(self.screen_width)
        jittered_perspective_projection_matrix.entries[7] = (jitter_y * self.jitter_scale) / float(self.screen_height)

        return perspective_projection_matrix * view_matrix, jittered_perspective_projection_matrix * view_matrix, perspective_projection_matrix, jittered_perspective_projection_matrix


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
        
        # save previous view projection matrix
        self.prev_view_projection_matrix = self.view_projection_matrix
        self.view_projection_matrix = view_projection_matrix

        self.prev_jittered_view_projection_matrix = self.jittered_view_projection_matrix
        self.jittered_view_projection_matrix = jittered_view_projection_matrix

        # default uniform data
        update_default_data(
            device = self.device,
            device_buffer = self.default_uniform_buffer,
            screen_width = self.screen_width,
            screen_height = self.screen_height,
            frame_index = self.frame_index,
            num_meshes = self.num_large_meshes,
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
            ambient_occlusion_distance_threshold = 1.0
        )

        self.update_render_job_user_data()

        # current presentable swapchain texture
        current_present_texture = self.present_context.get_current_texture()

        curr_job_group_run_count = 0
        start_job_group = None
        end_job_group = None 
        start_job_group_index = -1
        render_job_index = 0

        start_time = datetime.datetime.now()
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

                            load_op = render_job.attachment_load_ops[attachment_name]
                            store_op = render_job.attachment_store_ops[attachment_name]

                            # don't clear on input-output attachment
                            if render_job.attachment_types[attachment_name] == 'TextureInputOutput':
                                load_op = wgpu.LoadOp.load

                            attachment_view = render_job.attachment_views[attachment_name]
                            color_attachments.append({
                                'view': attachment_view,
                                'resolve_target': None,
                                'clear_value': (0, 0, 0, 0),
                                'load_op': load_op,
                                'store_op': store_op
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

                    # get the number of draw calls from the second mesh culling pass
                    draw_call_count_buffer = self.device.queue.read_buffer(
                        self.render_job_dict['Mesh Culling Compute'].attachments['Draw Indexed Call Count']
                    ).tobytes()
                    self.num_indirect_draw_calls = struct.unpack('I', draw_call_count_buffer[0:4])[0]

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

                    draw_call_count_buffer = self.device.queue.read_buffer(
                        self.render_job_dict['Light Mesh Culling Compute 0'].attachments['Draw Indexed Call Count']
                    ).tobytes()
                    self.num_light_indirect_draw_calls[0] = struct.unpack('I', draw_call_count_buffer[0:4])[0]

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

                    draw_call_count_buffer = self.device.queue.read_buffer(
                        self.render_job_dict['Light Mesh Culling Compute 1'].attachments['Draw Indexed Call Count']
                    ).tobytes()
                    self.num_light_indirect_draw_calls[1] = struct.unpack('I', draw_call_count_buffer[0:4])[0]

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

                    draw_call_count_buffer = self.device.queue.read_buffer(
                        self.render_job_dict['Light Mesh Culling Compute 2'].attachments['Draw Indexed Call Count']
                    ).tobytes()
                    self.num_light_indirect_draw_calls[2] = struct.unpack('I', draw_call_count_buffer[0:4])[0]

                    indirect_offset = 0
                    for draw_call_index in range(self.num_light_indirect_draw_calls[2]):
                        render_pass.draw_indexed_indirect(
                            indirect_buffer =  self.render_job_dict['Light Mesh Culling Compute 2'].attachments['Draw Indexed Calls'],
                            indirect_offset = indirect_offset
                        )

                        indirect_offset += (4 * 5)

                else:

                    # vertex and index buffer based on the pass type
                    vertex_buffer = self.device_large_vertex_buffer
                    index_buffer = self.device_large_index_buffer
                    index_size = 0 
                    if render_job.pass_type == 'Swap Chain Full Triangle' or render_job.pass_type == 'Full Triangle':
                        index_buffer = self.full_screen_triangle_index_buffer
                        vertex_buffer = self.full_screen_triangle_vertex_buffer
                        index_size = 3

                    # render pass pipeline, vertex and index buffers, bind groups, and draw
                    render_pass.set_pipeline(render_job.render_pipeline)
                    render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)
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
                    
                    assert(dest_attachment_name in render_job.copy_attachments)
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

            #render_time_delta = datetime.datetime.now() - start_render_job_time
            #print('render job end encoding {} ({}) time elapsed {} milliseconds'.format(
            #   render_job_name, render_job_index, 
            #   render_time_delta.microseconds / 1000))
            #start_render_job_time = datetime.datetime.now()

            self.device.queue.submit([command_encoder.finish()])

            if render_job_name == 'Skinning Compute':
                self.set_voxelize_dispatch_size()

            #render_time_delta = datetime.datetime.now() - start_render_job_time
            #print('render job run {} ({}) time elapsed {} milliseconds'.format(
            #   render_job_name, render_job_index, 
            #   render_time_delta.microseconds / 1000))

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 4: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        #self.device.queue.submit([command_encoder.finish()])
        self.canvas.request_draw()

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 5: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        self.frame_index += 1
        
        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 6: {} milliseconds'.format(time_delta.microseconds / 1000))
        #start_time = datetime.datetime.now()

        self.update_data_per_frame()

        #time_delta = datetime.datetime.now() - start_time
        #print('time delta 7: {} milliseconds'.format(time_delta.microseconds / 1000))

        #if self.frame_index == 200:
        #    irradiance_cache_bytes = self.device.queue.read_buffer(
        #        self.render_job_dict['Build Irradiance Cache Compute'].attachments['Irradiance Cache']
        #    ).tobytes()
        #    irradiance_cache_file = open('d:\\Downloads\\screen-shots\\bistro-irradiance-cache.bin', 'wb')
        #    irradiance_cache_file.write(irradiance_cache_bytes)
        #    irradiance_cache_file.close()

    ##
    def update_data_per_frame(self):
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Mesh Culling Compute'].attachments['Draw Indexed Call Count'],
            buffer_offset = 0,
            data = self.zero_bytes)
        
        temporal_restir_uniform_data_bytes = b''
        temporal_restir_uniform_data_bytes = struct.pack('I', 4)
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[0],
            buffer_offset = 0,
            data = temporal_restir_uniform_data_bytes
        )

        self.queue_texture_pages()
        #self.device.queue.write_buffer(
        #    buffer = self.render_job_dict['Texture Page Queue Compute'].attachments['Counters'],
        #    buffer_offset = 0,
        #    data = self.zero_bytes)

        self.select_swap_chain_output()


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
    def load_scene_textures(
        self,
        texture_paths):

        texture_array = []
        texture_view_array = []
        texture_dimensions = []
        scale_image_size = 128
        for texture_path in texture_paths:
            image = Image.open(texture_path, mode = 'r')
            
            texture_dimensions.append((image.width, image.height))

            # scale down the image
            flipped_image = None
            if image.width >= scale_image_size or image.width >= scale_image_size:
                scale_width = float(scale_image_size) / float(image.width)
                scale_height = float(scale_image_size) / float(image.height)
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

        return texture_array, texture_view_array, texture_dimensions
    
    ## TLAS TEST ##
    def setup_TLASTest_data(self):
        
        if not 'TLAS Test Graphics' in self.render_job_dict:
            return

        tlas_file_path = os.path.join(self.options['mesh-directory'], 'bvh-obj', self.options['mesh-name'] + '-triangle-tlas-bvh.bin')
        bvh_file = open(tlas_file_path, 'rb')
        bistro_tlas_bvh_bytes = bvh_file.read()
        bvh_file.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[1], 
            buffer_offset = 0,
            data = bistro_tlas_bvh_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[1], 
            buffer_offset = 0,
            data = bistro_tlas_bvh_bytes
        )

        blas_node_partition_indices_file_path = os.path.join(self.options['mesh-directory'], 'bvh-obj', self.options['mesh-name'] + '-triangle-blas-node-partition-indices.bin')
        bvh_file = open(blas_node_partition_indices_file_path, 'rb')
        bistro_blas_node_index_bytes = bvh_file.read()
        bvh_file.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[4], 
            buffer_offset = 0,
            data = bistro_blas_node_index_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[4], 
            buffer_offset = 0,
            data = bistro_blas_node_index_bytes
        )

        blas_bvh_file_path_0 = os.path.join(self.options['mesh-directory'], 'bvh-obj', self.options['mesh-name'] + '-triangle-blas-bvh-0.bin')
        bvh_file_0 = open(blas_bvh_file_path_0, 'rb')
        bistro_blas_bvh_bytes_0 = bvh_file_0.read()
        bvh_file_0.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[2], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_0
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[2], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_0
        )

        blas_bvh_file_path_1 = os.path.join(self.options['mesh-directory'], 'bvh-obj', self.options['mesh-name'] + '-triangle-blas-bvh-1.bin')
        bvh_file_1 = open(blas_bvh_file_path_1, 'rb')
        bistro_blas_bvh_bytes_1 = bvh_file_1.read()
        bvh_file_1.close()
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[3], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_1
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[3], 
            buffer_offset = 0,
            data = bistro_blas_bvh_bytes_1
        )

        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[5], 
            buffer_offset = 0,
            data = self.large_vertex_buffer_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[5], 
            buffer_offset = 0,
            data = self.large_vertex_buffer_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['TLAS Test Graphics'].uniform_buffers[6], 
            buffer_offset = 0,
            data = self.large_index_buffer_bytes
        )
        self.device.queue.write_buffer(
            buffer = self.render_job_dict['Temporal Restir Graphics'].uniform_buffers[6], 
            buffer_offset = 0,
            data = self.large_index_buffer_bytes
        )

        vertex_buffer_byte_size = len(self.large_vertex_buffer_bytes)
        index_buffer_byte_size = len(self.large_index_buffer_bytes)

    ##
    def select_swap_chain_output(self):

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

    ##
    def get_mip_page_info(
        self,
        mip_page_request_bytes,
        mip_texture_page_hash_bytes,
        num_requests):

        texture_page_size = 64
        self.mip_texture_page_info = {}

        curr_pos = 0
        for i in range(num_requests):
            page_x_y = struct.unpack('i', mip_page_request_bytes[curr_pos:curr_pos+4])[0]    
            texture_id = struct.unpack('i', mip_page_request_bytes[curr_pos+4:curr_pos+8])[0] 
            hash_id = struct.unpack('i', mip_page_request_bytes[curr_pos+8:curr_pos+12])[0]
            mip_level = struct.unpack('i', mip_page_request_bytes[curr_pos+12:curr_pos+16])[0] & 0xff
            
            page_x = (0xffff & page_x_y)
            page_y = (page_x_y >> 16)

            mip_denom = pow(2, mip_level)
            texture_dimension = self.albedo_texture_dimensions[texture_id]

            mip_texture_dimension = [
                int(texture_dimension[0] / mip_denom),
                int(texture_dimension[1] / mip_denom)
            ]
    
            num_div_x = max(int(mip_texture_dimension[0] / texture_page_size), 1)
            num_div_y = max(int(mip_texture_dimension[1] / texture_page_size), 1)

            # texture page coordinate
            x_coord = abs(page_x) % num_div_x
            y_coord = abs(page_y) % num_div_y

            # wrap around for negative coordinate
            if page_x < 0:
                x_coord = num_div_x - x_coord
                if x_coord >= num_div_x:
                    x_coord = x_coord % num_div_x
            if page_y < 0:
                y_coord = num_div_y - y_coord
                if y_coord >= num_div_y:
                    y_coord = y_coord % num_div_x

            # image coordinate
            x_coord *= texture_page_size
            y_coord *= texture_page_size

            if texture_dimension[0] < texture_page_size:
                x_coord = 0
                y_coord = 0         

            # group by texture id and mip level
            key = str(texture_id) + '-' + str(mip_level)
            if not key in self.mip_texture_page_info:
                self.mip_texture_page_info[key] = []
            self.mip_texture_page_info[key].append((x_coord, y_coord, hash_id, texture_id, mip_level))

            curr_pos += 16   

        self.prev_num_mip_texture_page_requests = num_requests

        ##
        def sort_second(val):
            return val[1]

        ##
        def sort_first(val):
            return val[0]

        # sort list by y and x coordinate
        for key in self.mip_texture_page_info:
            self.mip_texture_page_info[key].sort(key = sort_second)
        for key in self.mip_texture_page_info:
            self.mip_texture_page_info[key].sort(key = sort_first)


    ##
    def queue_texture_pages(self):

        # get texture page requests
        self.mip_page_request_bytes = bytearray(self.device.queue.read_buffer(
            self.render_job_dict['Texture Page Queue Compute'].attachments['Texture Page Queue MIP']
        ).tobytes())

        self.request_count_bytes = self.device.queue.read_buffer(
            self.render_job_dict['Texture Page Queue Compute'].attachments['Counters']
        ).tobytes()

        self.mip_texture_page_hash_bytes = bytearray(self.device.queue.read_buffer(
            self.render_job_dict['Texture Page Queue Compute'].attachments['MIP Texture Page Hash Table']
        ).tobytes())

        # copy given texture pages 
        if self.texture_page_loaded == True:
            texture_page_size = 64
            pages_per_dimension = int(8192 / texture_page_size)
            for page_index in range(len(self.copy_texture_pages)):
                texture_page_info = self.copy_texture_pages[page_index]
                copy_page_size = texture_page_size, texture_page_size, 1
                atlas_texture = self.render_job_dict['Texture Page Queue Compute'].attachments[texture_page_info['attachment-key']]
                
                # texture page info
                atlas_texture_x = texture_page_info['x']
                atlas_texture_y = texture_page_info['y']
                page_index = texture_page_info['curr-mip-page-index']
                hash_index = texture_page_info['hash-index']
                byte_position = hash_index * 16

                # copy texture page
                if page_index < pages_per_dimension * pages_per_dimension:
                    # valid texture page 
                    
                    self.device.queue.write_texture(
                        {
                            'texture': atlas_texture,
                            'mip_level': 0,
                            'origin': (atlas_texture_x, atlas_texture_y, 0),
                        },
                        texture_page_info['image-bytes'],
                        {
                            'offset': 0,
                            'bytes_per_row': texture_page_size * 4
                        },
                        copy_page_size
                    )

                    # update with the page index
                    self.mip_texture_page_hash_bytes[byte_position+4:byte_position+8] = struct.pack(
                        'I', 
                        texture_page_info['curr-mip-page-index'])
                
                else:
                    # invalid texture page, just un-register page

                    self.mip_texture_page_hash_bytes[byte_position+4:byte_position+8] = struct.pack(
                        'I', 
                        0xffffffff)
                
            # update the hash table with the updated page index
            self.device.queue.write_buffer(
                buffer = self.render_job_dict['Texture Page Queue Compute'].attachments['MIP Texture Page Hash Table'],
                buffer_offset = 0,
                data = self.mip_texture_page_hash_bytes)
            
            self.copy_texture_pages = []
            self.texture_page_loaded = False

    ##
    def load_mip_texture_pages(self):
        # curr_mip_load_texture_page index mean:
        # 0: original texture key index
        # 1: page index in the current texture
        # 2: overall total page index
        # 3: total atlas texture index 
        # 4: overall total page index mip 0 
        # 5: overall total page index mip 1 
        # 6: overall total page index mip 2 
        # 7: overall total texture index for mip 0
        # 8: overall total texture index for mip 1
        # 9: overall total texture index for mip 2
        
        self.num_texture_pages_per_load = 384

        texture_atlas_dimension = 8192
        texture_page_size = 64
        hash_entry_size = 16

        num_pages_per_dimension = int(texture_atlas_dimension / texture_page_size)
        max_num_pages_per_texture = int(num_pages_per_dimension * num_pages_per_dimension)

        keys = list(self.mip_texture_page_info.keys())
        num_keys = len(keys)
        
        mip_image = None
        num_page_loaded = 0
        while True:
            # load texture

            if mip_image != None:
                mip_image.close()
                mip_image = None

            # past num pages in current texture
            if self.curr_mip_load_texture_page[0] >= num_keys:
                break
            
            # current image
            key = keys[self.curr_mip_load_texture_page[0]]
            page_info = self.mip_texture_page_info[key]
            texture_id = page_info[0][3]
            texture_path = self.albedo_texture_paths[texture_id]
            mip_level = page_info[0][4]
            #image = Image.open(texture_path, mode = 'r')
            #flipped_image = image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
            mip_denom = pow(2, mip_level)
            #mip_image_size = int(image.width / mip_denom), int(image.height / mip_denom)
            #mip_image = flipped_image.resize(mip_image_size)
            #mip_image_bytes = mip_image.tobytes()
            num_pages = len(self.mip_texture_page_info[key])    

            # texture dimensions for mip level
            texture_index = page_info[0][3]
            #start_page_index = self.total_num_albedo_per_mip_texture[mip_level][texture_index]
            texture_dimension = self.albedo_texture_dimensions[texture_index]
            mip_texture_dimension = [0, 0]
            mip_texture_dimension[0] = int(texture_dimension[0] / mip_denom)
            mip_texture_dimension[1] = int(texture_dimension[1] / mip_denom)
            #mip_image_page_size = int(min(mip_image.width / mip_denom, texture_page_size))
            mip_image_page_size = int(min(texture_dimension[0] / mip_denom, texture_page_size))
            num_page_x = int(mip_texture_dimension[0] / texture_page_size)
            num_pages = len(self.mip_texture_page_info[key])
            
            image_loaded = False
            mip_image_bytes = None

            num_curr_texture_page_loaded = 0
            while True:
                # load pages from texture

                # past number of pages to load per frame
                if num_page_loaded >= self.num_texture_pages_per_load:
                    break
                
                # move to next texture to load pages
                if self.curr_mip_load_texture_page[1] >= num_pages:
                    self.curr_mip_load_texture_page[0] += 1
                    self.curr_mip_load_texture_page[1] = 0
                    if mip_image != None:
                        mip_image.close()
                        mip_image = None

                    break
                
                # current page to load info
                page_info = self.mip_texture_page_info[key][self.curr_mip_load_texture_page[1]]

                # check if page has already been loaded
                hash_byte_pos = page_info[2] * 16
                page_index = struct.unpack('I', self.mip_texture_page_hash_bytes[hash_byte_pos+4:hash_byte_pos+8])[0]
                if page_index != 0xffffffff:
                    if page_index < self.curr_mip_load_texture_page[2]:
                        self.curr_mip_load_texture_page[1] += 1
                        continue
                
                # load image
                if image_loaded == False:
                    image = Image.open(texture_path, mode = 'r')
                    flipped_image = image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
                    mip_image_size = int(image.width / mip_denom), int(image.height / mip_denom)
                    mip_image = flipped_image.resize(mip_image_size)
                    mip_image_bytes = mip_image.tobytes()
                    image_loaded = True

                # load page image data
                mip_page_image_bytes = self.load_texture_page_image_data(
                    image = mip_image, 
                    texture_page_size = texture_page_size,
                    image_page_size = mip_image_page_size,
                    image_width = mip_texture_dimension[0],
                    page_info = page_info,
                    image_bytes = mip_image_bytes)
                
                # determine the atlas texture to copy into based on mip level and coordinate in the atlas texture
                attachment_key = 'Texture Atlas ' + str(int(page_info[4]))
                page_data = mip_page_image_bytes
                copy_page_size = texture_page_size, texture_page_size, 1
                atlas_texture = self.render_job_dict['Texture Page Queue Compute'].attachments[attachment_key]
                curr_mip_page = self.curr_mip_load_texture_page[mip_level+4]
                mip_texture_atlas_x = int((curr_mip_page % num_pages_per_dimension) * texture_page_size)
                mip_texture_atlas_y = int(int(curr_mip_page / num_pages_per_dimension) * texture_page_size)
                self.device.queue.write_texture(
                    {
                        'texture': atlas_texture,
                        'mip_level': 0,
                        'origin': (mip_texture_atlas_x, mip_texture_atlas_y, 0),
                    },
                    page_data,
                    {
                        'offset': 0,
                        'bytes_per_row': texture_page_size * 4
                    },
                    copy_page_size
                )

                # update hash page index
                hash_index = page_info[2]
                byte_position = hash_index * hash_entry_size
                self.mip_texture_page_hash_bytes[byte_position+4:byte_position+8] = struct.pack(
                    'I', 
                    self.curr_mip_load_texture_page[mip_level+4] + 1) 
                
                # update mip texture page in mip level
                self.curr_mip_load_texture_page[2] += 1
                self.curr_mip_load_texture_page[mip_level+4] += 1

                # update mip texture index for mip level
                if self.curr_mip_load_texture_page[mip_level+4] >= max_num_pages_per_texture:
                    self.curr_mip_load_texture_page[mip_level+7] += 1

                num_page_loaded += 1
                self.curr_mip_load_texture_page[1] += 1

            if num_page_loaded >= self.num_texture_pages_per_load:
                if mip_image != None:
                    mip_image.close()
                    mip_image = None

                break

    ##
    def load_texture_page_image_data(
        self, 
        image,
        texture_page_size, 
        image_page_size, 
        image_width,
        page_info,
        image_bytes):

        # load page row by row
        page_image_bytes = b''
        zero_byte = struct.pack('I', 0)
        num_left_over = texture_page_size - image_page_size
        for y in range(image_page_size):
            coord_x = page_info[0]
            coord_y = page_info[1] + y
            start_index = (coord_y * image_width + coord_x) * 4
            
            page_image_bytes += image_bytes[start_index:start_index + image_page_size * 4]
            
            # padding
            for i in range(num_left_over):
                page_image_bytes += zero_byte

        # row padding
        for y in range(num_left_over):
            for i in range(texture_page_size):
                page_image_bytes += zero_byte

        #test_image_size = texture_page_size, texture_page_size
        #output_image = Image.frombytes(mode = 'RGBA', size = test_image_size, data = page_image_bytes)
        #output_image.show()

        return page_image_bytes

    ##
    def load_initial_scaled_textures(self):

        texture_dimension = 512
        page_size = 8

        texture_atlas_x = 0
        texture_atlas_y = 0

        # albedo textures
        for texture_path in self.albedo_texture_paths:
            image = Image.open(texture_path, mode = 'r')
            flipped_image = image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
            mip_image_size = 8, 8
            mip_image = flipped_image.resize(mip_image_size)
            mip_image_bytes = mip_image.tobytes()

            attachment_key = 'Initial Texture Atlas'
            page_data = mip_image_bytes
            copy_page_size = page_size, page_size, 1
            atlas_texture = self.render_job_dict['Texture Page Queue Compute'].attachments[attachment_key]
            self.device.queue.write_texture(
                {
                    'texture': atlas_texture,
                    'mip_level': 0,
                    'origin': (texture_atlas_x, texture_atlas_y, 0),
                },
                page_data,
                {
                    'offset': 0,
                    'bytes_per_row': page_size * 4
                },
                copy_page_size
            )

            texture_atlas_x += page_size 
            if texture_atlas_x >= texture_dimension:
                texture_atlas_y += page_size
                texture_atlas_x = 0

        # normal textures
        for texture_path in self.normal_texture_paths:
            image = Image.open(texture_path, mode = 'r')
            flipped_image = image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
            mip_image_size = 8, 8
            mip_image = flipped_image.resize(mip_image_size)
            if mip_image.mode == 'RGB':
                mip_image = mip_image.convert(mode = 'RGBA')
            mip_image_bytes = mip_image.tobytes()

            attachment_key = 'Initial Texture Atlas'
            page_data = mip_image_bytes
            copy_page_size = page_size, page_size, 1
            atlas_texture = self.render_job_dict['Texture Page Queue Compute'].attachments[attachment_key]
            self.device.queue.write_texture(
                {
                    'texture': atlas_texture,
                    'mip_level': 0,
                    'origin': (texture_atlas_x, texture_atlas_y, 0),
                },
                page_data,
                {
                    'offset': 0,
                    'bytes_per_row': page_size * 4
                },
                copy_page_size
            )

            texture_atlas_x += page_size 
            if texture_atlas_x >= texture_dimension:
                texture_atlas_y += page_size
                texture_atlas_x = 0

##
if __name__ == "__main__":
    app = MyApp()
    app.init_data()
    app.init_draw()

    run()