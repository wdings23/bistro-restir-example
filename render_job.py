import os
import json
import wgpu
import io
from PIL import Image

##
class RenderJob(object):

    ##
    def __init__(
        self,
        name,
        device, 
        present_context,
        render_job_file_path,
        canvas_width,
        canvas_height,
        curr_render_jobs,
        extra_attachment_info):

        self.output_size = canvas_width, canvas_height, 1

        self.present_context = present_context

        # render job file
        file = open(render_job_file_path, 'rb')
        file_content = file.read()
        file.close()
        self.render_job_dict = json.loads(file_content)
        
        self.uniform_data = None

        self.name = name
        self.type = self.render_job_dict['Type']
        self.pass_type = self.render_job_dict['PassType']

        self.delayed_attachments = []
        
        self.view_port = [0, 0, canvas_width, canvas_height]

        self.min_depth = 0.0
        self.max_depth = 1.0

        self.draw_enabled = True

        # shader file
        if self.type != 'Copy':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.shader_path = os.path.join(dir_path, 'shaders', self.render_job_dict['Shader'])
            file = open(self.shader_path, 'rb')
            file_content = file.read()
            file.close()
            shader_source = file_content.decode('utf-8')
            self.shader = device.create_shader_module(
                code=shader_source
            )
        
        self.dispatch_size = [1, 1, 1]
        if self.type == 'Compute':
            if 'Dispatch' in self.render_job_dict:
                self.dispatch_size = self.render_job_dict['Dispatch'] 

        self.depth_texture = None
        self.depth_attachment_parent_info = None

        # attachments
        attachment_info = self.render_job_dict["Attachments"]
        self.create_attachments(
            attachment_info,
            device,
            canvas_width, 
            canvas_height,
            curr_render_jobs,
            extra_attachment_info)

        self.group_prev = None
        self.group_next = None
        self.group_run_count = 0

        self.shader_resource_user_data = []

    ##
    def finish_attachments_and_pipeline(
        self, 
        device, 
        total_render_jobs,
        default_uniform_buffer,
        albedo_texture_array_views,
        normal_texture_array_views,
        albedo_texture_array_paths,
        normal_texture_array_paths):
        
        self.albedo_texture_array_paths = albedo_texture_array_paths
        self.normal_texture_array_paths = normal_texture_array_paths
        self.albedo_texture_array_views = albedo_texture_array_views
        self.normal_texture_array_views = normal_texture_array_views

        # process the attachments that were delayed due to ordering of the render jobs
        self.process_delayed_attachments(total_render_jobs)

        if self.type != 'Copy':
            # pipeline data
            self.shader_resources = self.render_job_dict['ShaderResources']
            self.create_pipeline_data(
                shader_resource_dict = self.shader_resources, 
                curr_render_jobs = total_render_jobs,
                device = device)

            # pipeline binding and layout
            self.init_pipeline_layout(
                shader_resource_dict = self.shader_resources, 
                device = device,
                default_uniform_buffer = default_uniform_buffer)

            self.depth_texture = None
            self.depth_texture_view = None
            self.render_pipeline = None
            self.compute_pipeline = None

            if self.type == 'Graphics':
                self.init_render_pipeline(
                    render_job_dict = self.render_job_dict,
                    device = device)

            elif self.type == 'Compute':
                self.init_compute_pipeline(
                    render_job_dict = self.render_job_dict,
                    device = device)
                
            # any depth attachments from parent
            self.process_depth_attachments(total_render_jobs)

    ##
    def create_attachments(
        self,
        attachment_info,
        device,
        width,
        height,
        curr_render_jobs,
        extra_attachment_info):

        self.attachments = {}
        self.attachment_views = {}
        self.attachment_formats = {}
        self.attachment_types = {}

        self.attachment_info = attachment_info

        # swap chain uses presentable texture
        if self.pass_type == 'Swap Chain' or self.pass_type == "Swap Chain Full Triangle":
            texture_size = width, height, 1
            attachment_format = self.present_context.get_preferred_format(device.adapter)
            attachment_name = self.name + ' Output'
            self.attachments[attachment_name] = None
            self.attachment_formats[attachment_name] = attachment_format
            self.attachment_info.append(
                {
                    'Name': attachment_name,
                    'Type': 'TextureOutput',
                    'ParentJobName': 'This',
                    'Format': 'bgra8unorm-srgb',
                }
            )
        
        # create attachments
        attachment_index = 0
        for info in attachment_info:
            attachment_name = info['Name']
            attachment_type = info['Type']
            
            image_width = width
            attachment_scale_width = 1.0
            if 'ScaleWidth' in info:
                attachment_scale_width = info['ScaleWidth']
                image_width = width * attachment_scale_width
                self.view_port[2] = int(image_width)

            image_height = height
            attachment_scale_height = 1.0
            if 'ScaleHeight' in info:
                attachment_scale_height = info['ScaleHeight']
                image_height = height * attachment_scale_height
                self.view_port[3] = int(image_height)

            if 'ImageWidth' in info:
                image_width = info['ImageWidth']
                self.view_port[2] = int(image_width)

            if 'ImageHeight' in info:
                image_height = info['ImageHeight']
                self.view_port[3] = int(image_height)

            attachment_width = int(image_width)
            attachment_height = int(image_height)

            # create texture for output texture
            if attachment_type == 'TextureOutput':
                texture_size = attachment_width, attachment_height, 1
                attachment_format_str = info['Format']
                
                attachment_format = wgpu.TextureFormat.rgba8unorm
                if attachment_format_str == 'rgba32float':
                    attachment_format = wgpu.TextureFormat.rgba32float
                elif attachment_format_str == 'bgra8unorm-srgb':
                    attachment_format = wgpu.TextureFormat.bgra8unorm_srgb
                elif attachment_format_str == 'rgba16float':
                    attachment_format = wgpu.TextureFormat.rgba16float
                elif attachment_format_str == 'r32float':
                    attachment_format = wgpu.TextureFormat.r32float
                elif attachment_format_str == 'r32float':
                    attachment_format = wgpu.TextureFormat.r16float
                elif attachment_format_str == 'r16float':
                    attachment_format = wgpu.TextureFormat.r32float
                elif attachment_format_str == 'rg16float':
                    attachment_format = wgpu.TextureFormat.rg16float
                elif attachment_format_str == 'rg16float':
                    attachment_format = wgpu.TextureFormat.rg16float

                texture_usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
                if self.type == 'Copy':
                    texture_usage |= wgpu.TextureUsage.COPY_DST
                elif self.type == 'Compute':
                    texture_usage |= wgpu.TextureUsage.STORAGE_BINDING

                self.attachments[attachment_name] = device.create_texture(
                    size = texture_size,
                    usage = texture_usage,
                    dimension = wgpu.TextureDimension.d2,
                    format = attachment_format,
                    mip_level_count = 1,
                    sample_count = 1,
                    label = attachment_name
                )

                self.attachment_views[attachment_name] = self.attachments[attachment_name].create_view()
                self.attachment_formats[attachment_name] = attachment_format

                self.output_size = attachment_width, attachment_height, 1 

            elif attachment_type == 'TextureInput':
                # input texture

                # override attachment from top render job json
                for extra in extra_attachment_info:
                    if extra['index'] == attachment_index:
                        attachment_info[attachment_index]['ParentJobName'] = extra['parent_job_name']
                        attachment_info[attachment_index]['Name'] = extra['name']
                        break

                parent_job_name = info['ParentJobName']
                parent_attachment_name = info['Name']

                # find the parent job for this input
                parent_job = None
                for render_job in curr_render_jobs:
                    if render_job.name == parent_job_name:
                        parent_job = render_job
                        break
                
                # set view and format, attachment = None signals that it's an input attachment
                if parent_job != None and parent_attachment_name in parent_job.attachments:
                    new_attachment_name = parent_job_name + "-" + info['Name']
                    self.attachments[new_attachment_name] = None
                    self.attachment_views[new_attachment_name] = parent_job.attachment_views[parent_attachment_name]
                    self.attachment_formats[new_attachment_name] = parent_job.attachment_formats[parent_attachment_name]
                else:
                    new_attachment_name = parent_job_name + "-" + info['Name']
                    self.attachments[new_attachment_name] = None
                    self.delayed_attachments.append(
                        {
                            'Name': info['Name'],
                            'ParentJobName': parent_job_name,
                            'ParentName': parent_attachment_name
                        }
                    )

            elif attachment_type == 'BufferOutput':
                usage = wgpu.BufferUsage.STORAGE

                if 'Usage' in info:
                    if info['Usage'] == 'Vertex':
                        usage |= wgpu.BufferUsage.VERTEX

                buffer_size = 0
                if self.type == 'Copy':
                    parent_job_name = info['ParentJobName']
                    parent_attachment_name = info['ParentName']

                    # find the parent job for this input
                    parent_job = None
                    for render_job in curr_render_jobs:
                        if render_job.name == parent_job_name:
                            parent_job = render_job
                            break
                    
                    assert(parent_job != None)
                    assert(parent_attachment_name in parent_job.attachments)
                    buffer_size = parent_job.attachments[parent_attachment_name].size

                else:
                    buffer_size = info['Size']

                self.attachments[attachment_name] = device.create_buffer(
                    size = buffer_size, 
                    usage = usage | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.INDIRECT,
                    label = attachment_name
                )

            elif attachment_type == 'BufferInput':
                usage = wgpu.BufferUsage.STORAGE

                parent_job_name = info['ParentJobName']
                parent_attachment_name = info['Name']

                # find the parent job for this input
                parent_job = None
                for render_job in curr_render_jobs:
                    if render_job.name == parent_job_name:
                        parent_job = render_job
                        break
                
                if parent_job != None:
                    new_attachment_name = parent_job_name + "-" + info['Name']
                    self.attachments[new_attachment_name] = parent_job.attachments[parent_attachment_name]
                    self.attachment_views[new_attachment_name] = None
                    info['Size'] = self.attachments[new_attachment_name].size
                else:
                    new_attachment_name = parent_job_name + "-" + info['Name']
                    self.attachments[new_attachment_name] = None
                    self.delayed_attachments.append(
                        {
                            'Name': attachment_name,
                            'ParentJobName': parent_job_name,
                            'ParentName': parent_attachment_name
                        }
                    )

            elif attachment_type == 'BufferInputOutput':
                usage = wgpu.BufferUsage.STORAGE

                parent_job_name = info['ParentJobName']
                parent_attachment_name = info['Name']

                # find the parent job for this input
                parent_job = None
                for render_job in curr_render_jobs:
                    if render_job.name == parent_job_name:
                        parent_job = render_job
                        break
                
                if parent_job != None and parent_attachment_name in parent_job.attachments:
                    new_attachment_name = parent_job_name + "-" + info['Name']
                    self.attachments[new_attachment_name] = parent_job.attachments[parent_attachment_name]
                    self.attachment_views[new_attachment_name] = None
                    info['Size'] = self.attachments[new_attachment_name].size
                else:
                    # check if this is correct
                    assert(0)   
                    new_attachment_name = parent_job_name + "-" + info['Name']
                    self.attachments[new_attachment_name] = None
                    self.delayed_attachments.append(
                        {
                            'Name': attachment_name,
                            'ParentJobName': parent_job_name,
                            'ParentName': parent_attachment_name
                        }
                    )

            elif attachment_type == 'TextureInputOutput':
                parent_job_name = info['ParentJobName']
                parent_attachment_name = info['Name']
                
                # find the parent job for this input
                parent_job = None
                for render_job in curr_render_jobs:
                    if render_job.name == parent_job_name:
                        parent_job = render_job
                        break
                
                # check if this render job is using parent's depth texture for its own 
                if parent_attachment_name == 'Depth Output':
                    self.depth_attachment_parent_info = (parent_job_name, parent_attachment_name)
                else:
                    # writable attachment from parent, use the same name
                    if parent_job != None and parent_attachment_name in parent_job.attachments:
                        self.attachments[parent_attachment_name] = parent_job.attachments[parent_attachment_name]
                        self.attachment_views[parent_attachment_name] = parent_job.attachment_views[parent_attachment_name]
                        self.attachment_formats[parent_attachment_name] = parent_job.attachment_formats[parent_attachment_name]
                        
            self.attachment_types[attachment_name] = attachment_type

            attachment_index += 1

        # save the attachment link to the parent job and attachment
        if self.type == 'Copy':
            self.copy_attachments = {}
            for attachment_index in range(len(attachment_info)):
                info = attachment_info[attachment_index]
                attachment_name = info['Name']

                parent_job_name = info['ParentJobName']
                parent_attachment_name = info['ParentName']
                for parent_job in curr_render_jobs:
                    if parent_job.name == parent_job_name:
                        self.copy_attachments[attachment_name] = parent_job.attachments[parent_attachment_name]
                        break


    ##
    def create_pipeline_data(
        self, 
        shader_resource_dict, 
        curr_render_jobs,
        device):

        self.uniform_buffers = []
        self.textures = []
        self.texture_views = []

        # shader resources
        shader_resource_index = 0
        for shader_resource_entry in shader_resource_dict:
            name = shader_resource_entry['name']
            shader_resource_type = shader_resource_entry['type']
            shader_resource_usage = shader_resource_entry['usage']
            if shader_resource_type == 'buffer':
                # shader buffer
                
                shader_resource_size = 0 
                if 'size' in shader_resource_entry:
                    shader_resource_size = shader_resource_entry['size']
                else:
                    assert('parent_job' in shader_resource_entry)       # must have parent job to allow size to not be specified

                # attach parent job's uniform buffer to this one
                skip_creation = False
                if 'parent_job' in shader_resource_entry:
                    shader_resource_name = shader_resource_entry['name']

                    parent_job_name = shader_resource_entry['parent_job']
                    for render_job in curr_render_jobs:
                        if render_job.name == parent_job_name:
                            for resource_index in range(len(render_job.shader_resources)):
                                if render_job.shader_resources[resource_index]['name'] == shader_resource_name:
                                    self.uniform_buffers.append(render_job.uniform_buffers[resource_index])
                                    shader_resource_size = render_job.uniform_buffers[resource_index].size       
                                    skip_creation = True
                                    break
                            
                            if skip_creation == False:
                                for attachment_key in render_job.attachments:
                                    if attachment_key == shader_resource_name:
                                        self.uniform_buffers.append(render_job.attachments[attachment_key])
                                        shader_resource_size = render_job.attachments[attachment_key].size     
                                        skip_creation = True
                                        break


                            break
                    
                assert(shader_resource_size > 0)
                shader_resource_entry['size'] = shader_resource_size

                usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.STORAGE
                if shader_resource_usage == 'read_only_storage' or shader_resource_usage == 'storage':
                    usage = wgpu.BufferUsage.STORAGE

                if skip_creation == False:
                    self.uniform_buffers.append(device.create_buffer(
                        size = shader_resource_size, 
                        usage = usage | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
                        label = shader_resource_entry['name']
                    ))

                if 'resource_data' in shader_resource_entry:
                    for data in shader_resource_entry['resource_data']:
                        data_type = 'int'
                        if 'type' in data:
                            data_type = data['type']
                        copy_data = [shader_resource_index, data['data'], data['offset'], data_type]
                        self.shader_resource_user_data.append(copy_data)


            elif shader_resource_type == 'texture2d':
                # shader texture

                texture_width = 0
                texture_height = 0
                texture_size = 0, 0, 1
                texture_format = wgpu.TextureFormat.rgba8unorm
                image_byte_array = None
                if 'file_path' in shader_resource_entry:
                    file_path = shader_resource_entry['file_path']
                    image = Image.open(file_path, mode = 'r')
                    image_byte_array = image.tobytes()

                    texture_width = image.width
                    texture_height = image.height
                    texture_size = texture_width, texture_height, 1

                    shader_resource_entry['size'] = len(image_byte_array)
                    
                else:
                    texture_width = shader_resource_entry['width']
                    texture_height = shader_resource_entry['height']
                    texture_size = texture_width, texture_height, 1
                    texture_format = shader_resource_entry['format']

                    shader_resource_entry['size'] = texture_width * texture_height

                texture = device.create_texture(
                    size=texture_size,
                    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
                    dimension=wgpu.TextureDimension.d2,
                    format=texture_format,
                    mip_level_count=1,
                    sample_count=1,
                    label = name
                )

                self.textures.append(texture)
                self.texture_views.append(self.textures[len(self.textures) - 1].create_view())

                if image_byte_array != None:
                    device.queue.write_texture(
                        {
                            'texture': self.textures[len(self.textures) - 1],
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

            shader_resource_index += 1

    ##
    def init_pipeline_layout(
        self,
        shader_resource_dict, 
        device,
        default_uniform_buffer):
        
        bind_group_index = 0

        print('"{}"'.format(self.name))

        if self.name == 'Deferred Indirect Offscreen Graphics':
            print('')

        # We always have two bind groups, so we can play distributing our
        # resources over these two groups in different configurations.
        bind_groups_entries = [[]]
        bind_groups_layout_entries = [[]]

        # attachment bindings at group 0
        num_input_attachments = 0
        num_input_texture_attachments = 0
        num_input_buffer_attachments = 0
        added_sampler = False
        for attachment_index in range(len(self.attachments)):
            
            key = self.attachment_info[attachment_index]['Name']
            attachment_info = self.attachment_info[attachment_index]

            # render target doesn't need binding
            if attachment_info['Type'] == 'TextureOutput':
                if self.type == 'Compute':
                    bind_group_info = {
                        "binding": num_input_attachments,
                        "resource": self.attachment_views[key]
                    }
                    bind_groups_entries[bind_group_index].append(bind_group_info)

                    # binding layout
                    shader_stage = wgpu.ShaderStage.COMPUTE
                    sample_type = wgpu.TextureSampleType.unfilterable_float
                    bind_group_layout_info = {
                        "binding": num_input_attachments,
                        "visibility": shader_stage,
                        "storage_texture": {
                            "format": wgpu.TextureFormat.rgba32float,
                            "view_dimension": wgpu.TextureViewDimension.d2,
                            "access": wgpu.StorageTextureAccess.write_only
                        }
                    }
                    bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                    num_input_texture_attachments += 1

                    print('\tattachment texture: "{}" binding group: {}, binding: {}'.format(
                        key,
                        bind_group_index,
                        num_input_attachments))

                else:
                    continue

            # binding group
            if attachment_info['Type'] == 'TextureInput':
                key = self.attachment_info[attachment_index]['ParentJobName'] + '-' + self.attachment_info[attachment_index]['Name']
                assert(key in self.attachment_views)

                bind_group_info = {
                    "binding": num_input_attachments,
                    "resource": self.attachment_views[key]
                }
                bind_groups_entries[bind_group_index].append(bind_group_info)

                # binding layout
                shader_stage = wgpu.ShaderStage.FRAGMENT
                sample_type = wgpu.TextureSampleType.unfilterable_float
                if self.type == 'Compute':
                    shader_stage = wgpu.ShaderStage.COMPUTE
                    sample_type = wgpu.TextureSampleType.unfilterable_float

                bind_group_layout_info = {
                    "binding": num_input_attachments,
                    "visibility": shader_stage,
                    "texture": {
                        "sample_type": sample_type,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False
                    }
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                num_input_texture_attachments += 1

                print('\tattachment texture: "{}" binding group: {}, binding: {}'.format(
                    key,
                    bind_group_index,
                    num_input_attachments))

            elif attachment_info['Type'] == 'BufferOutput':
                bind_group_info = {
                    "binding": num_input_attachments,
                    "resource": {
                        "buffer": self.attachments[key],
                        "offset": 0,
                        "size": self.attachment_info[attachment_index]['Size']
                    }
                }
                bind_groups_entries[bind_group_index].append(bind_group_info)

                visibility_flag = wgpu.ShaderStage.COMPUTE
                if self.type == 'Graphics':
                    visibility_flag = wgpu.ShaderStage.FRAGMENT

                # binding layout
                bind_group_layout_info = {
                    "binding": num_input_attachments,
                    "visibility": visibility_flag,
                    "buffer": {
                        "type": wgpu.BufferBindingType.storage
                    }
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)
                
                num_input_buffer_attachments += 1

                print('\tattachment buffer output: "{}" binding group: {}, binding: {} type: storage'.format(
                    key,
                    bind_group_index,
                    num_input_attachments))
            
            elif attachment_info['Type'] == 'BufferInput':
                key = self.attachment_info[attachment_index]['ParentJobName'] + '-' + self.attachment_info[attachment_index]['Name']
                assert(key in self.attachment_views)

                bind_group_info = {
                    "binding": num_input_attachments,
                    "resource": {
                        "buffer": self.attachments[key],
                        "offset": 0,
                        "size": self.attachment_info[attachment_index]['Size']
                    }
                }
                bind_groups_entries[bind_group_index].append(bind_group_info)

                # binding layout
                bind_group_layout_info = {
                    "binding": num_input_attachments,
                    "visibility": wgpu.ShaderStage.COMPUTE | wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage
                    }
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                print('\tattachment buffer input: "{}" binding group: {}, binding: {} type: readonly'.format(
                    key,
                    bind_group_index,
                    num_input_attachments))

            elif attachment_info['Type'] == 'BufferInputOutput':
                key = self.attachment_info[attachment_index]['ParentJobName'] + '-' + self.attachment_info[attachment_index]['Name']
                assert(key in self.attachment_views)

                bind_group_info = {
                    "binding": num_input_attachments,
                    "resource": {
                        "buffer": self.attachments[key],
                        "offset": 0,
                        "size": self.attachment_info[attachment_index]['Size']
                    }
                }
                bind_groups_entries[bind_group_index].append(bind_group_info)

                # binding layout
                bind_group_layout_info = {
                    "binding": num_input_attachments,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": wgpu.BufferBindingType.storage
                    }
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                print('\tattachment buffer input: "{}" binding group: {}, binding: {} type: storage'.format(
                    key,
                    bind_group_index,
                    num_input_attachments))


            num_input_attachments += 1

        if num_input_attachments > 0:
            bind_group_index += 1
            bind_groups_entries.append([])
            bind_groups_layout_entries.append([])

        # create group bindings and group layout
        texture_index = 0
        binding_index = 0
        uniform_buffer_index = 0
        for shader_resource_entry in shader_resource_dict:
            
            # shader stage (vertex/fragment/compute)
            shader_stage = wgpu.ShaderStage.VERTEX
            if shader_resource_entry['shader_stage'] == 'fragment':
                shader_stage |= wgpu.ShaderStage.FRAGMENT
            elif shader_resource_entry['shader_stage'] == 'compute':
                shader_stage = wgpu.ShaderStage.COMPUTE
            elif shader_resource_entry['shader_stage'] == 'all':
                shader_stage |= wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
                

            # usage
            usage = wgpu.BufferBindingType.uniform
            if shader_resource_entry['usage'] == 'storage':
                usage = wgpu.BufferBindingType.storage
            elif shader_resource_entry['usage'] == 'read_only_storage':
                usage = wgpu.BufferBindingType.read_only_storage

            # build binding group and binding layout
            if shader_resource_entry['type'] == 'buffer':
                # buffer

                # binding group
                bind_group_info = {
                    "binding": binding_index,
                    "resource": 
                    {
                        "buffer": self.uniform_buffers[uniform_buffer_index],
                        "offset": 0,
                        "size": shader_resource_entry['size'],
                    },
                }
                
                bind_groups_entries[bind_group_index].append(bind_group_info)

                # binding layout
                bind_group_layout_info = {
                    "binding": binding_index,
                    "visibility": shader_stage,
                    "buffer": { "type": usage},
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                print('\tresource buffer: "{}" binding group: {}, binding: {}, uniform index: {}, size: {}, visibility: {}, usage "{}"'.format(
                    shader_resource_entry['name'],
                    bind_group_index,
                    binding_index,
                    uniform_buffer_index,
                    shader_resource_entry['size'], 
                    shader_stage, 
                    usage))

                uniform_buffer_index += 1

            elif shader_resource_entry['type'] == 'texture2d':
                # texture

                # binding group
                bind_group_info = {
                    "binding": binding_index,
                    "resource": self.texture_views[texture_index]
                }
                bind_groups_entries[bind_group_index].append(bind_group_info)

                # binding layout
                bind_group_layout_info = {
                    "binding": binding_index,
                    "visibility": shader_stage,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    }
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                print('\tresource texture: "{}" binding group: {}, binding: {}, size: {}, visibility: {}, usage "{}"'.format(
                    shader_resource_entry['name'],
                    bind_group_index,
                    binding_index,
                    shader_resource_entry['size'], 
                    shader_stage, 
                    usage))
            
            binding_index += 1
        
        # create sampler for textures
        num_attachment_binding_groups = len(bind_groups_entries[0])
        if len(self.texture_views) > 0 or num_input_texture_attachments > 0:
            
            # nearest point sampler 
            self.nearest_sampler = device.create_sampler(
                address_mode_u = wgpu.AddressMode.mirror_repeat,
                address_mode_v = wgpu.AddressMode.mirror_repeat
            )
            bind_groups_entries[0].append(
                {
                    "binding": num_attachment_binding_groups,
                    "resource": self.nearest_sampler
                }
            )

            # linear sampler
            self.linear_sampler = device.create_sampler(
                min_filter = wgpu.FilterMode.linear,
                mag_filter =  wgpu.FilterMode.linear,
                address_mode_u = wgpu.AddressMode.repeat,
                address_mode_v = wgpu.AddressMode.repeat
            )
            bind_groups_entries[0].append(
                {
                    "binding": num_attachment_binding_groups + 1,
                    "resource": self.linear_sampler
                }
            )

            # layout of nearest sampler
            shader_stage = wgpu.ShaderStage.FRAGMENT
            if self.type == 'Compute':
                shader_stage = wgpu.ShaderStage.COMPUTE
            bind_groups_layout_entries[0].append(
                {
                    "binding": num_attachment_binding_groups,
                    "visibility": shader_stage,
                    "sampler": {
                        "type": wgpu.SamplerBindingType.non_filtering
                    }
                }
            )

            # layout for linear sampler
            bind_groups_layout_entries[0].append(
                {
                    "binding": num_attachment_binding_groups + 1,
                    "visibility": shader_stage,
                    "sampler": {
                        "type": wgpu.SamplerBindingType.filtering
                    }
                }
            )

            added_sampler = True

            print('\tnearest sampler: group: 0, binding: {}'.format(
                    num_attachment_binding_groups))

            print('\tlinear sampler: group: 0, binding: {}'.format(
                    num_attachment_binding_groups + 1))

        # layout for default uniform buffer
        #bind_group_index = len(bind_groups_entries) - 1
        shader_stage = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
        bind_groups_entries[bind_group_index].append(
            {
                'binding': binding_index,
                'resource':
                {
                    "buffer": default_uniform_buffer,
                    "offset": 0,
                    "size": 1024,
                }
            }
        )
        bind_groups_layout_entries[bind_group_index].append(
            {
                "binding": binding_index,
                "visibility": shader_stage,
                "buffer": 
                { 
                    "type": wgpu.BufferBindingType.uniform
                },
            }
        )
        num_attachment_binding_groups += 1

        print('\tdefault uniform buffer: group {}, binding: {}'.format(bind_group_index, binding_index))

        has_texture_array = False
        if 'UseGlobalTextures' in self.render_job_dict:
            if self.render_job_dict['UseGlobalTextures'] == 'True':
                has_texture_array = True 

        if has_texture_array == True:
            
            if added_sampler == False:
                # nearest point sampler 
                self.nearest_sampler = device.create_sampler(
                    address_mode_u = wgpu.AddressMode.mirror_repeat,
                    address_mode_v = wgpu.AddressMode.mirror_repeat
                )
                bind_groups_entries[0].append(
                    {
                        "binding": num_attachment_binding_groups,
                        "resource": self.nearest_sampler
                    }
                )

                # linear sampler
                self.linear_sampler = device.create_sampler(
                    min_filter = wgpu.FilterMode.linear,
                    mag_filter =  wgpu.FilterMode.linear,
                    address_mode_u = wgpu.AddressMode.repeat,
                    address_mode_v = wgpu.AddressMode.repeat
                )
                bind_groups_entries[0].append(
                    {
                        "binding": num_attachment_binding_groups + 1,
                        "resource": self.linear_sampler
                    }
                )

                # layout of nearest sampler
                shader_stage = wgpu.ShaderStage.FRAGMENT
                if self.type == 'Compute':
                    shader_stage = wgpu.ShaderStage.COMPUTE
                bind_groups_layout_entries[0].append(
                    {
                        "binding": num_attachment_binding_groups,
                        "visibility": shader_stage,
                        "sampler": {
                            "type": wgpu.SamplerBindingType.non_filtering
                        }
                    }
                )

                # layout for linear sampler
                bind_groups_layout_entries[0].append(
                    {
                        "binding": num_attachment_binding_groups + 1,
                        "visibility": shader_stage,
                        "sampler": {
                            "type": wgpu.SamplerBindingType.filtering
                        }
                    }
                )

                print('\tnearest sampler: group: 0, binding: {}'.format(
                        num_attachment_binding_groups))

                print('\tlinear sampler: group: 0, binding: {}'.format(
                        num_attachment_binding_groups + 1))

            self.add_textures_to_pass(
                device = device, 
                albedo_texture_array_views = self.albedo_texture_array_views,
                normal_texture_array_views = self.normal_texture_array_views,
                albedo_texture_paths = self.albedo_texture_array_paths,
                normal_texture_paths = self.normal_texture_array_paths,
                bind_groups = bind_groups_entries,
                bind_group_layouts = bind_groups_layout_entries)

        # Create the wgpu binding objects
        self.bind_group_layouts = []
        self.bind_groups = []

        for entries, layout_entries in zip(bind_groups_entries, bind_groups_layout_entries):
            bind_group_layout = device.create_bind_group_layout(entries=layout_entries)
            self.bind_group_layouts.append(bind_group_layout)
            self.bind_groups.append(
                device.create_bind_group(layout=bind_group_layout, entries=entries)
            )

        self.pipeline_layout = device.create_pipeline_layout(bind_group_layouts=self.bind_group_layouts)

    ##
    def init_compute_pipeline(
        self,
        render_job_dict,
        device):

        self.compute_pipeline = device.create_compute_pipeline(
            layout = self.pipeline_layout,
            compute = {
                'module': self.shader, 
                'entry_point': 'cs_main'
            })

    ##
    def init_render_pipeline(
        self,
        render_job_dict, 
        device):

        cull_mode_str = render_job_dict['RasterState']['CullMode']
        front_face_str = render_job_dict['RasterState']['FrontFace']
        depth_enabled_str = render_job_dict['DepthStencilState']['DepthEnable']
        depth_func_str = render_job_dict['DepthStencilState']['DepthFunc']

        # cull mode
        cull_mode = wgpu.CullMode.none
        if cull_mode_str == 'Front':
            cull_mode = wgpu.CullMode.front
        elif cull_mode_str == 'Back':
            cull_mode = wgpu.CullMode.back
        
        # front face
        front_face = wgpu.FrontFace.ccw
        if front_face_str == 'Clockwise':
            front_face = wgpu.FrontFace.cw

        # depth toggle
        depth_enabled = False
        if depth_enabled_str == "True":
            depth_enabled = True

        # depth comparison function
        depth_func = wgpu.CompareFunction.never
        if depth_func_str == "Less":
            depth_func = wgpu.CompareFunction.less
        elif depth_func_str == "LessEqual":
            depth_func = wgpu.CompareFunction.less_equal
        elif depth_func_str == "Greater":
            depth_func = wgpu.CompareFunction.greater
        elif depth_func_str == "GreaterEqual":
            depth_func = wgpu.CompareFunction.greater_equal
        elif depth_func_str == "Equal":
            depth_func = wgpu.CompareFunction.equal
        elif depth_func_str == "NotEqual":
            depth_func = wgpu.CompareFunction.not_equal
        elif depth_func_str == "Always":
            depth_func = wgpu.CompareFunction.always

        self.depth_enabled = depth_enabled

        # attachment info
        render_target_info = []
        for attachment_name in self.attachments:
            
            if (attachment_name in self.attachment_views and 
                self.attachment_views[attachment_name] != None and 
                self.attachments[attachment_name] is not None):
                
                attachment_format = self.attachment_formats[attachment_name]

                if attachment_format == 'rgba32float' or attachment_format == 'r32float':
                    render_target_info.append(
                        {
                            "format": attachment_format,
                        }
                    )
                else:
                    render_target_info.append(
                        {
                            "format": attachment_format,
                            "blend": {
                                "alpha": (
                                    wgpu.BlendFactor.one,
                                    wgpu.BlendFactor.zero,
                                    wgpu.BlendOperation.add,
                                ),
                                "color": (
                                    wgpu.BlendFactor.one,
                                    wgpu.BlendFactor.zero,
                                    wgpu.BlendOperation.add,
                                ),
                            },
                        }
                    )

                print('\toutput attachment: "{}", format: {}'.
                    format(
                        attachment_name, 
                        attachment_format))

        # vertex input assembly
        self.vertex_component_data_size = []
        self.vertex_component_format = []
        self.vertex_num_components = []
        self.vertex_stride_size = 0

        if "VertexFormat" in render_job_dict:
            vertex_format_list = render_job_dict['VertexFormat']
            self.vertex_stride_size = 0
            for entry in vertex_format_list:
                if entry == 'Vec4':
                    component_data_size = 4
                    num_components = 4
                    format = wgpu.VertexFormat.float32x4
                elif entry == 'Vec3':
                    component_data_size = 4
                    num_components = 3
                    format = wgpu.VertexFormat.float32x3
                elif entry == 'Vec2':
                    component_data_size = 4
                    num_components = 2
                    format = wgpu.VertexFormat.float32x2
                elif entry == 'Float':
                    component_data_size = 4
                    num_components = 1
                    format = wgpu.VertexFormat.float32

                self.vertex_stride_size += num_components * component_data_size
                self.vertex_component_data_size.append(component_data_size)
                self.vertex_num_components.append(num_components)
                self.vertex_component_format.append(format)            

        attributes = []
        offset = 0
        for i in range(len(self.vertex_component_format)):
            attribute = {
                "format": self.vertex_component_format[i],
                "offset": offset,
                "shader_location": i
            }
            attributes.append(attribute)

            offset += self.vertex_component_data_size[i] * self.vertex_num_components[i]

        default_buffer_info = {
            "array_stride": 4 * 10,
            "step_mode": wgpu.VertexStepMode.vertex,
            "attributes": [
                {
                    "format": wgpu.VertexFormat.float32x4,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": wgpu.VertexFormat.float32x2,
                    "offset": 4 * 4,
                    "shader_location": 1,
                },
                {
                    "format": wgpu.VertexFormat.float32x4,
                    "offset": 4 * 4 + 4 * 2,
                    "shader_location": 2,
                },
            ]
        }
        
        buffer_info = None
        if "VertexFormat" in render_job_dict:
            buffer_info = {
                "array_stride": self.vertex_stride_size,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": attributes
            }
        else:
            buffer_info = default_buffer_info
            self.vertex_stride_size = 4 * 10
            self.vertex_component_data_size = [16, 8, 16]
            self.vertex_component_format = [
                wgpu.VertexFormat.float32x4,
                wgpu.VertexFormat.float32x2,
                wgpu.VertexFormat.float32x4
            ]
            self.vertex_num_components = [4, 2, 4]


        depth_stencil = None
        if depth_enabled == True: 
            depth_stencil = {
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": depth_enabled,
                "depth_compare": depth_func,
            }

        self.render_pipeline = device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex = {
                "module": self.shader,
                "entry_point": "vs_main",
                "buffers": [
                    buffer_info
                ]
            },
            primitive = {
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": front_face,
                "cull_mode": cull_mode,
            },
            multisample=False,
            fragment = {
                "module": self.shader,
                "entry_point": "fs_main",
                "targets": render_target_info,
            },
            depth_stencil = depth_stencil
        )

        print('\tpipeline: depth enabled: {}, depth func: {}, front face: {}, cull mode: {}, vertex buffer array stride: {}'.format(
            depth_enabled,
            depth_func,
            front_face,
            cull_mode,
            self.vertex_stride_size
        ))

        # create depth texture if depth test is enabled
        # and only if not using depth texture attachment from parent job
        if depth_enabled == True and self.depth_texture == None:

            if self.depth_attachment_parent_info == None:
                depth_texture_size = self.output_size
                self.depth_texture = device.create_texture(
                    size=depth_texture_size,
                    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
                    dimension=wgpu.TextureDimension.d2,
                    format=wgpu.TextureFormat.depth24plus,
                    mip_level_count=1,
                    sample_count=1,
                )

                self.depth_texture_view = self.depth_texture.create_view()
            
    ##
    def process_delayed_attachments(
        self, 
        total_render_jobs):
        
        # process delayed attachments, waited for all the render jobs to be created
        for delayed_attachment_info in self.delayed_attachments:
            parent_job_name = delayed_attachment_info['ParentJobName']
            parent_attachment_name = delayed_attachment_info['ParentName']
            attachment_name = delayed_attachment_info['Name']

            # find parent job
            parent_job = None
            for render_job in total_render_jobs:
                if render_job.name == parent_job_name:
                    parent_job = render_job
                    break 
            
            # new attachment name
            assert(parent_job != None)
            new_attachment_name = parent_job_name + "-" + attachment_name
            self.attachments[new_attachment_name] = None

            # attachment index in the info
            attachment_type = None
            for attachment_index in range(len(self.attachment_info)):
                attachment = self.attachment_info[attachment_index]
                if attachment['Name'] == attachment_name:
                    attachment_type = attachment['Type']
                    break
            assert(attachment_type != None)

            # create view for texture and set directly for buffer
            if attachment_type == 'TextureInput':
                if parent_attachment_name == 'Depth Output':
                    self.attachment_views[new_attachment_name] = parent_job.depth_texture_view
                    self.attachment_formats[new_attachment_name] = 'depth24plus'
                else:
                    self.attachment_views[new_attachment_name] = parent_job.attachment_views[parent_attachment_name]
                    self.attachment_formats[new_attachment_name] = parent_job.attachment_formats[parent_attachment_name]
            elif attachment_type == 'BufferInput':
                self.attachments[new_attachment_name] = parent_job.attachments[attachment_name]
                self.attachment_info[attachment_index]['Size'] = self.attachments[new_attachment_name].size
                self.attachment_views[new_attachment_name] = None

    ##
    def process_depth_attachments(
        self,
        total_render_jobs):

        # set the depth texture to paren't depth texture
        if self.depth_attachment_parent_info != None:
            parent_job_name = self.depth_attachment_parent_info[0]

            parent_job = None
            for render_job in total_render_jobs:
                if render_job.name == parent_job_name:
                    parent_job = render_job
                    break 

            assert(parent_job != None)
            self.depth_texture = parent_job.depth_texture
            self.depth_texture_view = self.depth_texture.create_view()

    ##
    def create_texture_bindings(
        self,
        device,
        texture_views,
        texture_paths,
        max_bindings_per_group):

        bind_groups_entries = [[]]
        bind_groups_layout_entries = [[]]

        texture_array = []
        texture_view_array = []

        max_bindings_per_group = 100

        texture_index = 0
        binding_index = 0
        curr_bind_group_entry = bind_groups_entries[0]
        curr_bind_group_layout_entry = bind_groups_layout_entries[0]
        curr_group_index = 0
        for texture_view in texture_views:
            
            # binding group
            bind_group_info = {
                "binding": binding_index % max_bindings_per_group,
                "resource": texture_view
            }
            curr_bind_group_entry.append(bind_group_info)

            # binding layout
            shader_stage = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
            bind_group_layout_info = {
                "binding": binding_index % max_bindings_per_group,
                "visibility": shader_stage,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                }
            }
            curr_bind_group_layout_entry.append(bind_group_layout_info)

            texture_index += 1
            binding_index += 1

            curr_group_index = int(binding_index / max_bindings_per_group)
            if curr_group_index >= len(bind_groups_entries):
                bind_groups_entries.append([])
                bind_groups_layout_entries.append([])
                curr_bind_group_entry = bind_groups_entries[curr_group_index]
                curr_bind_group_layout_entry = bind_groups_layout_entries[curr_group_index]

        return bind_groups_entries, bind_groups_layout_entries, texture_array, texture_view_array

    ##
    def add_textures_to_pass(
        self,
        device,
        albedo_texture_array_views,
        normal_texture_array_views,
        albedo_texture_paths,
        normal_texture_paths,
        bind_groups,
        bind_group_layouts):
        
        '''
        bind_groups_entries = [[]]
        bind_groups_layout_entries = [[]]

        max_bindings_per_group = 100

        start_binding_index = 0
        texture_index = 0
        binding_index = 0
        curr_bind_group_entry = bind_groups_entries[0]
        curr_bind_group_layout_entry = bind_groups_layout_entries[0]
        curr_group_index = 0
        for texture_path in albedo_texture_paths:
            image = Image.open(texture_path, mode = 'r')
            
            # scale down the image
            if image.width >= 256 or image.width >= 256:
                scale_width = 256.0 / float(image.width)
                scale_height = 256.0 / float(image.height)
                scaled_image = image.resize((int(image.width * scale_width), int(image.height * scale_height)))
                flipped_image = scaled_image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
                image = flipped_image
                
            # create texture
            image_byte_array = image.tobytes()
            texture_width = image.width
            texture_height = image.height
            texture_size = texture_width, texture_height, 1
            texture_format = wgpu.TextureFormat.rgba8unorm
            texture = device.create_texture(
                size=texture_size,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
                dimension=wgpu.TextureDimension.d2,
                format=texture_format,
                mip_level_count=1,
                sample_count=1,
                label = texture_path
            )
            self.textures.append(texture)

            # upload texture data
            device.queue.write_texture(
                {
                    'texture': self.textures[len(self.textures) - 1],
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

            self.texture_views.append(self.textures[len(self.textures) - 1].create_view())

            # binding group
            bind_group_info = {
                "binding": binding_index % max_bindings_per_group,
                "resource": self.texture_views[texture_index]
            }
            curr_bind_group_entry.append(bind_group_info)

            # binding layout
            shader_stage = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
            bind_group_layout_info = {
                "binding": binding_index % max_bindings_per_group,
                "visibility": shader_stage,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                }
            }
            curr_bind_group_layout_entry.append(bind_group_layout_info)

            texture_index += 1
            binding_index += 1

            curr_group_index = int(binding_index / max_bindings_per_group)
            if curr_group_index >= len(bind_groups_entries):
                bind_groups_entries.append([])
                bind_groups_layout_entries.append([])
                curr_bind_group_entry = bind_groups_entries[curr_group_index]
                curr_bind_group_layout_entry = bind_groups_layout_entries[curr_group_index]
        '''

        # albedo texture bindings
        max_bindings_per_group = 100
        albedo_bind_groups_entries, albedo_bind_groups_layout_entries, albedo_texture_array, albedo_texture_view_array = self.create_texture_bindings(
            device = device,
            texture_views = albedo_texture_array_views,
            texture_paths = albedo_texture_paths,
            max_bindings_per_group = max_bindings_per_group)
        
        # normal texture bindings
        normal_bind_groups_entries, normal_bind_groups_layout_entries, normal_texture_array, normal_texture_view_array = self.create_texture_bindings(
            device = device,
            texture_views = normal_texture_array_views,
            texture_paths = normal_texture_paths,
            max_bindings_per_group = max_bindings_per_group)

        for bind_group, bind_group_layout in zip(albedo_bind_groups_entries, albedo_bind_groups_layout_entries):
            bind_groups.append(bind_group)
            bind_group_layouts.append(bind_group_layout)

            print('\tadd textures to group {}, binding start: {} to binding: {}'.format(
                len(bind_groups) - 1,
                bind_group[0]['binding'],
                bind_group[len(bind_group) - 1]['binding']
            ))

        for bind_group, bind_group_layout in zip(normal_bind_groups_entries, normal_bind_groups_layout_entries):
            bind_groups.append(bind_group)
            bind_group_layouts.append(bind_group_layout)

            print('\tadd textures to group {}, binding start: {} to binding: {}'.format(
                len(bind_groups) - 1,
                bind_group[0]['binding'],
                bind_group[len(bind_group) - 1]['binding']
            ))

        

        