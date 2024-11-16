import threading
import time
import struct 
from PIL import Image

##
def load_mip_texture_pages(
    app,
    mip_texture_page_info,
    curr_mip_load_texture_page,
    albedo_texture_paths,
    albedo_texture_dimensions,
    normal_texture_paths,
    normal_texture_dimensions,
    num_texture_pages_per_load,
    mip_texture_page_hash_bytes
):
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
    
    # page info
    # 0: page x, y encoded
    # 1: texture id
    # 2: hash index
    # 3: mip level 
    
    texture_atlas_dimension = 8192
    texture_page_size = 64
    hash_entry_size = 16

    num_pages_per_dimension = int(texture_atlas_dimension / texture_page_size)
    max_num_pages_per_texture = int(num_pages_per_dimension * num_pages_per_dimension)

    keys = list(mip_texture_page_info.keys())
    num_keys = len(keys)
    
    mip_image = None
    num_page_loaded = 0
    while True:
        # load texture

        if mip_image != None:
            mip_image.close()
            mip_image = None

        # past num pages in current texture
        if curr_mip_load_texture_page[0] >= num_keys:
            break
        
        # current image
        key = keys[curr_mip_load_texture_page[0]]
        page_info = mip_texture_page_info[key]
        texture_id = page_info[0][3]
        texture_path = ''
        if texture_id >= 65536:
            texture_path = normal_texture_paths[texture_id - 65536]
        else:
            texture_path = albedo_texture_paths[texture_id]
        mip_level = page_info[0][4]
        mip_denom = pow(2, mip_level)
        num_pages = len(mip_texture_page_info[key])    

        # texture dimensions for mip level
        texture_index = page_info[0][3]
        texture_dimension = [0, 0]
        if texture_id >= 65536:
            texture_dimension = normal_texture_dimensions[texture_index - 65536]
        else:
            texture_dimension = albedo_texture_dimensions[texture_index]
        mip_texture_dimension = [0, 0]
        mip_texture_dimension[0] = int(texture_dimension[0] / mip_denom)
        mip_texture_dimension[1] = int(texture_dimension[1] / mip_denom)
        mip_image_page_size = int(min(texture_dimension[0] / mip_denom, texture_page_size))
        num_pages = len(mip_texture_page_info[key])
        
        image_loaded = False
        mip_image_bytes = None

        num_curr_texture_page_loaded = 0
        while True:
            # load pages from texture

            # past number of pages to load per frame
            if num_page_loaded >= num_texture_pages_per_load:
                break
            
            # move to next texture to load pages
            if curr_mip_load_texture_page[1] >= num_pages:
                curr_mip_load_texture_page[0] += 1
                curr_mip_load_texture_page[1] = 0
                if mip_image != None:
                    mip_image.close()
                    mip_image = None

                break
            
            # current page to load info
            page_info = mip_texture_page_info[key][curr_mip_load_texture_page[1]]
            hash_index = page_info[2]

            # check if page has already been loaded
            register_page_to_load = True
            hash_byte_pos = page_info[2] * 16
            page_index = struct.unpack('I', mip_texture_page_hash_bytes[hash_byte_pos+4:hash_byte_pos+8])[0]
            if page_index != 0xffffffff and page_index < curr_mip_load_texture_page[2]:
                register_page_to_load = False

                if hash_index == 51316:
                    print('!!! hash 51316 already loaded at page {} texture id {}'.format(page_index, texture_id))

            # load image
            if register_page_to_load == True:
                if image_loaded == False:
                    image = Image.open(texture_path, mode = 'r')
                    flipped_image = image.transpose(method = Image.Transpose.FLIP_TOP_BOTTOM)
                    mip_image_size = max(int(image.width / mip_denom), texture_page_size), max(int(image.height / mip_denom), texture_page_size)
                    mip_image = flipped_image.resize(mip_image_size)
                    if mip_image.mode == 'RGB':
                        mip_image = mip_image.convert(mode = 'RGBA')
                    mip_image_bytes = mip_image.tobytes()
                    image_loaded = True

                # load page image data
                mip_page_image_bytes = load_texture_page_image_data(
                    texture_page_size = texture_page_size,
                    image_page_size = mip_image_page_size,
                    image_width = mip_texture_dimension[0],
                    page_info = page_info,
                    image_bytes = mip_image_bytes)
                
                if len(mip_page_image_bytes) <= 0:
                    length = len(mip_image_bytes)
                    mip_page_image_bytes = load_texture_page_image_data(
                        texture_page_size = texture_page_size,
                        image_page_size = mip_image_page_size,
                        image_width = mip_texture_dimension[0],
                        page_info = page_info,
                        image_bytes = mip_image_bytes)

                # determine the atlas texture to copy into based on mip level and coordinate in the atlas texture
                attachment_key = 'Texture Atlas ' + str(int(page_info[4]))
                curr_mip_page = curr_mip_load_texture_page[mip_level+4]
                mip_texture_atlas_x = int((curr_mip_page % num_pages_per_dimension) * texture_page_size)
                mip_texture_atlas_y = int(int(curr_mip_page / num_pages_per_dimension) * texture_page_size)
                
                # swap out old pages with the new one
                swap_page_index = 0
                if curr_mip_load_texture_page[mip_level+4] >= max_num_pages_per_texture:
                   
                    last_used_page_index = -1
                    earliest_frame_accessed = app.frame_index
                    num_candidate_swap_index = 0
                    for i in range(5000):
                        hash_index = (page_info[2] + i) % 5000
                        buffer_pos = hash_index * 16
                        update_frame_index = struct.unpack('I', mip_texture_page_hash_bytes[buffer_pos+12:buffer_pos+16])[0]
                        if update_frame_index > 0 and update_frame_index < app.frame_index - min(50, app.frame_index):
                            swap_page_index = struct.unpack('I', mip_texture_page_hash_bytes[buffer_pos+4:buffer_pos+8])[0]
                            if swap_page_index >= max_num_pages_per_texture:
                                continue

                            if earliest_frame_accessed > update_frame_index:
                                earliest_frame_accessed = update_frame_index
                                last_used_page_index = swap_page_index
                                num_candidate_swap_index += 1
                            
                            if num_candidate_swap_index >= 5:
                                break
                            
                    if earliest_frame_accessed < app.frame_index and last_used_page_index >= 0:
                        mip_texture_atlas_x = int(int(last_used_page_index % num_pages_per_dimension) * texture_page_size)
                        mip_texture_atlas_y = int(int(last_used_page_index / num_pages_per_dimension) * texture_page_size)

                        if hash_index == 51316 or hash_index == 3960:
                            print('!!! SWAP OUT {} !!!'.format(hash_index))

                #if mip_level == 2:
                #    print('mip 2 page {}'.format(curr_mip_load_texture_page[mip_level+4]))

                encoded = int(mip_texture_atlas_x / texture_page_size) | int(int(mip_texture_atlas_y / texture_page_size) << 16)
                if hash_index == 3960 or hash_index == 51316:
                    print('wtf hash index {} texture id {} ({}, {}) mip page index {} mip {}'.format(
                        hash_index,
                        texture_id, 
                        mip_texture_atlas_x, 
                        mip_texture_atlas_y, 
                        curr_mip_load_texture_page[mip_level+4] + 1,
                        mip_level))

                if len(mip_image_bytes) <= 0:
                    print('wtf')

                if mip_texture_atlas_x < texture_atlas_dimension and mip_texture_atlas_y < texture_atlas_dimension:
                    app.copy_texture_pages.append(
                        {
                            'attachment-key': attachment_key,
                            'x': mip_texture_atlas_x,
                            'y': mip_texture_atlas_y,
                            'image-bytes': mip_page_image_bytes,
                            'hash-index': hash_index,
                            'curr-mip-page-index': curr_mip_load_texture_page[mip_level+4] + 1
                        }
                    )

                # update mip texture page in mip level
                curr_mip_load_texture_page[mip_level+4] += 1

                # update mip texture index for mip level
                if curr_mip_load_texture_page[mip_level+4] >= max_num_pages_per_texture:
                    curr_mip_load_texture_page[mip_level+7] += 1

            num_page_loaded += 1
            curr_mip_load_texture_page[1] += 1
            curr_mip_load_texture_page[2] += 1

        if num_page_loaded >= num_texture_pages_per_load:
            if mip_image != None:
                mip_image.close()
                mip_image = None

            break

##
def load_texture_page_image_data(
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

    return page_image_bytes

##
def get_mip_page_info(
    app,
    mip_texture_page_info,
    mip_page_request_bytes,
    num_requests,
    texture_page_size):

    curr_pos = 0
    for i in range(num_requests):
        page_x_y = struct.unpack('i', mip_page_request_bytes[curr_pos:curr_pos+4])[0]    
        texture_id = struct.unpack('i', mip_page_request_bytes[curr_pos+4:curr_pos+8])[0] 
        hash_id = struct.unpack('i', mip_page_request_bytes[curr_pos+8:curr_pos+12])[0]
        mip_level = struct.unpack('i', mip_page_request_bytes[curr_pos+12:curr_pos+16])[0] & 0xff
        
        normal_texture_id = -1
        if texture_id >= 65536:
            normal_texture_id = texture_id - 65536

        page_x = (0xffff & page_x_y)
        page_y = (page_x_y >> 16)

        mip_denom = pow(2, mip_level)
        texture_dimension = [0, 0] 
        if normal_texture_id >= 0:
            texture_dimension = app.albedo_texture_dimensions[normal_texture_id]
            texture_id = normal_texture_id + 65536
        else:
            texture_dimension = app.albedo_texture_dimensions[texture_id]

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
        if not key in mip_texture_page_info:
            mip_texture_page_info[key] = []
        mip_texture_page_info[key].append((x_coord, y_coord, hash_id, texture_id, mip_level))

        curr_pos += 16   

    ##
    def sort_second(val):
        return val[1]

    ##
    def sort_first(val):
        return val[0]

    ##
    def sort_mip(val):
        return val[4]

    # sort list by y and x coordinate
    for key in mip_texture_page_info:
        mip_texture_page_info[key].sort(key = sort_second)
    for key in mip_texture_page_info:
        mip_texture_page_info[key].sort(key = sort_first)
    for key in mip_texture_page_info:
        mip_texture_page_info[key].sort(key = sort_mip)

##
def thread_load_texture_pages(app):

    curr_mip_load_texture_page = [0] * 16

    prev_num_page_info_requests = 0

    num_texture_pages_per_load = 1024
    texture_page_size = 64

    while app.frame_index < 2:
        time.sleep(0.000001)

    while app.request_count_bytes == None:
        time.sleep(0.000001)

    while True:
        if app.canvas.is_closed():
            break

        while app.texture_page_loaded != False:
            time.sleep(0.000001)

        mip_texture_page_info = {}
        num_requests = struct.unpack('I', app.request_count_bytes[0:4])[0]

        # convert texture page info and load the pages
        get_mip_page_info(
            app = app,
            mip_texture_page_info = mip_texture_page_info,
            mip_page_request_bytes = app.mip_page_request_bytes,
            num_requests = num_requests,
            texture_page_size = texture_page_size
        )
        
        # record the texture page info
        load_mip_texture_pages(
            app = app,
            mip_texture_page_info = mip_texture_page_info,
            curr_mip_load_texture_page = curr_mip_load_texture_page,
            albedo_texture_paths = app.albedo_texture_paths,
            albedo_texture_dimensions = app.albedo_texture_dimensions,
            normal_texture_paths = app.normal_texture_paths,
            normal_texture_dimensions = app.normal_texture_dimensions,
            num_texture_pages_per_load = num_texture_pages_per_load,
            mip_texture_page_hash_bytes = app.mip_texture_page_hash_bytes
        ) 

        app.texture_page_loaded = True

        # restart checking page to load from the start
        if app.frame_index > 20 and curr_mip_load_texture_page[0] >= len(mip_texture_page_info):
            curr_mip_load_texture_page[0] = 0
            curr_mip_load_texture_page[1] = 0

        prev_num_page_info_requests = num_requests