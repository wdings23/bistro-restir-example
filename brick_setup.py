from vec import *

import struct

##
def brick_setup(
    device,
    bbox_device_buffer,
    output_device_buffer,
    position_scale, 
    brick_dimension):

    bounding_box_buffer = device.queue.read_buffer(bbox_device_buffer)

    max_bounding_box = float3(
        float(struct.unpack('i', bounding_box_buffer[0:4])[0]) * 0.001,
        float(struct.unpack('i', bounding_box_buffer[4:8])[0]) * 0.001,
        float(struct.unpack('i', bounding_box_buffer[8:12])[0]) * 0.001
    )

    min_bounding_box = float3(
        float(struct.unpack('i', bounding_box_buffer[12:16])[0]) * 0.001,
        float(struct.unpack('i', bounding_box_buffer[16:20])[0]) * 0.001,
        float(struct.unpack('i', bounding_box_buffer[24:28])[0]) * 0.001
    )

    max_bbox = float3(
        math.ceil(max_bounding_box.x),
        math.ceil(max_bounding_box.y),
        math.ceil(max_bounding_box.z)
    ) * position_scale

    min_bbox = float3(
        math.floor(min_bounding_box.x),
        math.floor(min_bounding_box.y),
        math.floor(min_bounding_box.z)
    ) * position_scale

    dimension = max_bbox - min_bbox

    byte_buffer = b''

    for z in range(int(dimension.z)):
        for y in range(int(dimension.y)):
            for x in range(int(dimension.x)):
                brick_index = z * int(dimension.x * dimension.y) + y * int(dimension.x) + x
                brick_position = min_bbox + float3(
                    x * brick_dimension,
                    y * brick_dimension,
                    z * brick_dimension)
                byte_buffer += struct.pack('f', brick_position.x)
                byte_buffer += struct.pack('f', brick_position.y)
                byte_buffer += struct.pack('f', brick_position.z)
                byte_buffer += struct.pack('I', brick_index)

    device.queue.write_buffer(
        buffer = output_device_buffer,
        buffer_offset = 0,
        data = byte_buffer)