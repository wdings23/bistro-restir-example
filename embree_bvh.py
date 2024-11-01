import os

os.add_dll_directory('D:\\projects\\embree3\\bin')

import embree


class Embree(object):

    ##
    def __init__(self):
        self.device = embree.Device()

    ##
    def init_scene(self):
        self.scene = self.device.make_scene()
        self.geometry = self.device.make_geometry(embree.GeometryType.Triangle)
        print('')

##
def embree_test():
    embree_obj = Embree()
    embree_obj.init_scene()


if __name__ == '__main__':
    embree_test()