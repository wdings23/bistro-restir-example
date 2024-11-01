import os



##
def main():

    total_content = ''

    for root, dirs, files in os.walk('D:\\test\\python-webgpu\\shaders'):
        for file in files:
            full_path = os.path.join(root, file)
            file = open(full_path, 'r')
            file_content = file.read()
            file.close()

            start = file_content.find('struct UniformData')
            if start >= 0:
                start = file_content.find('{', start) + 1
                end = file_content.find('};', start + 1)
                struct_content = file_content[start:end]

                total_content += struct_content + '\n'
                #print('{}'.format(struct_content))

    output_file = open('d:\\test\\python-webgpu\\uniform-data.txt', 'w')
    output_file.write(total_content)
    output_file.close()

    print('{}'.format(total_content))
    print('')


##
if __name__ == '__main__':
    main()