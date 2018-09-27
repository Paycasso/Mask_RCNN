import os

IMAGE_EXTENSIONS = ('.png', '.PNG', '.jpg', '.JPG', '.JPEG', '.jpeg')

def get_all_file_list(root_dirs):
    '''
    Given a directory path, form a list of paths of all images contained within
    '''
    matches = []
    for root_dir in root_dirs:
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(IMAGE_EXTENSIONS):
                    matches.append(os.path.join(root, filename))
    return matches