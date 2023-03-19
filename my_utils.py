import sys, os
import cv2
import numpy as np
import scipy.io
import json
from find_rigid_alignment import *
from torchvision import transforms


im_normalization = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
                
im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img
    

def read_data(dirname, index):
    # color image
    im_file = os.path.join(dirname, 'color-%06d.jpg' % index)
    im = cv2.imread(im_file)
    print(im_file)
    
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_tensor = im_transform(im_rgb)
    
    # label image
    filename = os.path.join(dirname, 'label-%06d.png' % index)
    label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)    

    # score image   
    filename = os.path.join(dirname, 'score-%06d.png' % index)
    score = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)    

    # depth image    
    filename = os.path.join(dirname, 'depth-%06d.png' % index)
    depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    
    # meta data
    filename = os.path.join(dirname, 'meta-%06d.mat' % index)
    data = scipy.io.loadmat(filename)
    factor_depth = data['factor_depth']
    depth /= factor_depth
    
    # contruct data
    data['image'] = im
    data['image_tensor'] = im_tensor
    data['im_file'] = im_file
    data['depth'] = depth
    data['label'] = label
    data['score'] = score
    
    # compute xyz image    
    height = depth.shape[0]
    width = depth.shape[1]
    intrinsics = data['intrinsic_matrix']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    px = intrinsics[0, 2]
    py = intrinsics[1, 2]
    data['xyz_img'] = compute_xyz(depth, fx, fy, px, py, height, width)
    
    # pushing data
    filename = os.path.join(dirname, 'push-%02d-%02d.json' % (index, index+1))
    data['push_id'] = -1
    if os.path.exists(filename):
        f = open(filename, 'r')
        s = f.read()
        f.close()
        if len(s) > 0:
            push_data = json.loads(s)
            if 'push_id' in push_data:
                data['push_id'] = push_data['push_id']
    
    return data
    
    
# ransac to estimate SE(3) before two point clouds
def ransac_se3(p1, p2):

    iterations = 1000
    threshold = 0.01
    count = np.zeros((iterations, ), dtype=np.float32)
    poses = []
    num = p1.shape[0]
    K = 3
    for i in range(iterations):

        # randomly select K points    
        index = np.random.permutation(num)[:K]
        A = p1[index, :]
        B = p2[index, :]

        # compute pose  (target -> source)
        R, t = find_rigid_alignment(B.copy(), A.copy())
        poses.append((R, t))  
                
        #  compute agreement
        pnew = R @ p1.T + t.reshape(3, 1)
        pnew = pnew.T
        
        distances = np.linalg.norm(p2 - pnew, axis=1)
        count[i] = np.sum(distances < threshold)

    best = np.argmax(count)
    R, t = poses[best]
    return R, t
