import sys, os
import cv2
import numpy as np
import scipy.io
import glob
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt
# from find_rigid_alignment import *
from my_utils import *
from flow_utils import FlowUtils
from mask import visualize_segmentation
from nms import nms

# import libraries from XMem
sys.path.append('./XMem')
import torch
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore


class Graph:
 
    # Constructor
    def __init__(self, data, flow, flow_backward):
 
        # default dictionary to store graph
        self.data = data
        self.flow = flow
        self.flow_backward = flow_backward
        self.graph = defaultdict(list)
        kernel = np.ones((3, 3), np.uint8)

        # scan objects
        objects = []
        num = len(data)
        for i in range(num):
            label = data[i]['label']
            score = data[i]['score']
            push_id = data[i]['push_id']
            mask_ids = np.unique(label)
            if mask_ids[0] == 0:
                mask_ids = mask_ids[1:]
    
            for mask_id in mask_ids:
                mask = np.array(label == mask_id).astype(np.uint8)
                mask = cv2.erode(mask, kernel)
                if np.sum(mask) < 400:
                    continue            
	    
                s = np.mean(score[mask == 1]) / 100.0
                if mask_id == push_id:
                    pushed = 1
                else:
                    pushed = 0
                    
                objects.append((i, mask_id, s, mask, pushed))   # image index, mask id, score, mask, pushed
        self.objects = objects
            
        # build graph in reverse order as the image sequence
        count = len(objects)
        print('%d objects in total' % count)    
        for i in range(count):
            self.graph[i] = []
            o1 = objects[i]
            for j in range(count):
                if j == i:
                    continue
                o2 = objects[j]
        
                # use optical flow to compue the edge cost
                index1 = o1[0]
                mask1 = o1[3]
                index2 = o2[0]
                mask2 = o2[3]
                if index1 == index2 + 1:    # only consider adjacent frames
                    # forward flow
                    flow_img = flow[index2]
                    score_forward = self.compute_score(flow_img, mask2, mask1)
                    # backward flow
                    flow_img = flow_backward[index2]
                    score_backward = self.compute_score(flow_img, mask1, mask2)
                    score = (score_forward + score_backward) / 2
                    if score > 0.4:
                        self.graph[i].append((j, score))
        print(self.graph)
        
        
    def compute_score(self, flow_img, mask1, mask2):
    
        index = np.where(mask1 > 0)
        num = len(index[0])
        pixels = np.zeros((num, 2), dtype=np.float32)
        pixels[:, 0] = index[1]    # x
        pixels[:, 1] = index[0]    # y
        
        flows = flow_img[mask1 > 0, :]
        # add flows to pixels
        pixels_new = pixels + flows
        # obtain the mask on the other image
        xys = pixels_new.astype(np.int32)
        xys = np.unique(xys, axis=0)        
                    
        # bound
        height = mask2.shape[0]
        width = mask2.shape[1]
        index = (xys[:, 1] >= 0) & (xys[:, 1] < height) & (xys[:, 0] >= 0) & (xys[:, 0] < width)
        xys = xys[index, :]
        mask_other = mask2[xys[:, 1], xys[:, 0]]
                    
        # compute matching score
        inter = np.sum(mask_other)
        score = inter / (np.sum(mask1) + np.sum(mask2) - inter)
        return score
    
    
    # A function used by DFS
    def DFSUtil(self, v, visited, count):
 
        # Mark the current node as visited and print it
        visited[v]= count
 
        # Recur for all the vertices adjacent to this vertex
        index = []
        score = []
        for e in self.graph[v]:
            if visited[e[0]] == 0:
                index.append(e[0])
                score.append(e[1])

        if len(score) > 0:            
            selected = np.argmax(score)
            self.DFSUtil(index[selected], visited, count)
 
 
    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, visited, count):

        # find the first node not visitied
        index = np.where(visited == 0)[0]
        
        if len(index) > 0:
        
            image_ids = []
            for i in range(len(index)):
                ind = index[i]
                im_id, mask_id, s, mask, pushed = self.objects[ind]
                image_ids.append(im_id)
                
            # start with the maximum image id
            im_id = np.max(image_ids)
            index = index[image_ids == im_id]
        
            # sort according to scores
            score = np.zeros((len(index), ), dtype=np.float32)
            for i in range(len(index)):
                ind = index[i]
                edges = self.graph[ind]
                maxs = 0
                for e in edges:
                    s = e[1]
                    if s > maxs:
                        maxs = s
                score[i] = maxs

            # visit the max score first
            i = np.argmax(score)
            self.DFSUtil(index[i], visited, count)

        return visited
                

    # a greedy algorithm to find object trajectories
    def greedy_search(self):

        trajectories = []
        num = len(self.graph)  # total vertices
        # Mark all the vertices as not visited
        visited = np.zeros((num, ), dtype=np.int32)
        count = 0
        while not (visited > 0).all():
            count += 1
            visited = self.DFS(visited, count)
            
        ids = np.unique(visited)
        for i in ids:
            tra = np.where(visited == i)[0]
            if len(tra) > 1:
                trajectories.append(tra)
        return trajectories
        
        
    def visualize_trajectories(self, trajectories, scores, selected_objects):

        for i in range(len(trajectories)):
            fig = plt.figure()
            tra = trajectories[i]
            print(tra, scores[i])
            n = len(tra)
            count = 1
            for j in range(n):
                index, mask_id, s, mask, pushed = self.objects[tra[j]]
                ax = fig.add_subplot(int(n**0.5)+1, int(n**0.5)+1, count)
                im = self.data[index]['image'][:, :, (2, 1, 0)]
                im_label = visualize_segmentation(im, mask, return_rgb=True)
                plt.imshow(im_label)
                if j == selected_objects[i]:
                    plt.title('im %d, %.2f, pushed %d (selected)' % (index, scores[i][j], pushed))
                else:
                    plt.title('im %d, %.2f, pushed %d' % (index, scores[i][j], pushed))
                plt.axis('off')
                count += 1
            plt.show()


    # compute scores for objects in each trajectory
    def score_objects(self, trajectories):
    
        scores = []
        for i in range(len(trajectories)):
            tra = trajectories[i]
            print('trajectory', i, tra)
            n = len(tra)
            score = np.zeros((n, ), dtype=np.float32)
            for i in range(n):
                ind = tra[i]
                im_id1, mask_id1, s1, mask1, pushed1 = self.objects[ind]
                
                # compute forward score
                if i < n - 1:
                    j = i + 1
                    ind = tra[j]
                    im_id2, mask_id2, s2, mask2, pushed2 = self.objects[ind]
                    # forward flow
                    flow_forward = self.flow[im_id1]                    
                    score_forward = self.compute_score(flow_forward, mask1, mask2)
                else:
                    score_forward = 0.5
                    
                # compute backward score
                if i > 0:
                    j = i - 1
                    ind = tra[j]
                    im_id2, mask_id2, s2, mask2, pushed2 = self.objects[ind]
                    # forward flow
                    flow_backward = self.flow_backward[im_id2]                    
                    score_backward = self.compute_score(flow_backward, mask1, mask2)             
                else:
                    score_backward = 0.5

                # score[i] = (score_forward + score_backward) / 2
                score[i] = min(score_forward, score_backward)
            print('score', score)
            scores.append(score)
        return scores
        
        
def process_xmem_mask(im_id, mask, prev_mask, flow_img, connectivity=4):
    # find the connected components in mask
    """ Run connected components algorithm and return mask of the matching one
        @param mask: a [H x W] numpy array 
        @return: a [H x W] numpy array of same type as input
    """

    # Run connected components algorithm
    num_components, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    
    if num_components <= 2 or np.sum(prev_mask) == 0:
        return mask

    # propagate prev mask
    index = np.where(prev_mask > 0)
    num = len(index[0])
    prev_size = num
    pixels = np.zeros((num, 2), dtype=np.float32)
    pixels[:, 0] = index[1]    # x
    pixels[:, 1] = index[0]    # y
        
    flows = flow_img[prev_mask > 0, :]
    # add flows to pixels
    pixels_new = pixels + flows
    # obtain the mask on the other image
    xys = pixels_new.astype(np.int32)
    xys = np.unique(xys, axis=0)
                    
    # bound
    height = mask.shape[0]
    width = mask.shape[1]
    index = (xys[:, 1] >= 0) & (xys[:, 1] < height) & (xys[:, 0] >= 0) & (xys[:, 0] < width)
    xys = xys[index, :]
    mask_other = np.zeros((height, width), dtype=np.uint8)
    mask_other[xys[:, 1], xys[:, 0]] = 1
    
    # select component
    max_score = -1
    selected = -1
    for j in range(1, num_components):
        foreground = np.array(components == j).astype(np.uint8)
        current_size = np.sum(foreground)
        inter = np.sum(np.multiply(mask_other, foreground))
        ovr = inter / (np.sum(mask_other) + np.sum(foreground) - inter)
        score = 0.5 * ovr + 0.5 * min(current_size / prev_size, prev_size / current_size)
        if score >= max_score:
            max_score = score
            selected = j
    print('select connected component %d' % selected)
    return (components == selected).astype(mask.dtype)    
        

# run the XMem network for each trajectory
def run_XMem(network, config, data, flow, flow_backward, objects, trajectory, score, threshold=0.7):

    # run XMem for each images
    n = len(data)
    height = data[0]['image'].shape[0]
    width = data[0]['image'].shape[1]
    xmem_masks = [np.zeros((height, width), dtype=np.uint8) for i in range(n)]

    # no need to count usage for LT if the video is not that long anyway
    config['enable_long_term_count_usage'] = (
        config['enable_long_term'] and
        (n / (config['max_mid_term_frames']-config['min_mid_term_frames'])
            * config['num_prototypes'])
        >= config['max_long_term_elements']
    )
    
    # push information
    pushed = []
    for i in trajectory:
        im_id, mask_id, s, msk, p = objects[i]
        pushed.append(p)
    
    # find the maximum score object that is pushed
    tmp = np.array(score).copy()
    index = np.where((np.array(pushed) == 1) & (tmp > threshold))[0]    
    tmp[index] += 1
    smax = np.max(tmp)
    index = np.where(np.array(tmp) == smax)[0]
    if len(index) > 1:
        index = index[-1]
    index = int(index)
    
    im_id, mask_id, s, msk, pushed = objects[trajectory[index]]
    mapper = MaskMapper()
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False      
    
    for k in range(2):
        if k == 0:
            interval = np.arange(im_id, -1, -1)
        else:
            interval = np.arange(im_id, n)
        if len(interval) == 1:
            continue
            
        prev_mask = None
        for i in interval:
            rgb = data[i]['image_tensor'].cuda()
            flow_img = None
            if k == 0:
                if i >= 0 and i < len(flow_backward):
                    flow_img = flow_backward[i]
            else:
                if i - 1 >= 0:
                    flow_img = flow[i - 1]
        
            if not first_mask_loaded:
                # im_id, mask_id, s, msk, pushed = objects[trajectory[index]]                
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue
            else:
                msk = None
                
            # Map possibly non-continuous labels to continuous ones
            if msk is not None:
                msk, labels = mapper.convert_mask(msk)
                msk = torch.Tensor(msk).cuda()
                processor.set_all_labels(list(mapper.remappings.values()))
            else:
                labels = None

            # Run the model on this frame
            prob = processor.step(rgb, msk, labels, end=(i==interval[-1]))
        
            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
            
            # find the matching connected component
            if prev_mask is not None:
                new_mask = process_xmem_mask(i, out_mask, prev_mask, flow_img)
            else:
                new_mask = out_mask
            
            '''
            fig = plt.figure()
            ax = fig.add_subplot(2, 3, 1)
            im = data[i]['image']
            im_label = visualize_segmentation(im, new_mask, return_rgb=True)
            plt.imshow(im_label[:, :, (2, 1, 0)])
            plt.title('image %d' % i)

            ax = fig.add_subplot(2, 3, 2)
            if prev_mask is not None:
                plt.imshow(prev_mask)
            plt.title('previous mask')    
            ax = fig.add_subplot(2, 3, 3)
            plt.imshow(out_mask)
            plt.title('before')
            ax = fig.add_subplot(2, 3, 4)
            plt.imshow(new_mask)
            plt.title('after')
            if flow_img is not None:
                ax = fig.add_subplot(2, 3, 5)
                plt.imshow(flow_img[:, :, 0])
                plt.title('flow x')
                ax = fig.add_subplot(2, 3, 6)
                plt.imshow(flow_img[:, :, 1])
                plt.title('flow y')                            
            plt.show()
            #'''
            
            # save the new mask
            prev_mask = new_mask.copy()
            xmem_masks[i] = new_mask            
            
    return xmem_masks, index
            
            
def combine_masks(mask, score):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
    """
    
    # non-maximum suppression
    # keep = nms(mask, score, thresh=0.7).astype(int)
    # mask = mask[keep]
    # score = score[keep]
    
    num, h, w = mask.shape
    bin_mask = np.zeros((h, w))
    num_instance = len(mask)
    # if there is not any instance, just return a mask full of 0s.
    if num_instance == 0:
        return bin_mask

    for m, object_label in zip(mask, range(2, 2+num_instance)):
        label_pos = np.nonzero(m)
        bin_mask[label_pos] = object_label

    # filtering         
    kernel = np.ones((3, 3), np.uint8)
    for l in np.unique(bin_mask):
        mask = np.array(bin_mask == l).astype(np.uint8)
        mask = cv2.erode(mask, kernel)
        if np.sum(bin_mask == l) < 200 or np.sum(mask) < 200:
            bin_mask[bin_mask == l] = 0
            
    # remapping
    labels = np.unique(bin_mask)
    mask_new = np.zeros((h, w))
    for i in range(len(labels)):
        mask_new[bin_mask == labels[i]] = i
        
    return mask_new
            

# save the final masks
def save_results(data, scores, xmem_masks, visualize=False):
    num_images = len(data)
    num_trajectories = len(xmem_masks)
    
    # sort trajectroies according to scores
    score = np.array([np.sum(s) for s in scores])
    index = np.argsort(score).astype(np.int32)
    print('sorting trajectories as')
    print(score)
    print(index)
    
    print('save data to:')
    for i in range(num_images):
        filename = data[i]['im_file']
        masks = []
        for j in index:
            if len(xmem_masks[j][i]) > 0:
                masks.append(xmem_masks[j][i])
        masks = np.stack(masks)
        label = combine_masks(masks, score[index])
        
        # save the label image
        im = data[i]['image']
        im_label = visualize_segmentation(im, label, return_rgb=True)
        
        gtname = filename.replace('color', 'gt-final')
        cv2.imwrite(gtname, im_label)
        
        labelname = filename.replace('color', 'label-final')
        labelname = labelname.replace('jpg', 'png')
        cv2.imwrite(labelname, label.astype(np.uint8))
        print('save to file', labelname)
        print('labels', np.unique(label))
        
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            plt.imshow(im_label[:, :, (2, 1, 0)])
            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(label)
            plt.show()
        
def parse_args():
    """
    Parse input arguments
    """
                 
    parser = ArgumentParser()
    parser.add_argument('--model', default='./XMem/saves/XMem-s012.pth')
    parser.add_argument('--dirname', default='all')

    # Data options
    parser.add_argument('--d16_path', default='../DAVIS/2016')
    parser.add_argument('--d17_path', default='../DAVIS/2017')
    parser.add_argument('--y18_path', default='../YouTube2018')
    parser.add_argument('--y19_path', default='../YouTube')
    parser.add_argument('--lv_path', default='../long_video_set')
    # For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
    parser.add_argument('--generic_path')

    parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G', default='D17')
    parser.add_argument('--split', help='val/test', default='val')
    parser.add_argument('--output', default=None)
    parser.add_argument('--save_all', action='store_true', 
            help='Save all frames. Useful only in YouTubeVOS/long-time video', )

    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
        
    # Long-term memory options
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

    # Multi-scale options
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=480, type=int, 
            help='Resize the shorter side to this size. -1 to use original resolution. ')  

    args = parser.parse_args()
    return args


def run_directory(dirname, network, flow_helper, visualize=False):
    
    # list all the images
    filenames = sorted(list(Path(dirname).glob('color-*.jpg')))
    num = len(filenames)
    print(filenames)
    
    # read all the data
    data = []
    for i in range(num):
        data.append(read_data(dirname, i))

    # compute optical flow
    flow = []
    flow_backward = []
    for i in range(num - 1):
        flow_img = flow_helper.calculate_flow(data[i]['image'], data[i+1]['image'])
        flow.append(flow_img.copy())
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(2, 3, 1)
        plt.imshow(data[i]['image'][:, :, (2, 1, 0)])
        plt.title('image 1')
    
        ax = fig.add_subplot(2, 3, 4)
        plt.imshow(data[i+1]['image'][:, :, (2, 1, 0)])
        plt.title('image 2')    
        
        ax = fig.add_subplot(2, 3, 2)
        plt.imshow(data[i]['label'])
        plt.title('segmentation 1')    
    
        ax = fig.add_subplot(2, 3, 5)
        plt.imshow(data[i+1]['label'])    
        plt.title('segmentation 2')        
    
        ax = fig.add_subplot(2, 3, 3)
        plt.imshow(flow_img[:, :, 0])
        plt.title('flow x')        
    
        ax = fig.add_subplot(2, 3, 6)
        plt.imshow(flow_img[:, :, 1])
        plt.title('flow y')            
            
        plt.show()
        #'''

        flow_img = flow_helper.calculate_flow(data[i+1]['image'], data[i]['image'])
        flow_backward.append(flow_img.copy())        
            
    # build a graph for min-cost flow
    graph = Graph(data, flow, flow_backward)
    
    # search trajectories
    trajectories = graph.greedy_search()
    scores = graph.score_objects(trajectories)
    
    # run XMem network
    xmem_masks_all = []
    selected_objects = []
    for i in range(len(trajectories)):
        print('run XMem on trajectory %d' % i)
        xmem_masks, index = run_XMem(network, config, data, flow, flow_backward, graph.objects, trajectories[i], scores[i])
        xmem_masks_all.append(xmem_masks)
        selected_objects.append(index)
        
    # graph.visualize_trajectories(trajectories, scores, selected_objects)
    
    # save results
    save_results(data, scores, xmem_masks_all, visualize)
    
    
if __name__ == "__main__":

    args = parse_args()
    
    if args.dirname == 'all':
        # list all the directories
        files = os.listdir('./data')
        dirnames = []
        for f in files:
            filename = os.path.join('./data', f)
            if os.path.isdir(filename):
                dirnames.append(filename)
        dirnames = sorted(dirnames)
    else:
        dirnames = ['./data/' + args.dirname]
    print(dirnames)
    
    # Set up XMem
    torch.autograd.set_grad_enabled(False)
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']
    
    network = XMem(config, args.model).cuda().eval()
    if args.model is not None:
        model_weights = torch.load(args.model)
        network.load_weights(model_weights, init_as_zero_if_needed=True)
        print('loaded model from %s' % args.model)
    else:
        print('No model loaded.') 
        
    # set up flow
    mmflow_model_config_path = "raft_8x2_100k_flyingthings3d_sintel_368x768.py"
    mmflow_model_checkpoint_path = "raft_8x2_100k_flyingthings3d_sintel_368x768.pth"
    flow_helper = FlowUtils(mmflow_model_config_path, mmflow_model_checkpoint_path)
    
    for dirname in dirnames:
        print('run on %s' % dirname)
        run_directory(dirname, network, flow_helper, visualize=True)
