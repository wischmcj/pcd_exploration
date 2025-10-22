from collections import defaultdict
import numpy as np
import pickle
from glob import glob
import os
from open3d.ml.vis import o3d

from open3d.visualization import draw_geometries as draw
from scipy.spatial import cKDTree
from viz.viz_utils import color_continuous_map


def to_o3d(coords=None, colors=None, labels=None, las=None):
    if las is not None:
        las = np.asarray(las)
        coords = las[:, :3]
        if las.shape[1]>3:
            labels = las[:, 3]
        if las.shape[1]>4:
            colors = las[:, 4:7]       
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    elif labels is not None:
        pcd, _ = color_continuous_map(pcd,labels)
    # labels = labels.astype(np.int32)
    return pcd

def load_kdtrees(label_list):
    print(f'loading kdtrees')
    kdtrees = []
    for label in label_list:
        with open(f'data/kdtree/{label}_kd_tree.pkl', 'rb') as f:
             kdtrees.append((label, pickle.load(f)))
    return kdtrees

def load_adjacency_dict():
    print(f'loading adjacency dict')
    with open('data/adjacency/adj.pkl', 'rb') as f:
        adj = pickle.load(f)
    return adj

def create_kdtrees(coords, labels):
    print(f'creating kdtrees')
    kdtrees = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        print(f'creating kdtree for {label}')
        pt_sample = coords[labels == label][::10]
        kdtree = cKDTree(pt_sample)
        with open(f'data/kdtree/{label}_kd_tree.pkl', 'wb') as f:
            pickle.dump(kdtree, f)
        kdtrees.append((label, kdtree))
    return kdtrees

def determine_cluster_adjacency(src_file:str, decent_trees:list[int], threshold:float=0.35):
    # load or create kdtrees
    try:
        kdtrees = load_kdtrees(unique_labels)
    except:
        _, unique_labels, labels, coords = labeled_clusters_from_pw_results(src_file, return_src_arrays=True)
        # create from scratch
        kdtrees = create_kdtrees(coords, labels)
    
    # load or create adjacency dict
    try:
        adj = load_adjacency_dict()
    except:
        # create from scratch
        if coords is None or labels is None:
            _, unique_labels, labels, coords = labeled_clusters_from_pw_results(src_file, return_src_arrays=True)
        adj = map_clusters(decent_trees, kdtrees, threshold)
    return adj


def map_clusters(label_list, kdtrees=None, threshold=0.35):
    # Given clusters as a list of arrays of points (N x 3), determine adjacency
    # Two clusters are adjacent if any points are within a small threshold distance

    adjacency_dict = {label: {} for label in label_list}

    print(f'mapping clusters')
    for label_i, tree_i in kdtrees:
        if label_i in label_list:
            print(f'{label_i} in label_list, getting neighbors')
            print(f'mapping cluster {label_i}')
            for label_j, tree_j in kdtrees:
                if label_j== label_i or label_j in label_list or label_j == 0:
                    continue
                # Query cluster i against cluster j for close neighbors
                # For efficiency, only check shortest pairwise distance
                pts_and_dists = tree_i.sparse_distance_matrix(tree_j, threshold, output_type='ndarray')

                if pts_and_dists.shape[0] > 0:
                    min_dist = pts_and_dists['v'].min()
                    adjacency_dict[label_i][label_j] = min_dist 
    with open('data/adjacency/adj.pkl', 'wb') as f:
        pickle.dump(adjacency_dict, f)
    return adjacency_dict
    
def cluster_color(pcd,labels):
    import matplotlib.pyplot as plt
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    orig_colors = np.array(pcd.colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, orig_colors

EPS_DEFAULT = 1.2
MIN_POINTS_DEFAULT = 175
def user_cluster(pcd, src_pcd, eps=EPS_DEFAULT, min_points=MIN_POINTS_DEFAULT, filter_non_clusters=True, draw_result=True):
    print('clustering')
    choosing_inputs = True
    while choosing_inputs:
        user_input = input(f"eps? (default={eps})").strip().lower()
        try:
            if float(user_input) >3:
                print(f'eps too large, setting to {EPS_DEFAULT}')
                eps = EPS_DEFAULT
        except:
            print(f'error parsing min_points input, setting to {EPS_DEFAULT}')
            eps = EPS_DEFAULT
        eps = float(user_input or 1)

        user_input = input(f"min_points? (default={min_points})").strip().lower()
        try:
            min_points = int(user_input or MIN_POINTS_DEFAULT)
            eps = EPS_DEFAULT
        except:
            print(f'error parsing min_points input, setting to {MIN_POINTS_DEFAULT}')
            min_points = MIN_POINTS_DEFAULT
        # Cluster and draw
        labels =  np.array( pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
        print(f'{labels.shape=}')
        non_cluster_idxs = np.where(labels < 0)[0]
        if len(labels) == 0:
            print(f'no clusters found, trying again')
            continue
        new_pcd, orig_colors = cluster_color(pcd,labels)
        new_pcd = new_pcd.select_by_index(non_cluster_idxs, invert=True)
        draw([new_pcd])
        draw([new_pcd, src_pcd])
        # Finalize
        user_input = 'draw_again'
        while user_input == 'draw_again':
            user_input = input("Accept? (default=draw_again)").strip().lower()
        if user_input == 'y':
            choosing_inputs = False
    return labels, eps, min_points


def loop_and_ask(src_lbl,src_pts, labeled_clusters, 
                save_file=False, return_chosen_pts=False):
    """
        src_pts (np.array[N,3]): points of the source cluster
        src_lbls (np.array[N]): labels of the source cluster
        labeled_clusters (list[tuple[int, np.array[N]]]): list of tuples of labels and points of the labeled clusters
    """
    src_lbls = [src_lbl]*len(src_pts)
    final_pts = [src_pts]
    final_lbls = [src_lbls]
    pt_lens = [len(src_pts)]
    src_pcd = to_o3d(coords=src_pts,colors= [[0,0,1]]*len(src_pts))
    inputs = []
    for lbl, pts in labeled_clusters:
        if len(pts)>50:
            if return_chosen_pts: 
                print(f'running algo for subcluster {lbl}')
            else:
                print(f'processing cluster {lbl}')
            lbls = [lbl]*len(pts)
            comp_pcd = to_o3d(coords=pts, colors= [[1,0,0]]*len(pts))
            draw([comp_pcd, src_pcd])
            user_input = input("Add points to_cluster? (y/n/r): ").strip().lower()
            in_dict = {'user_input': user_input}
            if user_input == 'y':
                final_pts.append(pts)
                final_lbls.append(lbls)
            if user_input == 'r':
                print(f'recursing on cluster {lbl}')
                # split the cluster into subclusters
                sub_labels, eps, min_points = user_cluster(comp_pcd, src_pcd)
                in_dict['eps'] = eps
                in_dict['min_points'] = min_points
                unique_sub_labels = np.unique(sub_labels)
                labeled_subclusters = [(sub_lbl, pts[sub_labels == sub_lbl]) for sub_lbl in unique_sub_labels]
                # run algo to choose which subclusters to add to the final cluster
                chosen_pts_list, chosen_sub_lbls, inputs = loop_and_ask(src_lbl, src_pts, labeled_subclusters, return_chosen_pts=True)
                in_dict['recurse_inputs'] = inputs
                # generate compound labels, add chosen to final
                for chosen_pts, chosen_sub_lbl in zip(chosen_pts_list, chosen_sub_lbls):
                    final_pts.append(chosen_pts)
                    compound_lbl = f'{lbl}pt{chosen_sub_lbl}'
                    final_lbls.append(compound_lbl)
                    pt_lens.append(len(chosen_pts))
            inputs.append(in_dict)
    if return_chosen_pts:
        return final_pts[1:], final_lbls[1:], inputs
    if save_file:
        try:
            with open(f'data/kdtree/{src_lbl}_inputs.pkl', 'wb') as f: pickle.dump(inputs, f)
        except:
            breakpoint()
            print(f'failed to save inputs')
        try:
            pt_lens = np.array(pt_lens)
            final_pts = np.vstack(final_pts)
            final_lbls = np.array(final_lbls)
            pcd = to_o3d(coords=final_pts)
            draw([pcd])
            # file_name = '_'.join(final_lbls)'
            np.savez_compressed(f'data/cluster_joining/{lbl}_joined.npz', points=final_pts, labels=final_lbls, pt_lens=pt_lens)
        except:
            breakpoint()
            print(f'failed to save inputs')

def labeled_clusters_from_pw_results(results_file:str,
                                    coords_file:str='coords',
                                    labels_file:str='instance_preds',
                                    return_src_arrays:bool=False):
    data = np.load(results_file)
    labels = data[labels_file]
    coords = data[coords_file]
    unique_labels = np.unique(labels)
    label_to_points = {label:coords[labels == label] for label in unique_labels}
    if return_src_arrays:
        return label_to_points, unique_labels, np.array(labels), np.array(coords)
    
    return label_to_points, unique_labels, None, None

def join_clusters(base_clusters:list[int],
                    num_closest:int=10,
                    threshold:float=0.35):
    
    # run_name = 'full_collective_diverse'
    # file = f'/media/penguaman/tosh2b/lidar_sync/adjacency_dict/{run_name}/pipeline/results/pointwise_results/pointwise_results.npz'
    file = '/media/penguaman/overflow/pointwise_results.npz'
    label_to_points, _, _, _ = labeled_clusters_from_pw_results(file, return_src_arrays=False)


    adj = determine_cluster_adjacency(file, base_clusters, threshold)

    print(f'building nbrhood pcds')
    # Now select and draw the clusters for each cluster's five closest (and itself)
    results_files = glob('data/cluster_joining/*.npz')
    finished = [os.path.basename(file).split('_')[0] for file in results_files]
    breakpoint()
    for label in base_clusters:
        if label not in finished:
            src_pts=label_to_points[label]

            # Get the <num_closest> closest other clusters
            adj_labels, dists = zip(*adj[label].items())
            low_dist_idxs = np.array(dists).argsort()[:num_closest]
            closest_lbls = np.array(adj_labels)[low_dist_idxs]

            # Loop over the closest labels and allow the user
            #   to select all/part of each to include in the final cluster
            print(f'processing cluster {label}')
            close_pts = [(lbl, label_to_points[lbl]) for lbl in closest_lbls]
            loop_and_ask(label, 
                            src_pts, 
                            close_pts[1:], 
                            save_file=True)
            breakpoint()


if __name__ == '__main__':
    num_closest = 10
    threshold = 0.35
    # Get closest clusters to each tree cluster
    decent_trees = [6, 9, 14, 15, 23, 34, 49, 68, 69, 151, 154, 157, 158, 163, 167, 188, 191, 192, 202,
                    220, 223, 236, 240,  241,
                    246, 283, 297,306,307, 377, 460, 490, 493, 512,
                    556, 669]
    join_clusters(decent_trees, num_closest, threshold)