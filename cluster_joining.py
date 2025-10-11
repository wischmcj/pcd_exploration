from collections import defaultdict
import numpy as np
import pickle

from open3d.ml.vis import o3d

from demo import to_o3d
from open3d.visualization import draw_geometries as draw
from scipy.spatial import cKDTree



def load_kdtrees(label_list):
    print(f'loading kdtrees')
    kdtrees = []
    for label in label_list:
        with open(f'scripts/clust_kdtrees/{label}_kd_tree.pkl', 'rb') as f:
             kdtrees.append((label, pickle.load(f)))
    return kdtrees

def load_adjacency_dict():
    print(f'loading adjacency dict')
    with open('scripts/clust_kdtrees/adj.pkl', 'rb') as f:
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
        with open(f'scripts/clust_kdtrees/{label}_kd_tree.pkl', 'wb') as f:
            pickle.dump(kdtree, f)
        kdtrees.append((label, kdtree))
    return kdtrees

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
    with open('scripts/clust_kdtrees/adj.pkl', 'wb') as f:
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

def user_cluster(pcd, eps=0.03, min_points=15, filter_non_clusters=True, draw_result=True):
    print('clustering')
    choosing_inputs = True
    while choosing_inputs:
        user_input = input("eps? ").strip().lower()
        eps = float(user_input)
        user_input = input("min_points? ").strip().lower()
        min_points = int(user_input)
        # Cluster and draw
        labels =  np.array( pcd.cluster_dbscan(eps=eps, min_points=min_points,print_progress=True))
        print(f'{labels.shape=}')
        non_cluster_idxs = np.where(labels < 0)[0]
        pcd, orig_colors = cluster_color(pcd,labels)
        pcd = pcd.select_by_index(non_cluster_idxs, invert=True)
        draw([pcd])
        # Finalize
        user_input = input("again? (y/n)").strip().lower()
        if user_input == 'n':
            choosing_inputs = False
    return labels


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
    src_pcd = to_o3d(coords=src_pts,colors= [[0,0,1]]*len(src_pts))

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
            if user_input == 'y':
                final_pts.append(pts)
                final_lbls.append(lbls)
            if user_input == 'r':
                print(f'recursing on cluster {lbl}')
                # split the cluster into subclusters
                sub_labels = user_cluster(comp_pcd)
                unique_sub_labels = np.unique(sub_labels)
                labeled_subclusters = [(sub_lbl, pts[sub_labels == sub_lbl]) for sub_lbl in unique_sub_labels]
                # run algo to choose which subclusters to add to the final cluster
                chosen_pts_list, chosen_sub_lbls = loop_and_ask(src_lbl, src_pts, labeled_subclusters, return_chosen_pts=True)
                # generate compound labels, add chosen to final
                for chosen_pts, chosen_sub_lbl in zip(chosen_pts_list, chosen_sub_lbls):
                    final_pts.append(chosen_pts)
                    compound_lbl = f'{lbl}pt{chosen_sub_lbl}'
                    final_lbls.append([compound_lbl]*len(chosen_pts))
    if return_chosen_pts:
        return final_pts[1:], final_lbls[1:]
    if save_file:
        final_pts = np.vstack(final_pts)
        final_lbls = np.hstack(final_lbls)
        pcd = to_o3d(coords=final_pts, labels=final_lbls)
        draw([pcd])
        file_name = '_'.join(final_lbls)
        np.savez_compressed(f'scripts/joined_trees/{file_name}_joined.npz', points=final_pts, labels=final_lbls)

# def join_clusters(pcds):
#     joined_pcd = join_pcds(pcds)
#     draw([joined_pcd])
#     breakpoint()
#     return joined_pcd


if __name__ == '__main__':
    run_name = 'full_collective_diverse'
    # file = f'/media/penguaman/tosh2b/lidar_sync/adjacency_dict/{run_name}/pipeline/results/pointwise_results/pointwise_results.npz'
    file = '/media/penguaman/overflow/pointwise_results.npz'
    coords_file = 'coords'
    labels_file = 'instance_preds'
    num_closest = 10
    threshold = 0.35
    # READ IN DATA
    data = np.load(file)
    labels = data[labels_file]
    coords = data[coords_file]
    unique_labels = np.unique(labels)
    label_to_points = {label:coords[labels == label] for label in unique_labels}

    # Get closest clusters to each tree cluster
    decent_trees = [6, 9, 14, 15, 23, 34, 49, 68, 69, 151, 154, 157, 158, 163, 167, 188, 191, 192, 202,
                    220, 223, 236, 240,  241,
                    246, 283, 297,306,307, 377, 460, 490, 493, 512,
                    556, 669]

    # load saved data
    kdtrees = load_kdtrees(unique_labels)
    adj = load_adjacency_dict()
    # create from scratch
    # kdtrees = create_kdtrees(coords, labels)
    # adj = map_clusters(decent_trees, kdtrees, threshold)

    print(f'getting closest labels')
    label_to_closest_labels = defaultdict(list)
    for tree_label in decent_trees:
        adj_labels, dists = zip(*adj[tree_label].items())
        low_dist_idxs = np.array(dists).argsort()[:num_closest]
        closest_lbls = np.array(adj_labels)[low_dist_idxs]
        label_to_closest_labels[tree_label] = closest_lbls
    

    print(f'building nbrhood pcds')
    # Now select and draw the clusters for each cluster's five closest (and itself)
    for label, close_labels in label_to_closest_labels.items():
        cluster_labels = [label] + list(close_labels)
        close_pts = [(lbl, label_to_points[lbl]) for lbl in cluster_labels]
        close_details = [([lbl]*len(pts),pts) for lbl,pts in close_pts]

        # # Create single pcd colored by labels
        # nbr_lbls=np.hstack([lbls for lbls,pts in close_details])
        # nbr_pts = np.vstack([pts for _,pts in close_pts])
        # pcd = to_o3d(coords=nbr_pts, labels=nbr_lbls)
        # draw([pcd])

        # Draw clusters and src one at a time 
        src_pts = close_pts[0][1]
        # src_lbls = [label]*len(src_pts)
        # create pcds for each cluster
        loop_and_ask(label, 
                        src_pts, 
                        close_pts[1:], 
                        save_file=True)
        breakpoint()