import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_mesh(path):
    """
    Input: The file path to the object.
    Output: The open3d mesh object of the object with computed vert4ex normals 
    """
    if __name__ == "__main__":
        print("Testing mesh in open3d ...")
        mesh = o3d.io.read_triangle_mesh(path)#here one maybe needs to change the data path of the input data
        print(mesh)
        print(np.asarray(mesh.vertices))
        print(np.asarray(mesh.triangles))
        print("")
        mesh.compute_vertex_normals()
        return(mesh)

def reduce_mesh_size(mesh, factor):
    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / factor
    mesh_smp = mesh.simplify_vertex_clustering(voxel_size=voxel_size,contraction=o3d.geometry.SimplificationContraction.Average)
    return(mesh_smp)

def vis_as_png(mesh,scale = 1,tr_x=0, tr_y=0, title = ""):
    vis = o3d.visualization.Visualizer()
    
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    
    ctr = vis.get_view_control()
    #ctr.change_field_of_view(step=fov_step)
    ctr.translate(tr_x,tr_y)
    ctr.set_zoom(1/scale)
    
    img = vis.capture_screen_float_buffer(True)
    A = plt.imshow(np.asarray(img))
    #A.set_title(title)
    A.axes.set_title(title)
    A.axes.get_xaxis().set_visible(False)
    A.axes.get_yaxis().set_visible(False)
    return(A)

def put_noise_mesh(mesh, rho):
    mesh_noise = copy.deepcopy(mesh)
    vertices = np.asarray(mesh_noise.vertices)
    vertices+=np.random.uniform(0,rho, size = vertices.shape)
    mesh_noise.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_noise.compute_vertex_normals()
    return(mesh_noise)

def SNR(mesh1, mesh2):
    """
    Input: Two meshes with the same number of Vertices and the same number of faces. (Like in our denoising 
    case). mesh1 should be the target mesh, mesh2 should be the denoised mesh.
    Output: The SNR value for the 2 meshes
    """
    total_mesh1 = 0
    total_diff = 0
    for j in range(len(mesh.vertices)):
        total_diff+= np.linalg.norm(mesh1.vertices[j]-mesh2.vertices[j])
        total_mesh1+= np.linalg.norm(mesh1.vertices[j])
    return(-20*np.log10(total_diff/total_mesh1))
    

def denoising_filter_distance(mesh):
    """
    Input: A mesh datastructure
    Return: None, the input value just becomes modified by applying the distance filter
    """
    mesh.compute_adjacency_list()
    mesh_old = copy.deepcopy(mesh)
    
    #iterate over the vertices
    for j in tqdm(range(len(mesh_old.vertices))):
        vect_value = np.zeros(3) #set the value for the averaging
        dist = 0
        vertex_star_old = mesh_old.adjacency_list[j]
        #now iterate over the vertex star of the center vertex
        vtx_center = mesh_old.vertices[j]
        for vtx_index in vertex_star_old:
            l = 1/(np.linalg.norm(vtx_center - mesh_old.vertices[vtx_index])**2)
            dist+=l
            vect_value+=mesh_old.vertices[vtx_index]*l
        if dist!=0:
            vect_value/=dist # averaging with the surrounding values
        
        mesh.vertices[j] = vect_value
            

def denoising_average(mesh,nb_itr = 1):
    """
    Input: A mesh datastructure
    Return: None, the input value just becomes modified
    """
    mesh.compute_adjacency_list()
    mesh_old = copy.deepcopy(mesh)
    
    for j in tqdm(range(len(mesh_old.vertices))):
        vect_value = mesh_old.vertices[j]
        
        vertex_star = mesh_old.adjacency_list[j]
        for vtx_index in vertex_star:
            vect_value+=mesh.vertices[vtx_index]
        vect_value = vect_value/(len(vertex_star)+1)
        mesh.vertices[j] = vect_value
            
        
def plotting_the_denoising_avg(mesh_original,mesh_noisy,folder_path,picture_name, scale_cam=1,tr_x=0, tr_y=0, iterations = 1):
    """
    Input: the original mesh, the noisy mesh and the number of iterations. Important: The picture name must not contain the .png ending.
    Output: the list of the snr values, the images just become saved in the specific folder 
    """
    snr_values = []
    m_orig = vis_as_png(mesh_original,title="Original mesh before denoising")
    plt.savefig(folder_path+"/"+picture_name+".png")
    m_noisy = vis_as_png(mesh_noisy,scale=scale_cam, tr_x=tr_x, tr_y=tr_y,title="Mesh with noise")
    snr_values.append(SNR(mesh_original,mesh_noisy))
    plt.savefig(folder_path+"/mesh_with_noise.png")
    for j in range(iterations):
        denoising_average(mesh_noisy)
        vis_as_png(mesh_noisy,scale=scale_cam,tr_x=tr_x, tr_y=tr_y,title="Mesh after {} denoising iterations".format(j+1))
        plt.savefig(folder_path+"/"+picture_name+"_itr{}.png".format(j))
        snr_values.append(SNR(mesh_original,mesh_noisy))
    return(snr_values)

def plot_the_snr_ev_avg(mesh_original,mesh_noisy,folder_path,iterations = 1):
    snr_values = []
    snr_values.append(SNR(mesh_original,mesh_noisy))
    for j in range(iterations):
        denoising_average(mesh_noisy)
        snr_values.append(SNR(mesh_original,mesh_noisy))
        
    plt.plot(snr_values)
    plt.title("Evolution of the SNR value over the iterations")
    plt.savefig(folder_path+"/evolution.png")


def plotting_the_denoising_dst(mesh_original,mesh_noisy,iterations = 1):
    """
    Input:
    Output: 
    """
    #snr_values = []
    m_orig = vis_as_png(mesh_original,title = "Original mesh before denoising")
    plt.savefig("plotting_f16_with_miss/f16_w_m.png")
    m_noisy = vis_as_png(mesh_noisy,title="Mesh with noise")
    snr_values.append(SNR(mesh_original,mesh_noisy))
    plt.savefig("plotting_f16_with_miss/f16_w_m_noisy.png")
    for j in range(iterations):
        denoising_filter_distance(mesh_noisy)
        vis_as_png(mesh_noisy, title = "Mesh after {} denoising iterations".format(j+1))
        plt.savefig("plotting_f16_with_miss/f16_w_m_dst_itr{}.png".format(j))
        snr_values.append(SNR(mesh_original,mesh_noisy))


def plot_the_snr_dst_ev(mesh_original,mesh_noisy,iterations = 1):
    snr_values = []
    snr_values.append(SNR(mesh_original,mesh_noisy))
    for j in range(iterations):
        denoising_filter_distance(mesh_noisy)
        snr_values.append(SNR(mesh_original,mesh_noisy))
    plt.plot(snr_values)
    plt.title("Evolution of the SNR value over the iterations with the distance filter")
    plt.savefig("plotting_f16_without_miss/evolution_dst_without_missile.png")

def compute_the_cotan_weights(mesh):
    """ Input: The mesh datastructure,
    Output: The Matrix W and an array D consisting of the Cotan-Weights and the corresponding weights 
    for each vertex.
    """
    nvert = len(mesh.vertices)
    W = np.zeros([nvert,nvert])
    D = np.zeros(nvert)
    mesh.compute_adjacency_list()
    for vertex in tqdm(range(nvert)):
        neighb_vtx = mesh.adjacency_list[vertex]
        if len(neighb_vtx)!=0:
            for vtx_n in neighb_vtx:
                if W[vertex,vtx_n]==0:#check whether we already passed
                    neighbors_vtx_2 = mesh.adjacency_list[vtx_n]
                    """we now consider the adjacency list of one vertex. We loop over this list.
                    We want to obtain the vertices that are opposite to the line between vertex and vtx_n.
                    Since the adjacency lists are sets in pythpn we can intersect these sets
                    """
                    pt_of_interest = neighb_vtx.intersection(neighbors_vtx_2)
                    #print('points',vertex, vtx_n, pt_of_interest)
                    if len(pt_of_interest)==2:#for the small testing cases important
                        start1 = pt_of_interest.pop()
                        start2 = pt_of_interest.pop()

                        #get the geometric vector positions 
                        v_vertex = mesh.vertices[vertex]
                        v_opp = mesh.vertices[vtx_n]
                        v_start1 = mesh.vertices[start1]
                        v_start2 = mesh.vertices[start2]

                        #get the geometry information of the combining vector
                        v1 = v_vertex-v_start1
                        v2 = v_opp-v_start1
                        w1 = v_vertex-v_start2
                        w2 = v_opp-v_start2
                        ctan1 = np.dot(v1,v2)/np.linalg.norm(np.cross(v1,v2))
                        ctan2 = np.dot(w1,w2)/np.linalg.norm(np.cross(w1,w2))
                        ctan_weight = ctan1+ctan2
                        W[vertex,vtx_n]=ctan1+ctan2
                        W[vtx_n,vertex] = ctan1+ctan2
                        D[vertex]+=ctan_weight
                        D[vtx_n]+=ctan_weight
                        #print('vertex', vertex)
                        #print(v_vertex)
    return(W,D)

def denoise_with_laplace(mesh,delta):
    vertices = mesh.vertices
    mesh.compute_adjacency_list()
    print(type(vertices))
    vertices = np.asarray(vertices)
    W,D = compute_the_cotan_weights(mesh)
    #tilde_W =np.diag(1/D)@W
    for j in range(len(vertices)):
        vect_value = np.zeros(3)
        vertex_star = mesh.adjacency_list[j]
        v_vtx_center = mesh.vertices[j]
        factor = D[j]
        for vtx_ind in vertex_star:
            vect_value += delta*mesh.vertices[vtx_ind]*W[vtx_ind,j]
        if factor!=0:
            vect_value/=factor
        vect_value += (1-delta)*mesh.vertices[vtx_ind]
        mesh.vertices[j] = vect_value




#todo!!
def denoise_with_heat_diffusion(mesh_noise, mesh, delta, steps):
    """
    Input: The mesh with noise, the size of the time step, the number of time steps.
    Output: An array with the evolution of the SNR value and an array with the steps
    The mesh won't be returned but modified!
    """
    vertices = mesh_noise.vertices
    mesh_noise.compute_adjacency_list()
    print(type(vertices))
    vertices = np.asarray(vertices)
    print(vertices)
    W,D = compute_the_cotan_weights(mesh)
    D+=np.random.uniform(0.00001, 10**(-10))
    print('how much zeros',np.count_nonzero(D==0.))
    tilde_W =np.diag(1/D)@W
    snr_values = np.zeros(steps+1)
    times = delta*np.arange(steps+1)
    snr_values[0] = SNR(mesh,mesh_noise)

    for i in range(steps):
        for j in range(len(mesh_noise.vertices)):
            vertex = mesh_noise.vertices[j]
            neighbours = mesh_noise.adjacency_list[j]
            vertex_new = 0
            for vtx in neighbours:
                vertex_new+= delta*tilde_W[vtx,j]*vertex
            vertex_new+=(1-delta)*vertex
            mesh_noise.vertices[j] = vertex_new
        vis_as_png(mesh_noise,scale = 1.5,title = 'Itr going on')
        plt.show()

        snr_values[i] = SNR(mesh,mesh_noise)
    return(times, snr_values)


mesh = read_mesh("figures/bunny.obj")
#mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius = 2)
#o3d.visualization.draw_geometries([mesh])
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()
mesh = reduce_mesh_size(mesh, 15)
mesh_original = copy.deepcopy(mesh)
mesh = put_noise_mesh(mesh, .1)
print(SNR(mesh_original,mesh))
vis_as_png(mesh, scale=1.5)
plt.show()
denoise_with_laplace(mesh,0)
print(SNR(mesh_original,mesh))
vis_as_png(mesh, scale=1.5)
plt.show()

