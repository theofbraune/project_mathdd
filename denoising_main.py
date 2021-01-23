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
    """
    Input: A mesh datastructure, the camera scale, the x-translation, the y-translation and the 
    title on top of the picture.
    Output: A matplotlib figure of the scene
    """
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
    #mesh_noise.compute_vertex_normals()
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
    

def denoising_filter_distance(mesh, delta = 1):
    """
    Input: A mesh datastructure and a time step delta 
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
        
        mesh.vertices[j] = (1-delta)*mesh.vertices[j] + delta*vect_value
            
def denoising_average(mesh,delta = 1):
    """
    Input: A mesh datastructure and a value of a timestep delta 
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
        mesh.vertices[j] = delta*vect_value + (1-delta)*mesh.vertices[j]
            
        
def plotting_the_denoising_avg(mesh_original,mesh_noisy,folder_path,picture_name, delta_n=1, scale_cam=1,tr_x=0, tr_y=0, iterations = 1):
    """
    Input: the original mesh, the noisy mesh, the folder path as a string, the picture name. 
    The picture name must not contain the .png ending. Further one has optional the stepsize for the heat diffusion, 
    the scale of the camera, the translation in x direction tr_x,the translation in y direction tr_y and the number of iterations. 
    Output: the list of the snr values, the images just become saved in the specific folder 
    """
    snr_values = []
    m_orig = vis_as_png(mesh_original,title="Original mesh before denoising")
    plt.savefig(folder_path+"/"+picture_name+".png")
    m_noisy = vis_as_png(mesh_noisy,scale=scale_cam, tr_x=tr_x, tr_y=tr_y,title="Mesh with noise")
    snr_values.append(SNR(mesh_original,mesh_noisy))
    plt.savefig(folder_path+"/mesh_with_noise.png")
    for j in range(iterations):
        denoising_average(mesh_noisy,delta=delta_n)
        vis_as_png(mesh_noisy,scale=scale_cam,tr_x=tr_x, tr_y=tr_y,title="Mesh after {} denoising iterations".format(j+1))
        plt.savefig(folder_path+"/"+picture_name+"_itr{}.png".format(j))
        plt.close()
        snr_values.append(SNR(mesh_original,mesh_noisy))
    print('snr_values', snr_values)
    return(snr_values)

def plotting_the_denoising_dst(mesh_original,mesh_noisy,folder_path,picture_name,delta_n = 1,scale_cam=1,tr_x=0, tr_y=0, iterations = 1):
    """
    Input: The original mesh, the mesh with noise,the step size delta_n, the number of iterations, 
    Output: array of the SNR values, the mesh wont be returned, but modified with the distance laplace.

    Note: The folder name and folder path need to be renamed each time, s.th the pictures will be saved 
    in the right folder. 
    """
    snr_values = []
    m_orig = vis_as_png(mesh_original,title = "Original mesh before denoising",scale=scale_cam)
    plt.savefig(folder_path+"/"+picture_name+".png")
    m_noisy = vis_as_png(mesh_noisy,title="Mesh with noise",scale=scale_cam)
    snr_values.append(SNR(mesh_original,mesh_noisy))
    plt.savefig(folder_path+"/"+picture_name+"_noisy.png")
    for j in range(iterations):
        denoising_filter_distance(mesh_noisy, delta = delta_n) #{:.2f}
        vis_as_png(mesh_noisy, title = "Mesh after {} iterations of denoising".format(delta_n*(j+1)),scale=2.5)
        plt.savefig(folder_path+"/"+picture_name+"_itr{}.png".format(j))
        snr_values.append(SNR(mesh_original,mesh_noisy))
        plt.close()
        print('snr_values',snr_values)
    return(snr_values)

#####--- cotangent laplace start here ---#####

def denoise_with_laplace(mesh,delta=1):
    """
    Input: The mesh and the timestep delta
    Output:
    """
    nvert = len(mesh.vertices)
    mesh.compute_adjacency_list()
    tr_list = compute_the_triangle_topology(mesh)
    #this is not optimal at all !! Several computations are done twice
    for j in tqdm(range(len(mesh.vertices))):
        neighb_vtx = mesh.adjacency_list[j]
        vect_value = 0
        sum = 0
        for n_vtx in neighb_vtx:
            if ctan_weight(mesh,tr_list, j, n_vtx)== False:
                pass
            else:
                vect_value += mesh.vertices[n_vtx]*ctan_weight(mesh,tr_list, j, n_vtx)
                sum+= ctan_weight(mesh,tr_list, j, n_vtx)
        if sum !=0:
            vect_value = vect_value/sum
            mesh.vertices[j] = (1-delta)*mesh.vertices[j] + delta*vect_value
        else:#if the sum is zero color the point and don't change it
            mesh.vertex_colors[j] = np.array([0.,1.,0.])


def compute_the_triangle_topology(mesh):
    """
    Input: The mesh 

    Output: An Array that has len of the vertices. For each position/vertex we create an array with the 
    triangles containing the vertex.
    """
    mesh.remove_duplicated_triangles()
    triangle_vtx = [[] for j in range(len(mesh.vertices))]

    for j in range(len(mesh.triangles)):
        triangle = mesh.triangles[j]
        triangle_vtx[triangle[0]].append(j)
        triangle_vtx[triangle[1]].append(j)
        triangle_vtx[triangle[2]].append(j)
        
    return(triangle_vtx)

def ctan_weight(mesh, tr_list, vertex1, vertex2):
    """
    Input: The mesh, the topological information about the connecting triangles and two vertex indices 
    Output: The corrsponding cotangent weight
    """

    #take the triangles that include the corr vertices
    tr_1 = tr_list[vertex1]
    tr_2 = tr_list[vertex2]

    #take the intersection. We obtain two triangles
    def intersection(lst1, lst2): 
        return list(set(lst1) & set(lst2)) 
    #look which two triangles share both vertex1 and vertex2
    tr_of_interest = intersection(tr_1,tr_2)
    vtc_1 = mesh.triangles[tr_of_interest[0]]
    vtc_2 = mesh.triangles[tr_of_interest[1]]

    opp_points = set(vtc_1).union(set(vtc_2))
    #remove the vertices, such that only the opposing vertices remain
    opp_points.remove(vertex1)
    opp_points.remove(vertex2)


    if len(opp_points)==2:
        start1 = opp_points.pop()
        start2 = opp_points.pop()
    else:
        #color the points where we fail!! Only for debugging purpose.
        
        mesh.vertex_colors[vertex1] = np.array([0.,0.,0.])
        mesh.vertex_colors[vertex2] = np.array([0.,0.,0.])
        print('tr vertex1', tr_1)
        print('tr vertex2', tr_2)
        print('tr interest', tr_of_interest)
        print('triangle 1', vtc_1)
        print('triangle 2', vtc_2)
        print(vertex1, vertex2)
        print(opp_points)
        return(False)
    #get the geometric vector positions 
    v_vertex = mesh.vertices[vertex1]
    v_opp = mesh.vertices[vertex2]
    v_start1 = mesh.vertices[start1]
    v_start2 = mesh.vertices[start2]
    v1 = v_vertex-v_start1
    v2 = v_opp-v_start1
    w1 = v_vertex-v_start2
    w2 = v_opp-v_start2
    ctan1 = np.dot(v1,v2)/np.linalg.norm(np.cross(v1,v2))
    ctan2 = np.dot(w1,w2)/np.linalg.norm(np.cross(w1,w2))
    ctan_weight = ctan1+ctan2
    return(ctan_weight)

def plt_evolution_laplace_ctan(mesh_original,mesh_noisy,folder_path,picture_name ,iterations = 1, delta_n=1,scale_cam=1):
    """
    Input: the original mesh, the noisy mesh, the folder path as a string, the picture name. 
    The picture name must not contain the .png ending. Further one has optional the stepsize for the heat diffusion, 
    the scale of the camera, the translation in x direction tr_x,the translation in y direction tr_y and the number of iterations. 
    Output: the list of the snr values (it has length iterations+1), the images just become saved in the specific folder 
    """
    snr_values = []
    m_orig = vis_as_png(mesh_original,title = "Original mesh before denoising",scale=scale_cam)
    plt.savefig(folder_path+"/"+picture_name+".png")
    m_noisy = vis_as_png(mesh_noisy,title="Mesh with noise",scale=scale_cam)
    snr_values.append(SNR(mesh_original,mesh_noisy))
    plt.savefig(folder_path+"/mesh_with_noise.png")
    for j in range(iterations):
        denoise_with_laplace(mesh_noisy, delta = delta_n) #{:.2f}
        vis_as_png(mesh_noisy, title = "Mesh after {:.2f} time units of denoising".format(delta_n*(j+1)),scale=scale_cam)
        plt.savefig(folder_path+"/"+picture_name+"_itr{}.png".format(j))
        plt.close()
        snr_values.append(SNR(mesh_original,mesh_noisy))
    
    print(snr_values)
    return(snr_values)



def rotate_mesh(mesh):
    """
    Use this method to turn the mesh. Since it was a bit complicated to change the camera position,
    this method does the trick too.
    """
    rot_90_z = np.transpose(np.array([[0,1,0],[-1,0,0],[0,0,1]]))
    rot_90_x = np.transpose(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
    rot_90_y = np.transpose(np.array([[0,0,1],[0,1,0],[-1,0,0]]))
    angle = np.radians(-30)
    angle2 = np.radians(10)
    rot_angle_x = np.array([[1,0,0],[0,np.cos(angle2),-np.sin(angle2)],[0,np.sin(angle2),np.cos(angle2)]])
    rot_angle_y = np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
    for j in range(len(mesh.vertices)):
        mesh.vertices[j] = (mesh.vertices[j]@rot_90_y)@(np.transpose(rot_90_z))@rot_angle_y@rot_angle_x
        #mesh.vertices[j]*=3

######---READ THE MESH---######
#mesh = read_mesh("figures/bunny.obj")
#mesh = o3d.io.read_triangle_mesh("figures/f16_without_missiles.obj")
mesh = read_mesh("figures/knot.ply")
#mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius = 2)
#o3d.visualization.draw_geometries([mesh])# a good method to analyze the mesh, because one can turn the mesh around in the output
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

######---REDUCE THE SIZE---######
#mesh = reduce_mesh_size(mesh, 65)

######---TRY TO REPAIR THE TOPOLOGICAL DAMAGE---######
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()

mesh.compute_adjacency_list()
#rotate_mesh(mesh) #optional, only to move the mesh
mesh_original = copy.deepcopy(mesh)

######---ADD NOISE TO THE MESH---######
vis_as_png(mesh,scale = 1.8, title = "Original mesh")
plt.show()
plt.close()

mesh = put_noise_mesh(mesh, 0.07)

vis_as_png(mesh,scale = 1.8, title = "Noisy mesh")
print('SNR before', SNR(mesh_original,mesh))
plt.show()
plt.close()


######---START THE DENOISING PROCESS WITH THE ALGORITHM OF CHOICE---######

Y = plt_evolution_laplace_ctan(mesh_original, mesh,"checklaplace","knot_heat", iterations=5, delta_n=0.3,scale_cam=1.5)

######---PLOT THE EVOLUTION OF THE SNR VALUE---######
X = [0.2*j for j in range(5+1)]

plt.plot(X, Y, label='cotangent')
plt.legend()
plt.title("Evolution of the SNR value over the time.")
plt.show()

