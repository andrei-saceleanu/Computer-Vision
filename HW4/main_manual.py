import numpy as np
import open3d as o3d
import cv2
from maps import *


np.random.seed(42)

# configs for enhancing cloud and filtering from mesh afterwards
# 200, 20, densities quantile 0.05
# 100, 10, densities quantile 0.1

# uvs = []
# vertices = np.array(mesh.vertices)
# triangles = np.array(mesh.triangles)

# # for num_tr in range(triangles.shape[0]):
# #     for i in range(3):
# #         pt = vertices[triangles[num_tr, i]]
# #         pt[1] *= (-1)
# #         proj_pt = P @ np.array(pt.tolist()+[1]).reshape(4,1)
# #         proj_pt /= proj_pt[2]
# #         x, y = proj_pt[0], proj_pt[1]
# #         uvs.append([1-x[0]/w,1-y[0]/h])

# mesh.textures = [o3d.geometry.Image(img)]
# mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(uvs,dtype=np.float64))
# mesh.triangle_material_ids = o3d.utility.IntVector([0] * triangles.shape[0])       



def find_3d_intersection(start_points, end_points):

    n = (end_points - start_points) / np.linalg.norm(end_points - start_points, axis=1)[:, np.newaxis]
    projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis, :]

    R = projs.sum(axis=0)
    q = (projs @ start_points[:, :, np.newaxis]).sum(axis=0)

    p = np.linalg.lstsq(R, q, rcond=None)[0]

    return p.reshape(1, 3)[0]


def get_system_mat(image_map, compute_error_stats=True, compute_separate_params=False):

    mat = []
    for point_id, point_data in image_map.items():
        u, v = point_data["2d"]
        x, y, z = point_data["3d"]
        point_entries = [
            [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u],
            [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
        ]
        mat.extend(point_entries)

    mat = np.array(mat)
    ata = mat.T @ mat

    errs = []
    P = np.linalg.eig(ata)[1][:,-1].reshape(3,4)
    for point_id in image_map:
        proj = P @ np.array(image_map[point_id]["3d"]+[1],dtype=np.float64)
        err = np.linalg.norm((proj/proj[2])[:2]-np.array(image_map[point_id]["2d"]))
        errs.append(err)
        
    if compute_error_stats:
        print(np.mean(errs))
    

    if compute_separate_params:
        p3x3 = np.array(P[:3,:3][::-1,:])
        qtilde, rtilde = np.linalg.qr(p3x3.T)
        q = qtilde.T[::-1,:]
        r = rtilde.T[::-1,:][:,::-1]
    
    m_3x3 = P[:, :3]
    p4_3x1 = P[:, 3]
    m_inv_3x3 = np.linalg.inv(m_3x3)
    camera_center_3x1 = np.expand_dims(-m_inv_3x3 @ p4_3x1, 1)
    

    return camera_center_3x1, m_inv_3x3, P

def interpolate_new_points(pts, pool, pool2, pool3):
    for _ in range(100):
        a,b = np.random.choice(np.arange(len(pool)),2,replace=False)
        for _ in range(10):
            t = np.random.rand()
        
            pts = np.concatenate([pts,(t * np.array(pool[a]) + (1-t) * np.array(pool[b])).reshape(1,3)])

    for _ in range(100):
        a,b = np.random.choice(np.arange(len(pool2)),2,replace=False)
        for _ in range(10):
            t = np.random.rand()
        
            pts = np.concatenate([pts,(t * np.array(pool2[a]) + (1-t) * np.array(pool2[b])).reshape(1,3)])


    for _ in range(100):
        a,b = np.random.choice(np.arange(len(pool3)),2,replace=False)
        for _ in range(10):
            t = np.random.rand()
        
            pts = np.concatenate([pts,(t * np.array(pool3[a]) + (1-t) * np.array(pool3[b])).reshape(1,3)])
    return pts


def texture_mesh(P, pts, img, h, w):
    colors = []
    for pt in pts:
        pt[1] *= (-1)
        proj_pt = P @ np.array(pt.tolist()+[1]).reshape(4,1)
        proj_pt /= proj_pt[2]
        x, y = int(proj_pt[0]), int(proj_pt[1])
        colors.append((img[y, x]/255).tolist())
        
    return colors

def main():

    cc0, mat0, _ = get_system_mat(image0_map)
    cc1, mat1, P = get_system_mat(image1_map)


    start_points = np.concatenate([cc0.T,cc1.T]) # camera centers in 3D world coordinates

    errs = []
    pts = []
    pool = []
    pool2 = []
    pool3 = []

    for point_id in image0_map_all:
        if point_id in image1_map_all:

            curr_point0 = np.array(image0_map_all[point_id]["2d"] +[1]).reshape(3,1)
            curr_point1 = np.array(image1_map_all[point_id]["2d"] +[1]).reshape(3,1)

            ray_dir0 = mat0 @ curr_point0
            ray_dir0 /= ray_dir0[2]
            ray_dir1 = mat1 @ curr_point1
            ray_dir1 /= ray_dir1[2]
            end_points = np.concatenate([
                (cc0+100*ray_dir0).T,
                (cc1+100*ray_dir1).T
            ]) # distant point along ray passing through point of interest

            backproj_3d =[
                round(elem, 3)
                for elem in find_3d_intersection(start_points, end_points).tolist()
            ]
            real_3d = image0_map_all[point_id]["3d"]
            
            pts.append(backproj_3d)
            if point_id < 24:
                pool.append(backproj_3d)
            elif 24 <= point_id < 46:
                pool2.append(backproj_3d)
            else:
                pool3.append(backproj_3d)
            
            err = np.linalg.norm(np.array(backproj_3d)-np.array(real_3d))
            errs.append(err)
    print(np.mean(errs))

    # reflection on Oy in order to match Open3D convention with our convention
    refl_y = np.eye(4)
    refl_y[1,1] = -1
    geometry_list = []

    # coordinate frame
    org = np.array([0,0,0])
    org_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=org)
    org_axis = org_axis.transform(refl_y)
    geometry_list.append(org_axis)


    pts = np.array(pts)
    pts = interpolate_new_points(pts, pool, pool2, pool3)
    pts = (refl_y @ np.hstack((pts,np.ones((pts.shape[0],1)))).T).T[:,:3]
    
    # add point cloud to list of geometries to be rendered
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_covariances()
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    geometry_list.append(pcd)


    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh = mesh.subdivide_midpoint(number_of_iterations=3)
    mesh.compute_vertex_normals()
    print(mesh)
    
    # add mesh to geometries to be rendered, map texture to its vertices
    pts = np.array(mesh.vertices)
    img = cv2.cvtColor(cv2.imread("image1.jpg"),cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    colors = texture_mesh(P, pts, img, h, w)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    geometry_list.append(mesh)

    # plot results
    o3d.visualization.draw_geometries(
        geometry_list,
        window_name="Backprojected 3D points and textured mesh",
        mesh_show_back_face=False,
        point_show_normal=False
    )

if __name__ == "__main__":
    main()