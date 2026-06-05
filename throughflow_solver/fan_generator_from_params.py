from DOE2_rotor_only_fan_generator import create_doe2_rotor_only_fan_parametric
from run_sleq_solver import run_solver_from_sleq
import numpy as np
from stl import mesh
import pandas as pd
import time

## Helper Functions

def convert_to_cartesian(section):
    Z_cart = section['x'] # Axial becomes Z axis
    theta = section['theta']
    r = section['r']
    X_cart = r * np.cos(theta)
    Y_cart = r * np.sin(theta)
    
    return X_cart, Y_cart, Z_cart

def convert_to_cylindrical(section):
    x,y,z = section[:,0], section[:,1], section[:,2]
    Z_cyl = z
    theta = np.arctan(y/z)
    r = np.sqrt(x**2+y**2)
    
    return r,theta,Z_cyl

def reorder_with_te_last(side, other_te):
    d0 = np.linalg.norm(side[0] - other_te)
    d1 = np.linalg.norm(side[1] - other_te)

    if d0 < d1:
        return side[::-1]
    return side

def trailing_edge_arc_from_sides(P_side, S_side, n_points=40):
    """
    Robust circular trailing edge arc construction.

    Parameters
    ----------
    P_side : (2,3) array
        Two pressure-side points near TE.
        P_side[1] must be the TE point.
    S_side : (2,3) array
        Two suction-side points near TE.
        S_side[1] must be the TE point.
    n_points : int
        Number of arc points.

    Returns
    -------
    arc : (n_points,3) ndarray
        Circular TE arc.
    """
    P_side = reorder_with_te_last(P_side, S_side[1]) # this assumes s_side[1] is the TE point
    S_side = reorder_with_te_last(S_side, P_side[1])
    

    P1, P2 = np.asarray(P_side[0], float), np.asarray(P_side[1], float)
    S1, S2 = np.asarray(S_side[0], float), np.asarray(S_side[1], float)

    # ---------------------------
    # 1️⃣ Fit best TE plane (SVD)
    # ---------------------------
    pts = np.vstack([P1, P2, S1, S2])
    centroid = pts.mean(axis=0)

    A = pts - centroid
    _, _, vh = np.linalg.svd(A)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    # ---------------------------
    # 2️⃣ Project TE endpoints to plane
    # ---------------------------
    def project_to_plane(point):
        return point - np.dot(point - centroid, normal) * normal

    P_te = project_to_plane(P2)
    S_te = project_to_plane(S2)

    # ---------------------------
    # 3️⃣ Circle geometry
    # ---------------------------
    chord = S_te - P_te
    thickness = np.linalg.norm(chord)

    if thickness < 1e-12:
        raise ValueError("Trailing edge thickness is zero or degenerate.")

    radius = thickness / 2.0
    center = 0.5 * (P_te + S_te)

    # First in-plane direction (radial toward P_te)
    e1 = (P_te - center) / radius

    # Second orthogonal in-plane direction
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)

    # ---------------------------
    # 4️⃣ Ensure correct bulge direction
    # ---------------------------
    # Use average tangent to determine outward direction
    t_ps = P2 - P1
    t_ss = S2 - S1
    avg_tangent = 0.5 * (
        t_ps / np.linalg.norm(t_ps) +
        t_ss / np.linalg.norm(t_ss)
    )

    # If arc bulges into blade, flip e2
    if np.dot(e2, avg_tangent) < 0:
        e2 = -e2

    # ---------------------------
    # 5️⃣ Generate semicircle
    # ---------------------------
    theta = np.linspace(0, np.pi, n_points)

    arc = np.array([
        center + radius * (np.cos(t) * e1 + np.sin(t) * e2)
        for t in theta
    ])

    return arc

def smooth_trailing_edges(sections):
    smoothed_sections = []
    for section in sections:
        new_points = trailing_edge_arc_from_sides(section[-2:,:], section[:2,:])
        new_section = np.vstack((section, new_points))
        
        smoothed_sections.append(new_section)
    
    return np.array(smoothed_sections)

def make_tip_points(section):
    n = len(section)//2
    te_points = 40
    faces = []
    ps = np.vstack((section[:n+1,:],section[-te_points//2:,:]))
    ss = section[n+1:-te_points//2,:][::-1]

    for j in range(len(ps)):
        try:
            p1 = ps[j]
            p2 = ps[j+1]
            p3 = ss[j]
            p4 = ss[j+1]
            
            # Triangle 1
            faces.append([p1, p2, p3])
            # Triangle 2
            faces.append([p1, p3, p4])
        except IndexError:
            try:
                p1 = ps[j]
                p2 = ps[j+1]
                p3 = ss[j]
                
                faces.append([p1, p2, p3])
            except IndexError:
                break
            
    return faces

def generate_stl_faces(sections, Nblade = 7): 
    faces = []
    n_pts = len(sections[0])
    n_sec = len(sections)

    for i in range(n_sec - 1):
        for j in range(n_pts - 1):
            # Define 4 corners of a "rectangle" between two radial layers
            p1 = sections[i][j]
            p2 = sections[i][j+1]
            p3 = sections[i+1][j+1]
            p4 = sections[i+1][j]
            
            # Triangle 1
            faces.append([p1, p2, p3])
            # Triangle 2
            faces.append([p1, p3, p4])
            
    faces += make_tip_points(sections[-1])

    # Convert to numpy array for STL
    faces_array = np.array(faces)
    # Flip blade section 180 degrees
    faces = [] 
    rotation_angle_x = 180 
    rotation_matrix_x = np.array([[1, 0, 0], 
                                  [0, np.cos(np.radians(rotation_angle_x)), -np.sin(np.radians(rotation_angle_x))], 
                                  [0, np.sin(np.radians(rotation_angle_x)), np.cos(np.radians(rotation_angle_x))]]) 
    for face in faces_array:
        rotated_face = np.matmul(rotation_matrix_x, face.T).T 
        faces.append(rotated_face) 
        
    faces_array = np.array(faces)

    # Rotate the blade section around the Z-axis to create the full fan
    all_faces = []
    for i in range(Nblade):
        rotation_angle = i * (360 / Nblade)
        rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle)), 0],
                                    [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle)), 0],
                                    [0, 0, 1]])
        for face in faces_array:
            rotated_face = np.matmul(rotation_matrix, face.T).T
            all_faces.append(rotated_face)
            
    all_faces = np.array(all_faces)
    
    return all_faces

def generate_fan_stl(
    n_blades = 7,
    rhub=0.015,              # 15 mm hub radius
    rtip=0.040,              # 40 mm tip radius
    RPM=3000.0,              # 3000 rpm
    mdot=0.05,               # 0.05 kg/s (further reduced for better blade effect)
    DHmid=0.9,               # De Haller number = 0.9 (diffusion)
    INC=2.0,                 # 2° incidence
    Vexp=0,                # Vortex exponent (between free and rigid)
    lean_max = 0.0,
    lean_weighting = 1.0,
    lean_straight = 0,
    index = None, 
    ):
    
    start = time.time()
    tip_clearance_percent = np.round((0.04-rtip)/0.04 *100,3)
    num_streamlines=20
    num_internal_stations=5
    rotor_chord=0.015

    # Set corresponding filename for later retrieval
    filename = f"inverse_design_test_{index}_{n_blades}_{mdot:.4f}_{DHmid:.4f}_{INC:.2f}_{Vexp:.4f}_{lean_max:.4f}_{lean_weighting}_{lean_straight:.2f}_{tip_clearance_percent:.3f}"
    print(filename)

    # # Run fan design
    create_doe2_rotor_only_fan_parametric(
            rhub=rhub,              # 15 mm hub radius
            rtip=rtip,              # 40 mm tip radius
            RPM=RPM,              # 3000 rpm
            mdot=mdot,               # 0.05 kg/s (further reduced for better blade effect)
            DHmid=DHmid,               # De Haller number = 0.9 (diffusion)
            INC=INC,                 # 2° incidence
            Vexp=Vexp,                # Vortex exponent (between free and rigid)
            num_streamlines=num_streamlines,
            num_internal_stations=num_internal_stations,
            rotor_chord=rotor_chord,
            rotor_blades=n_blades,
            enable_curtis=True,      # ENABLE CURTIS OPTIMIZATION
            lean_max = lean_max,
            lean_weighting = lean_weighting,
            lean_straight=lean_straight,
            filename='rotor_files\\' + filename+".dat"
        )
    print("============================================= \n")
    print('Fan design complete')
    
    # Sleq params
    sleq_filename = filename+".dat"
    verbose = False
    max_iterations = 50
    plot_results = False
    tabbed_interface = False
    
    # Run sleq throughflow w/ CURTIS
    run_solver_from_sleq(sleq_filename, verbose=verbose, max_iterations=max_iterations, 
                                    plot_results=plot_results, tabbed_interface=tabbed_interface)
    
    print("============================================= \n")
    print('Sleq run complete')

    end = time.time()
    print("============================================= \n")
    print(f'Time taken: {end-start}s')
    print("============================================= \n")
    # Load output
    output_filename = 'throughflow_output\\' + filename+"_blade_sections.npz"
    section_data = np.load(output_filename, allow_pickle=True)['sections'].item()['R1']
    
    XYZ_sections = []
    # Convert all section data to cartesian coordinates for stl generation
    for i, section in enumerate(section_data):
        X_cart, Y_cart, Z_cart = convert_to_cartesian(section)
        XYZ_sections.append(np.stack([X_cart, Y_cart, Z_cart], axis=1))

    # Make trailing edges
    XYZ_sections_complete = smooth_trailing_edges(XYZ_sections)
    
    # midspan_index = len(XYZ_sections_complete)//2
    # midspan_shape = XYZ_sections_complete[midspan_index]
    # print(midspan_shape, np.shape(midspan_shape))
    
    # r,theta,z = convert_to_cylindrical(midspan_shape)
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # fig, ax = plt.subplots()
    # ax.plot(z,np.multiply(r,theta))
    
    # inset = inset_axes(ax, width="30%", height="30%", loc="upper right")
    # inset.plot(z,np.multiply(r,theta), color="coral")
    # inset.axis("off")
    
    # ax.grid()
    # plt.show()
    
    # # Generate STL faces for mesh
    all_faces = generate_stl_faces(XYZ_sections_complete, Nblade=int(n_blades))
    
    # Import Hub as mesh
    hub_file = r"MarketDesign1 - ChrisEyeball.stl"
    hub_mesh = mesh.Mesh.from_file(hub_file)
    
    # Create the mesh object
    blade_mesh = mesh.Mesh(np.zeros(all_faces.shape[0]+hub_mesh.vectors.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(all_faces):
        blade_mesh.vectors[i] = f

    # # Translate hub to align with blade section
    diff = blade_mesh.z.min() - hub_mesh.z.min() - 0.002 # Clearance
    copy_hub_mesh = mesh.Mesh(hub_mesh.data.copy())
    copy_hub_mesh.points[:, (2,5,8)] += diff

    for j, g in enumerate(copy_hub_mesh.vectors):
        blade_mesh.vectors[i+j] = g
        
    # Save to disk
    mesh_filename = 'stl_files\\' + filename + "blade.stl"
    blade_mesh.save(mesh_filename)
    
    print("\n=============================================")
    # print('Saved stl to file')

def main():
    
    # Take params from csv
    file = 'm_dot_dh_variation.csv'
    data = pd.read_csv(file, index_col=None)
    # data = data.iloc[-5:]
    print(data.head())
    stopping = input("press enter to continue")
    for i, row in data.iterrows():
        # rows: ['mdot', 'DH_mid', 'incidence', 'Vexp', 'lean_compound', 'lean_straight', 'tip_clearance', 'n_blade']
        ### input params
        index = i
        n_blades = row['n_blade']
        rhub=0.015             # 15 mm hub radius
        rcase=0.040             # 40 mm case radius
        tip_clearance_percent = row['tip_clearance']
        tip_clearance_abs = tip_clearance_percent/100*rcase
        rtip = rcase-tip_clearance_abs
        RPM=3000.0             # 3000 rpm
        mdot=row['mdot']              # 0.05 kg/s (further reduced for better blade effect)
        DHmid=row['DH_mid']              # De Haller number = 0.9 (diffusion)
        INC=row['incidence']                # 2° incidence
        Vexp=row['Vexp']               # Vortex exponent 
        lean_max = row['lean_compound']        # Angle (degrees) of max lean
        lean_weighting = 1   # Form of lean curve (r*(1-r))**lean_weighting
        lean_straight = row['lean_straight'] # Straight lean angle (degrees)
        
        generate_fan_stl(n_blades, rhub, rtip, RPM, mdot, DHmid, INC, Vexp, lean_max, lean_weighting, lean_straight, index = 101+i)
        break

    



if __name__ == '__main__':
    main()